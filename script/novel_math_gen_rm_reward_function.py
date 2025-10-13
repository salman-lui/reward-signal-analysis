# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Debug flag - controlled by environment variable DEBUG
import os
import re
DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'

# Reward model type - controlled by environment variable REWARD_MODEL_TYPE
REWARD_MODEL_TYPE = os.environ['REWARD_MODEL_TYPE'].upper()  # ORM, PRM, PRM_STEP_AVG, PRM_STEP_LEVEL_ADVANTAGE, PRM_COT_OUTCOME, PRM_COT_OUTCOME_FORMAT, PRM_OUTCOME_FORMAT, PRM_HYBRID_STEP_AVG_FORMAT, PRM_COT_HYBRID_STEP_AVG_FORMAT, THINK_PRM_OUTCOME, RULE_BASED, INTUITOR, MAJORITY_VOTE, MAJORITY_VOTE_CHUNK, MAJORITY_50_DIFFICULT_RULE_50_EASY, or REVERSE_MAJORITY_50_EASY_RULE_50_DIFFICULT

# FIXED_REWARD flag - controlled by environment variable
FIXED_REWARD = os.environ.get('FIXED_REWARD', 'False').lower() == 'true'

from concurrent.futures import ThreadPoolExecutor
from time import sleep
from collections import Counter, defaultdict

from openai import OpenAI
import numpy as np

# GenRM setup - only initialize if using GenRM-based models (not RULE_BASED, INTUITOR, INTUITOR_50_DIFFICULT_RULE_50_EASY, MAJORITY_VOTE, MAJORITY_VOTE_CHUNK, MAJORITY_50_DIFFICULT_RULE_50_EASY, or REVERSE_MAJORITY_50_EASY_RULE_50_DIFFICULT)
if REWARD_MODEL_TYPE not in ['RULE_BASED', 'INTUITOR', 'INTUITOR_50_DIFFICULT_RULE_50_EASY', 'MAJORITY_VOTE', 'MAJORITY_VOTE_CHUNK', 'MAJORITY_50_DIFFICULT_RULE_50_EASY', 'REVERSE_MAJORITY_50_EASY_RULE_50_DIFFICULT']:
    # GenRM setup - read from environment variables
    ENDPOINT = os.environ['GENRM_ENDPOINT']
    MODEL_PATH = os.environ['GENRM_MODEL_PATH']
    API_KEY = os.environ['GENRM_API_KEY']
    MAX_RETRIES = int(os.environ['GENRM_MAX_RETRIES'])
    BASE_DELAY = int(os.environ['GENRM_BASE_DELAY'])
    MAX_WORKERS = int(os.environ['GENRM_MAX_WORKERS'])

    # Initialize OpenAI client
    client = OpenAI(api_key=API_KEY, base_url=ENDPOINT)

    # Fixed parameters for GenRM - adjust max_tokens for THINK_PRM models
    max_tokens_value = 8192 if "THINK_PRM" in REWARD_MODEL_TYPE else 4096

    common_payload = {
        "model": MODEL_PATH,
        "temperature": 0,
        "max_tokens": max_tokens_value,
        "top_p": 0.95,
    }
else:
    # Non-GenRM modes - no GenRM setup needed
    ENDPOINT = None
    MODEL_PATH = None
    API_KEY = None
    MAX_RETRIES = 0
    BASE_DELAY = 0
    MAX_WORKERS = 1  # Not used in RULE_BASED mode
    client = None
    common_payload = None

# ORM (Outcome Reward Model) Template - Direct Yes/No judgment
ORM_PROMPT = r"""Problem: {question}

Solution: {solution}

Is the answer correct (Yes/No)?"""

# PRM (Process Reward Model) Template - Step-by-step verification 
PRM_PROMPT = r"""Problem: {question}

Student Solution: {solution}

Let's verify step by step."""

# Select prompt template based on environment variable
CURRENT_PROMPT_TEMPLATE = PRM_PROMPT if REWARD_MODEL_TYPE in ['PRM', 'PRM_STEP_AVG', 'PRM_STEP_LEVEL_ADVANTAGE', 'PRM_COT_OUTCOME', 'PRM_COT_OUTCOME_FORMAT', 'PRM_OUTCOME_FORMAT', 'PRM_HYBRID_STEP_AVG_FORMAT', 'PRM_COT_HYBRID_STEP_AVG_FORMAT', 'THINK_PRM_OUTCOME'] else ORM_PROMPT


def query_qwen(prompt, common_payload):
   try:
       # For instruct models, use chat completions
       res = client.chat.completions.create(
           **common_payload,
           messages=[{"role": "user", "content": prompt}]
       )
       return res
   except Exception as e:
       print(f"Error: {e}")
       return None


def get_response(problem, solution_str):
   prompt = CURRENT_PROMPT_TEMPLATE.format(question=problem, solution=solution_str)
  
   for attempt in range(MAX_RETRIES):
       try:
           # Use query_qwen function matching production format
           res = query_qwen(prompt, common_payload)
           if res is not None:
               response = res.choices[0].message.content
              
               # Response will be shown in main debug output
              
               return response
           else:
               raise Exception("query_qwen returned None")
              
       except Exception as e:
           if attempt < MAX_RETRIES - 1:
               print("Exception: ", repr(e))
               delay = BASE_DELAY * (2**attempt)
               print(f"Retrying in {delay} seconds...")
               sleep(delay)
           else:
               print(f"Failed after {MAX_RETRIES} attempts. Error: {e}")

   raise ConnectionRefusedError(f"Failed to run the model for {prompt}!")


def compute_reward_orm(response):
   """
   For ORM: GenRM directly outputs Yes or No
   Yes -> 1.0 reward, No -> 0.0 reward
   """
   reward_score = 0.0
   try:
       response_clean = response.strip().lower()
       if response_clean == "yes":
           reward_score = 1.0
       elif response_clean == "no":
           reward_score = 0.0
       # Default: 0.0 for any other response
   except Exception as e:
       print(f"Error parsing ORM reward: {e}")
   return reward_score


def compute_reward_prm(response):
   """
   For PRM: GenRM outputs step-by-step verification ending with:
   **Verification: Is the answer correct (Yes/No)? Yes**
  
   Extract the final Yes/No from this format.
   Yes -> 1.0 reward, No -> 0.0 reward
   """
   reward_score = 0.0
   try:
       # Multiple patterns to match different verification formats
       patterns = [
           r"\*\*Verification:\s*Is the answer correct.*?\?\s*(Yes|No)\*\*",
           r"Verification:\s*Is the answer correct.*?\?\s*(Yes|No)",
           r"Is the answer correct.*?\?\s*(Yes|No)",
           r"\?\s*(Yes|No)\s*\*\*\s*$",
           r"\?\s*(Yes|No)\s*$"
       ]
      
       for pattern in patterns:
           match = re.search(pattern, response, re.IGNORECASE)
           if match:
               answer = match.group(1).strip().lower()
               if answer == "yes":
                   reward_score = 1.0
               elif answer == "no":
                   reward_score = 0.0
               break
              
   except Exception as e:
       print(f"Error parsing PRM reward: {e}")
   return reward_score


def compute_reward_prm_step_avg(response):
   """
   For PRM Step Average: Calculate reward based on step-by-step accuracy ratio.
  
   Expected format:
   Step 1: Correct
   Step 2: Incorrect 
   Step 3: Correct
   ...
   **Verification: Is the answer correct (Yes/No)? Yes**
  
   Returns: correct_steps / total_steps
   """
   reward_score = 0.0
   try:
       # Find all step patterns: "Step X: Correct/Incorrect"
       step_pattern = r"Step\s+\d+:\s*(Correct|Incorrect)"
       matches = re.findall(step_pattern, response, re.IGNORECASE)
      
       if matches:
           total_steps = len(matches)
           correct_steps = sum(1 for match in matches if match.lower() == "correct")
           reward_score = correct_steps / total_steps
          
           if DEBUG:
               print(f"Matches found: {matches}")
               print(f"Correct steps: {correct_steps}")
               print(f"Total steps: {total_steps}")
               print(f"Step analysis: {correct_steps}/{total_steps} correct steps")
       else:
           # Fallback: if no step pattern found, use original PRM logic
           reward_score = compute_reward_prm(response)
           if DEBUG:
               print("No step pattern found, falling back to original PRM logic")
              
   except Exception as e:
       print(f"Error parsing PRM step average reward: {e}")
   return reward_score


def compute_reward_prm_cot_outcome(response):
   """
   For PRM_COT_OUTCOME: GenRM outputs detailed step-by-step verification ending with:
   **Verification: Is the answer correct (Yes/No)? Yes**
  
   Extract the final Yes/No from this format.
   Yes -> 1.0 reward, No -> 0.0 reward
   """
   reward_score = 0.0
   try:
       # Multiple patterns to match different verification formats
       patterns = [
           r"\*\*Verification:\s*Is the answer correct.*?\?\s*(Yes|No)\*\*",
           r"Verification:\s*Is the answer correct.*?\?\s*(Yes|No)",
           r"Is the answer correct.*?\?\s*(Yes|No)",
           r"\?\s*(Yes|No)\s*\*\*\s*$",
           r"\?\s*(Yes|No)\s*$"
       ]
      
       for pattern in patterns:
           match = re.search(pattern, response, re.IGNORECASE)
           if match:
               answer = match.group(1).strip().lower()
               if answer == "yes":
                   reward_score = 1.0
               elif answer == "no":
                   reward_score = 0.0
               break
              
   except Exception as e:
       print(f"Error parsing PRM_COT_OUTCOME reward: {e}")
   return reward_score


def compute_reward_think_prm_outcome(response):
   """
   For THINK_PRM_OUTCOME: Look for final_verification> from the end.
   If not found, fallback to **Verification: Is the answer correct (Yes/No)? Yes/No**
   If verification correct → 1.0, if not → 0.0
   """
   reward_score = 0.0
   try:
       # Primary: Look for final_verification> pattern from the end
       pattern = r"final_verification>\s*(.*?)$"
       match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
       
       if match:
           verification_content = match.group(1).strip().lower()
           
           # Clean up verification content to handle both \boxed and \\boxed
           # Remove \boxed{} or \\boxed{} wrapper to get the inner content
           boxed_patterns = [
               r'\\\\boxed\{([^}]*)\}',  # Matches \\boxed{content}
               r'\\boxed\{([^}]*)\}',    # Matches \boxed{content}
               r'boxed\{([^}]*)\}'       # Matches boxed{content} (fallback)
           ]
           
           inner_content = verification_content
           for boxed_pattern in boxed_patterns:
               boxed_match = re.search(boxed_pattern, verification_content)
               if boxed_match:
                   inner_content = boxed_match.group(1).strip().lower()
                   break
           
           # Check if it indicates correct answer (whole word matching on inner content)
           if (re.search(r'\bcorrect\b', inner_content) or 
               "1.0" in inner_content or 
               "yes" in inner_content):
               reward_score = 1.0
           else:
               reward_score = 0.0
               
           if DEBUG:
               print(f"Found final_verification: '{verification_content}'")
               print(f"Extracted inner content: '{inner_content}' → reward: {reward_score}")
       else:
           # Fallback: Look for PRM-style verification pattern
           fallback_patterns = [
               r"\*\*Verification:\s*Is the answer correct.*?\?\s*(Yes|No)\*\*",
               r"Verification:\s*Is the answer correct.*?\?\s*(Yes|No)",
               r"Is the answer correct.*?\?\s*(Yes|No)",
               r"\?\s*(Yes|No)\s*\*\*\s*$",
               r"\?\s*(Yes|No)\s*$"
           ]
           
           for fallback_pattern in fallback_patterns:
               fallback_match = re.search(fallback_pattern, response, re.IGNORECASE)
               if fallback_match:
                   answer = fallback_match.group(1).strip().lower()
                   if answer == "yes":
                       reward_score = 1.0
                   elif answer == "no":
                       reward_score = 0.0
                   
                   if DEBUG:
                       print(f"Found fallback verification: '{fallback_match.group(0)}' → answer: '{answer}' → reward: {reward_score}")
                   break
           
           if not any(re.search(p, response, re.IGNORECASE) for p in fallback_patterns):
               if DEBUG:
                   print("No final_verification> or fallback verification pattern found")
               reward_score = 0.0
               
   except Exception as e:
       print(f"Error parsing THINK_PRM_OUTCOME reward: {e}")
   return reward_score


def format_validity(solution_str):
   """
   Check if solution follows the required format for novel setup.
   Returns 1.0 if valid, 0.0 if invalid.
   
   Novel format rules (stricter boxed requirement):
   1. Exactly ONE \boxed{} or \\boxed{} (no double boxed, no missing boxed)
   2. Exactly ONE <answer>...</answer> tag
   3. No text after </answer>
   
   The model is instructed to provide final answer in boxed format,
   so we give 0 reward if there's no boxed or multiple boxed answers.
   """
   try:
       # 1. Must have exactly ONE boxed answer (handles both \boxed and \\boxed)
       boxed_patterns = [r'\\\\boxed\{.*?\}', r'\\boxed\{.*?\}']  # \\boxed and \boxed
       total_boxed = sum(len(re.findall(pattern, solution_str)) for pattern in boxed_patterns)
       
       # Strict boxed requirement: exactly 1, no more, no less
       if total_boxed == 0:
           if DEBUG:
               print(f"❌ FORMAT INVALID: No \\boxed{{}} found (required exactly 1)")
           return 0.0
       elif total_boxed > 1:
           if DEBUG:
               print(f"❌ FORMAT INVALID: Multiple \\boxed{{}} found ({total_boxed}, required exactly 1)")
           return 0.0
       
       # 2. Must have exactly ONE <answer>...</answer> section
       answer_count = len(re.findall(r'<answer>.*?</answer>', solution_str, re.DOTALL))
       if answer_count != 1:
           if DEBUG:
               print(f"❌ FORMAT INVALID: Found {answer_count} <answer> tags (required exactly 1)")
           return 0.0
       
       # 3. No text after </answer>
       if re.search(r'</answer>\s*\S', solution_str):
           if DEBUG:
               print(f"❌ FORMAT INVALID: Text found after </answer> tag")
           return 0.0
       
       if DEBUG:
           print(f"✅ FORMAT VALID: Exactly 1 \\boxed{{}}, 1 <answer> tag, no text after </answer>")
       return 1.0
   
   except Exception as e:
       if DEBUG:
           print(f"❌ FORMAT ERROR: Exception during validation: {e}")
       return 0.0


def extract_valid_answer(solution_str):
   """
   Extract everything from beginning up to </answer>
   """
   try:
       answer_match = re.search(r'(.*</answer>)', solution_str, re.DOTALL)
       if answer_match:
           return answer_match.group(1).strip()
       else:
           return solution_str
   except Exception:
       return solution_str


def compute_reward_prm_outcome_format(solution_str, genrm_response, problem=None, valid_answer=None, ground_truth=None):
   """
   PRM_OUTCOME_FORMAT: Format validation + PRM evaluation
   """
   # Check format validity first
   if format_validity(solution_str) == 0.0:
       return 0.0  # Format penalty
   
   # If format is valid, use standard PRM evaluation
   reward_score = compute_reward_prm(genrm_response)
   
   return reward_score


def compute_reward_prm_cot_outcome_format(solution_str, genrm_response, problem=None, valid_answer=None, ground_truth=None):
   """
   PRM_COT_OUTCOME_FORMAT: Format validation + PRM COT evaluation
   """
   # Check format validity first
   if format_validity(solution_str) == 0.0:
       return 0.0  # Format penalty
   
   # If format is valid, use standard PRM COT evaluation
   reward_score = compute_reward_prm_cot_outcome(genrm_response)
   
   return reward_score


def compute_reward_prm_hybrid_step_avg_format(solution_str, genrm_response, problem=None, valid_answer=None, ground_truth=None):
   """
   PRM_HYBRID_STEP_AVG_FORMAT: Format validation + Hybrid scoring (40% step avg + 60% outcome)
   
   Flow:
   1. Format validation (same as PRM_OUTCOME_FORMAT)
   2. Parse step-by-step rewards from GenRM response
   3. Parse final outcome (Yes/No) from GenRM response
   4. Combine: 40% * step_avg + 60% * outcome
   
   Example:
   - 5 steps: 3 correct, 2 incorrect → step_avg = 3/5 = 0.6
   - Final outcome: "No" → outcome = 0.0
   - Final reward: 0.6 * 0.4 + 0.0 * 0.6 = 0.24
   
   Returns: dict with final_reward, step_avg_score, outcome_score for debug consistency
   """
   # Check format validity first
   if format_validity(solution_str) == 0.0:
       return {"final_reward": 0.0, "step_avg_score": 0.0, "outcome_score": 0.0}  # Format penalty
   
   # Parse step-by-step rewards (same logic as compute_reward_prm_step_avg)
   step_avg_score = 0.0
   try:
       # Find all step patterns: "Step X: Correct/Incorrect"
       step_pattern = r"Step\s+\d+:\s*(Correct|Incorrect)"
       matches = re.findall(step_pattern, genrm_response, re.IGNORECASE)
       
       if matches:
           total_steps = len(matches)
           correct_steps = sum(1 for match in matches if match.lower() == "correct")
           step_avg_score = correct_steps / total_steps
       else:
           # If no step pattern found, default to 0.0
           step_avg_score = 0.0
           
   except Exception as e:
       if DEBUG:
           print(f"Error parsing step average in hybrid: {e}")
       step_avg_score = 0.0
   
   # Parse final outcome (same logic as compute_reward_prm)
   outcome_score = 0.0
   try:
       # Multiple patterns to match different verification formats
       patterns = [
           r"\*\*Verification:\s*Is the answer correct.*?\?\s*(Yes|No)\*\*",
           r"Verification:\s*Is the answer correct.*?\?\s*(Yes|No)",
           r"Is the answer correct.*?\?\s*(Yes|No)",
           r"\?\s*(Yes|No)\s*\*\*\s*$",
           r"\?\s*(Yes|No)\s*$"
       ]
       
       for pattern in patterns:
           match = re.search(pattern, genrm_response, re.IGNORECASE)
           if match:
               answer = match.group(1).strip().lower()
               if answer == "yes":
                   outcome_score = 1.0
               elif answer == "no":
                   outcome_score = 0.0
               break
               
   except Exception as e:
       if DEBUG:
           print(f"Error parsing outcome in hybrid: {e}")
       outcome_score = 0.0
   
   # Hybrid scoring: 40% step average + 60% outcome
   STEP_WEIGHT = 0.4
   OUTCOME_WEIGHT = 0.6
   final_reward = step_avg_score * STEP_WEIGHT + outcome_score * OUTCOME_WEIGHT
   
   return {"final_reward": final_reward, "step_avg_score": step_avg_score, "outcome_score": outcome_score}


def compute_reward_prm_cot_hybrid_step_avg_format(solution_str, genrm_response, problem=None, valid_answer=None, ground_truth=None):
   """
   PRM_COT_HYBRID_STEP_AVG_FORMAT: Format validation + Hybrid scoring (40% step avg + 60% outcome)
   
   Same as PRM_HYBRID_STEP_AVG_FORMAT but parses PRM CoT format:
   - Step patterns: "**This step is correct.**" or "**This step is incorrect.**"
   - Outcome: "**Verification: Is the answer correct (Yes/No)? Yes/No**"
   
   Flow:
   1. Format validation (same as PRM_OUTCOME_FORMAT)
   2. Parse step-by-step rewards from GenRM CoT response
   3. Parse final outcome (Yes/No) from GenRM response
   4. Combine: 40% * step_avg + 60% * outcome
   
   Returns: dict with final_reward, step_avg_score, outcome_score for debug consistency
   """
   # Check format validity first
   if format_validity(solution_str) == 0.0:
       return {"final_reward": 0.0, "step_avg_score": 0.0, "outcome_score": 0.0}  # Format penalty
   
   # Parse step-by-step rewards (PRM CoT format)
   step_avg_score = 0.0
   try:
       # Find all CoT step patterns: "**This step is correct/incorrect.**"
       step_pattern = r"\*\*This step is (correct|incorrect)\.\*\*"
       matches = re.findall(step_pattern, genrm_response, re.IGNORECASE)
       
       if matches:
           total_steps = len(matches)
           correct_steps = sum(1 for match in matches if match.lower() == "correct")
           step_avg_score = correct_steps / total_steps
       else:
           # If no step pattern found, default to 0.0
           step_avg_score = 0.0
           
   except Exception as e:
       if DEBUG:
           print(f"Error parsing CoT step average in hybrid: {e}")
       step_avg_score = 0.0
   
   # Parse final outcome (same logic as regular PRM)
   outcome_score = 0.0
   try:
       # Multiple patterns to match different verification formats
       patterns = [
           r"\*\*Verification:\s*Is the answer correct.*?\?\s*(Yes|No)\*\*",
           r"Verification:\s*Is the answer correct.*?\?\s*(Yes|No)",
           r"Is the answer correct.*?\?\s*(Yes|No)",
           r"\?\s*(Yes|No)\s*\*\*\s*$",
           r"\?\s*(Yes|No)\s*$"
       ]
       
       for pattern in patterns:
           match = re.search(pattern, genrm_response, re.IGNORECASE)
           if match:
               answer = match.group(1).strip().lower()
               if answer == "yes":
                   outcome_score = 1.0
               elif answer == "no":
                   outcome_score = 0.0
               break
               
   except Exception as e:
       if DEBUG:
           print(f"Error parsing CoT outcome in hybrid: {e}")
       outcome_score = 0.0
   
   # Hybrid scoring: 40% step average + 60% outcome
   STEP_WEIGHT = 0.4
   OUTCOME_WEIGHT = 0.6
   final_reward = step_avg_score * STEP_WEIGHT + outcome_score * OUTCOME_WEIGHT
   
   return {"final_reward": final_reward, "step_avg_score": step_avg_score, "outcome_score": outcome_score}


def compute_reward_prm_step_wise_advantage(response):
   """
   For Step-Wise Advantage: Extract individual step rewards for step-level GRPO.
  
   Expected format:
   Step 1: Correct
   Step 2: Incorrect 
   Step 3: Correct
   ...
   **Verification: Is the answer correct (Yes/No)? Yes**
  
   Returns: dict with step_rewards and step_count (no overall score)
   """
   # Fixed reward: return hardcoded values without LLM inference
   if FIXED_REWARD:
       print("fixed reward method is used")
       return {
           "step_rewards": [0.0, 1.0, 0.0],
           "step_count": 3
       }
  
   try:
       # Find all step patterns: "Step X: Correct/Incorrect"
       step_pattern = r"Step\s+\d+:\s*(Correct|Incorrect)"
       matches = re.findall(step_pattern, response, re.IGNORECASE)
      
       if matches:
           total_steps = len(matches)
           # Create individual step rewards (1.0 for correct, 0.0 for incorrect)
           step_rewards = [1.0 if match.lower() == "correct" else 0.0 for match in matches]
          
           if DEBUG:
               print(f"Step-wise advantage - Matches found: {matches}")
               print(f"Step rewards: {step_rewards}")
               print(f"Total steps: {total_steps}")
          
           return {
               "step_rewards": step_rewards,
               "step_count": total_steps
           }
       else:
           # Fallback: if no step pattern found, treat as single step
           if DEBUG:
               print("Step-wise advantage - No step pattern found, treating as single step")
          
           return {
               "step_rewards": [0.0],  # Conservative: unparseable = incorrect
               "step_count": 1
           }
              
   except Exception as e:
       print(f"Error parsing step-wise advantage: {e}")
       return {
           "step_rewards": [0.0],
           "step_count": 1
       }


def compute_score(data_source, solution_str, ground_truth, extra_info):
   # RULE_BASED: Use math_verify for enhanced accuracy
   if REWARD_MODEL_TYPE == 'RULE_BASED':
       from verl.utils.reward_score import math_verify
       
       # Handle ground truth format: convert list to string if needed
       if isinstance(ground_truth, list) and len(ground_truth) > 0:
           processed_ground_truth = str(ground_truth[0])
       else:
           processed_ground_truth = str(ground_truth)
       
       reward_score = math_verify.compute_score(solution_str, processed_ground_truth)
       
       if DEBUG:
           print("\n" + "="*100)
           print("🔍 RULE_BASED (RLVR) DEBUG")
           print("="*100)
           print(f"📝 MODEL RESPONSE:\n{solution_str}")
           print(f"\n✂️  EXTRACTED: N/A (rule-based)")
           print(f"\n✅ VALIDITY: N/A (rule-based)")
           print(f"\n📤 PRM PROMPT: N/A (rule-based)")
           print(f"\n🤖 PRM RESPONSE: N/A (rule-based)")
           print(f"\n🎯 GROUND TRUTH: {ground_truth}")
           print(f"\n🏆 FINAL REWARD: {reward_score}")
           print("="*100 + "\n")
       
       return reward_score
   
   # Get question from extra_info
   problem = extra_info["question"]
  
   # Fixed reward: return hardcoded values without any model inference
   # ONLY when BOTH conditions are true: FIXED_REWARD=True AND REWARD_MODEL_TYPE='PRM_STEP_LEVEL_ADVANTAGE'
   if FIXED_REWARD and REWARD_MODEL_TYPE == 'PRM_STEP_LEVEL_ADVANTAGE':
       print("fixed reward method is used")
       return {
           "step_rewards": [1.0, 0.0, 1.0],
           "step_count": 3
       }
  
   # Special handling for format-validated types: send only clean solution to PRM
   valid_answer = None  # Initialize for all reward types
   if REWARD_MODEL_TYPE in ['PRM_OUTCOME_FORMAT', 'PRM_COT_OUTCOME_FORMAT', 'PRM_HYBRID_STEP_AVG_FORMAT', 'PRM_COT_HYBRID_STEP_AVG_FORMAT']:
       # First extract clean solution (up to </answer>)
       valid_answer = extract_valid_answer(solution_str)
       
       # Then validate the clean solution
       format_valid = format_validity(valid_answer)
       if format_valid == 0.0:
           if DEBUG:
               validity_status = "INVALID"
               prompt_to_prm = "N/A (format invalid)"
               prm_response = "N/A (format invalid)"
               final_reward = 0.0
               debug_title = f"🔍 {REWARD_MODEL_TYPE} DEBUG"
               
               print("\n" + "="*100)
               print(debug_title)
               print("="*100)
               print(f"📝 MODEL RESPONSE:\n{solution_str}")
               print(f"\n✂️  EXTRACTED:\n{valid_answer}")
               print(f"\n✅ VALIDITY: {validity_status}")
               print(f"\n📤 PRM PROMPT:\n{prompt_to_prm}")
               print(f"\n🤖 PRM RESPONSE:\n{prm_response}")
               print(f"\n🎯 GROUND TRUTH: {ground_truth}")
               print(f"\n🏆 FINAL REWARD: {final_reward}")
               print("="*100 + "\n")
           return {"score": 0.0, "ground_truth": ground_truth, "reward_method": REWARD_MODEL_TYPE}  # Format penalty
       
       # Send clean solution to PRM
       response = get_response(problem, valid_answer)
   else:
       # For all other reward types, send full solution
       response = get_response(problem, solution_str)
  
   if response is not None:
       # Use appropriate reward function based on model type
       if REWARD_MODEL_TYPE == 'PRM':
           reward_score = compute_reward_prm(response)
       elif REWARD_MODEL_TYPE == 'PRM_STEP_AVG':
           reward_score = compute_reward_prm_step_avg(response)
       elif REWARD_MODEL_TYPE == 'PRM_STEP_LEVEL_ADVANTAGE':
           reward_score = compute_reward_prm_step_wise_advantage(response)
       elif REWARD_MODEL_TYPE == 'PRM_COT_OUTCOME':
           reward_score = compute_reward_prm_cot_outcome(response)
       elif REWARD_MODEL_TYPE == 'PRM_COT_OUTCOME_FORMAT':
           reward_score = compute_reward_prm_cot_outcome_format(solution_str, response, problem, valid_answer, ground_truth)
           
           # Clean debug output for valid responses
           if DEBUG:
               validity_status = "VALID"
               prompt_to_prm = PRM_PROMPT.format(question=problem, solution=valid_answer)
               
               print("\n" + "="*100)
               print("🔍 PRM_COT_OUTCOME_FORMAT DEBUG")
               print("="*100)
               print(f"📝 MODEL RESPONSE:\n{solution_str}")
               print(f"\n✂️  EXTRACTED:\n{valid_answer}")
               print(f"\n✅ VALIDITY: {validity_status}")
               print(f"\n📤 PRM PROMPT:\n{prompt_to_prm}")
               print(f"\n🤖 PRM RESPONSE:\n{response}")
               print(f"\n🎯 GROUND TRUTH: {ground_truth}")
               print(f"\n🏆 FINAL REWARD: {reward_score}")
               print("="*100 + "\n")
       elif REWARD_MODEL_TYPE == 'PRM_OUTCOME_FORMAT':
           reward_score = compute_reward_prm_outcome_format(solution_str, response, problem, valid_answer, ground_truth)
           
           # Clean debug output for valid responses
           if DEBUG:
               validity_status = "VALID"
               prompt_to_prm = PRM_PROMPT.format(question=problem, solution=valid_answer)
               
               print("\n" + "="*100)
               print("🔍 PRM_OUTCOME_FORMAT DEBUG")
               print("="*100)
               print(f"📝 MODEL RESPONSE:\n{solution_str}")
               print(f"\n✂️  EXTRACTED:\n{valid_answer}")
               print(f"\n✅ VALIDITY: {validity_status}")
               print(f"\n📤 PRM PROMPT:\n{prompt_to_prm}")
               print(f"\n🤖 PRM RESPONSE:\n{response}")
               print(f"\n🎯 GROUND TRUTH: {ground_truth}")
               print(f"\n🏆 FINAL REWARD: {reward_score}")
               print("="*100 + "\n")
       elif REWARD_MODEL_TYPE == 'PRM_HYBRID_STEP_AVG_FORMAT':
           hybrid_result = compute_reward_prm_hybrid_step_avg_format(solution_str, response, problem, valid_answer, ground_truth)
           reward_score = hybrid_result["final_reward"]
           
           # Clean debug output for valid responses with hybrid breakdown using ACTUAL computed values
           if DEBUG:
               validity_status = "VALID"
               prompt_to_prm = PRM_PROMPT.format(question=problem, solution=valid_answer)
               
               # Use the EXACT same values that were computed in the function
               step_avg_score = hybrid_result["step_avg_score"]
               outcome_score = hybrid_result["outcome_score"]
               
               print("\n" + "="*100)
               print("🔍 PRM_HYBRID_STEP_AVG_FORMAT DEBUG")
               print("="*100)
               print(f"📝 MODEL RESPONSE:\n{solution_str}")
               print(f"\n✂️  EXTRACTED:\n{valid_answer}")
               print(f"\n✅ VALIDITY: {validity_status}")
               print(f"\n📤 PRM PROMPT:\n{prompt_to_prm}")
               print(f"\n🤖 PRM RESPONSE:\n{response}")
               print(f"\n🎯 GROUND TRUTH: {ground_truth}")
               print(f"\n📊 HYBRID BREAKDOWN:")
               print(f"    Step Average: {step_avg_score:.3f} (40% weight)")
               print(f"    Outcome Score: {outcome_score:.3f} (60% weight)")
               print(f"    Formula: {step_avg_score:.3f} × 0.4 + {outcome_score:.3f} × 0.6 = {reward_score:.3f}")
               print(f"\n🏆 FINAL REWARD: {reward_score}")
               print("="*100 + "\n")
       elif REWARD_MODEL_TYPE == 'PRM_COT_HYBRID_STEP_AVG_FORMAT':
           hybrid_result = compute_reward_prm_cot_hybrid_step_avg_format(solution_str, response, problem, valid_answer, ground_truth)
           reward_score = hybrid_result["final_reward"]
           
           # Clean debug output for valid responses with CoT hybrid breakdown using ACTUAL computed values
           if DEBUG:
               validity_status = "VALID"
               prompt_to_prm = PRM_PROMPT.format(question=problem, solution=valid_answer)
               
               # Use the EXACT same values that were computed in the function
               step_avg_score = hybrid_result["step_avg_score"]
               outcome_score = hybrid_result["outcome_score"]
               
               print("\n" + "="*100)
               print("🔍 PRM_COT_HYBRID_STEP_AVG_FORMAT DEBUG")
               print("="*100)
               print(f"📝 MODEL RESPONSE:\n{solution_str}")
               print(f"\n✂️  EXTRACTED:\n{valid_answer}")
               print(f"\n✅ VALIDITY: {validity_status}")
               print(f"\n📤 PRM PROMPT:\n{prompt_to_prm}")
               print(f"\n🤖 PRM RESPONSE:\n{response}")
               print(f"\n🎯 GROUND TRUTH: {ground_truth}")
               print(f"\n📊 COT HYBRID BREAKDOWN:")
               print(f"    Step Average: {step_avg_score:.3f} (40% weight)")
               print(f"    Outcome Score: {outcome_score:.3f} (60% weight)")
               print(f"    Formula: {step_avg_score:.3f} × 0.4 + {outcome_score:.3f} × 0.6 = {reward_score:.3f}")
               print(f"\n🏆 FINAL REWARD: {reward_score}")
               print("="*100 + "\n")
       elif REWARD_MODEL_TYPE == 'THINK_PRM_OUTCOME':
           reward_score = compute_reward_think_prm_outcome(response)
       else:
           reward_score = compute_reward_orm(response)
   else:
       if REWARD_MODEL_TYPE == 'PRM_STEP_LEVEL_ADVANTAGE':
           reward_score = {"step_rewards": [0.0], "step_count": 1}
       else:
           reward_score = 0.0

   # Debug output for non-format-validated types (keep existing behavior)
   if DEBUG and REWARD_MODEL_TYPE not in ['PRM_OUTCOME_FORMAT', 'PRM_COT_OUTCOME_FORMAT', 'PRM_HYBRID_STEP_AVG_FORMAT', 'PRM_COT_HYBRID_STEP_AVG_FORMAT']:
       model_type = REWARD_MODEL_TYPE
       print("=" * 80)
       print(f"{model_type} REWARD:")
       print("-" * 40)
       if isinstance(reward_score, dict):
           print(f"Step rewards: {reward_score.get('step_rewards', [])}")
           print(f"Step count: {reward_score.get('step_count', 0)}")
       else:
           print(f"Final reward score: {reward_score}")
       print("=" * 80)

   # Return dict with score, ground truth, and reward method for automatic saving
   result = {
       "score": reward_score,
       "ground_truth": ground_truth,
       "reward_method": REWARD_MODEL_TYPE
   }
   
   # Debug: Check if we're accidentally returning a non-dict
   if not isinstance(result, dict):
       print(f"ERROR: compute_score returning non-dict: {result} (type: {type(result)})")
   
   return result


def compute_rule_based_score(solution_str, ground_truth):
   """
   Use math_verify for enhanced mathematical verification accuracy.
   Handles ground truth format conversion (list -> string).
   """
   try:
       from verl.utils.reward_score import math_verify
       
       # Handle ground truth format: convert list to string if needed
       if isinstance(ground_truth, list) and len(ground_truth) > 0:
           processed_ground_truth = str(ground_truth[0])
       else:
           processed_ground_truth = str(ground_truth)
       
       score = math_verify.compute_score(solution_str, processed_ground_truth)
       
       if DEBUG:
           print("\n" + "="*100)
           print("🔍 RULE_BASED COMPUTE DEBUG")
           print("="*100)
           print(f"📝 MODEL RESPONSE:\n{solution_str}")
           print(f"\n✂️  EXTRACTED: N/A (rule-based)")
           print(f"\n✅ VALIDITY: N/A (rule-based)")
           print(f"\n📤 PRM PROMPT: N/A (rule-based)")
           print(f"\n🤖 PRM RESPONSE: N/A (rule-based)")
           print(f"\n🎯 GROUND TRUTH: {ground_truth}")
           print(f"\n🏆 FINAL REWARD: {score}")
           print("="*100 + "\n")
       
       return score
   except Exception as e:
       if DEBUG:
           print(f"Error in math_verify scoring: {e}")
       return 0.0


def compute_validation_scores(data_sources, solution_strs, ground_truths, extra_infos, reward_method_name="VALIDATION"):
    """
    Common validation scoring function using math_verify.
    Used by all reward model types for validation data.
    
    Args:
        data_sources: List of data sources
        solution_strs: List of solution strings
        ground_truths: List of ground truths
        extra_infos: List of extra info dicts
        reward_method_name: Name to use in reward_method field
        
    Returns:
        List of result dicts with score, ground_truth, and reward_method
    """
    from verl.utils.reward_score import math_verify
    
    results = []
    for data_source, solution_str, ground_truth, extra_info in zip(
        data_sources, solution_strs, ground_truths, extra_infos, strict=True
    ):
        # For validation, use original ground truth if available (for MAJORITY_VOTE)
        if "MAJORITY_VOTE_VALIDATION" in reward_method_name and 'original_ground_truth' in extra_info:
            processed_gt = extra_info['original_ground_truth']
        else:
            processed_gt = ground_truth
            
        # Handle ground truth format: convert list to string if needed
        if isinstance(processed_gt, list) and len(processed_gt) > 0:
            processed_ground_truth = str(processed_gt[0])
        else:
            processed_ground_truth = str(processed_gt)
        
        score = math_verify.compute_score(solution_str, processed_ground_truth)
        
        # Enhanced return with both ground truths for MAJORITY_VOTE
        if "MAJORITY_VOTE" in reward_method_name:
            results.append({
                "score": score, 
                "majority_vote_gt": ground_truth,                    # Original majority vote GT
                "original_gt": processed_gt,                         # Original dataset GT (used for validation)
                "ground_truth": processed_gt,                        # Keep for backward compatibility
                "reward_method": reward_method_name
            })
        else:
            results.append({
                "score": score, 
                "ground_truth": processed_gt,
                "reward_method": reward_method_name
            })
    
    if DEBUG and results:
        correct_count = sum(1 for r in results if r["score"] > 0.5)
        total_count = len(results)
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        print(f"\n📊 {reward_method_name} SUMMARY: {correct_count}/{total_count} = {accuracy:.3f} accuracy\n")
    
    return results


def is_validation_data(data_sources):
    """
    Check if the batch contains validation data based on data source patterns.
    
    Args:
        data_sources: List of data sources to check
        
    Returns:
        bool: True if this is a validation batch, False otherwise
    """
    # validation_patterns = ['test-math-aime24', 'test-math-aime25', 'huggingfaceh4/math-500', 'test-math-']
    validation_patterns = ['test-math-aime24', 'test-math-aime25', 'huggingfaceh4/math-500', 'test-math-', 'test-amc']
    return any(
        any(pattern in str(data_source).lower() for pattern in validation_patterns)
        for data_source in data_sources
    )


def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos):
   print(f"DEBUG: compute_score_batch called with {len(data_sources)} items, REWARD_MODEL_TYPE={REWARD_MODEL_TYPE}")
   
   # MAJORITY_VOTE: TTRL-style majority voting rewards
   if REWARD_MODEL_TYPE == 'MAJORITY_VOTE':
       if is_validation_data(data_sources):
           # Validation: Use original ground truth (no majority voting)
           return compute_validation_scores(data_sources, solution_strs, ground_truths, extra_infos, "MAJORITY_VOTE_VALIDATION")
       else:
           # Training: Use majority vote ground truth (already replaced by TTRL)
           from verl.utils.reward_score import math_verify
           
           results = []
           for data_source, solution_str, ground_truth, extra_info in zip(
               data_sources, solution_strs, ground_truths, extra_infos, strict=True
           ):
               # Handle ground truth format (majority vote is already set as ground_truth)
               if isinstance(ground_truth, list) and len(ground_truth) > 0:
                   processed_ground_truth = str(ground_truth[0])
               else:
                   processed_ground_truth = str(ground_truth)
               
               # Use math_verify to compare solution against majority vote ground truth
               score = math_verify.compute_score(solution_str, processed_ground_truth)
               
               # Get original_gt from extra_info (now available in updated training data)
               original_gt = extra_info.get('original_ground_truth', None)
               
               # Handle ground truth format: convert list to string if needed
               if isinstance(original_gt, list) and len(original_gt) > 0:
                   original_gt = str(original_gt[0])
               elif original_gt is not None:
                   original_gt = str(original_gt)
               
               results.append({
                   "score": score, 
                   "majority_vote_gt": ground_truth,                    # Majority vote ground truth (clear naming)
                   "original_gt": original_gt,                          # Original dataset ground truth
                   "ground_truth": ground_truth,                        # Keep for backward compatibility
                   "reward_method": "MAJORITY_VOTE"
               })
           
           if DEBUG and results:
               correct_count = sum(1 for r in results if r["score"] > 0.5)
               total_count = len(results)
               accuracy = correct_count / total_count if total_count > 0 else 0.0
               
               # Show comparison between majority vote and original GT
               original_gts = [r.get('original_gt') for r in results if r.get('original_gt') is not None]
               majority_gts = [r.get('majority_vote_gt') for r in results]
               
               if original_gts and len(original_gts) == len(majority_gts):
                   matches = sum(1 for orig, maj in zip(original_gts, majority_gts) if str(orig) == str(maj))
                   agreement = matches / len(original_gts) if original_gts else 0.0
                   print(f"\n📊 MAJORITY_VOTE TRAINING SUMMARY:")
                   print(f"   Accuracy against majority vote GT: {correct_count}/{total_count} = {accuracy:.3f}")
                   print(f"   Majority vote ↔ Original GT agreement: {matches}/{len(original_gts)} = {agreement:.3f}")
               else:
                   print(f"\n📊 MAJORITY_VOTE TRAINING SUMMARY: {correct_count}/{total_count} = {accuracy:.3f} accuracy against majority vote GT")
               print()
           
           return results
   
   # MAJORITY_VOTE_CHUNK: TTRL-style majority voting with chunked reward processing for memory efficiency
   if REWARD_MODEL_TYPE == 'MAJORITY_VOTE_CHUNK':
       if is_validation_data(data_sources):
           # Validation: Use original ground truth (no majority voting)
           return compute_validation_scores(data_sources, solution_strs, ground_truths, extra_infos, "MAJORITY_VOTE_CHUNK_VALIDATION")
       else:
           # Training: Use majority vote ground truth with chunked processing to prevent OOM
           from verl.utils.reward_score import math_verify
           
           # Process in chunks to avoid memory explosion (4096 → 2 chunks of 2048)
           chunk_size = 2048
           all_results = []
           total_items = len(data_sources)
           
           if DEBUG:
               print(f"\n🔄 MAJORITY_VOTE_CHUNK: Processing {total_items} items in chunks of {chunk_size}")
           
           for chunk_start in range(0, total_items, chunk_size):
               chunk_end = min(chunk_start + chunk_size, total_items)
               chunk_results = []
               
               if DEBUG:
                   print(f"   Processing chunk {chunk_start//chunk_size + 1}/{(total_items + chunk_size - 1)//chunk_size}: items {chunk_start}-{chunk_end-1}")
               
               for i in range(chunk_start, chunk_end):
                   data_source = data_sources[i]
                   solution_str = solution_strs[i]
                   ground_truth = ground_truths[i]
                   extra_info = extra_infos[i]
                   
                   # Handle ground truth format (majority vote is already set as ground_truth)
                   if isinstance(ground_truth, list) and len(ground_truth) > 0:
                       processed_ground_truth = str(ground_truth[0])
                   else:
                       processed_ground_truth = str(ground_truth)
                   
                   # Use math_verify to compare solution against majority vote ground truth
                   score = math_verify.compute_score(solution_str, processed_ground_truth)
                   
                   # Get original_gt from extra_info (now available in updated training data)
                   original_gt = extra_info.get('original_ground_truth', None)
                   
                   # Handle ground truth format: convert list to string if needed
                   if isinstance(original_gt, list) and len(original_gt) > 0:
                       original_gt = str(original_gt[0])
                   elif original_gt is not None:
                       original_gt = str(original_gt)
                   
                   chunk_results.append({
                       "score": score, 
                       "majority_vote_gt": ground_truth,                    # Majority vote ground truth (clear naming)
                       "original_gt": original_gt,                          # Original dataset ground truth
                       "ground_truth": ground_truth,                        # Keep for backward compatibility
                       "reward_method": "MAJORITY_VOTE_CHUNK"
                   })
               
               all_results.extend(chunk_results)
               
               if DEBUG:
                   chunk_correct = sum(1 for r in chunk_results if r["score"] > 0.5)
                   chunk_total = len(chunk_results)
                   chunk_accuracy = chunk_correct / chunk_total if chunk_total > 0 else 0.0
                   print(f"   Chunk accuracy: {chunk_correct}/{chunk_total} = {chunk_accuracy:.3f}")
           
           if DEBUG and all_results:
               correct_count = sum(1 for r in all_results if r["score"] > 0.5)
               total_count = len(all_results)
               accuracy = correct_count / total_count if total_count > 0 else 0.0
               
               # Show comparison between majority vote and original GT
               original_gts = [r.get('original_gt') for r in all_results if r.get('original_gt') is not None]
               majority_gts = [r.get('majority_vote_gt') for r in all_results]
               
               if original_gts and len(original_gts) == len(majority_gts):
                   matches = sum(1 for orig, maj in zip(original_gts, majority_gts) if str(orig) == str(maj))
                   agreement = matches / len(original_gts) if original_gts else 0.0
                   print(f"\n📊 MAJORITY_VOTE_CHUNK TRAINING SUMMARY:")
                   print(f"   Total accuracy against majority vote GT: {correct_count}/{total_count} = {accuracy:.3f}")
                   print(f"   Majority vote ↔ Original GT agreement: {matches}/{len(original_gts)} = {agreement:.3f}")
                   print(f"   Memory optimization: Processed {total_count} items in {(total_count + chunk_size - 1)//chunk_size} chunks of {chunk_size}")
               else:
                   print(f"\n📊 MAJORITY_VOTE_CHUNK TRAINING SUMMARY: {correct_count}/{total_count} = {accuracy:.3f} accuracy against majority vote GT")
               print()
           
           return all_results
   
   # MAJORITY_50_DIFFICULT_RULE_50_EASY: Hybrid reward based on problem difficulty
   if REWARD_MODEL_TYPE == 'MAJORITY_50_DIFFICULT_RULE_50_EASY':
       if is_validation_data(data_sources):
           # Validation: Always use rule-based scoring
           return compute_validation_scores(data_sources, solution_strs, ground_truths, extra_infos, "MAJORITY_RULE_HYBRID_VALIDATION")
       else:
           # Training: Hybrid reward based on difficulty
           from verl.utils.reward_score import math_verify
           results = []
           majority_vote_count = 0
           rule_based_count = 0
           
           for data_source, solution_str, ground_truth, extra_info in zip(
               data_sources, solution_strs, ground_truths, extra_infos, strict=True
           ):
               # Extract difficulty for DeepSeek-R1-Distill-Qwen-1.5B
               try:
                   model_difficulty = extra_info.get('model_difficulty', {})
                   if not isinstance(model_difficulty, dict):
                       raise ValueError(f"model_difficulty is not a dict: {type(model_difficulty)} = {model_difficulty}")
                   
                   if 'DeepSeek-R1-Distill-Qwen-1.5B' not in model_difficulty:
                       raise KeyError(f"DeepSeek-R1-Distill-Qwen-1.5B not found in model_difficulty keys: {list(model_difficulty.keys())}")
                   
                   difficulty = model_difficulty['DeepSeek-R1-Distill-Qwen-1.5B']
                   if not isinstance(difficulty, (int, float)):
                       raise TypeError(f"Difficulty is not a number: {type(difficulty)} = {difficulty}")
                       
               except Exception as e:
                   print(f"ERROR: Failed to extract difficulty from extra_info: {e}")
                   print(f"extra_info: {extra_info}")
                   raise RuntimeError(f"Cannot determine difficulty for hybrid reward: {e}") from e
               
               if difficulty <= 9:
                   # Easy problems (0-9): Use rule-based reward (original ground truth)
                   original_gt = extra_info.get('original_ground_truth', None)
                   if isinstance(original_gt, list) and len(original_gt) > 0:
                       processed_original_gt = str(original_gt[0])
                   elif original_gt is not None:
                       processed_original_gt = str(original_gt)
                   else:
                       # Fallback to ground_truth if original_ground_truth not available
                       processed_original_gt = str(ground_truth)
                   
                   score = math_verify.compute_score(solution_str, processed_original_gt)
                   results.append({
                       "score": score,
                       "majority_vote_gt": ground_truth,                    # Majority vote ground truth (for reference)
                       "original_gt": original_gt,                          # Original dataset ground truth (used for scoring)
                       "ground_truth": processed_original_gt,               # Used ground truth for scoring
                       "reward_method": "RULE_BASED"
                   })
                   rule_based_count += 1
               else:
                   # Hard problems (10-16): Use majority vote reward
                   # Handle ground truth format (majority vote is already set as ground_truth)
                   if isinstance(ground_truth, list) and len(ground_truth) > 0:
                       processed_ground_truth = str(ground_truth[0])
                   else:
                       processed_ground_truth = str(ground_truth)
                   
                   # Use math_verify to compare solution against majority vote ground truth
                   score = math_verify.compute_score(solution_str, processed_ground_truth)
                   
                   # Get original_gt from extra_info
                   original_gt = extra_info.get('original_ground_truth', None)
                   if isinstance(original_gt, list) and len(original_gt) > 0:
                       original_gt = str(original_gt[0])
                   elif original_gt is not None:
                       original_gt = str(original_gt)
                   
                   results.append({
                       "score": score,
                       "majority_vote_gt": ground_truth,                    # Majority vote ground truth
                       "original_gt": original_gt,                          # Original dataset ground truth
                       "ground_truth": ground_truth,                        # Keep for backward compatibility
                       "reward_method": "MAJORITY_VOTE"
                   })
                   majority_vote_count += 1
           
           if DEBUG and results:
               print(f"\n📊 MAJORITY_50_DIFFICULT_RULE_50_EASY TRAINING SUMMARY:")
               print(f"   Rule-based (easy, 0-9): {rule_based_count} examples")
               print(f"   Majority vote (difficult, 10-16): {majority_vote_count} examples")
               print(f"   Total: {len(results)} examples\n")
           
           return results
   
   # REVERSE_MAJORITY_50_EASY_RULE_50_DIFFICULT: Reverse hybrid reward based on problem difficulty
   if REWARD_MODEL_TYPE == 'REVERSE_MAJORITY_50_EASY_RULE_50_DIFFICULT':
       if is_validation_data(data_sources):
           # Validation: Always use rule-based scoring
           return compute_validation_scores(data_sources, solution_strs, ground_truths, extra_infos, "MAJORITY_RULE_REVERSE_HYBRID_VALIDATION")
       else:
           # Training: Reverse hybrid reward based on difficulty
           from verl.utils.reward_score import math_verify
           results = []
           majority_vote_count = 0
           rule_based_count = 0
           
           for data_source, solution_str, ground_truth, extra_info in zip(
               data_sources, solution_strs, ground_truths, extra_infos, strict=True
           ):
               # Extract difficulty for DeepSeek-R1-Distill-Qwen-1.5B
               try:
                   model_difficulty = extra_info.get('model_difficulty', {})
                   if not isinstance(model_difficulty, dict):
                       raise ValueError(f"model_difficulty is not a dict: {type(model_difficulty)} = {model_difficulty}")
                   
                   if 'DeepSeek-R1-Distill-Qwen-1.5B' not in model_difficulty:
                       raise KeyError(f"DeepSeek-R1-Distill-Qwen-1.5B not found in model_difficulty keys: {list(model_difficulty.keys())}")
                   
                   difficulty = model_difficulty['DeepSeek-R1-Distill-Qwen-1.5B']
                   if not isinstance(difficulty, (int, float)):
                       raise TypeError(f"Difficulty is not a number: {type(difficulty)} = {difficulty}")
                       
               except Exception as e:
                   print(f"ERROR: Failed to extract difficulty from extra_info: {e}")
                   print(f"extra_info: {extra_info}")
                   raise RuntimeError(f"Cannot determine difficulty for reverse hybrid reward: {e}") from e
               
               if difficulty <= 9:
                   # Easy problems (0-9): Use majority vote reward
                   if isinstance(ground_truth, list) and len(ground_truth) > 0:
                       processed_ground_truth = str(ground_truth[0])
                   else:
                       processed_ground_truth = str(ground_truth)
                   
                   score = math_verify.compute_score(solution_str, processed_ground_truth)
                   
                   # Get original_gt from extra_info
                   original_gt = extra_info.get('original_ground_truth', None)
                   if isinstance(original_gt, list) and len(original_gt) > 0:
                       original_gt = str(original_gt[0])
                   elif original_gt is not None:
                       original_gt = str(original_gt)
                   
                   results.append({
                       "score": score,
                       "majority_vote_gt": ground_truth,                    # Majority vote ground truth
                       "original_gt": original_gt,                          # Original dataset ground truth
                       "ground_truth": ground_truth,                        # Keep for backward compatibility
                       "reward_method": "MAJORITY_VOTE"
                   })
                   majority_vote_count += 1
               else:
                   # Hard problems (10-16): Use rule-based reward (original ground truth)
                   original_gt = extra_info.get('original_ground_truth', None)
                   if isinstance(original_gt, list) and len(original_gt) > 0:
                       processed_original_gt = str(original_gt[0])
                   elif original_gt is not None:
                       processed_original_gt = str(original_gt)
                   else:
                       # Fallback to ground_truth if original_ground_truth not available
                       processed_original_gt = str(ground_truth)
                   
                   score = math_verify.compute_score(solution_str, processed_original_gt)
                   results.append({
                       "score": score,
                       "majority_vote_gt": ground_truth,                    # Majority vote ground truth (for reference)
                       "original_gt": original_gt,                          # Original dataset ground truth (used for scoring)
                       "ground_truth": processed_original_gt,               # Used ground truth for scoring
                       "reward_method": "RULE_BASED"
                   })
                   rule_based_count += 1
           
           if DEBUG and results:
               print(f"\n📊 REVERSE_MAJORITY_50_EASY_RULE_50_DIFFICULT TRAINING SUMMARY:")
               print(f"   Majority vote (easy, 0-9): {majority_vote_count} examples")
               print(f"   Rule-based (difficult, 10-16): {rule_based_count} examples")
               print(f"   Total: {len(results)} examples\n")
           
           return results
   
   # INTUITOR_50_DIFFICULT_RULE_50_EASY: Hybrid reward based on problem difficulty
   if REWARD_MODEL_TYPE == 'INTUITOR_50_DIFFICULT_RULE_50_EASY':
       if is_validation_data(data_sources):
           # Validation: Always use rule-based scoring
           return compute_validation_scores(data_sources, solution_strs, ground_truths, extra_infos, "INTUITOR_VALIDATION")
       else:
           # Training: Hybrid reward based on difficulty
           from verl.utils.reward_score import math_verify
           results = []
           rule_based_count = 0
           self_certainty_count = 0
           
           for data_source, solution_str, ground_truth, extra_info in zip(
               data_sources, solution_strs, ground_truths, extra_infos, strict=True
           ):
               # Extract difficulty for DeepSeek-R1-Distill-Qwen-1.5B
               try:
                   model_difficulty = extra_info.get('model_difficulty', {})
                   if not isinstance(model_difficulty, dict):
                       raise ValueError(f"model_difficulty is not a dict: {type(model_difficulty)} = {model_difficulty}")
                   
                   if 'DeepSeek-R1-Distill-Qwen-1.5B' not in model_difficulty:
                       raise KeyError(f"DeepSeek-R1-Distill-Qwen-1.5B not found in model_difficulty keys: {list(model_difficulty.keys())}")
                   
                   difficulty = model_difficulty['DeepSeek-R1-Distill-Qwen-1.5B']
                   if not isinstance(difficulty, (int, float)):
                       raise TypeError(f"Difficulty is not a number: {type(difficulty)} = {difficulty}")
                       
               except Exception as e:
                   print(f"ERROR: Failed to extract difficulty from extra_info: {e}")
                   print(f"extra_info: {extra_info}")
                   raise RuntimeError(f"Cannot determine difficulty for hybrid reward: {e}") from e
               
               if difficulty <= 9:
                   # Easy problems (0-9): Use rule-based reward
                   if isinstance(ground_truth, list) and len(ground_truth) > 0:
                       processed_ground_truth = str(ground_truth[0])
                   else:
                       processed_ground_truth = str(ground_truth)
                   
                   score = math_verify.compute_score(solution_str, processed_ground_truth)
                   results.append({
                       "score": score,
                       "ground_truth": ground_truth,
                       "reward_method": "RULE_BASED"
                   })
                   rule_based_count += 1
               else:
                   # Hard problems (10-16): Use self-certainty reward
                   results.append({
                       "score": 0.0,  # Placeholder - actual self-certainty applied in ray_trainer.py
                       "ground_truth": ground_truth,
                       "reward_method": "SELF_CERTAINTY"
                   })
                   self_certainty_count += 1
           
           if DEBUG and results:
               print(f"\n📊 INTUITOR_50_DIFFICULT_RULE_50_EASY TRAINING SUMMARY:")
               print(f"   Rule-based (easy, 0-9): {rule_based_count} examples")
               print(f"   Self-certainty (difficult, 10-16): {self_certainty_count} examples")
               print(f"   Total: {len(results)} examples\n")
           
           return results
   
   # INTUITOR: Smart reward handling for self-certainty training
   elif REWARD_MODEL_TYPE == 'INTUITOR':
       if is_validation_data(data_sources):
           # Validation: Use rule-based scoring
           return compute_validation_scores(data_sources, solution_strs, ground_truths, extra_infos, "INTUITOR_VALIDATION")
       else:
           # Training: Mark for self-certainty processing
           results = []
           for data_source, solution_str, ground_truth, extra_info in zip(
               data_sources, solution_strs, ground_truths, extra_infos, strict=True
           ):
               results.append({
                   "score": 0.0,  # Placeholder - actual self-certainty applied in ray_trainer.py
                   "ground_truth": ground_truth,
                   "reward_method": "SELF_CERTAINTY"
               })
           
           if DEBUG and results:
               print(f"\n📊 INTUITOR TRAINING SUMMARY: {len(results)} examples marked for self-certainty\n")
           
           return results
   
   # RULE_BASED: Force math_verify scoring for ALL data (training + validation)
   elif REWARD_MODEL_TYPE == 'RULE_BASED':
       return compute_validation_scores(data_sources, solution_strs, ground_truths, extra_infos, "RULE_BASED")
   
   # AUTO-DETECT: Validation vs Training (for GenRM modes only)
   # Validation data always uses rule-based scoring via default_compute_score
   if is_validation_data(data_sources):
       # For validation: ALWAYS use math_verify for enhanced accuracy (regardless of reward model type)
       return compute_validation_scores(data_sources, solution_strs, ground_truths, extra_infos, "GENRM_VALIDATION")
   else:
       # For training: use full generative model pipeline
      
       with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
           futures = []
           for data_source, solution_str, ground_truth, extra_info in zip(
               data_sources, solution_strs, ground_truths, extra_infos, strict=True
           ):
               future = executor.submit(compute_score, data_source, solution_str, ground_truth, extra_info)
               futures.append(future)

           results = []
           for i, future in enumerate(futures):
               try:
                   result = future.result()
                   if not isinstance(result, dict):
                       print(f"ERROR: Future {i} returned non-dict: {result} (type: {type(result)})")
                   results.append(result)
               except Exception as e:
                   print(f"Reward computation failed for future {i}: {e}")
                   # This should never happen since compute_score always returns a dict now
                   # But just in case, return a fallback
                   fallback = {"score": 0.0, "ground_truth": None}
                   print(f"Using fallback for future {i}: {fallback}")
                   results.append(fallback)

       return results