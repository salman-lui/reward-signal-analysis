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

import os
import re
DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'

if DEBUG:
    print(f"REWARD_MODEL_TYPE: {os.environ.get('REWARD_MODEL_TYPE', 'NOT_SET')}")

REWARD_MODEL_TYPE = os.environ['REWARD_MODEL_TYPE'].upper()
FIXED_REWARD = os.environ.get('FIXED_REWARD', 'False').lower() == 'true'

from concurrent.futures import ThreadPoolExecutor
from time import sleep

from openai import OpenAI





def print_highlight(response):
    if DEBUG:
        print("API Response:")
        print("-" * 50)
        if hasattr(response, 'choices') and len(response.choices) > 0:
            content = response.choices[0].message.content
            print(content)
        else:
            print(f"Raw response: {response}")
        print("-" * 50)

if REWARD_MODEL_TYPE not in ['RULE_BASED', 'RANDOM_REWARD']:
    ENDPOINT = os.environ['GENRM_ENDPOINT']
    MODEL_PATH = os.environ['GENRM_MODEL_PATH']
    API_KEY = os.environ['GENRM_API_KEY']
    MAX_RETRIES = int(os.environ['GENRM_MAX_RETRIES'])
    BASE_DELAY = int(os.environ['GENRM_BASE_DELAY'])
    MAX_WORKERS = int(os.environ['GENRM_MAX_WORKERS'])

    if DEBUG:
        DEBUG_PORT = os.environ.get('DEBUG_SGLANG_PORT', '1073')
        DEBUG_MODEL = os.environ.get('DEBUG_SGLANG_MODEL', '/local2/salman/model/step_cot')
        
        client = OpenAI(base_url=f"http://127.0.0.1:{DEBUG_PORT}/v1", api_key="None")
        common_payload = {
            "model": DEBUG_MODEL,
            "temperature": 0,
            "max_tokens": 4096,
        }
    else:
        client = OpenAI(api_key=API_KEY, base_url=ENDPOINT)
        max_tokens_value = 8192 if "THINK_PRM" in REWARD_MODEL_TYPE else 4096
        common_payload = {
            "model": MODEL_PATH,
            "temperature": 0,
            "max_tokens": max_tokens_value,
            "top_p": 0.95,
        }
else:
    ENDPOINT = None
    MODEL_PATH = None
    API_KEY = None
    MAX_RETRIES = 0
    BASE_DELAY = 0
    MAX_WORKERS = 1
    client = None
    common_payload = None

ORM_PROMPT = r"""Problem: {question}

Solution: {solution}

Is the answer correct (Yes/No)?"""

PRM_PROMPT = r"""Problem: {question}

Student Solution: {solution}

Let's verify step by step."""

CURRENT_PROMPT_TEMPLATE = PRM_PROMPT if REWARD_MODEL_TYPE in ['PRM', 'PRM_STEP_AVG', 'PRM_STEP_LEVEL_ADVANTAGE', 'PRM_COT_OUTCOME', 'PRM_COT_OUTCOME_FORMAT', 'PRM_OUTCOME_FORMAT', 'PRM_HYBRID_STEP_AVG_FORMAT', 'PRM_COT_HYBRID_STEP_AVG_FORMAT', 'PRM_COT_HYBRID_APPLY_STEP_GLOBAL_STAT_TANGO', 'THINK_PRM_OUTCOME'] else ORM_PROMPT


def query_qwen(prompt, common_payload):
   try:
       res = client.chat.completions.create(
           **common_payload,
           messages=[{"role": "user", "content": prompt}]
       )
       return res
   except Exception as e:
       if DEBUG:
           print(f"API Error: {e}")
       return None


def get_response(problem, solution_str):
   prompt = CURRENT_PROMPT_TEMPLATE.format(question=problem, solution=solution_str)
  
   for attempt in range(MAX_RETRIES):
       try:
           res = query_qwen(prompt, common_payload)
           if res is not None:
               response = res.choices[0].message.content
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
   reward_score = 0.0
   try:
       response_clean = response.strip().lower()
       if response_clean == "yes":
           reward_score = 1.0
       elif response_clean == "no":
           reward_score = 0.0
   except Exception as e:
       print(f"Error parsing ORM reward: {e}")
   return reward_score


def compute_reward_prm(response):
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
           reward_score = compute_reward_prm(response)
           if DEBUG:
               print("No step pattern found, falling back to original PRM logic")
              
   except Exception as e:
       print(f"Error parsing PRM step average reward: {e}")
   return reward_score


def compute_reward_prm_cot_outcome(response):
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
   reward_score = 0.0
   try:
       # Primary: Look for final_verification> pattern from the end
       pattern = r"final_verification>\s*(.*?)$"
       match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
       
       if match:
           verification_content = match.group(1).strip().lower()
           
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
           
           if (re.search(r'\bcorrect\b', inner_content) or 
               "1.0" in inner_content or 
               "yes" in inner_content):
               reward_score = 1.0
           else:
               reward_score = 0.0
               
           if DEBUG:
               print(f"Found final_verification: '{verification_content}'")
               print(f"Extracted inner content: '{inner_content}' -> reward: {reward_score}")
       else:
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
                       print(f"Found fallback verification: '{fallback_match.group(0)}' -> answer: '{answer}' -> reward: {reward_score}")
                   break
           
           if not any(re.search(p, response, re.IGNORECASE) for p in fallback_patterns):
               if DEBUG:
                   print("No final_verification> or fallback verification pattern found")
               reward_score = 0.0
               
   except Exception as e:
       print(f"Error parsing THINK_PRM_OUTCOME reward: {e}")
   return reward_score


def format_validity(solution_str):
   try:
       boxed_patterns = [r'\\\\boxed\{.*?\}', r'\\boxed\{.*?\}']
       total_boxed = sum(len(re.findall(pattern, solution_str)) for pattern in boxed_patterns)
       if total_boxed != 1:
           return 0.0
       
       answer_count = len(re.findall(r'<answer>.*?</answer>', solution_str, re.DOTALL))
       if answer_count != 1:
           return 0.0
       
       if re.search(r'</answer>\s*\S', solution_str):
           return 0.0
       
       return 1.0
   
   except Exception as e:
       return 0.0


def extract_valid_answer(solution_str):
   try:
       answer_match = re.search(r'(.*</answer>)', solution_str, re.DOTALL)
       if answer_match:
           return answer_match.group(1).strip()
       else:
           return solution_str
   except Exception:
       return solution_str


def compute_random_reward_score(solution_str, ground_truth):
   import random
   
   try:
       if format_validity(solution_str) == 0.0:
           if DEBUG:
               print("RANDOM_REWARD: Format invalid -> 0.0 reward")
           return 0.0
       
       random_score = random.choice([0.0, 1.0])
       
       if DEBUG:
           print(f"RANDOM_REWARD: Format valid -> Random score: {random_score}")
           print(f"   Solution format: VALID")
           print(f"   Ground truth: {ground_truth} (ignored for random scoring)")
           print(f"   Random result: {random_score}")
       
       return random_score
       
   except Exception as e:
       if DEBUG:
           print(f"RANDOM_REWARD: Error during processing: {e}")
       return 0.0


def compute_reward_orm_format(solution_str, genrm_response, problem=None, valid_answer=None, ground_truth=None):
   if format_validity(solution_str) == 0.0:
       return 0.0
   
   reward_score = compute_reward_orm(genrm_response)
   
   return reward_score


def compute_reward_prm_outcome_format(solution_str, genrm_response, problem=None, valid_answer=None, ground_truth=None):
   if format_validity(solution_str) == 0.0:
       return 0.0
   
   reward_score = compute_reward_prm(genrm_response)
   
   return reward_score


def compute_reward_prm_cot_outcome_format(solution_str, genrm_response, problem=None, valid_answer=None, ground_truth=None):
   if format_validity(solution_str) == 0.0:
       return 0.0
   
   reward_score = compute_reward_prm_cot_outcome(genrm_response)
   
   return reward_score


def compute_reward_prm_hybrid_step_avg_format(solution_str, genrm_response, problem=None, valid_answer=None, ground_truth=None):
   if format_validity(solution_str) == 0.0:
       return {"final_reward": 0.0, "step_avg_score": 0.0, "outcome_score": 0.0}
   
   step_avg_score = 0.0
   try:
       step_pattern = r"Step\s+\d+:\s*(Correct|Incorrect)"
       matches = re.findall(step_pattern, genrm_response, re.IGNORECASE)
       
       if matches:
           total_steps = len(matches)
           correct_steps = sum(1 for match in matches if match.lower() == "correct")
           step_avg_score = correct_steps / total_steps
       else:
           step_avg_score = 0.0
           
   except Exception as e:
       if DEBUG:
           print(f"Error parsing step average in hybrid: {e}")
       step_avg_score = 0.0
   
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
   
   STEP_WEIGHT = 0.4
   OUTCOME_WEIGHT = 0.6
   final_reward = step_avg_score * STEP_WEIGHT + outcome_score * OUTCOME_WEIGHT
   
   return {"final_reward": final_reward, "step_avg_score": step_avg_score, "outcome_score": outcome_score}


def compute_reward_prm_cot_hybrid_step_avg_format(solution_str, genrm_response, problem=None, valid_answer=None, ground_truth=None):
   if format_validity(solution_str) == 0.0:
       return {"final_reward": 0.0, "step_avg_score": 0.0, "outcome_score": 0.0}
   
   step_avg_score = 0.0
   try:
       step_pattern = r"\*\*This step is (correct|incorrect)\.\*\*"
       matches = re.findall(step_pattern, genrm_response, re.IGNORECASE)
       
       if matches:
           total_steps = len(matches)
           correct_steps = sum(1 for match in matches if match.lower() == "correct")
           step_avg_score = correct_steps / total_steps
       else:
           step_avg_score = 0.0
           
   except Exception as e:
       if DEBUG:
           print(f"Error parsing CoT step average in hybrid: {e}")
       step_avg_score = 0.0
   
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
   
   STEP_WEIGHT = 0.4
   OUTCOME_WEIGHT = 0.6
   final_reward = step_avg_score * STEP_WEIGHT + outcome_score * OUTCOME_WEIGHT
   
   return {"final_reward": final_reward, "step_avg_score": step_avg_score, "outcome_score": outcome_score}


def compute_reward_prm_cot_hybrid_apply_step_global_stat_tango(solution_str, genrm_response, problem=None, valid_answer=None, ground_truth=None):
   if format_validity(solution_str) == 0.0:
       return {
           "optional_outcome_and_avg_reward": 0.0,  # Consistent with line 676
           "step_avg_score": 0.0, 
           "outcome_score": 0.0,
           "step_count": 0,
           "step_wise_rewards": {},
           "step_rewards": [],
           "step_positions": [],
           "step_boundary_strategy": "none"
       }
   
   step_positions = []
   try:
       if not solution_str:
           if DEBUG:
               print("Empty solution text, no step positions found")
           step_positions = []
       else:
           step_patterns = [
                # Prioritize boundary detection for unclosed tags
                r'<step[^>]*>(.*?)(?=<step[^>]*>|</think>|</answer>|$)',
                r'<Step[^>]*>(.*?)(?=<Step[^>]*>|</think>|</answer>|$)',
                r'<STEP[^>]*>(.*?)(?=<STEP[^>]*>|</think>|</answer>|$)',
                r'<step>(.*?)</step>',
                r'<step[^>]*>(.*?)</step>',
                r'<Step>(.*?)</Step>',
                r'<STEP>(.*?)</STEP>',
                r'<(?:step|Step|STEP)[^>]*>(.*?)(?=<(?:step|Step|STEP)[^>]*>|</(?:step|Step|STEP)>|</think>|</answer>|$)',
                r'<step[^>]*>',
                r'<Step[^>]*>',
                r'<STEP[^>]*>',
                r'\*\*[^*]+\*\*:?\s*\n',
                r'##?\s*Step\s*\d+',
                r'Step\s*\d+:',
            ]
           
           for pattern in step_patterns:
               matches = list(re.finditer(pattern, solution_str, re.DOTALL | re.IGNORECASE))
               if matches:
                   for match in matches:
                       # Use tag end position for accurate step boundary placement
                       tag_start = match.start()  # Start of <step> tag
                       tag_end = match.end()      # End of </step> tag
                       
                       if 0 <= tag_start < len(solution_str) and 0 <= tag_end <= len(solution_str):
                           step_positions.append((tag_start, tag_end))
                           if DEBUG:
                               step_text = solution_str[tag_start:min(tag_start+20, len(solution_str))]
                               print(f"REWARD FUNC STEP {len(step_positions)-1}: chars({tag_start},{tag_end}) -> '{step_text}...'")
                   break
           
   except Exception as e:
       if DEBUG:
           print(f"Error extracting step positions: {e}")
       step_positions = []
   
   step_avg_score = 0.0
   step_count = 0
   step_wise_rewards = {}
   step_rewards = []
   penalty_applied = False
   
   try:
       step_pattern = r"\*\*This step is (correct|incorrect)\.\*\*"
       matches = re.findall(step_pattern, genrm_response, re.IGNORECASE)
       
       raw_step_rewards = []
       if matches:
           for match in matches:
               if match.lower() == "correct":
                   raw_step_rewards.append(1.0)
               else:
                   raw_step_rewards.append(-1.0)
       
       if DEBUG:
           print(f"PRM RESPONSE DEBUG:")
           print(f"   PRM response length: {len(genrm_response)} chars")
           print(f"   PRM response preview: {repr(genrm_response[:500])}...")
           print(f"   Pattern used: {step_pattern}")
           print(f"   Matches found: {matches}")
           print(f"   Number of matches: {len(matches)}")
           print(f"   Raw step rewards (original PRM): {raw_step_rewards}")
       
       if matches:
           model_step_count = len(step_positions)
           prm_step_count = len(matches)
           
           IS_STEP_PENALTY = REWARD_MODEL_TYPE == 'PRM_COT_HYBRID_APPLY_STEP_GLOBAL_STAT_TANGO_STEP_PENALTY'
           IS_HALF_DISCOUNT_STRICT = REWARD_MODEL_TYPE == 'PRM_COT_HYBRID_OUTCOME_DIFF_ADV_HALF_DISCOUNT'
           IS_REGULAR_STRICT = REWARD_MODEL_TYPE == 'PRM_COT_HYBRID_OUTCOME_DIFF_ADV'
           
           if DEBUG:
               print(f"STEP ANALYSIS: Model={model_step_count}, PRM={prm_step_count}, IS_STEP_PENALTY={IS_STEP_PENALTY}, IS_HALF_DISCOUNT_STRICT={IS_HALF_DISCOUNT_STRICT}, IS_REGULAR_STRICT={IS_REGULAR_STRICT}")
               print(f"PENALTY CONDITIONS CHECK:")
               print(f"   - Count mismatch: {model_step_count} != {prm_step_count} = {model_step_count != prm_step_count}")
               if IS_STEP_PENALTY:
                   print(f"   - Too few steps (STEP_PENALTY): {model_step_count} < 3 = {model_step_count < 3}")
               if IS_HALF_DISCOUNT_STRICT:
                   print(f"   - Too few steps (HALF_DISCOUNT_STRICT): {model_step_count} < 4 = {model_step_count < 4}")
               if IS_REGULAR_STRICT:
                   print(f"   - Too few steps (REGULAR_STRICT): {model_step_count} < 4 = {model_step_count < 4}")
               print(f"   - Too many steps: {model_step_count} > 20 = {model_step_count > 20}")
           
           penalty_applied = False
           penalty_reason = ""
           strict_penalty_applied = False
           count_mismatch_penalty_applied = False
           
           if IS_STEP_PENALTY:
               if model_step_count != prm_step_count:
                   penalty_applied = True
                   penalty_reason = f"count mismatch (model={model_step_count}, prm={prm_step_count})"
               
               elif model_step_count < 3:
                   penalty_applied = True
                   penalty_reason = f"too few steps ({model_step_count} < 3)"
               
               elif model_step_count > 20:
                   penalty_applied = True
                   penalty_reason = f"too many steps ({model_step_count} > 20)"
               
               if penalty_applied:
                   step_count = model_step_count
                   step_wise_rewards = {}
                   step_rewards = [0.0] * step_count
                   step_avg_score = 0.0
                   
                   if DEBUG:
                       print(f"STEP PENALTY APPLIED: All step rewards = 0.0 due to {penalty_reason}")
                       print(f"   -> Model steps: {model_step_count}, PRM steps: {prm_step_count}")
                       print(f"   -> Penalty step_rewards: {step_rewards}")
           
           elif IS_HALF_DISCOUNT_STRICT or IS_REGULAR_STRICT:
               if model_step_count < 4:
                   strict_penalty_applied = True
                   penalty_reason = f"too few steps ({model_step_count} < 4)"
               
               elif model_step_count > 20:
                   strict_penalty_applied = True
                   penalty_reason = f"too many steps ({model_step_count} > 20)"
               
               if strict_penalty_applied:
                   step_count = model_step_count
                   step_wise_rewards = {}
                   step_rewards = [0.0] * step_count
                   step_avg_score = 0.0
                   
                   variant_name = "HALF_DISCOUNT" if IS_HALF_DISCOUNT_STRICT else "REGULAR"
                   if DEBUG:
                       print(f"STRICT PENALTY APPLIED ({variant_name}): Both outcome and step rewards = 0.0 due to {penalty_reason}")
                       print(f"   -> Model steps: {model_step_count}, PRM steps: {prm_step_count}")
                       print(f"   -> Penalty step_rewards: {step_rewards}")
                       print(f"   -> Outcome will also be set to 0.0")
               
               elif IS_HALF_DISCOUNT_STRICT and model_step_count != prm_step_count:
                   count_mismatch_penalty_applied = True
                   step_count = model_step_count
                   step_wise_rewards = {}
                   step_rewards = [0.0] * step_count
                   step_avg_score = 0.0
                   penalty_reason = f"count mismatch (model={model_step_count}, prm={prm_step_count})"
                   
                   if DEBUG:
                       print(f"COUNT MISMATCH PENALTY (HALF_DISCOUNT): Only step rewards = 0.0 due to {penalty_reason}")
                       print(f"   -> Penalty step_rewards: {step_rewards}")
                       print(f"   -> Outcome will be parsed normally")
               
               elif IS_REGULAR_STRICT and model_step_count != prm_step_count:
                   count_mismatch_penalty_applied = True
                   step_count = model_step_count
                   step_wise_rewards = {}
                   step_rewards = [0.0] * step_count
                   step_avg_score = 0.0
                   penalty_reason = f"count mismatch (model={model_step_count}, prm={prm_step_count})"
                   
                   if DEBUG:
                       print(f"COUNT MISMATCH PENALTY (REGULAR): Only step rewards = 0.0 due to {penalty_reason}")
                       print(f"   -> Penalty step_rewards: {step_rewards}")
                       print(f"   -> Outcome will be parsed normally")
           
           # Handle step count assignment for non-penalty cases (both STEP_PENALTY and regular variants)
           if not penalty_applied and not strict_penalty_applied and not count_mismatch_penalty_applied and prm_step_count < model_step_count:
               missing_count = model_step_count - prm_step_count
               matches.extend(["incorrect"] * missing_count)
               step_count = model_step_count
               if DEBUG:
                   print(f"   -> PATH: Padding {missing_count} missing PRM evaluations with 'incorrect', step_count={step_count}")
                   
           elif not penalty_applied and not strict_penalty_applied and not count_mismatch_penalty_applied and prm_step_count > model_step_count:
               matches = matches[:model_step_count]
               step_count = model_step_count
               if DEBUG:
                   print(f"   -> PATH: Truncating {prm_step_count - model_step_count} extra PRM evaluations, step_count={step_count}")
                   
           elif not penalty_applied and not strict_penalty_applied and not count_mismatch_penalty_applied:
               step_count = len(matches)
               if DEBUG:
                   print(f"   -> PATH: Perfect match, step_count={step_count}")
           else:
               if DEBUG:
                   print(f"   -> PATH: UNHANDLED CASE! penalty_applied={penalty_applied}, strict_penalty_applied={strict_penalty_applied}, count_mismatch_penalty_applied={count_mismatch_penalty_applied}, model={model_step_count}, prm={prm_step_count}")
               step_count = model_step_count
           
           if not penalty_applied and not strict_penalty_applied and not count_mismatch_penalty_applied:
               correct_steps = 0
               
               if DEBUG:
                   print(f"SAFETY CHECK: step_count={step_count}, matches={len(matches)}, penalty_applied={penalty_applied}, strict_penalty_applied={strict_penalty_applied}, count_mismatch_penalty_applied={count_mismatch_penalty_applied}")
               
               if step_count == 0:
                   if DEBUG:
                       print(f"CRITICAL ERROR: step_count is 0 but no penalty applied! matches={len(matches)}, model_steps={model_step_count}, prm_steps={prm_step_count}")
                   step_count = max(len(matches), 1)
               
               for i, match in enumerate(matches, 1):
                   step_reward = (1.0 if match.lower() == "correct" else -1.0) / step_count
                   step_wise_rewards[f"step_{i}"] = step_reward
                   step_rewards.append(step_reward)
                   if match.lower() == "correct":
                       correct_steps += 1
               
               step_avg_score = correct_steps / step_count
           else:
               pass
           
           if DEBUG:
               print(f"Step-wise rewards: {step_wise_rewards}")
               print(f"Step rewards list: {step_rewards}")
               if not penalty_applied and not strict_penalty_applied and not count_mismatch_penalty_applied:
                   print(f"Step average: {correct_steps}/{step_count} = {step_avg_score}")
               elif penalty_applied:
                   print(f"Step average: 0/{step_count} = {step_avg_score} (step penalty applied)")
               elif strict_penalty_applied:
                   print(f"Step average: 0/{step_count} = {step_avg_score} (strict penalty applied)")
               elif count_mismatch_penalty_applied:
                   print(f"Step average: 0/{step_count} = {step_avg_score} (count mismatch penalty applied)")
       else:
           step_avg_score = 0.0
           step_count = 0
           step_wise_rewards = {}
           step_rewards = []
           penalty_applied = False
           strict_penalty_applied = False
           count_mismatch_penalty_applied = False
           penalty_reason = ""
           
   except Exception as e:
       if DEBUG:
           print(f"Error parsing CoT step-wise rewards: {e}")
       step_avg_score = 0.0
       step_count = 0
       step_wise_rewards = {}
       step_rewards = []
       raw_step_rewards = []
       penalty_applied = False
       strict_penalty_applied = False
       count_mismatch_penalty_applied = False
       penalty_reason = ""
   
   outcome_score = 0.0
   try:
       if strict_penalty_applied:
           outcome_score = 0.0
           if DEBUG:
               print(f"STRICT PENALTY: Outcome score forced to 0.0 due to {penalty_reason}")
       else:
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
           print(f"Error parsing CoT outcome in step-wise hybrid: {e}")
       outcome_score = 0.0
   
   STEP_WEIGHT = 0.4
   OUTCOME_WEIGHT = 0.6
   final_reward = step_avg_score * STEP_WEIGHT + outcome_score * OUTCOME_WEIGHT
   
   if DEBUG:
       print(f"TANGO_GRPO FINAL SCORES:")
       print(f"   step_count: {step_count}")
       print(f"   step_rewards: {step_rewards}")
       print(f"   outcome_score: {outcome_score:.3f}")
       print(f"   step_avg_score: {step_avg_score:.3f}")
       if strict_penalty_applied:
           print(f"   STRICT PENALTY: Both outcome and step = 0.0 due to {penalty_reason}")
           print(f"   hybrid_score (STRICT PENALTY): {final_reward:.3f}")
       elif penalty_applied:
           print(f"   STEP PENALTY: Only step rewards = 0.0 due to {penalty_reason}")
           print(f"   hybrid_score (40% step + 60% outcome): {final_reward:.3f}")
       elif count_mismatch_penalty_applied:
           print(f"   COUNT MISMATCH PENALTY: Only step rewards = 0.0 due to {penalty_reason}")
           print(f"   hybrid_score (40% step + 60% outcome): {final_reward:.3f}")
       else:
           print(f"   hybrid_score (40% step + 60% outcome): {final_reward:.3f}")
   
   return {
       "optional_outcome_and_avg_reward": final_reward,  # Not used in TANGO_GRPO - just for reference
       "step_avg_score": step_avg_score, 
       "outcome_score": outcome_score,
       "step_count": step_count,
       "step_wise_rewards": step_wise_rewards,
       "step_rewards": step_rewards,  # [1.0, 0.0, 1.0] format for tensor creation
       "step_positions": step_positions,  # [(start, end), ...] character positions (RL Tango approach)
       "step_boundary_strategy": "content_based" if step_count > 0 else "none",
       "model_step_count": len(step_positions),
       "prm_step_count": len(matches) if matches else 0,
       "step_penalty_applied": "Yes" if penalty_applied else "No",
       "strict_penalty_applied": "Yes" if strict_penalty_applied else "No",
       "count_mismatch_penalty_applied": "Yes" if count_mismatch_penalty_applied else "No",
       "penalty_reason": penalty_reason if (penalty_applied or strict_penalty_applied or count_mismatch_penalty_applied) else "",
       "raw_step_rewards": raw_step_rewards  # Raw PRM rewards: [1.0, -1.0, 1.0] for PRM_COT_HYBRID_OUTCOME_DIFF_ADV
   }




def compute_reward_prm_step_wise_advantage(response):
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
           if DEBUG:
               print("Step-wise advantage - No step pattern found, treating as single step")
          
           return {
               "step_rewards": [0.0],
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
       
       return reward_score
   
   # RANDOM_REWARD: Format validation + Random 50/50 scoring
   if REWARD_MODEL_TYPE == 'RANDOM_REWARD':
       # Handle ground truth format: convert list to string if needed
       if isinstance(ground_truth, list) and len(ground_truth) > 0:
           processed_ground_truth = str(ground_truth[0])
       else:
           processed_ground_truth = str(ground_truth)
       
       reward_score = compute_random_reward_score(solution_str, processed_ground_truth)
       
       if DEBUG:
           print("\n" + "="*100)
           print("🎲 RANDOM_REWARD DEBUG")
           print("="*100)
           print(f"📝 MODEL RESPONSE:\n{solution_str}")
           print(f"\n✂️  EXTRACTED: N/A (random scoring)")
           print(f"\n✅ VALIDITY: {'VALID' if format_validity(solution_str) == 1.0 else 'INVALID'}")
           print(f"\n📤 PRM PROMPT: N/A (random scoring)")
           print(f"\n🤖 PRM RESPONSE: N/A (random scoring)")
           print(f"\n🎯 GROUND TRUTH: {processed_ground_truth} (ignored)")
           print(f"\n🏆 FINAL REWARD: {reward_score} (random)")
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
  
   # Special handling for format-validated types: send only clean solution to PRM/ORM
   valid_answer = None  # Initialize for all reward types
   if REWARD_MODEL_TYPE in ['ORM_FORMAT', 'PRM_OUTCOME_FORMAT', 'PRM_COT_OUTCOME_FORMAT', 'PRM_HYBRID_STEP_AVG_FORMAT', 'PRM_COT_HYBRID_STEP_AVG_FORMAT', 'PRM_COT_HYBRID_APPLY_STEP_GLOBAL_STAT_TANGO', 'PRM_COT_HYBRID_APPLY_STEP_GLOBAL_STAT_TANGO_STEP_PENALTY', 'PRM_COT_HYBRID_OUTCOME_DIFF_ADV', 'PRM_COT_HYBRID_OUTCOME_DIFF_ADV_HALF_DISCOUNT']:
       # First extract clean solution (up to </answer>)
       valid_answer = extract_valid_answer(solution_str)
       
       # Then validate the clean solution
       format_valid = format_validity(valid_answer)
       if format_valid == 0.0:
           # For TANGO, return full dict structure even on format failure
           if REWARD_MODEL_TYPE in ['PRM_COT_HYBRID_APPLY_STEP_GLOBAL_STAT_TANGO', 'PRM_COT_HYBRID_APPLY_STEP_GLOBAL_STAT_TANGO_STEP_PENALTY', 'PRM_COT_HYBRID_OUTCOME_DIFF_ADV', 'PRM_COT_HYBRID_OUTCOME_DIFF_ADV_HALF_DISCOUNT']:
               result = {
                   "score": 0.0,
                   "ground_truth": ground_truth,
                   "step_rewards": [],
                   "step_count": 0,
                   "step_boundary_strategy": "none",
                   "step_positions": [],
                   "outcome_score": 0.0,
                   "step_avg_score": 0.0,
                   "hybrid_score": 0.0,  # Add missing hybrid_score for batch consistency
                   "prm_response": "N/A",  # Add prm_response for consistency
                   "reward_method": REWARD_MODEL_TYPE
               }
               # Add raw_step_rewards for PRM_COT_HYBRID_OUTCOME_DIFF_ADV variants
               if REWARD_MODEL_TYPE in ['PRM_COT_HYBRID_OUTCOME_DIFF_ADV', 'PRM_COT_HYBRID_OUTCOME_DIFF_ADV_HALF_DISCOUNT']:
                   result["raw_step_rewards"] = []
               return result
           else:
               return {"score": 0.0, "ground_truth": ground_truth, "reward_method": REWARD_MODEL_TYPE}  # Format penalty
       
       # Send clean solution to PRM/ORM
       response = get_response(problem, valid_answer)
   else:
       # For all other reward types, send full solution
       response = get_response(problem, solution_str)
  
   if response is not None:
       # Use appropriate reward function based on model type
       if REWARD_MODEL_TYPE == 'ORM':
           reward_score = compute_reward_orm(response)
       elif REWARD_MODEL_TYPE == 'ORM_FORMAT':
           reward_score = compute_reward_orm_format(solution_str, response, problem, valid_answer, ground_truth)
       elif REWARD_MODEL_TYPE == 'PRM':
           reward_score = compute_reward_prm(response)
       elif REWARD_MODEL_TYPE == 'PRM_STEP_AVG':
           reward_score = compute_reward_prm_step_avg(response)
       elif REWARD_MODEL_TYPE == 'PRM_STEP_LEVEL_ADVANTAGE':
           reward_score = compute_reward_prm_step_wise_advantage(response)
       elif REWARD_MODEL_TYPE == 'PRM_COT_OUTCOME':
           reward_score = compute_reward_prm_cot_outcome(response)
       elif REWARD_MODEL_TYPE == 'PRM_COT_OUTCOME_FORMAT':
           reward_score = compute_reward_prm_cot_outcome_format(solution_str, response, problem, valid_answer, ground_truth)
           
       elif REWARD_MODEL_TYPE == 'PRM_OUTCOME_FORMAT':
           reward_score = compute_reward_prm_outcome_format(solution_str, response, problem, valid_answer, ground_truth)
           
       elif REWARD_MODEL_TYPE == 'PRM_HYBRID_STEP_AVG_FORMAT':
           hybrid_result = compute_reward_prm_hybrid_step_avg_format(solution_str, response, problem, valid_answer, ground_truth)
           reward_score = hybrid_result["final_reward"]
           
       elif REWARD_MODEL_TYPE == 'PRM_COT_HYBRID_STEP_AVG_FORMAT':
           hybrid_result = compute_reward_prm_cot_hybrid_step_avg_format(solution_str, response, problem, valid_answer, ground_truth)
           reward_score = hybrid_result["final_reward"]
           
       elif REWARD_MODEL_TYPE in ['PRM_COT_HYBRID_APPLY_STEP_GLOBAL_STAT_TANGO', 'PRM_COT_HYBRID_APPLY_STEP_GLOBAL_STAT_TANGO_STEP_PENALTY', 'PRM_COT_HYBRID_OUTCOME_DIFF_ADV', 'PRM_COT_HYBRID_OUTCOME_DIFF_ADV_HALF_DISCOUNT']:
           hybrid_result = compute_reward_prm_cot_hybrid_apply_step_global_stat_tango(solution_str, response, problem, valid_answer, ground_truth)
           reward_score = hybrid_result  # Keep the full dict for step reward extraction
           
       elif REWARD_MODEL_TYPE == 'THINK_PRM_OUTCOME':
           reward_score = compute_reward_think_prm_outcome(response)
       else:
           # Default fallback for any unhandled reward types
           reward_score = compute_reward_orm(response)
   else:
       # API call failed - return fallback with reward_method
       if REWARD_MODEL_TYPE == 'PRM_STEP_LEVEL_ADVANTAGE':
           reward_score = {"step_rewards": [0.0], "step_count": 1}
       elif REWARD_MODEL_TYPE in ['PRM_COT_HYBRID_APPLY_STEP_GLOBAL_STAT_TANGO', 'PRM_COT_HYBRID_APPLY_STEP_GLOBAL_STAT_TANGO_STEP_PENALTY', 'PRM_COT_HYBRID_OUTCOME_DIFF_ADV', 'PRM_COT_HYBRID_OUTCOME_DIFF_ADV_HALF_DISCOUNT']:
           # For TANGO, return full dict structure even on API failure
           reward_score = {
               "optional_outcome_and_avg_reward": 0.0,
               "step_avg_score": 0.0, 
               "outcome_score": 0.0,
               "step_count": 0,
               "step_wise_rewards": {},
               "step_rewards": [],
               "step_positions": [],  # Empty but present
               "step_boundary_strategy": "none",
               "raw_step_rewards": []  # Add for PRM_COT_HYBRID_OUTCOME_DIFF_ADV variants
           }
       else:
           reward_score = 0.0


   result = {
       "score": reward_score,
       "ground_truth": ground_truth,
       "reward_method": REWARD_MODEL_TYPE
   }
   
   if "reward_method" not in result:
       print(f"CRITICAL ERROR: reward_method missing from result dict: {result}")
       result["reward_method"] = "EMERGENCY_FALLBACK"
   
   if REWARD_MODEL_TYPE in ['PRM_COT_HYBRID_APPLY_STEP_GLOBAL_STAT_TANGO', 'PRM_COT_HYBRID_APPLY_STEP_GLOBAL_STAT_TANGO_STEP_PENALTY', 'PRM_COT_HYBRID_OUTCOME_DIFF_ADV', 'PRM_COT_HYBRID_OUTCOME_DIFF_ADV_HALF_DISCOUNT'] and isinstance(reward_score, dict):
       pure_outcome_score = reward_score.get("outcome_score", 0.0)
       step_rewards = reward_score.get("step_rewards", [])
       step_count = reward_score.get("step_count", 0)
       step_boundary_strategy = reward_score.get("step_boundary_strategy", "none")
       
       result = {
           "score": pure_outcome_score,  # Use pure outcome score (1.0 or 0.0) for token-level rewards
           "ground_truth": ground_truth,
           "step_rewards": step_rewards,
           "step_count": step_count,
           "step_boundary_strategy": step_boundary_strategy,
           "step_positions": reward_score.get("step_positions", []),  # Add step positions (RL Tango)
           "prm_response": response if response else "N/A",  # Add PRM response for debug
           "reward_method": REWARD_MODEL_TYPE,
       }
       # Add raw_step_rewards for PRM_COT_HYBRID_OUTCOME_DIFF_ADV variants
       if REWARD_MODEL_TYPE in ['PRM_COT_HYBRID_OUTCOME_DIFF_ADV', 'PRM_COT_HYBRID_OUTCOME_DIFF_ADV_HALF_DISCOUNT']:
           result["raw_step_rewards"] = reward_score.get("raw_step_rewards", [])
   elif isinstance(reward_score, dict):
       # Handle other dict-type rewards (like PRM_STEP_LEVEL_ADVANTAGE)
       result["reward_method"] = REWARD_MODEL_TYPE
       for key, value in reward_score.items():
           if key != "score":
               result[key] = value
   else:
       # Handle scalar rewards (including API failures)
       result["reward_method"] = REWARD_MODEL_TYPE
   
   # Debug: Check if we're accidentally returning a non-dict
   if not isinstance(result, dict):
       print(f"ERROR: compute_score returning non-dict: {result} (type: {type(result)})")
   
   return result


def compute_rule_based_score(solution_str, ground_truth):
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


def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos):
   if DEBUG:
       print(f"BATCH INPUT DEBUG: data_sources={len(data_sources)}, solution_strs={len(solution_strs)}, ground_truths={len(ground_truths)}, extra_infos={len(extra_infos)}")
   if DEBUG and REWARD_MODEL_TYPE == 'PRM_COT_HYBRID_APPLY_STEP_GLOBAL_STAT_TANGO':
       print(f"BATCH DEBUG: Processing {len(solution_strs)} responses for TANGO_GRPO")
   
   # RULE_BASED: Force math_verify scoring for ALL data (training + validation)
   if REWARD_MODEL_TYPE == 'RULE_BASED':
       from verl.utils.reward_score import math_verify
       
       results = []
       for data_source, solution_str, ground_truth, extra_info in zip(
           data_sources, solution_strs, ground_truths, extra_infos, strict=True
       ):
           # Handle ground truth format: convert list to string if needed
           if isinstance(ground_truth, list) and len(ground_truth) > 0:
               processed_ground_truth = str(ground_truth[0])
           else:
               processed_ground_truth = str(ground_truth)
           
           score = math_verify.compute_score(solution_str, processed_ground_truth)
           results.append({"score": score, "ground_truth": ground_truth, "reward_method": "RULE_BASED"})
       
       return results
   
   # RANDOM_REWARD: Format validation + Random 50/50 scoring for TRAINING, RULE_BASED for VALIDATION
   if REWARD_MODEL_TYPE == 'RANDOM_REWARD':
       # Check if this is validation data
       validation_patterns = ['test-math-aime24', 'test-math-aime25', 'huggingfaceh4/math-500', 'test-math-']
       is_validation_batch = any(
           any(pattern in str(data_source).lower() for pattern in validation_patterns)
           for data_source in data_sources
       )
       
       if is_validation_batch:
           # For validation: Use RULE_BASED scoring for meaningful metrics
           from verl.utils.reward_score import math_verify
           
           results = []
           for data_source, solution_str, ground_truth, extra_info in zip(
               data_sources, solution_strs, ground_truths, extra_infos, strict=True
           ):
               # Handle ground truth format: convert list to string if needed
               if isinstance(ground_truth, list) and len(ground_truth) > 0:
                   processed_ground_truth = str(ground_truth[0])
               else:
                   processed_ground_truth = str(ground_truth)
               
               score = math_verify.compute_score(solution_str, processed_ground_truth)
               results.append({"score": score, "ground_truth": ground_truth, "reward_method": "RULE_BASED_VALIDATION"})
           
           if DEBUG and results:
               correct_count = sum(1 for r in results if r["score"] > 0.5)
               total_count = len(results)
               accuracy = correct_count / total_count if total_count > 0 else 0.0
               print(f"\nRANDOM_REWARD VALIDATION (using RULE_BASED): {correct_count}/{total_count} = {accuracy:.3f} actual accuracy\n")
           
           return results
       else:
           results = []
           for data_source, solution_str, ground_truth, extra_info in zip(
               data_sources, solution_strs, ground_truths, extra_infos, strict=True
           ):
               if isinstance(ground_truth, list) and len(ground_truth) > 0:
                   processed_ground_truth = str(ground_truth[0])
               else:
                   processed_ground_truth = str(ground_truth)
               
               score = compute_random_reward_score(solution_str, processed_ground_truth)
               results.append({"score": score, "ground_truth": ground_truth, "reward_method": "RANDOM_REWARD"})
           
           if DEBUG and results:
               correct_count = sum(1 for r in results if r["score"] > 0.5)
               total_count = len(results)
               accuracy = correct_count / total_count if total_count > 0 else 0.0
               print(f"\nRANDOM_REWARD TRAINING: {correct_count}/{total_count} = {accuracy:.3f} random accuracy (should be ~0.5)\n")
           
           return results
   
   # AUTO-DETECT: Validation vs Training (for GenRM modes only)
   # Validation data always uses rule-based scoring via default_compute_score
   validation_patterns = ['test-math-aime24', 'test-math-aime25', 'huggingfaceh4/math-500', 'test-math-']
   is_validation_batch = any(
       any(pattern in str(data_source).lower() for pattern in validation_patterns)
       for data_source in data_sources
   )
  
   if is_validation_batch:
       # For validation: ALWAYS use math_verify for enhanced accuracy (regardless of reward model type)
       from verl.utils.reward_score import math_verify
       
       results = []
       for data_source, solution_str, ground_truth, extra_info in zip(
           data_sources, solution_strs, ground_truths, extra_infos, strict=True
       ):
           # Handle ground truth format: convert list to string if needed
           if isinstance(ground_truth, list) and len(ground_truth) > 0:
               processed_ground_truth = str(ground_truth[0])
           else:
               processed_ground_truth = str(ground_truth)
           
           # Always use math_verify for eval data
           score = math_verify.compute_score(solution_str, processed_ground_truth)
           results.append({"score": score, "ground_truth": ground_truth, "reward_method": "RULE_BASED"})
      
      
       return results
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
                   # For TANGO_GRPO, ensure fallback has all required keys
                   if REWARD_MODEL_TYPE in ['PRM_COT_HYBRID_APPLY_STEP_GLOBAL_STAT_TANGO', 'PRM_COT_HYBRID_APPLY_STEP_GLOBAL_STAT_TANGO_STEP_PENALTY', 'PRM_COT_HYBRID_OUTCOME_DIFF_ADV', 'PRM_COT_HYBRID_OUTCOME_DIFF_ADV_HALF_DISCOUNT']:
                       fallback = {
                           "score": 0.0, 
                           "ground_truth": None, 
                           "reward_method": "FALLBACK",
                           "step_rewards": [],
                           "step_count": 0,
                           "step_boundary_strategy": "none",
                           "step_positions": [],
                           "outcome_score": 0.0,
                           "step_avg_score": 0.0,
                           "prm_response": "FALLBACK_ERROR"
                       }
                       # Add raw_step_rewards for PRM_COT_HYBRID_OUTCOME_DIFF_ADV variants
                       if REWARD_MODEL_TYPE in ['PRM_COT_HYBRID_OUTCOME_DIFF_ADV', 'PRM_COT_HYBRID_OUTCOME_DIFF_ADV_HALF_DISCOUNT']:
                           fallback["raw_step_rewards"] = []
                   else:
                       fallback = {"score": 0.0, "ground_truth": None, "reward_method": "FALLBACK"}
                   print(f"Using fallback for future {i}: {fallback}")
                   results.append(fallback)

       if DEBUG and REWARD_MODEL_TYPE in ['PRM_COT_HYBRID_APPLY_STEP_GLOBAL_STAT_TANGO', 'PRM_COT_HYBRID_APPLY_STEP_GLOBAL_STAT_TANGO_STEP_PENALTY', 'PRM_COT_HYBRID_OUTCOME_DIFF_ADV', 'PRM_COT_HYBRID_OUTCOME_DIFF_ADV_HALF_DISCOUNT']:
           print(f"RESULT DEBUG: Got {len(results)} results")
           for i, result in enumerate(results):
               has_prm_response = "prm_response" in result
               has_step_positions = "step_positions" in result
               print(f"  Result {i}: prm_response={has_prm_response}, step_positions={has_step_positions}, keys={list(result.keys())}")

       return results