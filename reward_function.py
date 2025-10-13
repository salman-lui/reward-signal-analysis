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


# =============================================================================
# FORMAT VALIDATION (for RANDOM_REWARD)
# =============================================================================

def format_validity(solution_str):
    """
    Check if solution has valid format:
    - Exactly one \\boxed{} or \\\\boxed{}
    - Exactly one <answer></answer>
    - Nothing after </answer>
    """
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
    """Extract the valid portion of answer (up to </answer> tag)."""
    try:
        answer_match = re.search(r'(.*</answer>)', solution_str, re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip()
        else:
            return solution_str
    except Exception:
        return solution_str


# =============================================================================
# RANDOM REWARD SCORING
# =============================================================================

def compute_random_reward_score(solution_str, ground_truth):
    """
    RANDOM_REWARD: Format validation + Random 50/50 scoring.
    Returns 0.0 if format is invalid, otherwise returns random 0.0 or 1.0.
    """
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


# =============================================================================
# RULE-BASED SCORING
# =============================================================================

def compute_rule_based_score(solution_str, ground_truth):
    """
    RULE_BASED: Uses math_verify for accurate math answer checking.
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
            print(f"\n🎯 GROUND TRUTH: {ground_truth}")
            print(f"\n🏆 FINAL REWARD: {score}")
            print("="*100 + "\n")
        
        return score
    except Exception as e:
        if DEBUG:
            print(f"Error in math_verify scoring: {e}")
        return 0.0


# =============================================================================
# MAIN SCORING FUNCTIONS
# =============================================================================

def compute_score(data_source, solution_str, ground_truth, extra_info):
    """
    Main scoring function for individual samples.
    Supports RULE_BASED and RANDOM_REWARD methods.
    """
    # RULE_BASED: Use math_verify for enhanced accuracy
    if REWARD_MODEL_TYPE == 'RULE_BASED':
        # Handle ground truth format: convert list to string if needed
        if isinstance(ground_truth, list) and len(ground_truth) > 0:
            processed_ground_truth = str(ground_truth[0])
        else:
            processed_ground_truth = str(ground_truth)
        
        reward_score = compute_rule_based_score(solution_str, processed_ground_truth)
        return {"score": reward_score, "ground_truth": ground_truth, "reward_method": "RULE_BASED"}
    
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
            print(f"\n✅ VALIDITY: {'VALID' if format_validity(solution_str) == 1.0 else 'INVALID'}")
            print(f"\n🎯 GROUND TRUTH: {processed_ground_truth} (ignored)")
            print(f"\n🏆 FINAL REWARD: {reward_score} (random)")
            print("="*100 + "\n")
        
        return {"score": reward_score, "ground_truth": ground_truth, "reward_method": "RANDOM_REWARD"}
    
    # Fallback for unknown reward types
    print(f"WARNING: Unknown REWARD_MODEL_TYPE: {REWARD_MODEL_TYPE}")
    return {"score": 0.0, "ground_truth": ground_truth, "reward_method": "UNKNOWN"}


# =============================================================================
# BATCH SCORING FUNCTION
# =============================================================================

def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos):
    """
    Batch scoring function. Supports RULE_BASED and RANDOM_REWARD methods.
    For RANDOM_REWARD, uses rule-based scoring for validation data.
    """
    if DEBUG:
        print(f"BATCH INPUT DEBUG: data_sources={len(data_sources)}, solution_strs={len(solution_strs)}, ground_truths={len(ground_truths)}, extra_infos={len(extra_infos)}")
    
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
    
    # Fallback for unknown reward types
    print(f"WARNING: Unknown REWARD_MODEL_TYPE in batch: {REWARD_MODEL_TYPE}")
    return [{"score": 0.0, "ground_truth": gt, "reward_method": "UNKNOWN"} for gt in ground_truths]
