import os
import sys
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json
from tqdm import tqdm
import numpy as np
import argparse
from pathlib import Path
import pandas as pd
from ast import literal_eval
from verl.utils.reward_score import math_verify

# Import eval functions
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))
from eval import calculate_metrics, calculate_summary_metrics

try:
    from math_verify.parser import parse, ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    print("To use Math-Verify parser, please install it first by running `pip install math-verify`.")

try:
    from reasoning_gym import get_score_answer_fn
    REASONING_GYM_AVAILABLE = True
except ImportError:
    REASONING_GYM_AVAILABLE = False
    print("Warning: reasoning_gym not available. Install it to use reasoning-gym verifier.")

PATH_hf_CACHE = '/scratch/js15262/hf_cache'
os.makedirs(PATH_hf_CACHE, exist_ok=True)

def parse_to_dict(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):  # NaN
        return None
    if isinstance(val, dict):
        return val
    if isinstance(val, (list, tuple)):  # sometimes stored as list/tuple
        return {"_value": list(val)}
    if not isinstance(val, str):
        return {"_value": val}
    s = val.strip()
    # Try JSON
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
        return {"_value": obj}
    except Exception:
        pass
    # Try Python literal
    try:
        obj = literal_eval(s)
        if isinstance(obj, dict):
            return obj
        return {"_value": obj}
    except Exception:
        return {"_unparsed": s}
    
def extract_answer_from_response(model_output: str) -> str:
    """Extract the answer from model output using math_verify parser"""
    try:
        parsed = parse(
            model_output,
            extraction_config=(ExprExtractionConfig(), LatexExtractionConfig()),
            fallback_mode="no_fallback",
            extraction_mode=["first_match"],
            parsing_timeout=5,
        )
        # parsed is a list of extracted answers, get the first one if available
        if parsed and len(parsed) > 0:
            return str(parsed[0])
    except Exception:
        pass
    return None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True, type=str, help="Path to the input parquet file")
    parser.add_argument("--output_path", default=None, type=str, help="Path to output JSON file (auto-generated from dataset name if not provided)")
    parser.add_argument("--output_dir", default="output", type=str, help="Directory to save output files")
    parser.add_argument("--model_name_or_path", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", type=str)
    parser.add_argument("--num_samples", default=32, type=int, help="Number of samples to generate per prompt.")
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--temperature", default=0.6, type=float)
    parser.add_argument("--top_p", default=0.95, type=float)
    parser.add_argument("--max_tokens", default=4096, type=int)
    parser.add_argument("--verifier", default="math-verify", type=str, 
                        choices=["math-verify", "reasoning-gym"],
                        help="Verification method: 'math-verify' (default) or 'reasoning-gym'")
    parser.add_argument("--save_frequency", default=10, type=int, 
                        help="Save results every N prompts (default: 10)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file instead of starting fresh")
    args = parser.parse_args()
    
    # Validate verifier choice
    if args.verifier == "reasoning-gym" and not REASONING_GYM_AVAILABLE:
        raise ImportError("reasoning_gym is not installed. Please install it or use --verifier math-verify")
    
    # Auto-generate output_path from dataset_path if not provided
    if args.output_path is None:
        # Extract dataset name from path (remove .parquet extension)
        dataset_path = Path(args.dataset_path)
        dataset_name = dataset_path.stem  # Gets filename without extension
        
        # Extract and sanitize model name
        model_name = args.model_name_or_path.split('/')[-1]
        model_name = model_name.replace(',', '_').replace(' ', '_')
        
        # Create output directory with model subdirectory
        output_dir = Path(args.output_dir) / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Combine dataset and model name in filename
        args.output_path = str(output_dir / f"{dataset_name}.json")
    
    return args

def setup(args):
    """Initialize vLLM model and tokenizer"""
    available_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    
    llm = LLM(
        model=args.model_name_or_path,
        download_dir=str(PATH_hf_CACHE),
        tensor_parallel_size=len(available_gpus) // args.pipeline_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.8,
        enforce_eager=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=PATH_hf_CACHE,
        use_fast=True
    )
    
    return llm, tokenizer

@torch.inference_mode()
def generate_vllm(llm, tokenizer, args, prompt, gt_answer, source_dataset=None, extra_info=None):
    """Generate K samples for a single prompt using vLLM"""
    # Process ground truth (same logic as parse.py)
    if isinstance(gt_answer, list) and len(gt_answer) > 0:
        processed_ground_truth = str(gt_answer[0])
    else:
        processed_ground_truth = str(gt_answer)
    
    # Setup verifier based on args
    if args.verifier == "reasoning-gym":
        # Get scoring function for reasoning-gym
        if source_dataset is None and extra_info is not None:
            if isinstance(extra_info, str):
                extra_info = parse_to_dict(extra_info)
            source_dataset = extra_info.get('source_dataset', 'math')
        elif source_dataset is None:
            source_dataset = 'math'  # Default fallback
        score_fn = get_score_answer_fn(source_dataset)
        print(f"Using reasoning-gym verifier for dataset: {source_dataset}")
        entry = {'question': prompt, 'metadata':extra_info,'answer':gt_answer}
    
    
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>", "<|end▁of▁sentence|>", "<｜end▁of▁sentence｜>"]
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=args.num_samples,  # Generate K samples for each prompt
        stop=stop_words,
    )
    
    prompt_str = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True,  # adds assistant tag if needed
    )
    prompts = [prompt_str]
        
    gen_output = llm.generate(prompts, sampling_params)
    results = []
    
    # Process all K samples from the single prompt
    for output in gen_output[0].outputs:
        text = output.text if output else ""

        # Extract reasoning tokens
        think_pos = text.find("</think>")
        reasoning_text = text[:think_pos] if think_pos != -1 else text
        reasoning_tok = len(tokenizer.encode(reasoning_text, add_special_tokens=False))

        extracted_answer = extract_answer_from_response(text)
        # Compute score using specified verifier
        if args.verifier == "reasoning-gym" and 'source_dataset' not in ['course_schedule', 'aiw','circuit_logic','self_reference','syllogism','zebra_puzzles']:
            score = score_fn(extracted_answer, entry)
        else:  # math-verify (default)
            score = math_verify.compute_score(text, processed_ground_truth)
        
        correct = bool(score == 1.0)  # score is 1.0 for correct, 0.0 for incorrect
        
        results.append({
            "reasoning_tokens": reasoning_tok,
            "correct": correct,
            "score": float(score),
            "response": text,
            "extracted_answer": extracted_answer
        })

    return results

def convert_to_serializable(obj):
    """Convert numpy arrays and other non-serializable objects to JSON-serializable types"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

def load_existing_indices(output_path):
    """Load the indices of already processed examples from existing output file"""
    if not os.path.exists(output_path):
        return set()
    
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if "results" in data and isinstance(data["results"], list):
            # Extract all processed indices
            indices = {item["index"] for item in data["results"] if "index" in item}
            return indices
    except Exception as e:
        print(f"Warning: Could not load existing indices from {output_path}: {str(e)}")
    
    return set()

def save_results(args, results, summary_metrics=None):
    """Save the results"""
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Convert any numpy arrays to lists before saving
    results = convert_to_serializable(results)
    summary_metrics = convert_to_serializable(summary_metrics) if summary_metrics is not None else None
    
    data_to_save = {
        "results": results,
        "config": {
            "model": args.model_name_or_path,
            "num_samples": args.num_samples,
            "dataset": args.dataset_path,
            "verifier": args.verifier
        }
    }
    
    # Add summary metrics if provided
    if summary_metrics is not None:
        data_to_save["summary"] = summary_metrics

    if os.path.exists(args.output_path):
        try:
            with open(args.output_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)

            # Merge the results (assuming both have "results" array)
            if isinstance(existing_data.get("results"), list):
                data_to_save["results"] = existing_data["results"] + data_to_save["results"]

        except json.JSONDecodeError:
            print(f"Warning: Could not parse existing JSON file at {args.output_path}. Overwriting.")
        except Exception as e:
            print(f"Warning: Error reading existing file: {str(e)}. Overwriting.")

    # Write the merged data back to file
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=2, ensure_ascii=False)

def main():
    args = parse_args()
    
    # Print configuration
    print(f"=== Configuration ===")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.output_path}")
    print(f"Model: {args.model_name_or_path}")
    print(f"Samples per prompt: {args.num_samples}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Verifier: {args.verifier}")
    print(f"Save frequency: Every {args.save_frequency} prompts")
    print(f"Resume mode: {args.resume}")
    print()
    
    # Handle resume mode
    processed_indices = set()
    if args.resume:
        processed_indices = load_existing_indices(args.output_path)
        if processed_indices:
            print(f"Resume mode: Found {len(processed_indices)} already processed examples")
            print(f"Will skip indices: {sorted(list(processed_indices))[:10]}{'...' if len(processed_indices) > 10 else ''}")
        else:
            print(f"Resume mode enabled but no existing results found at {args.output_path}")
    else:
        # Remove existing output file to overwrite (not append to) historical results
        if os.path.exists(args.output_path):
            print(f"Removing existing output file: {args.output_path}")
            os.remove(args.output_path)
    
    llm, tokenizer = setup(args)

    # Load dataset from parquet
    print(f"Loading dataset from {args.dataset_path}")
    df = pd.read_parquet(args.dataset_path)
    print(f"Loaded {len(df)} examples")

    results = []
    all_problem_metrics = []
    skipped_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating samples"):
        # Skip already processed examples in resume mode
        if idx in processed_indices:
            skipped_count += 1
            continue
            
        # Extract prompt and ground truth from the specified columns
        prompt = row["prompt"]
        gt_answer = row["reward_model"]["ground_truth"]
        
        # Extract extra_info for reasoning-gym verifier
        extra_info = row.get("extra_info", None) if args.verifier == "reasoning-gym" else None
        
        # Generate K samples for this prompt
        sample_results = generate_vllm(llm, tokenizer, args, prompt, gt_answer, extra_info=extra_info)
        
        # Calculate metrics for this problem
        metrics = calculate_metrics(sample_results)
        all_problem_metrics.append(metrics)
        print(f"   Metrics: Avg Score={metrics['average_score']:.3f}, Accuracy={metrics['accuracy']:.3f}, Pass@1={metrics.get('pass_at_1', 0):.3f}")
        
        # Print sample info
        for sample_idx, sample_result in enumerate(sample_results):
            print(f"   Sample {sample_idx+1}/{args.num_samples}: Correct={sample_result['correct']}, Score={sample_result['score']}, Answer={sample_result['extracted_answer']}, Tokens={sample_result['reasoning_tokens']}")
        
        # Create a single entry for this prompt with all responses
        prompt_entry = {
            "index": idx,
            "question": prompt,
            "ground_truth": gt_answer,
            "responses": [
                {
                    "sample_id": i,
                    "reasoning_tokens": sample["reasoning_tokens"],
                    "correct": sample["correct"],
                    "score": sample["score"],
                    "response": sample["response"],
                    "extracted_answer": sample["extracted_answer"]
                }
                for i, sample in enumerate(sample_results)
            ],
            "metrics": metrics
        }
        results.append(prompt_entry)

        # Save incrementally every N prompts instead of every prompt
        if (idx + 1) % args.save_frequency == 0 or (idx + 1) == len(df):
            save_results(args, results)
            print(f'--- Saved checkpoint at example {idx} (total saved: {idx + 1}) ----------------------------------------')
            results = []  # Clear for next batch
        else:
            print(f'--- Processed example {idx} ----------------------------------------')
    
    # Calculate and save overall summary metrics
    # In resume mode, we need to recalculate metrics across ALL results (old + new)
    if args.resume and os.path.exists(args.output_path):
        with open(args.output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Recalculate metrics for all results
        all_metrics_for_summary = []
        if "results" in data and isinstance(data["results"], list):
            for result in data["results"]:
                if "metrics" in result:
                    all_metrics_for_summary.append(result["metrics"])
        summary_metrics = calculate_summary_metrics(all_metrics_for_summary)
    else:
        summary_metrics = calculate_summary_metrics(all_problem_metrics)
    
    # Load existing data and add summary
    if os.path.exists(args.output_path):
        with open(args.output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["summary"] = summary_metrics
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n=== Evaluation Complete ===")
    print(f"Model: {args.model_name_or_path}")
    print(f"Total prompts in dataset: {len(df)}")
    if args.resume and skipped_count > 0:
        print(f"Skipped (already processed): {skipped_count}")
        print(f"Newly processed: {len(df) - skipped_count}")
    else:
        print(f"Total prompts processed: {len(df)}")
    print(f"Samples per prompt: {args.num_samples}")
    print(f"\n=== Summary Metrics ===")
    print(f"Overall Average Score: {summary_metrics['overall_average_score']:.3f}")
    print(f"Overall Accuracy: {summary_metrics['overall_accuracy']:.3f}")
    for k in [1, 5, 8, 16]:
        key = f"overall_pass_at_{k}"
        if key in summary_metrics:
            print(f"Overall Pass@{k}: {summary_metrics[key]:.3f}")
    print(f"\nOutput saved to: {args.output_path}")

if __name__ == "__main__":
    main()