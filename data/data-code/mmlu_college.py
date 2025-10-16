"""
Preprocess the MMLU College Science subsets to parquet format
"""

import os
import argparse
import random
from datasets import load_dataset, Dataset, concatenate_datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.data_process.utils import set_seed, sample_dataset, save_dataset


def get_datasets(subsets):
    """
    Loads specific MMLU subsets (college_biology, college_chemistry, college_physics).
    """
    all_datasets = []
    
    for subset in subsets:
        try:
            # MMLU has 'test', 'validation', 'dev' splits
            dataset = load_dataset("cais/mmlu", subset, split="test")
            print(f"MMLU {subset}: {len(dataset)} test examples")
            all_datasets.append(dataset)
        except Exception as e:
            print(f"Error loading {subset}: {e}")
    
    if all_datasets:
        combined = concatenate_datasets(all_datasets)
        print(f"\nTotal combined examples: {len(combined)}")
        return combined
    return None


def make_map_fn(split: str, data_source: str) -> callable:
    def process_fn(example, idx):
        def form_options(choices: list):
            """Format options similar to SuperGPQA style"""
            option_str = 'Options are:\n'
            opts = ['A', 'B', 'C', 'D']
            for choice, opt in zip(choices, opts):
                option_str += f'({opt}): {choice}\n'
            return option_str
        
        # MMLU format: question, choices (list of 4), answer (0-3)
        question = example["question"].strip()
        options = form_options(example["choices"])
        query = question + '\n' + options + '\n'
        
        # Convert answer index (0-3) to letter (A-D)
        answer_letter = ['A', 'B', 'C', 'D'][example["answer"]]
        
        # Prompt format similar to SuperGPQA
        prompt = (
            f"{query}"
            "Please reason step by step, and put your final answer option within \\boxed{}. "
            "Only put the letter in the box, e.g. \\boxed{A}. There is only one correct answer."
        )
        
        data = {
            "data_source": data_source,
            "prompt": [
                {"role": "user", "content": prompt}
            ],
            "ability": "stem",
            "apply_chat_template": True,
            "reward_model": {
                "style": "rule",
                "ground_truth": answer_letter,
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "original_prompt": prompt,
                "dataset": "cais/mmlu",
                "subset": example.get("subject", "unknown"),  # MMLU includes subject field
            },
        }
        
        if idx == 0 or idx == 1:
            print("\n" + "=" * 10 + f" {data_source} {split} {idx} " + "=" * 10)
            print(data)
            print(f'\nOne prompt example:\n{prompt}')
            
        return data

    return process_fn


if __name__ == '__main__':
    """Main script execution: parse args, load, process, and save MMLU college science datasets."""
    parser = argparse.ArgumentParser(description="Process and save MMLU college science datasets.")
    parser.add_argument('--output-dir', 
                        default='difficulty_estimation/datasets_with_difficulty',
                        help='Directory to save the processed data files.')
    parser.add_argument('--domain', default="stem",
                        help='Domain of the dataset.')
    parser.add_argument('--name', default="mmlu_college",
                        help='Name of the dataset.')
    parser.add_argument('--subsets', nargs='+', 
                        default=["college_biology", "college_chemistry", "college_physics"],
                        help='MMLU subsets to load.')
    parser.add_argument('--sample-size', type=int, default=None,
                        help='Number of samples to use from dataset. If None, use all samples.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    set_seed(args.seed)

    data_source = f"{args.domain}__{args.name}"
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the datasets
    print(f"Loading MMLU subsets: {args.subsets}")
    dataset = get_datasets(args.subsets)
    
    if dataset is None:
        print("Failed to load datasets. Exiting.")
        exit(1)

    # Process the dataset
    process_fn = make_map_fn('test', data_source)
    dataset = dataset.map(function=process_fn, with_indices=True)

    # Sample the dataset if specified
    if args.sample_size is not None:
        dataset = sample_dataset(dataset, args.sample_size)
    
    # Save the dataset
    output_path = save_dataset(
        dataset=dataset,
        output_dir=output_dir,
        filename_prefix=data_source,
        sample_size=len(dataset)
    )

    print(f"\n{'='*80}")
    print(f"Done!")
    print(f"Data source: {data_source}")
    print(f"Subsets: {args.subsets}")
    print(f"Data saved to: {output_path}")
    print(f"Total samples: {len(dataset)}")
    print(f"{'='*80}")

