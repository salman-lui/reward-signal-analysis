"""
Stratified Dataset Sampler by Difficulty

Creates multiple training datasets of different sizes while maintaining equal distribution 
across difficulty levels (1-15). Ensures cumulative sampling: smaller datasets are subsets 
of larger ones.

Usage:
    1. Set INPUT_PATH to your parquet file with difficulty scores
    2. Set OUTPUT_DIR where sampled datasets will be saved
    3. Set OUTPUT_PREFIX for the output filename prefix
    4. Set MODEL to the model name in the difficulty dictionary
    5. Run: python parquet_data_splitting_train.py

Output: Creates {OUTPUT_PREFIX}_{size}.parquet files with balanced difficulty distribution.
"""

import duckdb
import pandas as pd
import os
from pathlib import Path

# ============================================================================
# CONFIGURATION - CHANGE THESE FOR DIFFERENT DATASETS
# ============================================================================

# FOR MATH DATASET:
# INPUT_PATH = "/home/salman/reward-signal-analysis/difficulty_estimation/datasets_with_difficulty/train_novel_hybrid_8k_with_gt_with_difficulty.parquet"
# OUTPUT_DIR = "/home/salman/reward-signal-analysis/test_code/data_salman/math/train_v2/llama-3b"
# OUTPUT_PREFIX = "llama_sky_math"

# FOR SCP DATASET - LLAMA: (CURRENTLY ACTIVE - Updated dataset without "Show that"/"Prove that")
INPUT_PATH = "/home/salman/reward-signal-analysis/difficulty_estimation/datasets_with_difficulty/scp_updated_8620_with_difficulty.parquet"
OUTPUT_DIR = "/home/salman/reward-signal-analysis/test_code/data_salman/scp/train/llama-3b"
OUTPUT_PREFIX = "llama_scp"
MODEL = "Llama-3.2-3B-Instruct"

# FOR SCP DATASET - QWEN:
# INPUT_PATH = "/home/salman/reward-signal-analysis/difficulty_estimation/datasets_with_difficulty/scp_updated_8620_with_difficulty.parquet"
# OUTPUT_DIR = "/home/salman/reward-signal-analysis/test_code/data_salman/scp/train/qwen-3b"
# OUTPUT_PREFIX = "qwen_scp"
# MODEL = "Qwen2.5-3B"
SAMPLE_SIZES = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
DIFFICULTY_RANGE = range(1, 16)  # 1 to 15

# ============================================================================

print("="*80)
print(f"DATASET: {OUTPUT_PREFIX}")
print(f"MODEL: {MODEL}")
print("="*80)

# Load data with duckdb
print(f"\nLoading: {INPUT_PATH}")
conn = duckdb.connect()
df = conn.execute(f"SELECT * FROM '{INPUT_PATH}'").df()
print(f"Total records: {len(df)}")

# Filter for target difficulties (1-15)
filtered_rows = []
for idx, row in df.iterrows():
    difficulty_dict = row['difficulty']
    if isinstance(difficulty_dict, dict) and MODEL in difficulty_dict:
        diff_score = difficulty_dict[MODEL]
        # Handle both int and float (e.g., 0, 1, 0.0, 1.0)
        if isinstance(diff_score, (int, float)) and int(diff_score) in DIFFICULTY_RANGE:
            filtered_rows.append((idx, int(diff_score)))

print(f"Records with difficulty 1-15: {len(filtered_rows)}")

# Group by difficulty
difficulty_groups = {d: [] for d in DIFFICULTY_RANGE}
for idx, diff in filtered_rows:
    difficulty_groups[diff].append(idx)

# Print distribution
print("\nDifficulty distribution:")
for diff in DIFFICULTY_RANGE:
    print(f"  {diff}: {len(difficulty_groups[diff])}")

# Create output directory
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Sample data for each size - maintaining equal distribution
for size in SAMPLE_SIZES:
    # Strategy: Keep all difficulties at same count until one runs out
    all_indices = []
    counts = {d: 0 for d in DIFFICULTY_RANGE}
    remaining = size
    
    while remaining > 0:
        # Which difficulties still have data?
        active = [d for d in DIFFICULTY_RANGE if counts[d] < len(difficulty_groups[d])]
        if not active:
            break
        
        # How many can we add to each active difficulty?
        can_add_per_diff = remaining // len(active)
        if can_add_per_diff == 0:
            # Less than one per difficulty - add one to each in round-robin
            for d in active:
                if remaining <= 0:
                    break
                if counts[d] < len(difficulty_groups[d]):
                    start_idx = counts[d]
                    all_indices.extend(difficulty_groups[d][start_idx:start_idx + 1])
                    counts[d] += 1
                    remaining -= 1
            continue
        
        # Find bottleneck (difficulty that would run out first)
        min_available = min(len(difficulty_groups[d]) - counts[d] for d in active)
        add_per_diff = min(can_add_per_diff, min_available)
        
        # Add same amount to each active difficulty
        added_this_round = 0
        for d in active:
            take = min(add_per_diff, len(difficulty_groups[d]) - counts[d])
            start_idx = counts[d]
            all_indices.extend(difficulty_groups[d][start_idx:start_idx + take])
            counts[d] += take
            added_this_round += take
        
        remaining -= added_this_round
    
    # Create sampled dataframe
    sampled_df = df.loc[all_indices].reset_index(drop=True)
    
    # Save using duckdb
    output_file = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_{len(all_indices)}.parquet")
    conn.execute(f"COPY (SELECT * FROM sampled_df) TO '{output_file}' (FORMAT PARQUET)")
    print(f"\nSaved {len(all_indices)} samples -> {output_file}")
    
    # Show difficulty distribution
    diff_counts = {d: 0 for d in DIFFICULTY_RANGE}
    for idx in all_indices:
        diff_val = df.loc[idx, 'difficulty'][MODEL]
        # Handle both int and float
        diff_counts[int(diff_val)] += 1
    print(f"  Distribution: {list(diff_counts.values())}")

print("\n" + "="*80)
print("COMPLETE!")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Files created: {len(SAMPLE_SIZES)}")
print(f"Sample sizes: {SAMPLE_SIZES}")
print("="*80)

