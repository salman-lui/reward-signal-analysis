"""
Upsample Small Dataset by Repetition

Takes a small dataset and repeats it N times to create a larger training set.
Useful for data augmentation experiments.

Usage:
    1. Set INPUT_PATH to your small parquet file
    2. Set REPEAT_TIMES to how many times to repeat
    3. Set OUTPUT_NAME for the output filename
    4. Run: python upsampling_small_data.py

Output: Creates upsampled dataset with samples repeated N times.
"""

import duckdb
import pandas as pd
import os

# ============================================================================
# CONFIGURATION - CHANGE THESE FOR DIFFERENT DATASETS
# ============================================================================

# Base directory
BASE_DIR = "/home/salman/reward-signal-analysis/data/scp/train/llama-3b"

# Files to upsample: (input_filename, repeat_times, output_filename)
FILES_TO_UPSAMPLE = [
    ("llama_scp_8.parquet", 8, "llama_scp_8_upsample.parquet"),   # 8 * 8 = 64
    ("llama_scp_16.parquet", 4, "llama_scp_16_upsample.parquet"), # 16 * 4 = 64
    ("llama_scp_32.parquet", 2, "llama_scp_32_upsample.parquet"), # 32 * 2 = 64
]

# ============================================================================

print("="*80)
print("UPSAMPLING SMALL DATASETS")
print("="*80)

conn = duckdb.connect()

for input_file, repeat_times, output_file in FILES_TO_UPSAMPLE:
    print("\n" + "-"*80)
    input_path = os.path.join(BASE_DIR, input_file)
    output_path = os.path.join(BASE_DIR, output_file)
    
    # Load data
    print(f"Loading: {input_file}")
    df = conn.execute(f"SELECT * FROM '{input_path}'").df()
    print(f"  Original records: {len(df)}")
    
    # Repeat the dataset N times
    print(f"  Repeating {repeat_times} times...")
    upsampled_df = pd.concat([df] * repeat_times, ignore_index=True)
    print(f"  Upsampled records: {len(upsampled_df)}")
    
    # Save using duckdb
    conn.execute(f"COPY (SELECT * FROM upsampled_df) TO '{output_path}' (FORMAT PARQUET)")
    print(f"  ✓ Saved: {output_file}")

print("\n" + "="*80)
print("COMPLETE!")
print(f"Upsampled {len(FILES_TO_UPSAMPLE)} files")
print(f"Output directory: {BASE_DIR}")
print("="*80)

