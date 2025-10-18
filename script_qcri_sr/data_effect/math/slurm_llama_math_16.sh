#!/bin/bash
#
# =============================================================================
#  EXPERIMENT CONFIGURATION - CHANGE THESE FOR DIFFERENT RUNS
# =============================================================================
# EXPERIMENT_NAME: llama_math_16
# DATASET: data/math/train/llama-3b/llama_sky_math_16_upsample.parquet
# TOTAL_EPOCHS: 496
# =============================================================================

# GPU Options: Uncomment ONE set below

# Option 1: A100 GPUs (uncomment these 4 lines)
##SBATCH --gres=gpu:A100_80GB:2
##SBATCH --partition=gpu-A100
##SBATCH --account=A100
##SBATCH --qos=a100_qos

# Option 2: H200 GPUs (currently active)
#SBATCH --gres=gpu:H200_141GB:2
#SBATCH --partition=gpu-H200
#SBATCH --account=H200
#SBATCH --qos=h200_qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mem=300GB
#SBATCH --job-name=llama_math_16
#SBATCH --mail-type=END
#SBATCH --mail-user=salman@nyu.edu
#SBATCH --output=/export/alt-ai-agent/salman/slurm_log/llama_math_16_%j.out
#SBATCH --error=/export/alt-ai-agent/salman/slurm_log/llama_math_16_%j.err


# =============================================================================
#  Set experiment variables (matches config at top)
# =============================================================================
EXPERIMENT_NAME="llama_math_16"
DATASET_FILE="data/math/train/llama-3b/llama_sky_math_16_upsample.parquet"
TOTAL_EPOCHS=496

echo "=========================================="
echo "Starting job: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Experiment: $EXPERIMENT_NAME"
echo "Dataset: $DATASET_FILE"
echo "=========================================="

# Load required modules
module load slurm

# Change to project directory
cd /export/alt-ai-agent/salman/reward-signal-analysis
echo "Working directory: $(pwd)"

# Setup cache directories
mkdir -p /export/alt-ai-agent/salman/slurm_log
mkdir -p ~/.cache/triton
mkdir -p ~/.cache/torchinductor

# Activate conda environment
source ~/.bashrc
conda activate reward-signal

# FIX: Unset AMD GPU environment variable
unset ROCR_VISIBLE_DEVICES

echo "=========================================="
echo "Environment Information:"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "=========================================="

# Print GPU information
echo "GPU Information:"
nvidia-smi
echo "=========================================="

# Export variables for training script
export EXPERIMENT_NAME="$EXPERIMENT_NAME"
export TRAIN_DATA_PATH="$DATASET_FILE"
export TOTAL_EPOCHS=$TOTAL_EPOCHS

echo "Experiment Configuration:"
echo "  Name: $EXPERIMENT_NAME"
echo "  Training Data Path (relative): $TRAIN_DATA_PATH"
echo "  Full Path: $(pwd)/$TRAIN_DATA_PATH"
echo "  Total Epochs: $TOTAL_EPOCHS"
echo "=========================================="

# Run the training script
echo "Starting training..."
bash script_qcri_sr/data_effect/math/llama_data_effect_math_qcri.sh

echo "=========================================="
echo "Job completed at $(date)"
echo "=========================================="

