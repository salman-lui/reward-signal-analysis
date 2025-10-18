#!/bin/bash
#
# =============================================================================
#  FLEXIBLE GPU CONFIGURATION FOR A100/H200
# =============================================================================
# This script will work with either A100 or H200 GPUs (whichever is available)


# Option 1: A100 GPUs (uncomment these 4 lines)
## SBATCH --gres=gpu:A100_80GB:2
## SBATCH --partition=gpu-A100
## SBATCH --account=A100
## SBATCH --qos=a100_qos


# Option 2: H200 GPUs (uncomment these 4 lines)
#SBATCH --gres=gpu:H200_141GB:2
#SBATCH --partition=gpu-H200
#SBATCH --account=H200
#SBATCH --qos=h200_qos


# =============================================================================

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mem=300GB
#SBATCH --job-name=data-effect-llama-math-8
#SBATCH --mail-type=END
#SBATCH --mail-user=salman@nyu.edu
#SBATCH --output=/export/alt-ai-agent/salman/slurm_log/%x_%j.out
#SBATCH --error=/export/alt-ai-agent/salman/slurm_log/%x_%j.err


echo "=========================================="
echo "Starting job: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
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

# Run the training script
echo "Starting training..."
bash script_qcri_sr/data_effect/math/llama_data_effect_math_qcri.sh

echo "=========================================="
echo "Job completed at $(date)"
echo "=========================================="