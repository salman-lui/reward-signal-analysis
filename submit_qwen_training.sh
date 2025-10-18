#!/bin/bash -l

#SBATCH -J qwen_data_512           # Job name (matches EXPERIMENT_NAME)
#SBATCH -o logs/slurm_%j.out       # Standard output (%j = job ID)
#SBATCH -e logs/slurm_%j.err       # Standard error
#SBATCH -p gpu-all                 # Partition: gpu-all for long jobs
#SBATCH --gres=gpu:A100_80GB:4     # Request 4x A100 80GB GPUs
#SBATCH -c 32                      # 32 CPUs (8 per GPU is reasonable)
#SBATCH --mem=256G                 # 256GB RAM (64GB per GPU)
#SBATCH --time=3-00:00:00          # Max 3 days runtime

# Optional: If you have access to gpu-A100 partition (faster, dedicated)
# Uncomment these lines and change partition above to gpu-A100:
# #SBATCH -A A100
# #SBATCH -q a100_qos

# Load required modules
module load slurm

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start Time: $(date)"
echo "=========================================="

# Navigate to project directory
cd /export/alt-ai-agent/salman/reward-signal-analysis

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the training script
# The script already handles GPU assignment via CUDA_VISIBLE_DEVICES
bash script/data_effect/qwen_data_effect.sh

# Print completion info
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="

