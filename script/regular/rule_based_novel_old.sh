#!/bin/bash
# Multi-node VERL training script

# =============================================================================
#  EXPERIMENT CONFIGURATION
# =============================================================================
export REWARD_MODEL_TYPE=RULE_BASED
export DEBUG=False
export EXPERIMENT_NAME=salsrahm_rule_based_novel
export PROJECT_NAME='rule_based_rlvr_novel'
export RAY_HEAD_PORT=7015
# =============================================================================

CONDA_PATH="/code/salsrahm-sandbox/miniconda3"
ENV_NAME="open-prm"
export GENRM_API_KEY="None"

# Setup conda
source $CONDA_PATH/etc/profile.d/conda.sh
$CONDA_PATH/bin/conda config --add envs_dirs /code/salsrahm-sandbox/envs

# Activate environment with full path to ensure correct activation
conda activate /code/salsrahm-sandbox/envs/$ENV_NAME

# Verify correct Python is being used
if [[ $(which python) != "/code/salsrahm-sandbox/envs/$ENV_NAME/bin/python" ]]; then
echo "Warning: Python path mismatch. Re-activating environment..."
conda deactivate
conda activate /code/salsrahm-sandbox/envs/$ENV_NAME
fi

export DIST_NNODES="${REPLICA}"

# Add missing authentication tokens (from quick_setup.sh)
export WANDB_API_KEY="YOUR_WANDB_API_KEY_HERE"
export HF_TOKEN='YOUR_HUGGINGFACE_TOKEN_HERE'

# Verify conda environment is active and ray is available
echo "Current conda environment: $ENV_NAME"
which python3
which ray

# =============================================================================
# 🔧 TRAINING CONFIGURATION (Usually don't need to change)
# =============================================================================

# Model and Environment
export BASE_MODEL=/checkpoints/salsrahm-sandbox/DeepSeek-R1-Distill-Qwen-1.5B
export VLLM_ATTENTION_BACKEND=XFORMERS

# Storage and Logging Paths
CHECKPOINT_DIR="/checkpoints/salsrahm-sandbox/novel_hybrid/regular/$PROJECT_NAME"  # Checkpoint folder
LOG_FILE="/checkpoints/salsrahm-sandbox/novel_hybrid/regular/verl_logs_novel/$PROJECT_NAME/log_$PROJECT_NAME.log"
MLFLOW_DIR="/checkpoints/salsrahm-sandbox/novel_hybrid/regular/ml_flow_novel/${PROJECT_NAME}_ml_flows"
export MLFLOW_TRACKING_URI=file://$MLFLOW_DIR

# Training Data Paths - Updated for Novel Hybrid Dataset
TRAIN_DATA_PATH="/code/salsrahm-sandbox/data-reasoning/novel_hybrid_data/rl_data/train_novel_hybrid_16k.parquet"
EVAL_DATA_PATH_1="/code/salsrahm-sandbox/data-reasoning/novel_hybrid_data/rl_data/aime2024.parquet"
EVAL_DATA_PATH_2="/code/salsrahm-sandbox/data-reasoning/novel_hybrid_data/rl_data/aime2025.parquet"
EVAL_DATA_PATH_3="/code/salsrahm-sandbox/data-reasoning/novel_hybrid_data/rl_data/math500.parquet"
EVAL_DATA_PATH_4="/code/salsrahm-sandbox/data-reasoning/novel_hybrid_data/rl_data/test_novel_hybrid_200.parquet"

# Hyperparameters - UPDATED FOR REASONING MODEL (LONGER RESPONSES)
TRAIN_BATCH_SIZE=256
PPO_MINI_BATCH=256
MAX_PROMPT_LENGTH=2048                                       # Keep prompt length same
RES_LENGTH=8192                                             # Increased for long reasoning responses (4x longer)
GROUP_SIZE=16                                               # Scaled up from debug's 5

# Memory optimization calculations 
TP=1
SP=1
MAX_TOKEN_LEN=$((2*(RES_LENGTH + MAX_PROMPT_LENGTH) / SP))

# Learning Rate and Training
ACTOR_LR=1e-6
LR_WARMUP_RATIO=0.01
ENTROPY_COEFF=0.0
USE_KL_LOSS=True                                            # GRPO guideline: set to True
KL_LOSS_COEF=0.001                                          # GRPO guideline: default coefficient
KL_LOSS_TYPE=low_var_kl                                     # GRPO guideline: recommended type

# GPU and Memory Settings - MATCHING DEBUG CONFIG
N_GPUS_PER_NODE=8
PPO_MICRO_BATCH_SIZE_PER_GPU=16                             # Match debug reference
LOG_PROB_MICRO_BATCH_SIZE=64                                # Match config
TENSOR_MODEL_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.8

# Training Schedule - SCALED FROM DEBUG CONFIG
TOTAL_EPOCHS=25                                             # Updated to 30 epochs
SAVE_FREQ=50                                                # Match debug reference
TEST_FREQ=10                                                 # Match debug reference

# FSDP Settings - Now hardcoded in training command for memory optimization
REF_PARAM_OFFLOAD=True                                      # Only ref needs variable

# Settings - MATCHING DEBUG CONFIG
USE_REMOVE_PADDING=True
ENABLE_GRADIENT_CHECKPOINTING=True                          # Match debug reference

# =============================================================================
# 🔧 DERIVED PATHS AND SETUP
# =============================================================================
train_files="$TRAIN_DATA_PATH"
test_files="['$EVAL_DATA_PATH_1','$EVAL_DATA_PATH_2','$EVAL_DATA_PATH_3','$EVAL_DATA_PATH_4']"

apt update
apt-get install -y software-properties-common python3-dev cuda-minimal-build-12-5=12.5.1-1

# Create directories
mkdir -p $CHECKPOINT_DIR
mkdir -p "$(dirname $LOG_FILE)"
mkdir -p $MLFLOW_DIR

if [ "${HOSTNAME##*-}" -eq 0 ]; then
ray start --head --port=$RAY_HEAD_PORT
until [ "$(ray status | grep node_ | wc -l | awk '{print $1}')" -eq $DIST_NNODES ]; do
echo "waiting for all workers up..."
sleep 10
done
else
HEAD_ADDR="${HOSTNAME%-*}-0"
HEAD_PORT=$RAY_HEAD_PORT
echo "Waiting for head node (${HEAD_ADDR}:${HEAD_PORT}) to become reachable..."
until (echo > /dev/tcp/${HEAD_ADDR}/${HEAD_PORT}) >/dev/null 2>&1; do
sleep 5
done
echo "Head node is reachable, starting ray worker..."
ray start --address="${HEAD_ADDR}:${HEAD_PORT}" --block
fi
echo "Ray all worker nodes started"

if [ "${HOSTNAME##*-}" -eq 0 ]; then
# Command 1 - Following their reference format exactly
echo "Executing command 1 because DIST_NODE_RANK is 0"

# Replicate manual workflow: source quick_setup then cd to verl
source /code/salsrahm-sandbox/quick_setup.sh
cd /code/salsrahm-sandbox/open-prm/verl

python3 -m verl.trainer.main_ppo \
algorithm.adv_estimator=grpo \
data.train_files=$train_files \
data.val_files=$test_files \
data.train_batch_size=$TRAIN_BATCH_SIZE \
data.val_batch_size=512 \
data.max_prompt_length=$MAX_PROMPT_LENGTH \
data.max_response_length=$RES_LENGTH \
actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
data.filter_overlong_prompts=True \
data.filter_overlong_prompts_workers=8 \
data.truncation='error' \
actor_rollout_ref.model.path=$BASE_MODEL \
actor_rollout_ref.hybrid_engine=True \
actor_rollout_ref.actor.optim.lr=$ACTOR_LR \
actor_rollout_ref.model.use_remove_padding=$USE_REMOVE_PADDING \
actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
actor_rollout_ref.actor.use_dynamic_bsz=True \
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOKEN_LEN \
actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SP \
actor_rollout_ref.actor.use_kl_loss=$USE_KL_LOSS \
actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
actor_rollout_ref.actor.kl_loss_type=$KL_LOSS_TYPE \
actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
actor_rollout_ref.model.enable_gradient_checkpointing=$ENABLE_GRADIENT_CHECKPOINTING \
actor_rollout_ref.actor.fsdp_config.param_offload=True \
+actor_rollout_ref.actor.fsdp_config.grad_offload=True \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE \
actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_MODEL_PARALLEL_SIZE \
actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.rollout.mode="async" \
actor_rollout_ref.rollout.enforce_eager=False \
actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
actor_rollout_ref.rollout.n=$GROUP_SIZE \
actor_rollout_ref.rollout.val_kwargs.n=8 \
actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE \
actor_rollout_ref.ref.fsdp_config.param_offload=$REF_PARAM_OFFLOAD \
algorithm.use_kl_in_reward=False \
algorithm.kl_ctrl.kl_coef=0.001 \
reward_model.reward_manager=batch \
custom_reward_function.path=/code/salsrahm-sandbox/open-prm/verl/salman_scripts_novel_hybrid/novel_math_gen_rm_reward_function.py \
custom_reward_function.name=compute_score_batch \
trainer.critic_warmup=0 \
trainer.default_hdfs_dir=null \
trainer.default_local_dir=$CHECKPOINT_DIR \
trainer.logger='["console","mlflow"]' \
trainer.project_name=$PROJECT_NAME \
trainer.experiment_name=$EXPERIMENT_NAME \
trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
trainer.nnodes=$DIST_NNODES \
trainer.save_freq=$SAVE_FREQ \
trainer.test_freq=$TEST_FREQ \
trainer.total_epochs=$TOTAL_EPOCHS 2>&1 | tee $LOG_FILE
else
sleep infinity
fi










