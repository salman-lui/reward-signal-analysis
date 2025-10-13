#!/bin/bash
set -x

# Easy group: Difficulty 0-9 (3,859 problems)
# Hard group: Difficulty 10-16 (4,141 problems)

# Debug control - set to True for debugging, False for production
export DEBUG=False

# Change to verl directory (same as debug script)
cd /home/salman/verl
export REWARD_MODEL_TYPE=REVERSE_MAJORITY_50_EASY_RULE_50_DIFFICULT

# TTRL Configuration for Majority Voting (Easy Problems)
# TTRL is automatically enabled for easy problems (difficulty 0-9) in reverse hybrid mode
# Define parameters based on DEBUG mode
if [ "$DEBUG" = "True" ]; then
    export TTRL_N_VOTES_PER_PROMPT=8         # Generate 8 responses for voting (debug mode)
    export TTRL_N_SAMPLES_PER_PROMPT=8       # Use 4 responses for PPO training (debug mode)
else
    export TTRL_N_VOTES_PER_PROMPT=32        # Generate 32 responses for voting (production)
    export TTRL_N_SAMPLES_PER_PROMPT=8       # Use 8 responses for PPO training (production)
fi  

# GPU Configuration - Conditional based on DEBUG mode
if [ "$DEBUG" = "True" ]; then
    export CUDA_VISIBLE_DEVICES=6,7                            # Debug: Use only 2 GPUs (6,7)
else
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5                    # Production: Use 6 GPUs
fi

# Auto-calculate number of GPUs from CUDA_VISIBLE_DEVICES
N_GPUS_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "Auto-detected $N_GPUS_PER_NODE GPUs from CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"


# REVERSE_MAJORITY_50_EASY_RULE_50_DIFFICULT Configuration - Reverse Hybrid Training
# Easy problems (0-9): TTRL-style majority voting
# Hard problems (10-16): Rule-based rewards using original ground truth

# Authentication tokens
export WANDB_API_KEY="YOUR_WANDB_API_KEY_HERE"
export HF_TOKEN='YOUR_HUGGINGFACE_TOKEN_HERE'

# Model and Environment
export BASE_MODEL=/local2/salman/model/DeepSeek-R1-Distill-Qwen-1.5B
export VLLM_ATTENTION_BACKEND=XFORMERS

# Storage and Logging Paths
BASE_SAVE_PATH="/local2/salman/co-training/hybrid_training/reverse_easy_50_majority_hard_50_rule"
PROJECT_NAME='reverse_majority_50_easy_rule_50_difficult'
EXPERIMENT_NAME='salsrahm_reverse_majority_50_easy_rule_50_difficult'
CHECKPOINT_DIR="$BASE_SAVE_PATH/checkpoints/$PROJECT_NAME"  # Checkpoint folder
LOG_FILE="$BASE_SAVE_PATH/logs/log_$PROJECT_NAME.log"

# Rollout Data Directory - to save all responses per question with rewards
ROLLOUT_DATA_DIR="$BASE_SAVE_PATH/grpo_rollouts/${PROJECT_NAME}_rollouts/training"

# Validation Data Directory - to save validation rollouts (1 response per question, deterministic)
VALIDATION_DATA_DIR="$BASE_SAVE_PATH/grpo_rollouts/${PROJECT_NAME}_rollouts/validation"
MLFLOW_DIR="$BASE_SAVE_PATH/mlflow/${PROJECT_NAME}_ml_flows"
export MLFLOW_TRACKING_URI=file://$MLFLOW_DIR

# Training Data Paths
TRAIN_DATA_PATH="/home/salman/open-prm/data-skymath/rl_training_data/novel_hybrid/rl_training_no_step_instruction/train_novel_hybrid_8k_with_gt.parquet"
EVAL_DATA_PATH_1="/home/salman/open-prm/data-skymath/rl_training_data/novel_hybrid/rl_training_no_step_instruction/eval_data/aime2024.parquet"
EVAL_DATA_PATH_2="/home/salman/open-prm/data-skymath/rl_training_data/novel_hybrid/rl_training_no_step_instruction/eval_data/aime2025.parquet"
EVAL_DATA_PATH_3="/home/salman/open-prm/data-skymath/rl_training_data/novel_hybrid/rl_training_no_step_instruction/eval_data/math500.parquet"

# Hyperparameters - Conditional based on GPU count
if [ "$DEBUG" = "True" ]; then
    TRAIN_BATCH_SIZE=8                                      # Debug: Small batch divisible by 2 GPUs (8*4=32, 32%2=0)
    PPO_MINI_BATCH=8                                        # Debug: Match train batch size
else
    TRAIN_BATCH_SIZE=180                                    # Production: Training batch size (180*8=1440, divisible by 6)
    PPO_MINI_BATCH=180                                      # Production: Match train batch size
fi
if [ "$DEBUG" = "True" ]; then
    MAX_PROMPT_LENGTH=1024                                  # Debug: Shorter prompts for faster processing
    RES_LENGTH=2048                                         # Debug: Shorter responses for faster processing
    GROUP_SIZE=4                                            # Debug: Smaller group size for quicker batches
else
    MAX_PROMPT_LENGTH=2048                                  # Production: Increased for longer problems
    RES_LENGTH=4096                                         # Production: Reduced to 4096
    GROUP_SIZE=8                                            # Production: Scaled up from debug's 5
fi

# Learning Rate and Training
ACTOR_LR=1e-6
LR_WARMUP_RATIO=0.01
ENTROPY_COEFF=0.0
USE_KL_LOSS=True                                            # GRPO guideline: set to True
KL_LOSS_COEF=0.001                                          # GRPO guideline: default coefficient
KL_LOSS_TYPE=low_var_kl                                     # GRPO guideline: recommended type

# GPU and Memory Settings - OPTIMIZED FOR PERFORMANCE
PPO_MICRO_BATCH_SIZE_PER_GPU=4                             # Reduced from 4 to 2 to solve OOM
LOG_PROB_MICRO_BATCH_SIZE=8                                # Reduced from 16 to 8 to solve log_prob OOM
TENSOR_MODEL_PARALLEL_SIZE=1                                # Set to 1 for both debug (2 GPUs) and production (6 GPUs)
GPU_MEMORY_UTILIZATION=0.6                                 # VERL guideline: 0.5-0.7 balance, using 0.6 to avoid OOM

## Performance Optimization Parameters
MAX_NUM_BATCHED_TOKENS=16384                               # Higher throughput during generation
DISABLE_LOG_STATS=False                                    # Monitor rollout performance

# Training Schedule - SCALED FROM DEBUG CONFIG
TOTAL_EPOCHS=8                                             # Updated to 25 epochs
SAVE_FREQ=20                                                # Match debug reference
TEST_FREQ=25                                                 # Match debug reference

# Trainer Configuration
# N_GPUS_PER_NODE is auto-calculated above from CUDA_VISIBLE_DEVICES
NNODES=1                                                    # Number of nodes

# FSDP Settings - MATCHING DEBUG CONFIG
PARAM_OFFLOAD=False                                         # Match debug reference
OPTIMIZER_OFFLOAD=False                                     # Match debug reference
REF_PARAM_OFFLOAD=True                                      # Match debug reference

# Settings - OPTIMIZED FOR FASTER TRAINING
USE_REMOVE_PADDING=True                                     # Enable sequence packing for speedup
ENABLE_GRADIENT_CHECKPOINTING=True                          # Allows larger batch sizes
ENABLE_ACTIVATION_OFFLOAD=True                              # Memory optimization with gradient checkpointing

# Create necessary directories
mkdir -p $CHECKPOINT_DIR
mkdir -p "$(dirname $LOG_FILE)"
mkdir -p $MLFLOW_DIR
mkdir -p $ROLLOUT_DATA_DIR
mkdir -p $VALIDATION_DATA_DIR

# REVERSE_MAJORITY_50_EASY_RULE_50_DIFFICULT Training - Reverse Hybrid Approach
# Easy problems: TTRL-style majority voting, Hard problems: Rule-based rewards
# Single-node training - no Ray cluster setup needed
train_files="$TRAIN_DATA_PATH"
test_files="['$EVAL_DATA_PATH_1','$EVAL_DATA_PATH_2','$EVAL_DATA_PATH_3']"

python3 -m verl.trainer.main_ppo \
algorithm.adv_estimator=grpo \
data.train_files=$train_files \
data.val_files=$test_files \
data.train_batch_size=$TRAIN_BATCH_SIZE \
data.max_prompt_length=$MAX_PROMPT_LENGTH \
data.max_response_length=$RES_LENGTH \
data.filter_overlong_prompts=True \
data.truncation='error' \
actor_rollout_ref.model.path=$BASE_MODEL \
actor_rollout_ref.actor.optim.lr=$ACTOR_LR \
actor_rollout_ref.model.use_remove_padding=$USE_REMOVE_PADDING \
actor_rollout_ref.model.enable_gradient_checkpointing=$ENABLE_GRADIENT_CHECKPOINTING \
actor_rollout_ref.model.enable_activation_offload=$ENABLE_ACTIVATION_OFFLOAD \
actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
actor_rollout_ref.actor.use_kl_loss=$USE_KL_LOSS \
actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
actor_rollout_ref.actor.kl_loss_type=$KL_LOSS_TYPE \
actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
actor_rollout_ref.actor.fsdp_config.param_offload=$PARAM_OFFLOAD \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OPTIMIZER_OFFLOAD \
actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
actor_rollout_ref.actor.entropy_checkpointing=True \
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE \
actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_MODEL_PARALLEL_SIZE \
actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
actor_rollout_ref.rollout.disable_log_stats=$DISABLE_LOG_STATS \
actor_rollout_ref.rollout.n=$TTRL_N_SAMPLES_PER_PROMPT \
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE \
actor_rollout_ref.ref.fsdp_config.param_offload=$REF_PARAM_OFFLOAD \
actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
algorithm.use_kl_in_reward=False \
reward_model.reward_manager=batch \
custom_reward_function.path=/home/salman/verl/salman_scripts_novel_hybrid/novel_math_gen_rm_reward_function.py \
custom_reward_function.name=compute_score_batch \
trainer.critic_warmup=0 \
trainer.default_hdfs_dir=null \
trainer.default_local_dir=$CHECKPOINT_DIR \
trainer.logger='["console","mlflow"]' \
trainer.project_name=$PROJECT_NAME \
trainer.experiment_name=$EXPERIMENT_NAME \
trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
trainer.nnodes=$NNODES \
trainer.save_freq=$SAVE_FREQ \
trainer.test_freq=$TEST_FREQ \
trainer.rollout_data_dir=$ROLLOUT_DATA_DIR \
trainer.validation_data_dir=$VALIDATION_DATA_DIR \
trainer.total_epochs=$TOTAL_EPOCHS 2>&1 | tee $LOG_FILE


