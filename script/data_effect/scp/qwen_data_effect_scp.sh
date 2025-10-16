#!/bin/bash
# VERL training script for SCP dataset with Qwen2.5-3B
# Training: SCP balanced difficulty samples | Evaluation: Multiple benchmarks
#
# Usage:
#   DEBUG=True bash script/data_effect/scp/qwen_data_effect.sh   # Debug: 1 GPU, small batch
#   DEBUG=False bash script/data_effect/scp/qwen_data_effect.sh  # Production: 4 GPUs, full batch

# =============================================================================
#  EASY CONFIGURATION (MODIFY THESE FOR DIFFERENT EXPERIMENTS)
# =============================================================================
export DEBUG=${DEBUG:-False}  # Can be overridden: DEBUG=True bash script/rlvr_8k.sh
export REWARD_MODEL_TYPE=RULE_BASED  # Options: RULE_BASED, RANDOM_REWARD
export BASE_MODEL=/local2/salman/model/reward_signal_project/Qwen2.5-3B # CHANGE THIS

# Set save directory based on debug mode
if [ "$DEBUG" = "True" ]; then
export SAVE_DIR="/local2/salman/debug_save" # Debug save directory
else
export SAVE_DIR="/local2/salman/reward_signal_results/data_effect/scp/qwen3b" # CHANGE THIS
fi

export EXPERIMENT_NAME=qwen_scp_64 # CHANGE THIS

TOTAL_EPOCHS=496 # CHANGE THIS (496 for 64, 248 for 128, 124 for 256, 62 for 512, 31 for 1024, 15 for 2048) 
SAVE_FREQ=50
TEST_FREQ=20

# =============================================================================
#  TRAINING CONFIGURATION
# =============================================================================

if [ "$DEBUG" = "True" ]; then
unset VLLM_ATTENTION_BACKEND
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"2"}
N_GPUS_PER_NODE=1
else
export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"} # CHANGE THIS if using different GPUs
# export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"4,5,6,7"}
N_GPUS_PER_NODE=4
fi

if [ "$DEBUG" = "True" ]; then
TRAIN_DATA_PATH="$(pwd)/data/scp/train/qwen-3b/qwen_scp_64.parquet"
EVAL_DATA_PATH_1="$(pwd)/data/scp/eval_data/stem__gpqa_diamond_198.parquet"
# EVAL_DATA_PATH_2="$(pwd)/data/scp/eval_data/aime2024.parquet"
else
TRAIN_DATA_PATH="$(pwd)/data/scp/train/qwen-3b/qwen_scp_64.parquet" # CHANGE THIS
EVAL_DATA_PATH_1="$(pwd)/data/scp/eval_data/aime2024.parquet"
EVAL_DATA_PATH_2="$(pwd)/data/scp/eval_data/math500.parquet"
EVAL_DATA_PATH_3="$(pwd)/data/scp/eval_data/scp_test_difficult_1.parquet"
EVAL_DATA_PATH_4="$(pwd)/data/scp/eval_data/scp_test_medium_2_8.parquet"
EVAL_DATA_PATH_5="$(pwd)/data/scp/eval_data/scp_test_very_difficult_0.parquet"
EVAL_DATA_PATH_6="$(pwd)/data/scp/eval_data/stem__gpqa_diamond_198.parquet"
EVAL_DATA_PATH_7="$(pwd)/data/scp/eval_data/stem__mmlu_sci_college_346.parquet"
EVAL_DATA_PATH_8="$(pwd)/data/scp/eval_data/stem__mmlu_sci_high_school_300.parquet"
EVAL_DATA_PATH_9="$(pwd)/data/scp/eval_data/super_gpqa_in_domain_319.parquet"
EVAL_DATA_PATH_10="$(pwd)/data/scp/eval_data/super_gpqa_out_domain_250.parquet"
fi

CUSTOM_REWARD_PATH="$(pwd)/reward_function.py"
CHECKPOINT_DIR="$SAVE_DIR/$EXPERIMENT_NAME/checkpoints"
LOG_FILE="$SAVE_DIR/$EXPERIMENT_NAME/logs/log_$EXPERIMENT_NAME.log"
MLFLOW_DIR="$SAVE_DIR/$EXPERIMENT_NAME/mlflow"
ROLLOUT_DIR="$SAVE_DIR/$EXPERIMENT_NAME/rollouts/training"
VALIDATION_DIR="$SAVE_DIR/$EXPERIMENT_NAME/rollouts/validation"
export MLFLOW_TRACKING_URI=file://$MLFLOW_DIR

if [ "$DEBUG" = "True" ]; then
TRAIN_BATCH_SIZE=4
PPO_MINI_BATCH=4
MAX_PROMPT_LENGTH=1024
RES_LENGTH=1024
GROUP_SIZE=4
else
TRAIN_BATCH_SIZE=64
PPO_MINI_BATCH=64
MAX_PROMPT_LENGTH=2048
RES_LENGTH=2048
GROUP_SIZE=8
fi

ACTOR_LR=1e-6
LR_WARMUP_RATIO=0.01
ENTROPY_COEFF=0.0
USE_KL_LOSS=True
KL_LOSS_COEF=0.001
KL_LOSS_TYPE=low_var_kl

if [ "$DEBUG" = "True" ]; then
PPO_MICRO_BATCH_SIZE_PER_GPU=1
LOG_PROB_MICRO_BATCH_SIZE=1
TENSOR_MODEL_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.6
else
PPO_MICRO_BATCH_SIZE_PER_GPU=8
LOG_PROB_MICRO_BATCH_SIZE=16
TENSOR_MODEL_PARALLEL_SIZE=2
GPU_MEMORY_UTILIZATION=0.6
fi


PARAM_OFFLOAD=False
OPTIMIZER_OFFLOAD=False
REF_PARAM_OFFLOAD=True

USE_REMOVE_PADDING=True
ENABLE_GRADIENT_CHECKPOINTING=True

# =============================================================================
#  SETUP
# =============================================================================
train_files="$TRAIN_DATA_PATH"
if [ "$DEBUG" = "True" ]; then
test_files="['$EVAL_DATA_PATH_1']"
# test_files="['$EVAL_DATA_PATH_1','$EVAL_DATA_PATH_2']"
else
test_files="['$EVAL_DATA_PATH_1','$EVAL_DATA_PATH_2','$EVAL_DATA_PATH_3','$EVAL_DATA_PATH_4','$EVAL_DATA_PATH_5','$EVAL_DATA_PATH_6','$EVAL_DATA_PATH_7','$EVAL_DATA_PATH_8','$EVAL_DATA_PATH_9','$EVAL_DATA_PATH_10']"
fi

mkdir -p $CHECKPOINT_DIR
mkdir -p "$(dirname $LOG_FILE)"
mkdir -p $MLFLOW_DIR
mkdir -p $ROLLOUT_DIR
mkdir -p $VALIDATION_DIR

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
actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
actor_rollout_ref.actor.use_kl_loss=$USE_KL_LOSS \
actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
actor_rollout_ref.actor.kl_loss_type=$KL_LOSS_TYPE \
actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
actor_rollout_ref.model.enable_gradient_checkpointing=$ENABLE_GRADIENT_CHECKPOINTING \
actor_rollout_ref.actor.fsdp_config.param_offload=$PARAM_OFFLOAD \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OPTIMIZER_OFFLOAD \
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE \
actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_MODEL_PARALLEL_SIZE \
actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
actor_rollout_ref.rollout.n=$GROUP_SIZE \
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE \
actor_rollout_ref.ref.fsdp_config.param_offload=$REF_PARAM_OFFLOAD \
algorithm.use_kl_in_reward=False \
reward_model.reward_manager=batch \
custom_reward_function.path=$CUSTOM_REWARD_PATH \
custom_reward_function.name=compute_score_batch \
trainer.critic_warmup=0 \
trainer.default_hdfs_dir=null \
trainer.default_local_dir=$CHECKPOINT_DIR \
trainer.logger='["console","mlflow"]' \
trainer.project_name=$EXPERIMENT_NAME \
trainer.experiment_name=$EXPERIMENT_NAME \
trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
trainer.nnodes=1 \
trainer.save_freq=$SAVE_FREQ \
trainer.test_freq=$TEST_FREQ \
trainer.total_epochs=$TOTAL_EPOCHS \
trainer.rollout_data_dir=$ROLLOUT_DIR \
trainer.validation_data_dir=$VALIDATION_DIR 2>&1 | tee $LOG_FILE









