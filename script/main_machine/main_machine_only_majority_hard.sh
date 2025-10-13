#!/bin/bash
# Multi-mode VERL training script supporting 3 deployment scenarios
# SPECIALIZED FOR HARD PROBLEMS ONLY (difficulty 10-16) - 4K training samples with TTRL majority voting

# =============================================================================
#  MODE SELECTION - SET ONE OF THESE
# =============================================================================
# MODE 1: Debug mode (this machine, auto-detect GPUs from CUDA_VISIBLE_DEVICES)
# MODE 2: Production mode (this same machine, no debug, like rule_based_novel.sh)
# MODE 3: Production remote machine (production cluster with conda setup)

# Set your mode here:
export TRAINING_MODE=${TRAINING_MODE:-"MODE3"}  # Default to MODE1, can be overridden by environment

# Set CUDA_VISIBLE_DEVICES for MODE1 and MODE2 (this machine)
if [ "$TRAINING_MODE" = "MODE1" ] || [ "$TRAINING_MODE" = "MODE2" ]; then
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"1,2,3,4"}  # 4 GPUs to match reference
fi

echo "🚀 TRAINING MODE: $TRAINING_MODE"

# =============================================================================
#  EXPERIMENT CONFIGURATION - CHANGE THESE FOR DIFFERENT RUNS
# =============================================================================
export REWARD_MODEL_TYPE=MAJORITY_VOTE                        # TTRL-style majority voting for all modes
export GENRM_ENDPOINT="http://salsrahm-prm-cot-rm-1756423322-router.default.svc.cluster.local:8000/v1"
export GENRM_MODEL_PATH="/checkpoints/salsrahm-sandbox/sky-work/step-level-consistency/prm_cot"
export GENRM_MAX_RETRIES=10
export GENRM_BASE_DELAY=2
export GENRM_MAX_WORKERS=250

# Mode-specific DEBUG and naming - HARD PROBLEMS ONLY
if [ "$TRAINING_MODE" = "MODE1" ]; then
  export DEBUG=True
  export EXPERIMENT_NAME=debug_majority_vote_hard_only_ttrl
  export PROJECT_NAME='debug_majority_vote_hard_only_ttrl'
  export RAY_HEAD_PORT=6388
elif [ "$TRAINING_MODE" = "MODE2" ]; then
  export DEBUG=False
  export EXPERIMENT_NAME=prod_majority_vote_hard_only_ttrl
  export PROJECT_NAME='prod_majority_vote_hard_only_ttrl'
  export RAY_HEAD_PORT=6388
else  # MODE3
  export DEBUG=False
  export EXPERIMENT_NAME=majority_vote_hard_only_ttrl
  export PROJECT_NAME='majority_vote_hard_only_ttrl'
  export RAY_HEAD_PORT=6393
fi

# TTRL Configuration for Majority Voting
# TTRL is automatically enabled when REWARD_MODEL_TYPE=MAJORITY_VOTE
if [ "$TRAINING_MODE" = "MODE1" ]; then
  export TTRL_N_VOTES_PER_PROMPT=4         # Generate 4 responses for voting (debug mode)
  export TTRL_N_SAMPLES_PER_PROMPT=4       # Use 4 responses for PPO training (debug mode)
else
  export TTRL_N_VOTES_PER_PROMPT=32        # Generate 32 responses for voting (production)
  export TTRL_N_SAMPLES_PER_PROMPT=8       # Use 8 responses for PPO training (production)
fi
# =============================================================================

CONDA_PATH="/code/salsrahm-sandbox/miniconda3"
ENV_NAME="open-prm"
export GENRM_API_KEY="None"

# Mode-specific conda and environment setup
if [ "$TRAINING_MODE" = "MODE1" ] || [ "$TRAINING_MODE" = "MODE2" ]; then
  export DIST_NNODES=1
else
  # MODE3: Production remote machine - full conda setup
  source $CONDA_PATH/etc/profile.d/conda.sh
  $CONDA_PATH/bin/conda config --add envs_dirs /code/salsrahm-sandbox/envs
  conda activate /code/salsrahm-sandbox/envs/$ENV_NAME
   if [[ $(which python) != "/code/salsrahm-sandbox/envs/$ENV_NAME/bin/python" ]]; then
      conda deactivate
      conda activate /code/salsrahm-sandbox/envs/$ENV_NAME
  fi
  export DIST_NNODES="${REPLICA}"
fi

# Add missing authentication tokens (from quick_setup.sh)
export WANDB_API_KEY="YOUR_WANDB_API_KEY_HERE"
export HF_TOKEN='YOUR_HUGGINGFACE_TOKEN_HERE'


# =============================================================================
# 🔧 TRAINING CONFIGURATION (Usually don't need to change)
# =============================================================================

# Mode-specific model configuration
if [ "$TRAINING_MODE" = "MODE1" ]; then
  export BASE_MODEL=/local2/salman/model/Qwen2.5-Math-7B                     # Debug: Qwen2.5-Math-7B
elif [ "$TRAINING_MODE" = "MODE2" ]; then
  export BASE_MODEL=/local2/salman/model/Qwen2.5-Math-7B                     # Production: Qwen2.5-Math-7B (updated from DeepSeek)
else
  export BASE_MODEL=/checkpoints/salsrahm-sandbox/Qwen2.5-Math-7B            # Remote: Qwen2.5-Math-7B
fi

# Mode-specific vLLM and GPU configuration
if [ "$TRAINING_MODE" = "MODE1" ]; then
  unset VLLM_ATTENTION_BACKEND
  N_GPUS_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
elif [ "$TRAINING_MODE" = "MODE2" ]; then
  export VLLM_ATTENTION_BACKEND=XFORMERS
  N_GPUS_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
else
  export VLLM_ATTENTION_BACKEND=XFORMERS
  N_GPUS_PER_NODE=8
fi

# Mode-specific storage and logging paths
if [ "$TRAINING_MODE" = "MODE1" ]; then
  BASE_SAVE_PATH="/local2/salman/debug_save"
elif [ "$TRAINING_MODE" = "MODE2" ]; then
  BASE_SAVE_PATH="/local2/salman/co_train_qwen_2_5_math/only_majority_vote_hard_only"
else
  BASE_SAVE_PATH="/checkpoints/salsrahm-sandbox/novel_hybrid/only_majority_vote_hard_only"
fi

CHECKPOINT_DIR="$BASE_SAVE_PATH/checkpoints/$PROJECT_NAME"
LOG_FILE="$BASE_SAVE_PATH/logs/log_$PROJECT_NAME.log"
MLFLOW_DIR="$BASE_SAVE_PATH/mlflow/${PROJECT_NAME}_ml_flows"
ROLLOUT_DATA_DIR="$BASE_SAVE_PATH/grpo_rollouts/${PROJECT_NAME}_rollouts/training"
VALIDATION_DATA_DIR="$BASE_SAVE_PATH/grpo_rollouts/${PROJECT_NAME}_rollouts/validation"
export MLFLOW_TRACKING_URI=file://$MLFLOW_DIR

# Mode-specific training data paths - HARD PROBLEMS ONLY FOR MAJORITY VOTING
if [ "$TRAINING_MODE" = "MODE1" ]; then
  TRAIN_DATA_PATH="/home/salman/open-prm/data-skymath/rl_training_data/novel_hybrid/rl_training_no_step_instruction/easy_hard_train_data/hard_train_novel_hybrid_4k_with_gt.parquet"
  EVAL_DATA_PATH_1="/home/salman/open-prm/data-skymath/rl_training_data/novel_hybrid/rl_training_no_step_instruction/eval_data/aime2024.parquet"
  CUSTOM_REWARD_PATH="/home/salman/verl/salman_scripts_novel_hybrid/novel_math_gen_rm_reward_function.py"
elif [ "$TRAINING_MODE" = "MODE2" ]; then
  TRAIN_DATA_PATH="/home/salman/open-prm/data-skymath/rl_training_data/novel_hybrid/rl_training_no_step_instruction/easy_hard_train_data/hard_train_novel_hybrid_4k_with_gt.parquet"
  EVAL_DATA_PATH_1="/home/salman/open-prm/data-skymath/rl_training_data/novel_hybrid/rl_training_no_step_instruction/eval_data/aime2024.parquet"
  EVAL_DATA_PATH_2="/home/salman/open-prm/data-skymath/rl_training_data/novel_hybrid/rl_training_no_step_instruction/eval_data/aime2025.parquet"
  EVAL_DATA_PATH_3="/home/salman/open-prm/data-skymath/rl_training_data/novel_hybrid/rl_training_no_step_instruction/eval_data/math500.parquet"
  CUSTOM_REWARD_PATH="/home/salman/verl/salman_scripts_novel_hybrid/novel_math_gen_rm_reward_function.py"
else
  TRAIN_DATA_PATH="/checkpoints/salsrahm-sandbox/z_novel_rl_data_no_step_instruction/easy_hard_split/hard_train_novel_hybrid_4k_with_gt.parquet"
  EVAL_DATA_PATH_1="/checkpoints/salsrahm-sandbox/z_novel_rl_data_no_step_instruction/eval_data/aime2024.parquet"
  EVAL_DATA_PATH_2="/checkpoints/salsrahm-sandbox/z_novel_rl_data_no_step_instruction/eval_data/aime2025.parquet"
  EVAL_DATA_PATH_3="/checkpoints/salsrahm-sandbox/z_novel_rl_data_no_step_instruction/eval_data/math500.parquet"
  CUSTOM_REWARD_PATH="/code/salsrahm-sandbox/open-prm/verl/salman_scripts_novel_hybrid/novel_math_gen_rm_reward_function.py"
fi

# Mode-specific hyperparameters
if [ "$TRAINING_MODE" = "MODE1" ]; then
  TRAIN_BATCH_SIZE=4
  PPO_MINI_BATCH=4
  MAX_PROMPT_LENGTH=1024
  RES_LENGTH=1024
  GROUP_SIZE=4
elif [ "$TRAINING_MODE" = "MODE2" ]; then
  TRAIN_BATCH_SIZE=192                                    # Works: ÷4=48, ÷6=32
  PPO_MINI_BATCH=192
  MAX_PROMPT_LENGTH=2048
  RES_LENGTH=2048
  GROUP_SIZE=8
else
  TRAIN_BATCH_SIZE=192                                    # Works: ÷8=24, ÷64=3
  PPO_MINI_BATCH=192
  MAX_PROMPT_LENGTH=2048
  RES_LENGTH=2048
  GROUP_SIZE=8
fi

# Learning Rate and Training
ACTOR_LR=1e-6
LR_WARMUP_RATIO=0.01
ENTROPY_COEFF=0.0
USE_KL_LOSS=True                                            # GRPO guideline: set to True
KL_LOSS_COEF=0.001                                          # GRPO guideline: default coefficient
KL_LOSS_TYPE=low_var_kl                                     # GRPO guideline: recommended type

# Mode-specific GPU and memory settings
if [ "$TRAINING_MODE" = "MODE1" ]; then
  PPO_MICRO_BATCH_SIZE_PER_GPU=1
  LOG_PROB_MICRO_BATCH_SIZE=1
  TENSOR_MODEL_PARALLEL_SIZE=1
  GPU_MEMORY_UTILIZATION=0.5
elif [ "$TRAINING_MODE" = "MODE2" ]; then
  PPO_MICRO_BATCH_SIZE_PER_GPU=4
  LOG_PROB_MICRO_BATCH_SIZE=16
  TENSOR_MODEL_PARALLEL_SIZE=1
  GPU_MEMORY_UTILIZATION=0.6
else
  PPO_MICRO_BATCH_SIZE_PER_GPU=8
  LOG_PROB_MICRO_BATCH_SIZE=16
  TENSOR_MODEL_PARALLEL_SIZE=2
  GPU_MEMORY_UTILIZATION=0.6
fi

# Training Schedule
TOTAL_EPOCHS=15
SAVE_FREQ=25
TEST_FREQ=10

# FSDP Settings
PARAM_OFFLOAD=False
OPTIMIZER_OFFLOAD=False
REF_PARAM_OFFLOAD=True

# Settings
USE_REMOVE_PADDING=True
ENABLE_GRADIENT_CHECKPOINTING=True

# =============================================================================
# 🔧 DERIVED PATHS AND SETUP
# =============================================================================
train_files="$TRAIN_DATA_PATH"
if [ "$TRAINING_MODE" = "MODE1" ]; then
  test_files="['$EVAL_DATA_PATH_1']"
else
  test_files="['$EVAL_DATA_PATH_1','$EVAL_DATA_PATH_2','$EVAL_DATA_PATH_3']"
fi

# Mode-specific system setup
if [ "$TRAINING_MODE" = "MODE3" ]; then
  apt update
  apt-get install -y software-properties-common python3-dev cuda-minimal-build-12-5=12.5.1-1
fi

# Create directories
mkdir -p $CHECKPOINT_DIR
mkdir -p "$(dirname $LOG_FILE)"
mkdir -p $MLFLOW_DIR
mkdir -p $ROLLOUT_DATA_DIR
mkdir -p $VALIDATION_DATA_DIR

# Mode-specific Ray cluster setup
if [ "$TRAINING_MODE" = "MODE1" ] || [ "$TRAINING_MODE" = "MODE2" ]; then
  ray start --head --port=$RAY_HEAD_PORT
else
  # Multi-node setup for remote cluster
  if [ "${HOSTNAME##*-}" -eq 0 ]; then
      ray start --head --port=$RAY_HEAD_PORT
      until [ "$(ray status | grep node_ | wc -l | awk '{print $1}')" -eq $DIST_NNODES ]; do
          sleep 10
      done
  else
      HEAD_ADDR="${HOSTNAME%-*}-0"
      HEAD_PORT=$RAY_HEAD_PORT
      until (echo > /dev/tcp/${HEAD_ADDR}/${HEAD_PORT}) >/dev/null 2>&1; do
          sleep 5
      done
      ray start --address="${HEAD_ADDR}:${HEAD_PORT}" --block
  fi
fi

# Mode-specific training execution
if [ "$TRAINING_MODE" = "MODE1" ] || [ "$TRAINING_MODE" = "MODE2" ] || [ "${HOSTNAME##*-}" -eq 0 ]; then
  # Mode-specific setup and directory changes
  if [ "$TRAINING_MODE" = "MODE3" ]; then
      source /code/salsrahm-sandbox/quick_setup.sh
      cd /code/salsrahm-sandbox/open-prm/verl
  else
      cd /home/salman/verl
  fi

  # Training launch
  if [ "$TRAINING_MODE" = "MODE1" ]; then
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
      actor_rollout_ref.rollout.enforce_eager=False \
      actor_rollout_ref.rollout.free_cache_engine=False \
      actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
      actor_rollout_ref.rollout.n=$TTRL_N_SAMPLES_PER_PROMPT \
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
      trainer.project_name=$PROJECT_NAME \
      trainer.experiment_name=$EXPERIMENT_NAME \
      trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
      trainer.nnodes=$DIST_NNODES \
      trainer.save_freq=$SAVE_FREQ \
      trainer.test_freq=$TEST_FREQ \
      trainer.rollout_data_dir=$ROLLOUT_DATA_DIR \
      trainer.validation_data_dir=$VALIDATION_DATA_DIR \
      trainer.total_epochs=$TOTAL_EPOCHS 2>&1 | tee $LOG_FILE
   elif [ "$TRAINING_MODE" = "MODE2" ]; then
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
      actor_rollout_ref.rollout.n=$TTRL_N_SAMPLES_PER_PROMPT \
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
      trainer.project_name=$PROJECT_NAME \
      trainer.experiment_name=$EXPERIMENT_NAME \
      trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
      trainer.nnodes=$DIST_NNODES \
      trainer.save_freq=$SAVE_FREQ \
      trainer.test_freq=$TEST_FREQ \
      trainer.rollout_data_dir=$ROLLOUT_DATA_DIR \
      trainer.validation_data_dir=$VALIDATION_DATA_DIR \
      trainer.total_epochs=$TOTAL_EPOCHS 2>&1 | tee $LOG_FILE
    
  else
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
      actor_rollout_ref.rollout.n=$TTRL_N_SAMPLES_PER_PROMPT \
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
      trainer.project_name=$PROJECT_NAME \
      trainer.experiment_name=$EXPERIMENT_NAME \
      trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
      trainer.nnodes=$DIST_NNODES \
      trainer.save_freq=$SAVE_FREQ \
      trainer.test_freq=$TEST_FREQ \
      trainer.rollout_data_dir=$ROLLOUT_DATA_DIR \
      trainer.validation_data_dir=$VALIDATION_DATA_DIR \
      trainer.total_epochs=$TOTAL_EPOCHS 2>&1 | tee $LOG_FILE
  fi
else
  sleep infinity
fi















