# Understanding Reward Signal Quality in RL Training

We're systematically testing **when RL reward signals actually teach models better reasoning vs just overfitting**, by varying data scale, reward types, exploration, difficulty, and training methods across multiple models and domains.

## Repository Structure

```
reward-signal-analysis/
├── verl/                    # VERL RL framework
├── script/                  # Training scripts
├── reward_function.py       # Custom reward function
├── data/                    # Training/eval datasets
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/salman-lui/reward-signal-analysis.git
cd reward-signal-analysis

conda create -n reward-signal python=3.11
conda activate reward-signal

# Install VERL with vLLM backend (installs all dependencies)
pip install -e .[vllm]

# Install Flash Attention
pip install flash-attn==2.6.3 --no-build-isolation

# Install additional dependencies
pip install mlflow math-verify
```

## Configuration

**Before running, edit `script/old/rlvr_8k.sh`:**

1. **Required:** Set your model, reward type, experiment name, and save directories
```bash
export REWARD_MODEL_TYPE=RULE_BASED          # Options: RULE_BASED, RANDOM_REWARD
export BASE_MODEL=/path/to/your/model        # Add your base model path
export EXPERIMENT_NAME=your_experiment_name  # Add your experiment name

# Change save directories (lines 18 and 20):
# Debug mode:      export SAVE_DIR="/your/debug/save/path"
# Production mode: export SAVE_DIR="/your/production/save/path"
```

2. **For different domains:** Change train/eval data paths
```bash
TRAIN_DATA_PATH="$(pwd)/data/your_domain/train.parquet"
EVAL_DATA_PATH_1="$(pwd)/data/your_domain/eval/test1.parquet"
# Add more eval paths as needed
```

## Running Training

**Production mode (4 GPUs, batch=64) - Default:**
```bash
bash script/data_effect/qwen_data_effect.sh
```

**Debug mode (1 GPU, small batch):**
```bash
DEBUG=True bash script/data_effect/qwen_data_effect.sh
```

## Framework

Built on [VERL](https://github.com/volcengine/verl) - efficient RL training framework for LLMs ([HybridFlow paper](https://arxiv.org/abs/2409.19256v2))