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

## Part 1: Data Scale Effects

### Objective
Test the effect of training data quantity on RL learning signals. Specifically: does small-scale data (100-2000 samples) provide sufficient signal for effective learning across different model families and domains?

### Experimental Setup

**Models:**
- Qwen2.5-Math-1.5B (math-specialized)
- Qwen2.5-3B (general)
- Llama3.2-3B-Instruct (general)

**Domains:**
- Math (Skywork-OR1-RL-Data)
- Science (SCP-25K)
- Graph (Reasoning Gym)
- Logic (Reasoning Gym)

**Data Sizes:** 100, 500, 1000, 2000 samples

**Total Runs:** 36 (9 model-domain pairs × 4 data sizes)

### Running Experiments

#### For Qwen Models
Use script: `script/data_effect/qwen_data_effect.sh`

#### For Llama Models
Use script: `script/data_effect/llama_data_effect.sh`

### Configuration Steps

**1. Set Base Model Path (line 14):**
```bash
# For Qwen
export BASE_MODEL=/local2/salman/model/reward_signal_project/Qwen2.5-Math-1.5B

# For Llama
export BASE_MODEL=/local2/salman/model/reward_signal_project/Llama-3.2-3B-Instruct
```

**2. Set Save Directory (line 20):**
```bash
export SAVE_DIR="/local2/salman/reward_signal_results/data_effect/qwen"
# or
export SAVE_DIR="/local2/salman/reward_signal_results/data_effect/llama"
```

**3. Set Experiment Name (line 23):**
```bash
export EXPERIMENT_NAME=qwen_data_100
# Change for each run: qwen_data_100, qwen_data_500, llama_data_1000, etc.
```

**4. Set Training Epochs (line 25):**
```bash
# Match epochs to sample size:
TOTAL_EPOCHS=496  # for 64 or 100 samples
TOTAL_EPOCHS=71   # for 500 samples
TOTAL_EPOCHS=32   # for 1000 samples
TOTAL_EPOCHS=16   # for 2000 samples
```

**5. Set GPUs (line 39):**
```bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}
# Change GPU IDs as needed: "0,1,2,3" or "4,5,6,7"
```

**6. Set Training Data Path (line 49 for production, line 45 for debug):**
```bash
# For Qwen models:
TRAIN_DATA_PATH="$(pwd)/data/math/train/qwen-1-5b/qwen_sky_math_100.parquet"
# Available: qwen_sky_math_{64,100,500,1000,2000}.parquet

# For Llama models:
TRAIN_DATA_PATH="$(pwd)/data/math/train/llama-3b/qwen_sky_math_100.parquet"
# Available: qwen_sky_math_{64,100,500,1000,2000}.parquet
```

### Execution

```bash
bash script/data_effect/qwen_data_effect.sh
# or
bash script/data_effect/llama_data_effect.sh
```

### Data Locations

Training data:
- Qwen: `data/math/train/qwen-1-5b/`
- Llama: `data/math/train/llama-3b/`

Evaluation data (shared):
- AIME 2024: `data/math/eval_data/aime2024.parquet`
- AIME 2025: `data/math/eval_data/aime2025.parquet`
- MATH-500: `data/math/eval_data/math500.parquet`
- AMC Test: `data/math/eval_data/amc_test.parquet`
- SCP Tests: `data/math/eval_data/scp_test_{difficult,medium,very_difficult}_*.parquet`

---

## Framework

Built on [VERL](https://github.com/volcengine/verl) - efficient RL training framework for LLMs ([HybridFlow paper](https://arxiv.org/abs/2409.19256v2))