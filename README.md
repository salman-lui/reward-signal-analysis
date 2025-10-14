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

Full experimental details: [Hybrid Reward Document](https://docs.google.com/document/d/18asovJnrXA19ENS-YmS0-JopuGVjCCGncw_9K-DxHRw/edit?usp=sharing)

### Objective
Test if small-scale data (64-2000 samples) provides sufficient signal for RL learning across model families and domains.

### Setup

**Models:** Qwen2.5-Math-1.5B, Qwen2.5-3B, Llama3.2-3B-Instruct

**Domains:** Math (Skywork-OR1-RL-Data), Science (SCP-25K), Graph (Reasoning Gym)

**Data Sizes:** 64, 100, 500, 1000, 2000 samples

**Total Runs:** 45 (9 model-domain pairs × 5 data sizes)

### Configuration

Edit `script/data_effect/qwen_data_effect.sh` (Qwen) or `script/data_effect/llama_data_effect.sh` (Llama):

**1. Base Model (line 14):**
```bash
export BASE_MODEL=/path/to/your/model
```

**2. Save Directory (line 20):**
```bash
export SAVE_DIR="/path/to/save/results"
```

**3. Experiment Name (line 23):**
```bash
export EXPERIMENT_NAME=your_unique_experiment_name
```

**4. Training Epochs (line 25):**
```bash
TOTAL_EPOCHS=496  # 64/100 samples
TOTAL_EPOCHS=71   # 500 samples
TOTAL_EPOCHS=32   # 1000 samples
TOTAL_EPOCHS=16   # 2000 samples
```

**5. GPUs (line 39):**
```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"
```

**6. Training Data (line 49):**
```bash
TRAIN_DATA_PATH="$(pwd)/data/math/train/qwen-1-5b/qwen_sky_math_100.parquet" # for qwen 100 sample
```

Available training data:
- Qwen: `data/math/train/qwen-1-5b/qwen_sky_math_{64,100,500,1000,2000}.parquet`
- Llama: `data/math/train/llama-3b/qwen_sky_math_{64,100,500,1000,2000}.parquet`

### Evaluation Data

All experiments use the same evaluation benchmarks (configured in lines 50-56):

```
data/math/eval_data/
├── aime2024.parquet                    # AIME 2024
├── aime2025.parquet                    # AIME 2025
├── math500.parquet                     # MATH-500
├── amc_test.parquet                    # AMC Test
├── scp_test_difficult_1.parquet        # SCP Difficult
├── scp_test_very_difficult_0.parquet   # SCP Very Difficult
└── scp_test_medium_2_8.parquet         # SCP Medium
```

### Run

```bash
bash script/data_effect/qwen_data_effect.sh
```

---

## Framework

Built on [VERL](https://github.com/volcengine/verl) - efficient RL training framework for LLMs ([HybridFlow paper](https://arxiv.org/abs/2409.19256v2))