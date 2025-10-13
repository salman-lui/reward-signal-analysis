# Understanding Reward Signal Quality in RL Training

We're systematically testing **when RL reward signals actually teach models better reasoning vs just overfitting**, by varying data scale, reward types, exploration, difficulty, and training methods across multiple models and domains.

## Repository Structure

```
reward-signal-analysis/
├── verl/                    # VERL RL framework (GRPO/PPO training)
├── script/                  # Training scripts
│   └── llama/              # Llama-3.1-8B configurations
├── reward_function.py       # Custom reward function
├── data/                    # Training/eval datasets (8K samples)
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/salman-lui/reward-signal-analysis.git
cd reward-signal-analysis

conda create -n reward-signal python=3.10
conda activate reward-signal
pip install -e .
```

## Configuration

Set paths in `script/llama/m_family_main_full_rlvr_8k.sh`:
```bash
export TRAINING_MODE="MODE1"  # MODE1: debug, MODE2: prod-local, MODE3: cluster
export BASE_MODEL="/path/to/Llama-3.1-8B-Instruct"
```

## Running Training

**Debug mode (single GPU):**
```bash
export TRAINING_MODE="MODE1"
export CUDA_VISIBLE_DEVICES="0"
bash script/llama/m_family_main_full_rlvr_8k.sh
```

**Production mode (8 GPUs):**
```bash
export TRAINING_MODE="MODE2"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
bash script/llama/m_family_main_full_rlvr_8k.sh
```

## Framework

Built on [VERL](https://github.com/volcengine/verl) - efficient RL training framework for LLMs ([HybridFlow paper](https://arxiv.org/abs/2409.19256v2))