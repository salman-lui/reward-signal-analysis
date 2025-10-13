# Understanding Reward Signal Quality in RL Training

We're systematically testing **when RL reward signals actually teach models better reasoning vs just overfitting**, by varying data scale, reward types, exploration, difficulty, and training methods across multiple models and domains.

## Repository Structure

```
reward-signal-analysis/
├── verl/                    # VERL RL framework
├── script/old/             # Training scripts
│   └── rlvr_8k.sh          # GRPO training script
├── reward_function.py       # Custom reward function
├── data/                    # Training/eval datasets (8K train + eval sets)
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/salman-lui/reward-signal-analysis.git
cd reward-signal-analysis

conda create -n reward-signal python=3.11
conda activate reward-signal
pip install -e .
```

## Configuration

Edit the top of `script/old/rlvr_8k.sh` to configure your experiment:
```bash
export DEBUG=False  # True: 1 GPU debug, False: 4 GPU production
export BASE_MODEL=/local2/salman/model/Llama-3.1-8B-Instruct
export SAVE_DIR="/local2/salman/reward_signal_results"
export EXPERIMENT_NAME=llama_3_1_8b_rule_based_8k
```

## Running Training

**Debug mode (1 GPU, small batch):**
```bash
DEBUG=True bash script/old/rlvr_8k.sh
```

**Production mode (4 GPUs, batch=64):**
```bash
DEBUG=False bash script/old/rlvr_8k.sh
```

## Framework

Built on [VERL](https://github.com/volcengine/verl) - efficient RL training framework for LLMs ([HybridFlow paper](https://arxiv.org/abs/2409.19256v2))