# Understanding Reward Signal Quality in RL Training

We're systematically testing **when RL reward signals actually teach models better reasoning vs just overfitting**, by varying data scale, reward types, exploration, difficulty, and training methods across multiple models and domains.

## Repository Structure

```
reward-signal-analysis/
├── verl/                    # VERL RL framework (GRPO/PPO training)
├── script/                  # Training scripts and reward functions
│   ├── llama/              # Llama-3.1-8B configurations
│   └── novel_math_gen_rm_reward_function.py
├── data/                    # Training/eval datasets (8K samples)
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/salman-lui/reward-signal-analysis.git
cd reward-signal-analysis

conda create -n open-prm python=3.10
conda activate open-prm
pip install -e .
```

## Configuration

1. **Add tokens** to training scripts:
   ```bash
   export HF_TOKEN="your_huggingface_token"
   export WANDB_API_KEY="your_wandb_key"
   ```

2. **Set paths** in `script/llama/m_family_main_full_rlvr_8k.sh`:
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

## Key Features

- **Algorithm**: GRPO (Group Relative Policy Optimization)
- **Reward**: Rule-based mathematical verification (0/1 binary)
- **Training data**: 8K math problems (varying difficulty)
- **Evaluation**: AIME 2024/2025, MATH-500, AMC
- **Models**: Llama-3.1-8B-Instruct baseline

## Framework

Built on [VERL](https://github.com/volcengine/verl) - efficient RL training framework for LLMs ([HybridFlow paper](https://arxiv.org/abs/2409.19256v2))