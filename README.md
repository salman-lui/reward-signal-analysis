# Understanding Reward Signal Quality in RL Training

We're systematically testing **when RL reward signals actually teach models better reasoning vs just overfitting**, by varying data scale, reward types, exploration, difficulty, and training methods across multiple models and domains.

## Repository Structure

```
reward-signal-analysis/
├── verl/                    # VERL RL framework
├── script/old/             # Training scripts
│   └── rlvr_8k.sh          # GRPO training script
├── reward_function.py       # Custom reward function
├── data/old/               # Training/eval datasets
│   ├── train_novel_hybrid_8k_with_gt.parquet  # 8K training data
│   └── eval_data/          # AIME 2024/2025, MATH-500, AMC test sets
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

**Before running, edit the top of `script/old/rlvr_8k.sh` to set:**

```bash
export BASE_MODEL=/path/to/your/model        # Add your base model path
export EXPERIMENT_NAME=your_experiment_name  # Add your experiment name
```

**Optional:** Modify save directories (auto-set based on DEBUG mode):
- Debug mode: `/local2/salman/debug_save`
- Production mode: `/local2/salman/reward_signal_results`

All training/evaluation data is included in `data/old/` (no external dependencies).

## Running Training

**Production mode (4 GPUs, batch=64) - Default:**
```bash
bash script/old/rlvr_8k.sh
```

**Debug mode (1 GPU, small batch):**
```bash
DEBUG=True bash script/old/rlvr_8k.sh
```

## Framework

Built on [VERL](https://github.com/volcengine/verl) - efficient RL training framework for LLMs ([HybridFlow paper](https://arxiv.org/abs/2409.19256v2))