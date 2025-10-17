import matplotlib.pyplot as plt
import numpy as np
import os

"""
QUICK START GUIDE
=================
1. Update CONFIGURATION section below:
   - TRAINING_DOMAIN: e.g., "Math", "Science", "SCP"
   - MODEL_NAME: e.g., "Llama-3.2-3B-Instruct", "Qwen2.5-3B-Instruct"
   - EXPERIMENTS: Update paths to your MLflow metrics directories
   
2. Run the script:
   python plot/data_scale_effect/data_effect_result.py
   
3. Find plots in:
   plot/data_scale_effect/{domain}_{model}/
   
Example: Math + Llama → plot/data_scale_effect/math_llama/

Plots generated:
- reward_adv_length.png (critic metrics)
- entropy_grad_norm.png (actor metrics)  
- math_benchmark.png (MATH-500, AMC, AIME)
- scp_test.png (SCP difficulty levels)
- stem_benchmark.png (SuperGPQA, GPQA, MMLU - if available)
"""

# ============================================================================
# CONFIGURATION - CHANGE THESE FOR DIFFERENT RUNS
# ============================================================================
# Domain and Model
TRAINING_DOMAIN = "Science(SCP)"
MODEL_NAME = "Llama-3.2-3B-Instruct"

# Benchmark Names (for plot titles)
MATH_BENCHMARK_NAME = "Math Benchmark Performance"
SCP_BENCHMARK_NAME = "SCP Test Set Performance"
STEM_BENCHMARK_NAME = "STEM Benchmark Performance"


# # Experiments to plot (label, base_path) for Qwen Math
# EXPERIMENTS = [
#     ("64 samples", "/local2/salman/reward_signal_results/data_effect/qwen/qwen_data_64/mlflow/731139593762122798/d7be685f982b4bfbbd041a18ee551f50/metrics"),
#     ("128 samples", "/local2/salman/reward_signal_results/data_effect/qwen/qwen_data_128/mlflow/433222232317221588/650e9f701be84b878ac1bc5747a59b96/metrics"),
#     ("512 samples", "/local2/salman/reward_signal_results/data_effect/qwen/qwen_data_512/mlflow/973145789152362639/7f0099b9c2dd45ed83e1815ca18f9f26/metrics"),
#     ("1024 samples", "/local2/salman/reward_signal_results/data_effect/qwen/qwen_data_1024/mlflow/907235872503953152/9eb3e7be8b814cdb801f98f44c784b22/metrics"),
#     ("2048 samples", "/local2/salman/reward_signal_results/data_effect/qwen/qwen_data_2048/mlflow/490719113903161021/9d2611303e5d4cfb8b225c9681ac8e6c/metrics")
# ]

# # Experiments to plot (label, base_path) for Llama Math
# EXPERIMENTS = [
#     ("64 samples", "/local2/salman/reward_signal_results/data_effect/llama/llama_data_64/mlflow/184325052765024322/17c418f2c19b449081eeec0df8913786/metrics"),
#     ("128 samples", "/local2/salman/reward_signal_results/data_effect/llama/llama_data_128/mlflow/599100423772582426/0d0e26a0ec0d484b9aa98faa48994629/metrics"),
#     ("512 samples", "/local2/salman/reward_signal_results/data_effect/llama/llama_data_512/mlflow/343431917358588485/561906f95e174d089d8768a41a0d189a/metrics"),
#     ("1024 samples", "/local2/salman/reward_signal_results/data_effect/llama/llama_data_1024/mlflow/261221308158797635/13c80cd5ae8b4a74af1f508b50e710fe/metrics"),
#     ("2048 samples", "/local2/salman/reward_signal_results/data_effect/llama/llama_data_2048/mlflow/855510272267812287/945f5e3a3249447c92b4202c464c30ad/metrics")
# ]

# Experiments to plot (label, base_path) for Llama SCP
EXPERIMENTS = [
    ("64 samples", "/local2/salman/reward_signal_results/data_effect/scp/llama3b/llama_scp_64/mlflow/699028690186535534/d65cc97963374af89a121fa2d1bd25cc/metrics"),
    ("128 samples", "/local2/salman/reward_signal_results/data_effect/scp/llama3b/llama_scp_128/mlflow/440862638874843127/0abcc31a807a43aaa06d24a869ffb356/metrics"),
    ("512 samples", "/local2/salman/reward_signal_results/data_effect/scp/llama3b/llama_scp_512/mlflow/851771574621329669/efd6ead5d4da4a1da9a7e72a1b20ff09/metrics"),
    ("1024 samples", "/local2/salman/reward_signal_results/data_effect/scp/llama3b/llama_scp_1024/mlflow/188901959200617046/65590636728745daa9eb9820c726ebc7/metrics")
]

# Output configuration
BASE_OUTPUT_DIR = "/home/salman/reward-signal-analysis/plot/data_scale_effect"
# Automatically generate subdirectory and filenames from domain and model
model_short = MODEL_NAME.split('-')[0].lower()  # e.g., "Llama-3.2-3B-Instruct" -> "llama"
SUBDIR_NAME = f"{TRAINING_DOMAIN.lower()}_{model_short}"  # e.g., "math_llama"
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, SUBDIR_NAME)
OUTPUT_FILENAME_CRITIC = f"{model_short}_{TRAINING_DOMAIN.lower()}_reward_adv_length.png"
OUTPUT_FILENAME_ACTOR = f"{model_short}_{TRAINING_DOMAIN.lower()}_entropy_grad_norm.png"
OUTPUT_FILENAME_MATH_BENCH = f"{model_short}_{TRAINING_DOMAIN.lower()}_math_benchmark.png"
OUTPUT_FILENAME_SCP_TEST = f"{model_short}_{TRAINING_DOMAIN.lower()}_scp_test.png"
OUTPUT_FILENAME_STEM_BENCH = f"{model_short}_{TRAINING_DOMAIN.lower()}_stem_benchmark.png"
# ============================================================================


def read_mlflow_metric(file_path):
    """
    Read MLflow metric file.
    Format: timestamp value step
    Returns: steps (list), values (list)
    """
    steps = []
    values = []
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    timestamp = float(parts[0])
                    value = float(parts[1])
                    step = int(parts[2])
                    steps.append(step)
                    values.append(value)
        return steps, values
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return [], []
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return [], []


def main():
    # Read metrics for all experiments
    colors = ['purple', 'steelblue', 'forestgreen', 'coral', 'crimson']
    critic_data = []
    actor_data = []
    math_bench_data = []
    scp_test_data = []
    stem_bench_data = []
    
    # Load critic metrics: rewards, advantages, response_length
    print("Loading critic metrics...")
    for label, base_path in EXPERIMENTS:
        rewards_path = os.path.join(base_path, "critic/rewards/mean")
        advantages_path = os.path.join(base_path, "critic/advantages/mean")
        response_length_path = os.path.join(base_path, "response_length/mean")
        
        rewards_steps, rewards_values = read_mlflow_metric(rewards_path)
        advantages_steps, advantages_values = read_mlflow_metric(advantages_path)
        response_steps, response_values = read_mlflow_metric(response_length_path)
        
        if not rewards_steps or not advantages_steps or not response_steps:
            print(f"Warning: Failed to load critic data for {label}")
            continue
        
        critic_data.append({
            'label': label,
            'rewards': (rewards_steps, rewards_values),
            'advantages': (advantages_steps, advantages_values),
            'response_length': (response_steps, response_values)
        })
    
    # Load actor metrics: entropy, grad_norm
    print("Loading actor metrics...")
    for label, base_path in EXPERIMENTS:
        entropy_path = os.path.join(base_path, "actor/entropy")
        grad_norm_path = os.path.join(base_path, "actor/grad_norm")
        
        entropy_steps, entropy_values = read_mlflow_metric(entropy_path)
        grad_norm_steps, grad_norm_values = read_mlflow_metric(grad_norm_path)
        
        if not entropy_steps or not grad_norm_steps:
            print(f"Warning: Failed to load actor data for {label}")
            continue
        
        actor_data.append({
            'label': label,
            'entropy': (entropy_steps, entropy_values),
            'grad_norm': (grad_norm_steps, grad_norm_values)
        })
    
    # Load math benchmark validation metrics (flexible - only load what's available)
    print("Loading math benchmark metrics...")
    for label, base_path in EXPERIMENTS:
        math500_path = os.path.join(base_path, "val-core/HuggingFaceH4/MATH-500/reward/mean_at_1")
        amc_path = os.path.join(base_path, "val-core/test-amc/reward/mean_at_1")
        aime24_path = os.path.join(base_path, "val-core/test-math-aime24/reward/mean_at_1")
        aime25_path = os.path.join(base_path, "val-core/test-math-aime25/reward/mean_at_1")
        
        math500_steps, math500_values = read_mlflow_metric(math500_path)
        amc_steps, amc_values = read_mlflow_metric(amc_path)
        aime24_steps, aime24_values = read_mlflow_metric(aime24_path)
        aime25_steps, aime25_values = read_mlflow_metric(aime25_path)
        
        # Only add if at least one benchmark exists
        if math500_steps or amc_steps or aime24_steps or aime25_steps:
            math_bench_data.append({
                'label': label,
                'math500': (math500_steps, math500_values) if math500_steps else ([], []),
                'amc': (amc_steps, amc_values) if amc_steps else ([], []),
                'aime24': (aime24_steps, aime24_values) if aime24_steps else ([], []),
                'aime25': (aime25_steps, aime25_values) if aime25_steps else ([], [])
            })
        else:
            print(f"Warning: No math benchmark data found for {label}")
    
    # Load SCP test set validation metrics
    print("Loading SCP test set metrics...")
    for label, base_path in EXPERIMENTS:
        scp_diff_path = os.path.join(base_path, "val-core/test-math-scp-difficult/reward/mean_at_1")
        scp_med_path = os.path.join(base_path, "val-core/test-math-scp-medium/reward/mean_at_1")
        scp_vdiff_path = os.path.join(base_path, "val-core/test-math-scp-very-difficult/reward/mean_at_1")
        
        scp_diff_steps, scp_diff_values = read_mlflow_metric(scp_diff_path)
        scp_med_steps, scp_med_values = read_mlflow_metric(scp_med_path)
        scp_vdiff_steps, scp_vdiff_values = read_mlflow_metric(scp_vdiff_path)
        
        if not scp_diff_steps or not scp_med_steps or not scp_vdiff_steps:
            print(f"Warning: Failed to load SCP test data for {label}")
            continue
        
        scp_test_data.append({
            'label': label,
            'difficult': (scp_diff_steps, scp_diff_values),
            'medium': (scp_med_steps, scp_med_values),
            'very_difficult': (scp_vdiff_steps, scp_vdiff_values)
        })
    
    # Load STEM benchmark validation metrics (optional - only if they exist)
    print("Loading STEM benchmark metrics (if available)...")
    for label, base_path in EXPERIMENTS:
        supergpqa_path = os.path.join(base_path, "val-core/stem__supergpqa/reward/mean_at_1")
        gpqa_path = os.path.join(base_path, "val-core/stem__gpqa_diamond/reward/mean_at_1")
        mmlu_path = os.path.join(base_path, "val-core/mmlu_sci/reward/mean_at_2")
        
        supergpqa_steps, supergpqa_values = read_mlflow_metric(supergpqa_path)
        gpqa_steps, gpqa_values = read_mlflow_metric(gpqa_path)
        mmlu_steps, mmlu_values = read_mlflow_metric(mmlu_path)
        
        # Only add if at least one metric exists
        if supergpqa_steps or gpqa_steps or mmlu_steps:
            stem_bench_data.append({
                'label': label,
                'supergpqa': (supergpqa_steps, supergpqa_values) if supergpqa_steps else ([], []),
                'gpqa': (gpqa_steps, gpqa_values) if gpqa_steps else ([], []),
                'mmlu': (mmlu_steps, mmlu_values) if mmlu_steps else ([], [])
            })
    
    if not critic_data and not actor_data and not math_bench_data and not scp_test_data and not stem_bench_data:
        print("Error: No data loaded")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_title = f"Training Domain: {TRAINING_DOMAIN} | Model: {MODEL_NAME}"
    
    # ========================================================================
    # PLOT 1: Critic Metrics (Rewards, Advantages, Response Length)
    # ========================================================================
    if critic_data:
        print("\nCreating critic metrics plot...")
        fig1, axes1 = plt.subplots(1, 3, figsize=(24, 7))
        fig1.suptitle(plot_title, fontsize=24, fontweight='bold', y=1.02)
        
        # Plot 1.1: Mean Training Reward
        ax1 = axes1[0]
        for i, data in enumerate(critic_data):
            steps, values = data['rewards']
            ax1.plot(steps, values, color=colors[i], linewidth=3, marker='o', markersize=5, label=data['label'])
        ax1.set_xlabel('Training Steps', fontsize=18)
        ax1.set_ylabel('Mean Training Reward', fontsize=18)
        ax1.set_title('Mean Training Reward', fontsize=20, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', which='major', labelsize=16)
        ax1.legend(fontsize=16, loc='best')
        
        # Plot 1.2: Mean Advantage
        ax2 = axes1[1]
        for i, data in enumerate(critic_data):
            steps, values = data['advantages']
            ax2.plot(steps, values, color=colors[i], linewidth=3, marker='o', markersize=5, label=data['label'])
        ax2.set_xlabel('Training Steps', fontsize=18)
        ax2.set_ylabel('Mean Advantage', fontsize=18)
        ax2.set_title('Mean Advantage', fontsize=20, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', which='major', labelsize=16)
        ax2.legend(fontsize=16, loc='best')
        
        # Plot 1.3: Mean Response Length
        ax3 = axes1[2]
        for i, data in enumerate(critic_data):
            steps, values = data['response_length']
            ax3.plot(steps, values, color=colors[i], linewidth=3, marker='o', markersize=5, label=data['label'])
        ax3.set_xlabel('Training Steps', fontsize=18)
        ax3.set_ylabel('Mean Response Length', fontsize=18)
        ax3.set_title('Mean Response Length', fontsize=20, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='both', which='major', labelsize=16)
        ax3.legend(fontsize=16, loc='best')
        
        plt.tight_layout()
        output_path_critic = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME_CRITIC)
        fig1.savefig(output_path_critic, dpi=300, bbox_inches='tight')
        print(f"Saved critic plot to: {output_path_critic}")
        plt.close(fig1)
    
    # ========================================================================
    # PLOT 2: Actor Metrics (Entropy, Grad Norm)
    # ========================================================================
    if actor_data:
        print("\nCreating actor metrics plot...")
        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
        fig2.suptitle(plot_title, fontsize=24, fontweight='bold', y=1.02)
        
        # Plot 2.1: Entropy
        ax1 = axes2[0]
        for i, data in enumerate(actor_data):
            steps, values = data['entropy']
            ax1.plot(steps, values, color=colors[i], linewidth=3, marker='o', markersize=5, label=data['label'])
        ax1.set_xlabel('Training Steps', fontsize=18)
        ax1.set_ylabel('Entropy', fontsize=18)
        ax1.set_title('Entropy', fontsize=20, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', which='major', labelsize=16)
        ax1.legend(fontsize=16, loc='best')
        
        # Plot 2.2: Gradient Norm
        ax2 = axes2[1]
        for i, data in enumerate(actor_data):
            steps, values = data['grad_norm']
            ax2.plot(steps, values, color=colors[i], linewidth=3, marker='o', markersize=5, label=data['label'])
        ax2.set_xlabel('Training Steps', fontsize=18)
        ax2.set_ylabel('Gradient Norm', fontsize=18)
        ax2.set_title('Gradient Norm', fontsize=20, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', which='major', labelsize=16)
        ax2.legend(fontsize=16, loc='best')
        
        plt.tight_layout()
        output_path_actor = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME_ACTOR)
        fig2.savefig(output_path_actor, dpi=300, bbox_inches='tight')
        print(f"Saved actor plot to: {output_path_actor}")
        plt.close(fig2)
    
    # ========================================================================
    # PLOT 3: Math Benchmark Performance (flexible - plots only available benchmarks)
    # ========================================================================
    if math_bench_data:
        print("\nCreating math benchmark plot...")
        
        # Determine which benchmarks are available
        available_benchmarks = []
        if any(data['math500'][0] for data in math_bench_data):
            available_benchmarks.append(('math500', 'MATH-500'))
        if any(data['amc'][0] for data in math_bench_data):
            available_benchmarks.append(('amc', 'AMC'))
        if any(data['aime24'][0] for data in math_bench_data):
            available_benchmarks.append(('aime24', 'AIME-2024'))
        if any(data['aime25'][0] for data in math_bench_data):
            available_benchmarks.append(('aime25', 'AIME-2025'))
        
        if available_benchmarks:
            num_plots = len(available_benchmarks)
            fig3, axes3 = plt.subplots(1, num_plots, figsize=(7*num_plots, 7))
            if num_plots == 1:
                axes3 = [axes3]  # Make it iterable
            fig3.suptitle(f"{MATH_BENCHMARK_NAME} | {TRAINING_DOMAIN} | {MODEL_NAME}", fontsize=24, fontweight='bold', y=1.02)
            
            for idx, (bench_key, bench_title) in enumerate(available_benchmarks):
                ax = axes3[idx]
                for i, data in enumerate(math_bench_data):
                    steps, values = data[bench_key]
                    if steps:  # Only plot if data exists
                        values_percent = [v * 100 for v in values]  # Convert to percentage
                        ax.plot(steps, values_percent, color=colors[i], linewidth=3, marker='o', markersize=5, label=data['label'])
                ax.set_xlabel('Training Steps', fontsize=18)
                ax.set_ylabel('Accuracy (%)', fontsize=18)
                ax.set_title(bench_title, fontsize=20, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='both', which='major', labelsize=16)
                ax.legend(fontsize=16, loc='best')
            
            plt.tight_layout()
            output_path_math = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME_MATH_BENCH)
            fig3.savefig(output_path_math, dpi=300, bbox_inches='tight')
            print(f"Saved math benchmark plot to: {output_path_math}")
            plt.close(fig3)
    
    # ========================================================================
    # PLOT 4: SCP Test Set Performance (Difficult, Medium, Very Difficult)
    # ========================================================================
    if scp_test_data:
        print("\nCreating SCP test set plot...")
        fig4, axes4 = plt.subplots(1, 3, figsize=(24, 7))
        fig4.suptitle(f"{SCP_BENCHMARK_NAME} | {TRAINING_DOMAIN} | {MODEL_NAME}", fontsize=24, fontweight='bold', y=1.02)
        
        # Plot 4.1: SCP-Difficult
        ax1 = axes4[0]
        for i, data in enumerate(scp_test_data):
            steps, values = data['difficult']
            values_percent = [v * 100 for v in values]  # Convert to percentage
            ax1.plot(steps, values_percent, color=colors[i], linewidth=3, marker='o', markersize=5, label=data['label'])
        ax1.set_xlabel('Training Steps', fontsize=18)
        ax1.set_ylabel('Accuracy (%)', fontsize=18)
        ax1.set_title('SCP-Difficult', fontsize=20, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', which='major', labelsize=16)
        ax1.legend(fontsize=16, loc='best')
        
        # Plot 4.2: SCP-Medium
        ax2 = axes4[1]
        for i, data in enumerate(scp_test_data):
            steps, values = data['medium']
            values_percent = [v * 100 for v in values]  # Convert to percentage
            ax2.plot(steps, values_percent, color=colors[i], linewidth=3, marker='o', markersize=5, label=data['label'])
        ax2.set_xlabel('Training Steps', fontsize=18)
        ax2.set_ylabel('Accuracy (%)', fontsize=18)
        ax2.set_title('SCP-Medium', fontsize=20, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', which='major', labelsize=16)
        ax2.legend(fontsize=16, loc='best')
        
        # Plot 4.3: SCP-Very-Difficult
        ax3 = axes4[2]
        for i, data in enumerate(scp_test_data):
            steps, values = data['very_difficult']
            values_percent = [v * 100 for v in values]  # Convert to percentage
            ax3.plot(steps, values_percent, color=colors[i], linewidth=3, marker='o', markersize=5, label=data['label'])
        ax3.set_xlabel('Training Steps', fontsize=18)
        ax3.set_ylabel('Accuracy (%)', fontsize=18)
        ax3.set_title('SCP-Very-Difficult', fontsize=20, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='both', which='major', labelsize=16)
        ax3.legend(fontsize=16, loc='best')
        
        plt.tight_layout()
        output_path_scp = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME_SCP_TEST)
        fig4.savefig(output_path_scp, dpi=300, bbox_inches='tight')
        print(f"Saved SCP test set plot to: {output_path_scp}")
        plt.close(fig4)
    
    # ========================================================================
    # PLOT 5: STEM Benchmark Performance (SuperGPQA, GPQA Diamond, MMLU Science)
    # ========================================================================
    if stem_bench_data:
        print("\nCreating STEM benchmark plot...")
        fig5, axes5 = plt.subplots(1, 3, figsize=(24, 7))
        fig5.suptitle(f"{STEM_BENCHMARK_NAME} | {TRAINING_DOMAIN} | {MODEL_NAME}", fontsize=24, fontweight='bold', y=1.02)
        
        # Plot 5.1: SuperGPQA
        ax1 = axes5[0]
        for i, data in enumerate(stem_bench_data):
            steps, values = data['supergpqa']
            if steps:  # Only plot if data exists
                values_percent = [v * 100 for v in values]  # Convert to percentage
                ax1.plot(steps, values_percent, color=colors[i], linewidth=3, marker='o', markersize=5, label=data['label'])
        ax1.set_xlabel('Training Steps', fontsize=18)
        ax1.set_ylabel('Accuracy (%)', fontsize=18)
        ax1.set_title('SuperGPQA', fontsize=20, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', which='major', labelsize=16)
        ax1.legend(fontsize=16, loc='best')
        
        # Plot 5.2: GPQA Diamond
        ax2 = axes5[1]
        for i, data in enumerate(stem_bench_data):
            steps, values = data['gpqa']
            if steps:  # Only plot if data exists
                values_percent = [v * 100 for v in values]  # Convert to percentage
                ax2.plot(steps, values_percent, color=colors[i], linewidth=3, marker='o', markersize=5, label=data['label'])
        ax2.set_xlabel('Training Steps', fontsize=18)
        ax2.set_ylabel('Accuracy (%)', fontsize=18)
        ax2.set_title('GPQA Diamond', fontsize=20, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', which='major', labelsize=16)
        ax2.legend(fontsize=16, loc='best')
        
        # Plot 5.3: MMLU Science
        ax3 = axes5[2]
        for i, data in enumerate(stem_bench_data):
            steps, values = data['mmlu']
            if steps:  # Only plot if data exists
                values_percent = [v * 100 for v in values]  # Convert to percentage
                ax3.plot(steps, values_percent, color=colors[i], linewidth=3, marker='o', markersize=5, label=data['label'])
        ax3.set_xlabel('Training Steps', fontsize=18)
        ax3.set_ylabel('Accuracy (%)', fontsize=18)
        ax3.set_title('MMLU Science', fontsize=20, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='both', which='major', labelsize=16)
        ax3.legend(fontsize=16, loc='best')
        
        plt.tight_layout()
        output_path_stem = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME_STEM_BENCH)
        fig5.savefig(output_path_stem, dpi=300, bbox_inches='tight')
        print(f"Saved STEM benchmark plot to: {output_path_stem}")
        plt.close(fig5)


if __name__ == "__main__":
    main()

