import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Ensure standard UTF-8 stream coding for terminal consistency
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Plot styling configuration
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="white")

def reconstruct_normalized_heatmap(csv_path, num_layers=26):
    """
    Reconstruct the Welch's t-test significance matrix and apply local normalization 
    to bound the matrix elements to [-1.0, 1.0].
    """
    if not os.path.exists(csv_path):
        print(f"[WARN] File path does not exist: {csv_path}")
        return None
        
    df = pd.read_csv(csv_path)
    df_valid = df[df['Judge_Score'].isin([0, 1])].copy().reset_index(drop=True)
    group_0 = df_valid[df_valid['Judge_Score'] == 0]
    group_1 = df_valid[df_valid['Judge_Score'] == 1]
    
    alphas = np.round(np.arange(0.0, 1.1, 0.1), 1)
    heatmap_data = np.zeros((len(alphas), num_layers))
    
    for i, alpha in enumerate(alphas):
        for j, layer in enumerate(range(1, num_layers + 1)):
            col = f'L{layer}_A{alpha:.1f}_Ent'
            if col not in df_valid.columns:
                continue
            
            # Welch's t-test
            t_stat, p_val = stats.ttest_ind(group_0[col], group_1[col], equal_var=False)
            
            neg_log_p = -np.log10(p_val + 1e-300)
            direction = np.sign(t_stat)
            heatmap_data[i, j] = neg_log_p * direction
            
    max_val = np.max(np.abs(heatmap_data))
    if max_val == 0:
        max_val = 1.0
    norm_data = heatmap_data / max_val
    
    # Flip rows for proper display layout
    return norm_data[::-1, :]

def main():
    workspace_dir = r"E:\AI_Workspace"
    results_dir = os.path.join(workspace_dir, "结果图")
    output_path = os.path.join(results_dir, "composite_comparison_grid.png")
    os.makedirs(results_dir, exist_ok=True)
    
    # Grid configuration matrix: 3 columns (instruction formats) x 4 rows ( OOD datasets )
    grid_config = {
        "MCQ选择题": [
            ("事实冲撞_同质_MMLU_CollBio_alpha_grid_search.csv", "MMLU CollBio (Biology MCQ)"),
            ("数学逻辑_同质_MMLU_ElemMath_alpha_grid_search.csv", "MMLU ElemMath (Math MCQ)"),
            ("事实冲撞_同质_SciQ_alpha_grid_search.csv", "SciQ (Science MCQ)"),
            ("System1陷阱_同质_CSQA_alpha_grid_search.csv", "CSQA (Commonsense MCQ)")
        ],
        "主观陈述题": [
            ("数学逻辑_同质_MathQA_alpha_grid_search.csv", "MathQA (Math QA)"),
            ("事实冲撞_HaluEval_alpha_grid_search.csv", "HaluEval (Fact QA)"),
            ("事实冲撞_同质_TriviaQA_alpha_grid_search.csv", "TriviaQA (Trivia QA)"),
            ("System-1陷阱_TruthfulQA_alpha_grid_search.csv", "TruthfulQA (Truthful QA)")
        ],
        "CoT推理题": [
            ("数学逻辑_GSM8K_alpha_grid_search.csv", "GSM8K (CoT Math)"),
            ("数学逻辑_同质_MultiArith_alpha_grid_search.csv", "MultiArith (CoT Math)"),
            ("数学逻辑_同质_SVAMP_alpha_grid_search.csv", "SVAMP (CoT Math)"),
            (None, "LEGEND_CARD")
        ]
    }
    
    print("[*] Assembling and rendering OOD phase transition heatmaps...")
    
    fig, axes = plt.subplots(4, 3, figsize=(18, 20), sharex=True, sharey=True)
    columns = ["MCQ选择题", "主观陈述题", "CoT推理题"]
    
    for col_idx, col_name in enumerate(columns):
        configs = grid_config[col_name]
        for row_idx in range(4):
            ax = axes[row_idx, col_idx]
            file_info = configs[row_idx]
            
            if file_info[0] is None:
                # Plot legend inside the empty cell
                ax.axis('off')
                ax.text(0.5, 0.5, 
                        "【拓扑分析图例】\n\n"
                        "🔴 红色 (正值):\n错题组 SVD 谱熵更高\n(模型计算拓扑发散)\n\n"
                        "🔵 蓝色 (负值):\n对题组 SVD 谱熵更高\n(模型计算拓扑收敛)\n\n"
                        "⚪ 白色 (零值):\n对错组无显著性差异\n(相变平稳区)", 
                        ha='center', va='center', fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=1', facecolor='#F9F9F9', edgecolor='#DDDDDD'))
                continue
                
            csv_path = os.path.join(results_dir, file_info[0])
            if not os.path.exists(csv_path):
                csv_path = os.path.join(workspace_dir, file_info[0])
                
            norm_matrix = reconstruct_normalized_heatmap(csv_path)
            
            if norm_matrix is None:
                ax.text(0.5, 0.5, "Data Missing", ha='center', va='center', fontsize=12, color='red')
                continue
                
            # Plot the heatmap
            sns.heatmap(norm_matrix, cmap="RdBu_r", center=0, vmin=-1.0, vmax=1.0,
                        cbar=False, ax=ax, xticklabels=False, yticklabels=False)
            
            ax.set_title(file_info[1], fontsize=13, fontweight='bold', pad=8)
            ax.set_aspect('auto')
            
            # Left column Y labels (Alphas)
            if col_idx == 0:
                ax.set_yticks([0, 5, 10])
                ax.set_yticklabels(["1.0", "0.5", "0.0"], fontsize=10)
                ax.set_ylabel("Whitening Intensity α", fontsize=11, fontweight='bold')
                
            # Bottom row X labels (Layers)
            if row_idx == 3 or (row_idx == 2 and col_idx == 2):
                ax.set_xticks([0, 12, 25])
                ax.set_xticklabels(["L1", "L13", "L26"], fontsize=10)
                ax.set_xlabel("Transformer Layer", fontsize=11, fontweight='bold')
                
    # Composite colorbar configuration at the bottom
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(vmin=-1.0, vmax=1.0))
    sm._A = []
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Relative Polarization Strength (Local SVD Spectrum Alignment)", fontsize=13, fontweight='bold')
    cbar.ax.tick_params(labelsize=11)
    
    # Category titles
    fig.text(0.23, 0.95, "MCQ (Objective Multiple Choice)", fontsize=16, fontweight='bold', ha='center', color='#800020')
    fig.text(0.50, 0.95, "QA (Free-form Direct Answer)", fontsize=16, fontweight='bold', ha='center', color='#004080')
    fig.text(0.77, 0.95, "CoT (Chain-of-Thought Reasoning)", fontsize=16, fontweight='bold', ha='center', color='#006400')
    
    plt.suptitle("Gemma-2-2B-IT Latent Phase Transition Grid under Fractional Whitening", 
                 fontsize=18, fontweight='bold', y=0.98)
                 
    plt.subplots_adjust(top=0.92, bottom=0.10, left=0.08, right=0.95, hspace=0.25, wspace=0.15)
    
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"\n[SUCCESS] Composite grid successfully saved to: {output_path}")

if __name__ == "__main__":
    main()
