import os
import gc
import torch
import math
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.model_selection import train_test_split

# Environment and device setup
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "E:/HF_Models"
device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "E:/HF_Models/gemma-2-2b-it-local"
print(f"[*] Loading model from local path: {model_path}")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map="auto"
)
print("[*] Model loaded successfully.\n")

# Load and clean dataset
excel_path = r"E:\AI_Workspace\gemma_New_Validation_Ready_For_Judge_20260522_134606_Evaluated_Final.csv"
print(f"[*] Loading evaluated evaluation CSV: {excel_path}")
df = pd.read_csv(excel_path)

if 'Prompt' not in df.columns or 'Judge_Score' not in df.columns or 'Category' not in df.columns:
    raise ValueError("Missing 'Prompt', 'Judge_Score', or 'Category' in the input file.")

# Filter valid responses (0: incorrect/hallucination, 1: correct)
df_valid = df[df['Judge_Score'].isin([0, 1])].copy().reset_index(drop=True)
categories = df_valid['Category'].unique()

print(f"[SUCCESS] Loaded {len(df_valid)} valid probe samples.")
print(f"[*] Found {len(categories)} distinct categories for diagnostic scanning.")

alphas = np.round(np.arange(0.0, 1.1, 0.1), 1)

# Scan categories
for cat in categories:
    safe_cat_name = str(cat).replace('\r\n', '_').replace('\n', '_').replace('(', '').replace(')', '').replace(' ', '')
    
    print("\n" + "=" * 80)
    print(f"[*] Profiling category: {cat.replace(chr(10), ' ')}")
    print("=" * 80)

    cat_df_full = df_valid[df_valid['Category'] == cat].copy().reset_index(drop=True)
    N_full = len(cat_df_full)
    print(f"[*] Total samples in category: {N_full}")

    # Ensure sufficient sample size for Welch's t-test
    score_counts = cat_df_full['Judge_Score'].value_counts()
    if 0 not in score_counts or 1 not in score_counts or score_counts[0] < 2 or score_counts[1] < 2:
        print(f"[-] [WARN] Insufficient samples for 0/1 contrast. Skipping category.")
        continue

    # Stratified split-half indexing for reliability verification
    train_idx, _ = train_test_split(
        list(range(N_full)), 
        train_size=0.5, 
        stratify=cat_df_full['Judge_Score'], 
        random_state=42
    )
    N_split = len(train_idx)
    print(f"[*] Split-half subset sample size: {N_split}")

    # Extract layer-wise activations (Forward hook simulation)
    print("[*] Extracting hidden representations from model layers...")
    layer_hiddens = {}
    num_layers = model.config.num_hidden_layers
    prompts = cat_df_full['Prompt'].tolist()

    model.eval()
    with torch.no_grad():
        for prompt in tqdm(prompts, desc=f"Activations {safe_cat_name}"):
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([text], return_tensors="pt").to(model.device)

            outputs = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True
            )

            for layer_idx in range(1, num_layers + 1):
                last_token_hidden = outputs.hidden_states[layer_idx][0, -1, :].detach().cpu().float()
                if layer_idx not in layer_hiddens:
                    layer_hiddens[layer_idx] = []
                layer_hiddens[layer_idx].append(last_token_hidden)

    # Dimensional analysis: SVD with Fractional Whitening
    print(f"\n[*] Running SVD and grid scanning for 11 fractional alphas...")
    
    cat_df_full_results = cat_df_full.copy()
    cat_df_split_results = cat_df_full.iloc[train_idx].copy().reset_index(drop=True)

    heatmap_data_full = np.zeros((len(alphas), num_layers))
    heatmap_data_split = np.zeros((len(alphas), num_layers))

    for layer_idx in tqdm(range(1, num_layers + 1), desc=f"SVD Scanning {safe_cat_name}"):
        H_all = torch.stack(layer_hiddens[layer_idx])

        # ---------- Full Dataset SVD Analysis ----------
        H_mean_all = H_all.mean(dim=0, keepdim=True)
        H_centered_all = H_all - H_mean_all
        U_all, S_all, Vh_all = torch.linalg.svd(H_centered_all, full_matrices=False)
        
        mask_all = S_all > 1e-7
        S_valid_all = S_all[mask_all]
        Vh_valid_all = Vh_all[mask_all, :]
        P_all = torch.matmul(H_centered_all, Vh_valid_all.T)

        for i, alpha in enumerate(alphas):
            scale_factors = S_valid_all ** alpha
            W_frac = P_all / scale_factors.unsqueeze(0)
            W_sq = W_frac ** 2
            norm_sq = W_sq.sum(dim=1, keepdim=True)
            e = W_sq / (norm_sq + 1e-12)
            log_e = torch.log(e + 1e-12)
            entropies = -torch.sum(e * log_e, dim=1).numpy()
            
            col_name = f'L{layer_idx}_A{alpha:.1f}_Ent'
            cat_df_full_results[col_name] = entropies

        # ---------- Split-Half Dataset SVD Analysis ----------
        H_split = H_all[train_idx]
        H_mean_split = H_split.mean(dim=0, keepdim=True)
        H_centered_split = H_split - H_mean_split
        U_split, S_split, Vh_split = torch.linalg.svd(H_centered_split, full_matrices=False)

        mask_split = S_split > 1e-7
        S_valid_split = S_split[mask_split]
        Vh_valid_split = Vh_split[mask_split, :]
        P_split = torch.matmul(H_centered_split, Vh_valid_split.T)

        for i, alpha in enumerate(alphas):
            scale_factors = S_valid_split ** alpha
            W_frac = P_split / scale_factors.unsqueeze(0)
            W_sq = W_frac ** 2
            norm_sq = W_sq.sum(dim=1, keepdim=True)
            e = W_sq / (norm_sq + 1e-12)
            log_e = torch.log(e + 1e-12)
            entropies = -torch.sum(e * log_e, dim=1).numpy()

            col_name = f'L{layer_idx}_A{alpha:.1f}_Ent'
            cat_df_split_results[col_name] = entropies

    # Save SVD spectral entropy outputs
    csv_output_full = rf"E:\AI_Workspace\{safe_cat_name}_alpha_grid_search.csv"
    csv_output_split = rf"E:\AI_Workspace\{safe_cat_name}_SplitHalf_alpha_grid_search.csv"
    cat_df_full_results.to_csv(csv_output_full, index=False, encoding="utf-8-sig")
    cat_df_split_results.to_csv(csv_output_split, index=False, encoding="utf-8-sig")
    print(f"[SUCCESS] Saved full dataset SVD metrics to: {csv_output_full}")
    print(f"[SUCCESS] Saved split-half dataset SVD metrics to: {csv_output_split}")

    # ---------- Dynamic Phase Transition Mapping (Welch's t-test) ----------
    print("[*] Running Welch's t-test and plotting phase transition heatmaps...")
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    # Channel A: Full Dataset Welch T-Test
    group_0_f = cat_df_full_results[cat_df_full_results['Judge_Score'] == 0]
    group_1_f = cat_df_full_results[cat_df_full_results['Judge_Score'] == 1]
    for i, alpha in enumerate(alphas):
        for j, layer in enumerate(range(1, num_layers + 1)):
            col = f'L{layer}_A{alpha:.1f}_Ent'
            t_stat, p_val = stats.ttest_ind(group_0_f[col], group_1_f[col], equal_var=False)
            neg_log_p = -np.log10(p_val + 1e-300)
            heatmap_data_full[i, j] = neg_log_p * np.sign(t_stat)

    # Channel B: Split-Half Dataset Welch T-Test
    group_0_s = cat_df_split_results[cat_df_split_results['Judge_Score'] == 0]
    group_1_s = cat_df_split_results[cat_df_split_results['Judge_Score'] == 1]
    for i, alpha in enumerate(alphas):
        for j, layer in enumerate(range(1, num_layers + 1)):
            col = f'L{layer}_A{alpha:.1f}_Ent'
            t_stat, p_val = stats.ttest_ind(group_0_s[col], group_1_s[col], equal_var=False)
            neg_log_p = -np.log10(p_val + 1e-300)
            heatmap_data_split[i, j] = neg_log_p * np.sign(t_stat)

    # Plot Full Dataset Heatmap
    fig = plt.figure(figsize=(16, 8))
    heatmap_flipped = heatmap_data_full[::-1, :]
    alphas_flipped = alphas[::-1]
    max_val = np.max(np.abs(heatmap_flipped))
    if max_val == 0: max_val = 1.0

    ax = sns.heatmap(heatmap_flipped, cmap="RdBu_r", center=0, vmax=max_val, vmin=-max_val,
                     annot=False, xticklabels=range(1, num_layers + 1), yticklabels=[f"{a:.1f}" for a in alphas_flipped])
    colorbar = ax.collections[0].colorbar
    colorbar.set_label('Signed Significance Strength (Signed -log10 P-value)\n[ Red: Error/Incorrect Entropy > Correct | Blue: Correct Entropy > Error/Incorrect ]', fontsize=12)
    cat_title = str(cat).replace('\n', ' ')
    plt.title(f"{cat_title} Latent Phase Transition (Full Dataset)", fontsize=16, pad=15)
    plt.xlabel("Transformer Hidden Layer Index", fontsize=13)
    plt.ylabel("Fractional Whitening Intensity (Alpha)", fontsize=13)
    plt.text(num_layers + 0.5, len(alphas) - 0.5, "P=0.05 Threshold is approx. ±1.3", color="red", fontsize=10)
    
    fig_output_full = rf"E:\AI_Workspace\Heatmap_{safe_cat_name}.png"
    plt.tight_layout()
    plt.savefig(fig_output_full, dpi=300)
    plt.close(fig)
    print(f"[SUCCESS] Generated full dataset heatmap: {fig_output_full}")

    # Plot Split-Half Dataset Heatmap
    fig = plt.figure(figsize=(16, 8))
    heatmap_flipped_split = heatmap_data_split[::-1, :]
    max_val_split = np.max(np.abs(heatmap_flipped_split))
    if max_val_split == 0: max_val_split = 1.0

    ax = sns.heatmap(heatmap_flipped_split, cmap="RdBu_r", center=0, vmax=max_val_split, vmin=-max_val_split,
                     annot=False, xticklabels=range(1, num_layers + 1), yticklabels=[f"{a:.1f}" for a in alphas_flipped])
    colorbar = ax.collections[0].colorbar
    colorbar.set_label('Signed Significance Strength (Signed -log10 P-value)\n[ Red: Error/Incorrect Entropy > Correct | Blue: Correct Entropy > Error/Incorrect ]', fontsize=12)
    plt.title(f"{cat_title} Latent Phase Transition (Split-Half Reliability Check)", fontsize=16, pad=15)
    plt.xlabel("Transformer Hidden Layer Index", fontsize=13)
    plt.ylabel("Fractional Whitening Intensity (Alpha)", fontsize=13)
    plt.text(num_layers + 0.5, len(alphas) - 0.5, "P=0.05 Threshold is approx. ±1.3", color="red", fontsize=10)

    fig_output_split = rf"E:\AI_Workspace\Heatmap_{safe_cat_name}_SplitHalf.png"
    plt.tight_layout()
    plt.savefig(fig_output_split, dpi=300)
    plt.close(fig)
    print(f"[SUCCESS] Generated split-half dataset heatmap: {fig_output_split}")

    # Garbage collection and CUDA memory flush
    del layer_hiddens
    del H_all, H_split
    del U_all, S_all, Vh_all, P_all
    del U_split, S_split, Vh_split, P_split
    gc.collect()
    torch.cuda.empty_cache()

print("\n" + "=" * 80)
print("[SUCCESS] Automated FW-SE phase transition diagnostic profiling completed!")
print("==================================================")
