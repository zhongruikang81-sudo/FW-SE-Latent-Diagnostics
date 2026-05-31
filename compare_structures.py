import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from scipy import stats
from skimage.metrics import structural_similarity as ssim
from skimage.feature import hog
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity

# Support UTF-8 encoding for standard outputs
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Plot styling configuration
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="white")

def reconstruct_heatmap_data(csv_path, num_layers=26):
    """
    Reconstruct the 11x26 significance matrix from the SVD spectral entropy CSV file
    using Welch's t-test statistic.
    """
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
            
    # Flip rows to match standard plotting orientation (Alpha 1.0 at the top)
    return heatmap_data[::-1, :]

def get_normalized_color_map(heatmap_data):
    """
    Normalize the significance heatmap locally to scale in [-1.0, 1.0], 
    then project to [0, 255] RGB intensity bounds.
    """
    max_val = np.max(np.abs(heatmap_data))
    if max_val == 0:
        max_val = 1.0
    norm_data = heatmap_data / max_val
    scaled = (norm_data + 1.0) / 2.0 * 255.0
    return scaled

def preprocess_matrix_to_image(scaled_matrix, upscale_factor=20):
    """
    Bilinear upscale of the grid representation to simulate smoothed heatmap contours
    without axis labels or legend overlays.
    """
    img = Image.fromarray(scaled_matrix.astype(np.uint8))
    new_width = 26 * upscale_factor
    new_height = 11 * upscale_factor
    img_upscaled = img.resize((new_width, new_height), Image.Resampling.BILINEAR)
    return img_upscaled

def extract_hog_features(img):
    """
    Extract HOG representation to capture structural flow gradients.
    """
    arr = np.array(img)
    features = hog(arr, orientations=8, pixels_per_cell=(16, 16),
                   cells_per_block=(1, 1), visualize=False)
    return features

def calculate_nmi(img1, img2, bins=32):
    """
    Compute Normalized Mutual Information (NMI) on quantized pixel profiles.
    """
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    q1 = (arr1 // (256 // bins)).astype(np.int32).flatten()
    q2 = (arr2 // (256 // bins)).astype(np.int32).flatten()
    return normalized_mutual_info_score(q1, q2, average_method='arithmetic')

def main():
    workspace_dir = r"E:\AI_Workspace"
    results_dir = os.path.join(workspace_dir, "结果图")
    output_dir = os.path.join(results_dir, "相似性分析结果")
    os.makedirs(output_dir, exist_ok=True)
    
    print("[*] Scanning workspace for grid search CSV datasets...")
    # Scan E:\AI_Workspace\结果图 first, fallback to E:\AI_Workspace if empty
    csv_pattern = os.path.join(results_dir, "*_alpha_grid_search.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        csv_pattern = os.path.join(workspace_dir, "*_alpha_grid_search.csv")
        csv_files = glob.glob(csv_pattern)
        
    csv_files = [f for f in csv_files if "splithalf" not in os.path.basename(f).lower()]
    
    if not csv_files:
        print("[-] Error: No grid search CSV files found.")
        return
    
    print(f"[*] Found {len(csv_files)} core category CSV files.")
    
    names = []
    heatmap_data_dict = {}
    processed_images = {}
    hog_features_dict = {}
    
    for f in csv_files:
        basename = os.path.basename(f)
        name = basename.replace("_alpha_grid_search.csv", "")
        names.append(name)
        print(f"[*] Reconstructing phase boundaries: {name}")
        
        try:
            heatmap_data = reconstruct_heatmap_data(f)
            heatmap_data_dict[name] = heatmap_data
            
            scaled_matrix = get_normalized_color_map(heatmap_data)
            img = preprocess_matrix_to_image(scaled_matrix, upscale_factor=20)
            processed_images[name] = img
            
            hog_feat = extract_hog_features(img)
            hog_features_dict[name] = hog_feat
        except Exception as e:
            print(f"[-] Processing failed for {basename}: {e}")
            
    names = sorted(names)
    n = len(names)
    
    ssim_matrix = np.ones((n, n))
    hog_matrix = np.ones((n, n))
    nmi_matrix = np.ones((n, n))
    polarity_matrix = np.ones((n, n))
    composite_matrix = np.ones((n, n))
    
    print("\n[*] Evaluating structural and topological similarity matrices...")
    
    for i in range(n):
        for j in range(i, n):
            name1 = names[i]
            name2 = names[j]
            
            img1 = processed_images[name1]
            img2 = processed_images[name2]
            
            img1_arr = np.array(img1)
            img2_arr = np.array(img2)
            
            if i == j:
                ssim_val = 1.0
                hog_val = 1.0
                nmi_val = 1.0
                cos_sign = 1.0
                composite_val = 1.0
            else:
                # 1. Structural Similarity Index (SSIM)
                ssim_val = ssim(img1_arr, img2_arr, data_range=255)
                
                # 2. HOG Cosine Similarity (Gradient Flow Topology)
                feat1 = hog_features_dict[name1].reshape(1, -1)
                feat2 = hog_features_dict[name2].reshape(1, -1)
                hog_val = cosine_similarity(feat1, feat2)[0, 0]
                
                # 3. Normalized Mutual Information (NMI)
                nmi_val = calculate_nmi(img1, img2)
                
                # 4. Polarity Agreement (CSPS - Cosine Similarity of Phase Signs)
                # Filter significance threshold of p < 0.05 (-log10(p) > 1.301)
                h1 = heatmap_data_dict[name1]
                h2 = heatmap_data_dict[name2]
                sig_threshold = 1.301
                s1 = np.zeros_like(h1)
                s1[h1 > sig_threshold] = 1
                s1[h1 < -sig_threshold] = -1
                
                s2 = np.zeros_like(h2)
                s2[h2 > sig_threshold] = 1
                s2[h2 < -sig_threshold] = -1
                
                dot_prod = np.sum(s1 * s2)
                norm1 = np.sqrt(np.sum(s1**2))
                norm2 = np.sqrt(np.sum(s2**2))
                cos_sign = dot_prod / (norm1 * norm2) if (norm1 * norm2) > 0 else 0.0
                polarity_val = max(0.0, cos_sign)
                
                # 5. Composite Metric: CSSI
                composite_val = 0.4 * polarity_val + 0.3 * hog_val + 0.15 * ssim_val + 0.15 * nmi_val
                
            ssim_matrix[i, j] = ssim_val
            ssim_matrix[j, i] = ssim_val
            
            hog_matrix[i, j] = hog_val
            hog_matrix[j, i] = hog_val
            
            nmi_matrix[i, j] = nmi_val
            nmi_matrix[j, i] = nmi_val
            
            polarity_matrix[i, j] = cos_sign
            polarity_matrix[j, i] = cos_sign
            
            composite_matrix[i, j] = composite_val
            composite_matrix[j, i] = composite_val
            
            if i != j:
                print(f"[{name1}] vs [{name2}] -> CSPS Polarity: {cos_sign:.4f}, HOG: {hog_val:.4f}, SSIM: {ssim_val:.4f}, NMI: {nmi_val:.4f} | CSSI: {composite_val:.4f}")
                
    dfs = {
        "ssim": pd.DataFrame(ssim_matrix, index=names, columns=names),
        "hog": pd.DataFrame(hog_matrix, index=names, columns=names),
        "nmi": pd.DataFrame(nmi_matrix, index=names, columns=names),
        "polarity": pd.DataFrame(polarity_matrix, index=names, columns=names),
        "composite": pd.DataFrame(composite_matrix, index=names, columns=names)
    }
    
    for metric, df in dfs.items():
        csv_path = os.path.join(output_dir, f"similarity_{metric}_matrix.csv")
        df.to_csv(csv_path, encoding="utf-8-sig")
        print(f"[SUCCESS] Saved similarity matrix to: {csv_path}")
        
    metrics_config = {
        "ssim": {"title": "Structural Similarity Index (SSIM) Matrix", "cmap": "magma", "vmin": 0.3, "vmax": 1.0, "label": "SSIM"},
        "hog": {"title": "Gradient Topology Similarity (HOG Cosine) Matrix", "cmap": "viridis", "vmin": 0.3, "vmax": 1.0, "label": "HOG Cosine"},
        "nmi": {"title": "Normalized Mutual Information (NMI) Matrix", "cmap": "mako", "vmin": 0.0, "vmax": 1.0, "label": "NMI"},
        "polarity": {"title": "Cosine Similarity of Phase Signs (CSPS) Matrix", "cmap": "coolwarm", "vmin": -1.0, "vmax": 1.0, "label": "CSPS Polarity"},
        "composite": {"title": "Composite Spectral Similarity Index (CSSI) Matrix", "cmap": "rocket", "vmin": 0.2, "vmax": 1.0, "label": "CSSI"}
    }
    
    for metric, config in metrics_config.items():
        plt.figure(figsize=(12, 9.5))
        sns.heatmap(dfs[metric], annot=True, fmt=".2f", cmap=config["cmap"], 
                    vmin=config["vmin"], vmax=config["vmax"], square=True, 
                    cbar_kws={"shrink": .8, "label": config["label"]})
        plt.title(config["title"], fontsize=15, fontweight="bold", pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        
        img_path = os.path.join(output_dir, f"similarity_{metric}_heatmap.png")
        plt.savefig(img_path, dpi=300)
        plt.close()
        print(f"[SUCCESS] Generated similarity heatmap: {img_path}")
        
    print("\n" + "="*60)
    print(" Latent Representation Similarity Profiling (CSSI Summary)")
    print("="*60)
    
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((
                names[i], 
                names[j], 
                polarity_matrix[i, j],
                ssim_matrix[i, j], 
                hog_matrix[i, j], 
                nmi_matrix[i, j],
                composite_matrix[i, j]
            ))
            
    pairs_sorted = sorted(pairs, key=lambda x: x[6], reverse=True)
    
    print("\n[TOP 3 Similar Latent Configurations (Highest CSSI)]:")
    for i in range(min(3, len(pairs_sorted))):
        p = pairs_sorted[i]
        print(f"  {i+1}. {p[0]} <---> {p[1]}")
        print(f"     -> CSSI: {p[6]:.4f} (CSPS Polarity: {p[2]:.4f}, HOG: {p[4]:.4f}, SSIM: {p[3]:.4f}, NMI: {p[5]:.4f})")
        
    print("\n[TOP 3 Decoupled Latent Configurations (Lowest CSSI)]:")
    for i in range(1, min(4, len(pairs_sorted)) + 1):
        p = pairs_sorted[-i]
        print(f"  {i}. {p[0]} <---> {p[1]}")
        print(f"     -> CSSI: {p[6]:.4f} (CSPS Polarity: {p[2]:.4f}, HOG: {p[4]:.4f}, SSIM: {p[3]:.4f}, NMI: {p[5]:.4f})")
        
    print(f"\n[SUCCESS] Similarity matrices and heatmaps stored in: {output_dir}")

if __name__ == "__main__":
    main()
