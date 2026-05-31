import os
import gc
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "E:/HF_Models"
device = "cuda" if torch.cuda.is_available() else "cpu"

def extract_activations(model, tokenizer, prompts, num_layers=26):
    """
    Extract last-token hidden state activations for a set of prompts across all hidden layers.
    """
    model.eval()
    activations = {layer: [] for layer in range(1, num_layers + 1)}
    
    with torch.no_grad():
        for prompt in tqdm(prompts, desc="Extracting activations"):
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([text], return_tensors="pt").to(model.device)
            
            outputs = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True
            )
            
            for layer in range(1, num_layers + 1):
                # Shape: [hidden_size]
                hidden = outputs.hidden_states[layer][0, -1, :].detach().cpu().float()
                activations[layer].append(hidden)
                
    # Stack list to tensor: [N, d]
    for layer in activations:
        activations[layer] = torch.stack(activations[layer])
        
    return activations

def compute_subspace_alignment(H_A, H_B, k=5):
    """
    Compute the cosines of principal angles between the k-dimensional SVD subspaces
    spanned by activations H_A and H_B.
    """
    # 1. Center the activation matrices
    HA_centered = H_A - H_A.mean(dim=0, keepdim=True)
    HB_centered = H_B - H_B.mean(dim=0, keepdim=True)
    
    # 2. Run SVD on centered representations
    _, _, Vh_A = torch.linalg.svd(HA_centered, full_matrices=False)
    _, _, Vh_B = torch.linalg.svd(HB_centered, full_matrices=False)
    
    # SVD yields orthonormal rows. Take the top k singular vectors: [k, d]
    Q_A = Vh_A[:k, :]
    Q_B = Vh_B[:k, :]
    
    # 3. Compute inner product matrix of the two subspaces: [k, k]
    M = torch.matmul(Q_A, Q_B.T)
    
    # 4. SVD of the inner product matrix gives the cosines of principal angles
    _, S_angles, _ = torch.linalg.svd(M)
    
    # Singular values are bounded between 0 and 1, representing cos(theta)
    cosines = S_angles.clamp(0.0, 1.0).numpy()
    
    # Return cosines and their average (subspace alignment index)
    return cosines, np.mean(cosines)

def main():
    model_path = "E:/HF_Models/gemma-2-2b-it-local"
    csv_path = r"E:\AI_Workspace\gemma_New_Validation_Ready_For_Judge_20260522_134606_Evaluated_Final.csv"
    output_dir = r"E:\AI_Workspace\结果图\相似性分析结果"
    os.makedirs(output_dir, exist_ok=True)
    
    print("[*] Loading Gemma-2-2B-IT model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )
    print("[*] Model loaded.\n")
    
    print(f"[*] Reading evaluated prompts from: {csv_path}")
    df = pd.read_csv(csv_path)
    df_valid = df[df['Judge_Score'].isin([0, 1])].copy()
    
    # Print available categories
    categories = df_valid['Category'].unique()
    print("[*] Available categories:")
    for idx, cat in enumerate(categories):
        print(f"  {idx+1}. {cat.strip()}")
        
    # We will run 3 comparative pipelines to verify different formatting interactions:
    # 1. Same-format MCQ vs MCQ: Biology MCQ vs Math MCQ (Resonance expected)
    # 2. Same-format QA vs QA: Trivia QA vs Fact QA (Resonance expected)
    # 3. Different-format Math MCQ vs Math QA: Math MCQ vs Math QA (Decoupling expected)
    
    scenarios = [
        {
            "name": "MCQ_vs_MCQ",
            "title": "MCQ Format Resonance (Biology MCQ vs Math MCQ)",
            "cat_A": "事实冲撞_同质\n(MMLU_CollBio)",
            "cat_B": "数学逻辑_同质\n(MMLU_ElemMath)",
            "color": "red"
        },
        {
            "name": "QA_vs_QA",
            "title": "QA Format Resonance (Trivia QA vs HaluEval Fact QA)",
            "cat_A": "事实冲撞_同质\n(TriviaQA)",
            "cat_B": "事实冲撞\n(HaluEval)",
            "color": "blue"
        },
        {
            "name": "MCQ_vs_QA",
            "title": "Format Decoupling (Math MCQ vs Math QA)",
            "cat_A": "数学逻辑_同质\n(MMLU_ElemMath)",
            "cat_B": "数学逻辑_同质\n(MathQA)",
            "color": "purple"
        }
    ]
    
    num_layers = model.config.num_hidden_layers
    k = 5  # Dimensionality of the compared subspaces
    
    results = {}
    
    for sc in scenarios:
        print("\n" + "=" * 80)
        print(f"[*] Executing Subspace Comparison: {sc['title']}")
        print("=" * 80)
        
        df_A = df_valid[df_valid['Category'] == sc['cat_A']].sample(n=100, random_state=42)
        df_B = df_valid[df_valid['Category'] == sc['cat_B']].sample(n=100, random_state=42)
        
        print(f"[*] Extracting activations for Task A: {sc['cat_A'].strip()}")
        acts_A = extract_activations(model, tokenizer, df_A['Prompt'].tolist(), num_layers)
        
        print(f"[*] Extracting activations for Task B: {sc['cat_B'].strip()}")
        acts_B = extract_activations(model, tokenizer, df_B['Prompt'].tolist(), num_layers)
        
        alignments = []
        for layer in range(1, num_layers + 1):
            H_A = acts_A[layer]
            H_B = acts_B[layer]
            _, avg_cos = compute_subspace_alignment(H_A, H_B, k=k)
            alignments.append(avg_cos)
            print(f"  Layer {layer:02d} -> Mean Subspace Alignment (k={k}): {avg_cos:.4f}")
            
        results[sc['name']] = alignments
        
        # Flush memory
        del acts_A, acts_B
        gc.collect()
        torch.cuda.empty_cache()
        
    # ---------- Plotting Results ----------
    print("\n[*] Generating comparative subspace alignment curves...")
    plt.figure(figsize=(12, 6))
    
    layers = list(range(1, num_layers + 1))
    for sc in scenarios:
        plt.plot(layers, results[sc['name']], marker='o', linestyle='-', linewidth=2.5,
                 color=sc['color'], label=sc['title'])
                 
    plt.title(f"Task Latent Subspace Alignment (k={k} Principal Angles Cosine)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Transformer Hidden Layer Index", fontsize=12)
    plt.ylabel("Mean Subspace Alignment Index", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(layers)
    plt.ylim(0.0, 1.0)
    plt.legend(fontsize=10, loc="lower left")
    plt.tight_layout()
    
    fig_path = os.path.join(output_dir, "subspace_alignment_comparison.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    
    # Save numerical data to CSV
    df_out = pd.DataFrame(results, index=[f"Layer_{i}" for i in range(1, num_layers + 1)])
    csv_path = os.path.join(output_dir, "subspace_alignment_data.csv")
    df_out.to_csv(csv_path)
    
    print(f"\n[SUCCESS] Subspace comparison complete!")
    print(f"  -> Plot saved to: {fig_path}")
    print(f"  -> Numerical data saved to: {csv_path}")

if __name__ == "__main__":
    main()
