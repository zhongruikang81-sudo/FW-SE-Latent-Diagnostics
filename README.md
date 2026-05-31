# An Exploratory Perspective on Latent Representation Geometry: Analyzing Format-Induced Constraints and Phase Transitions in Large Language Models

<p align="center">
  <img src="assets/composite_comparison_grid.png" width="100%" alt="FW-SE Latent Geometry Cover" />
</p>

[![Language](https://img.shields.io/badge/Language-Python-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Interpretability](https://img.shields.io/badge/Domain-Mechanistic%20Interpretability-red.svg)]()
[![Hardware](https://img.shields.io/badge/Hardware-RTX%204060%20(8GB%20VRAM)-brightgreen.svg)]()

This repository contains the exploratory research codebase for analyzing **Latent Representation Geometry** in Large Language Models (LLMs). The project provides tools to investigate format-induced constraints and dynamic entropic trajectories (via Fractional Whitened Spectral Entropy) in intermediate layers, as presented in our exploratory manuscript.

---

## Abstract
This paper presents an exploratory perspective on analyzing the latent space hidden states of Large Language Models (LLMs) without relying on external reward models or intensive probe training. While conventional diagnostics often associate factual uncertainty with isotropic energy divergence (high spectral entropy) of output distributions, we discuss how instruction tuning and alignment paradigms may introduce localized probability concentration, thereby complicating direct surface-level entropy interpretations. To investigate these dynamics, we observe hidden state representations in relatively small-scale models (specifically Gemma-2-2B-IT) across approximately 11,000 evaluation prompts from 11 task domains. We observe that while static coordinates largely correlate with semantic clustering, the dynamic entropic trajectories in deeper layers are heavily constrained by target output structures (e.g., Multiple Choice Questions, MCQ), acting as a localized computational bottleneck. To characterize this behavior, we utilize Fractional Mahalanobis Whitening (FWM), Adaptive 90% Cumulative Variance Subspace Alignment, and a Composite Structural Similarity Index (CSSI). Our empirical measurements point to a stable representational divergence (Geometric Invariant Gap $\sim 0.115$) from Layer 10 to Layer 25, induced by structural differences. We present this analytical framework not as an immediate replacement for mature commercial alignment techniques, but as a supplementary diagnostic lens that may help understand how structural constraints interact with semantic representations in LLM intermediate layers.

---

## 1. Introduction and Theoretical Motivation

### 1.1 Observations on the Traditional Hallucination-Entropy Model
An intuitive starting point for unsupervised latent space diagnostics is the hypothesis that factual confidence corresponds to low-dimensional convergence of activation trajectories, whereas uncertainty or factual errors lead to high-dimensional energy leakage. Under this assumption, a model’s certainty when retrieving parametric knowledge should map to low spectral entropy ($H_{\text{spec}}$) in orthogonal coordinate spaces, while incorrect generations or hallucinations should manifest as elevated entropy due to noisy, multi-directional projection energy.

However, detailed exploratory scanning across different task distributions suggests that this relationship may be more nuanced and task-dependent:
* **Observation A (Task-Specific Monopolar Distributions)**: In direct question-answering formats like `MathQA`, we observe that incorrect responses exhibit consistently higher spectral entropy compared to correct ones across most layers and whitening bands, appearing as a uniform statistical contrast.
* **Observation B (Layer-Wise Phase Transitions)**: In multi-step logical reasoning datasets like `GSM8K`, the middle layers of the model exhibit a transitional phase where the spectral entropy of correct answers exceeds that of incorrect ones, suggesting a complex representational trajectory associated with active deduction.
* **Observation C (Representational Resonance Across Semantic Boundaries)**: We note that semantically distinct tasks utilizing similar answer formats—such as `MMLU_CollBio` (College Biology) and `MMLU_ElemMath` (Elementary Mathematics)—display highly similar latent phase transition trajectories ($\text{CSSI} = 0.5156$). Conversely, tasks within the same mathematical domain but structured differently, such as `MMLU_ElemMath` and `MathQA`, show lower correlation ($\text{CSSI} = 0.3086$).

### 1.2 The Format Constraints Perspective
To interpret these empirical behaviors, we consider a complementary viewpoint: the interaction between semantic coordinates and formatting constraints within the latent geometry:
1. **Static Geometry as a Semantic Map**: The spatial coordinates of static activations effectively map semantic relationships and vocabularies, organizing concepts into orthogonal sub-regions.
2. **Dynamic Trajectories under Formatting Constraints**: Highly constrained prompt designs (e.g., forcing multiple-choice selections among $\{A, B, C, D\}$) encourage representation vectors to contract into localized decision regions at the generation boundary. This behavior acts as a structural bottleneck, which can heavily shape and constrain the entropic path of intermediate hidden states.

Consequently, studying LLM representations may benefit from analyzing not only semantic contents but also how downstream structural templates shape intermediate computational pathways.

---

## 2. Related Work

### 2.1 Eigenspectrum Dynamics in Latent Space
Analyzing the spectral properties of latent representations has proven helpful in mechanistic interpretability. For instance, recent studies such as NerVE [1] investigate the spectral entropy of normalized eigenvalues to evaluate non-linear feature utilization in Feed-Forward Networks under various optimizers. While NerVE leverages these dynamics to study optimization, our exploratory analysis utilizes spectral properties to observe representation behaviors associated with factual errors and confidence. Unlike raw spectral metrics, which can be dominated by low-frequency, macroscopic properties, we apply Fractional Mahalanobis Whitening (FWM) to modulate the frequency bands and inspect quieter, high-frequency diagnostic components.

### 2.2 Interaction of Semantic and Alignment Representations
Prior work has studied how LLMs partition high-dimensional semantics and instruction templates into linearly separable, orthogonal sub-spaces [2, 3]. While these studies focus on the static separability of semantic and alignment features, our analysis explores their dynamic coupling. We observe that while static coordinates reflect semantic fields, the dynamic progression of activation vectors through deeper layers is significantly influenced by target formatting constraints. This localized bottleneck manifests as a stable representational divergence (Geometric Invariant Gap $\sim 0.115$) between different formats across several layers.

### 2.3 Whitening Methods and Subspace Alignment
Mahalanobis whitening ($\Sigma^{-1/2}$) and singular subspace alignment (using principal angles and singular vector cosines) are standard techniques for correcting anisotropy and comparing multi-dimensional spaces in representation learning and model merging [3–5]. While full whitening ($\alpha = 1$) and SVD-based alignment are theoretically robust, they can be sensitive to noise when applied directly to high-dimensional hidden spaces. In this work, we discuss the mathematical conditions under which full whitening may lead to isotropic noise decay, and how Leave-One-Out Cross-Validation (LOO-CV) can introduce numerical instability in high-rank representations. We mitigate these tendencies by employing fractional whitening $\alpha \in (0, 1)$ and adaptive 90% cumulative variance subspace truncation.

---

## 3. Mathematical Formulations

### 3.1 Activation Space Centering
Let $H \in \mathbb{R}^{N \times D}$ represent the activation matrix extracted from the final token of the hidden states at a given layer $l$, where $N$ is the number of evaluation queries, and $D$ is the hidden dimension ($D = 2048$ for `Gemma-2-2B-IT`).

We compute the mean activation vector $\mu_H \in \mathbb{R}^{1 \times D}$:
$$\mu_H = \frac{1}{N} \sum_{j=1}^N H_{j,\cdot}$$

To center the activation space and focus our singular vectors on the directional variance, we define:
$$H_c = H - \mathbf{1}_N \mu_H$$
where $\mathbf{1}_N \in \mathbb{R}^{N \times 1}$ is a column vector of ones.

### 3.2 Singular Value Decomposition (SVD) of the Covariance Structure
We apply thin Singular Value Decomposition (SVD) to the centered matrix $H_c$:
$$H_c = U S V^T$$
where:
* $U \in \mathbb{R}^{N \times r}$ is the left singular matrix, forming an orthonormal basis in the sample space.
* $S = \text{diag}(\sigma_1, \sigma_2, \dots, \sigma_r) \in \mathbb{R}^{r \times r}$ is the singular value diagonal matrix, sorted as $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 10^{-7}$ (pruning the numerical null space). The component variances are given by $\lambda_i = \sigma_i^2$.
* $V \in \mathbb{R}^{D \times r}$ is the right singular matrix, representing the orthonormal principal axes of the feature space.

The projection coordinates onto these principal axes are given by $P \in \mathbb{R}^{N \times r}$:
$$P = H_c V = U S$$

### 3.3 Fractional Mahalanobis Whitening & Spectral Tuning
To inspect the representation coordinates at different scales of variance, we introduce a fractional whitening parameter $\alpha \in [0, 1]$. The fractional whitened coordinates $W_{\alpha} \in \mathbb{R}^{N \times r}$ are formulated as:
$$W_{\alpha} = P S^{-\alpha} = U S^{1-\alpha}$$

We analyze how this parameter modulates the representation space at its boundaries:
1. **Case $\alpha = 0$ (Raw Projection - Low-Frequency Dominance)**:
$$W_0 = U S = P$$
At $\alpha = 0$, coordinates are weighted fully by their singular values. Because the spectrum decays rapidly ($\sigma_1 \gg \sigma_r$), dominant syntax and shared semantic structures explain the vast majority of the variance. Consequently, the projection energy $E_{j,i} = u_{j,i}^2 \sigma_i^2$ concentrates heavily on the first few head dimensions, which can compress overall spectral entropy and mask subtle signals residing in the tail components.

### 3.4 Analysis of Full Whitening ($\alpha = 1$)
When applying full Mahalanobis whitening ($\alpha = 1$), we scale the projections by the inverse singular value matrix $S^{-1}$:
$$W_1 = P S^{-1} = U$$

The coordinates correspond directly to the row vectors of the left singular matrix $U$:
$$w_j^{(1)} = u_j = [u_{j,1}, u_{j,2}, \dots, u_{j,r}]$$

In practical setups, after trimming the null space, $U$ is an $r \times r$ square orthogonal matrix (where $r \le N$ is the effective rank). Consequently, its row vectors $u_j$ form an orthonormal basis of $\mathbb{R}^r$. This introduces a geometric limitation: by completely removing the singular value matrix $S$, the coordinates $u_j$ lose the variance weights that distinguish the importance of different dimensions. The row vector $u_j$ becomes an isotropic unit vector satisfying:
$$\sum_{k=1}^r u_{j,k}^2 = \Vert u_j \Vert_2^2 = 1$$

Assuming symmetric expectation of energy projection in this unweighted space, the expectation of the normalized energy distribution $E_{j,i}$ trends toward uniform distribution:
$$\mathbb{E}[E_{j,i}] = \mathbb{E}\left[ \frac{u_{j,i}^2}{\Vert u_j \Vert_2^2} \right] = \frac{1}{r} \sum_{k=1}^r \mathbb{E}[u_{j,k}^2] = \frac{1}{r} \mathbb{E}[1] = \frac{1}{r}$$

Under these conditions, the spectral entropy of any query—regardless of factual correctness—converges toward its theoretical maximum:
$$H_{\text{spec}} \to \ln(r)$$

The coordinates behave similarly to isotropic noise, reducing the statistical contrast between correct and incorrect groups. Introducing $\alpha \in (0, 1)$ serves to balance these behaviors by tempering the influence of major components without completely standardizing tail variances.

### 3.5 Fractional Whitened Spectral Entropy (FW-SE) Calculation
For the $j$-th sample, the squared coordinate along the $i$-th principal component is:
$$W_{\text{sq},j,i} = (W_{\alpha,j,i})^2$$

Normalizing across the $r$ components yields the probability density $e_{j,i}$:
$$e_{j,i} = \frac{W_{\text{sq},j,i}}{\sum_{k=1}^r W_{\text{sq},j,k} + \epsilon}$$
where $\epsilon = 10^{-12}$ prevents division-by-zero. The Fractional Whitened Spectral Entropy is then calculated as:
$$H_{\text{spec},j} = - \sum_{i=1}^r e_{j,i} \ln(e_{j,i} + \epsilon)$$

### 3.6 Grid Significance Mapping via Bivariate Welch's t-test
For each layer $l \in [1, L]$ and whitening factor $\alpha \in [0.0, 1.0]$ (with step 0.1), we group the spectral entropy distributions by correctness $y_j \in \{0, 1\}$:
* **Incorrect Group ($y = 0$)**: $X_0 = \{H_{\text{spec},j} \mid y_j = 0\}$, size $n_0$, mean $\bar{X}_0$, variance $s_0^2$.
* **Correct Group ($y = 1$)**: $X_1 = \{H_{\text{spec},j} \mid y_j = 1\}$, size $n_1$, mean $\bar{X}_1$, variance $s_1^2$.

We apply Welch's t-test to compare the group means without assuming equal variances:
$$t = \frac{\bar{X}_0 - \bar{X}_1}{\sqrt{\frac{s_0^2}{n_0} + \frac{s_1^2}{n_1}}}$$

The effective degrees of freedom $\nu$ are approximated via:
$$\nu \approx \frac{\left( \frac{s_0^2}{n_0} + \frac{s_1^2}{n_1} \right)^2}{\frac{1}{n_0-1}\left( \frac{s_0^2}{n_0} \right)^2 + \frac{1}{n_1-1}\left( \frac{s_1^2}{n_1} \right)^2}$$

Using Student’s t-distribution, we find the two-tailed p-value and map the polarized significance grid:
$$\text{Grid}(l, \alpha) = -\log_{10}(p) \cdot \text{sign}(t)$$

---

## 4. Analysis of Cross-Validation and High-Rank Latent Spaces

We discuss how data co-dependency is handled in unsupervised subspace diagnostics, addressing why direct Leave-One-Out Cross-Validation (LOO-CV) can introduce numerical instabilities in high-dimensional representations.

### 4.1 The High-Rank Nature of LLM Latent Spaces
Our measurements of the activation covariance matrices suggest that representations in deep Transformer layers can be relatively high-rank. For instance, in Layer 24 of `Gemma-2-2B-IT`, preserving 90% of the total variance requires retaining approximately $r_{\text{eff}} = 381$ dimensions. This indicates that the evaluation sample activations are spread across a moderately high-dimensional space rather than being tightly restricted to a very low-dimensional manifold.

### 4.2 Observation on LOO-CV Instability in High-Rank Settings
Let $C_{\text{full}} = H_c^T H_c \in \mathbb{R}^{D \times D}$ be the full covariance matrix, and let $C_{\text{train}} = C_{\text{full}} - x_{\text{test}}^T x_{\text{test}}$ be the covariance matrix excluding a single test activation vector $x_{\text{test}} \in \mathbb{R}^{1 \times D}$.

In high-rank settings, the single test vector $x_{\text{test}}$ may contain a localized, orthogonal component $v_{\text{unique}}$ that is poorly represented in the remaining training set:
$$v_{\text{unique}} \perp \text{Span}(V_{\text{train}})$$

Consequently, projecting $x_{\text{test}}$ onto the training basis $V_{\text{train}} \in \mathbb{R}^{D \times r}$:
$$p_{\text{wrong}} = x_{\text{test}} V_{\text{train}}$$
can cause the unrepresented projection energy to bleed into the tail components of $V_{\text{train}}$ corresponding to very small singular values:
$$p_{\text{wrong}, j} \approx \delta > 0 \quad \text{for } j \to r_{\text{eff}}$$

When computing the fractional whitened coordinates:
$$\tilde{p}_j = \frac{p_{\text{wrong}, j}}{(\sigma_j^{(\text{train})})^\alpha}$$
the extremely small singular values $(\sigma_j^{(\text{train})})^\alpha$ in the denominator can act as a localized error amplifier for the leaked projection signal $\delta$, causing coordinate inflation:
$$\tilde{p}_j = \frac{\delta}{(\sigma_j^{(\text{train})})^\alpha} \gg 1 \quad \text{as } \sigma_j \to \epsilon$$

This inflation can flatten the normalized energy profile, driving the calculated spectral entropy $H_{\text{spec}}$ artificially toward its theoretical maximum $\ln(r)$ and masking the underlying factual uncertainty signal.

### 4.3 Alternative Implementation: Reference Subspace Saturation
To avoid recalculating full SVDs or introducing projection leakage, one can utilize a pre-computed reference basis derived from a sufficiently large and diverse offline anchor corpus.

We define the Reconstruction Error $E_{\text{recon}}$ of a new incoming activation $x_{\text{new}}$ on a reference basis $V_{\text{ref}}$ as:
$$E_{\text{recon}}(x_{\text{new}}) = \Vert x_{\text{new}} - x_{\text{new}} V_{\text{ref}} V_{\text{ref}}^T \Vert_2^2$$

In practice, due to the representational degeneration and anisotropy observed in autoregressive models, the effective latent space spanned by actual tokens is much smaller than the physical dimension $D$. Hence, when the offline anchor corpus size $N$ is moderately large, the reference subspace achieves a degree of saturation, reducing projection errors for inline queries:
$$\lim_{N \to \text{large}} E_{\text{recon}}(x_{\text{new}}) \to \epsilon \approx 0$$

Using a pre-computed reference basis stabilizes the coordinate projection and reduces the online computation to $O(D^2)$ projection operations, which is convenient for edge diagnostic implementations.

---

## 5. Subspace Alignment and Topological Similarity Metrics

### 5.1 Adaptive 90% Cumulative Variance Non-Square Subspace Alignment
To compare the latent representation geometry of different tasks without relying on a fixed, arbitrary truncation parameter $k$, we use a cumulative variance threshold.

For tasks $A$ and $B$, we perform SVD on their centered activation matrices to retrieve singular values $S_A, S_B$ and right singular matrices $V_A, V_B \in \mathbb{R}^{D \times r}$. We track the cumulative variance ratios:
$$\text{CumVar}_A(k) = \frac{\sum_{i=1}^k \sigma^2_{A,i}}{\sum_{j=1}^{r_A} \sigma^2_{A,j}}, \quad \text{CumVar}_B(k) = \frac{\sum_{i=1}^k \sigma^2_{B,i}}{\sum_{j=1}^{r_B} \sigma^2_{B,j}}$$

We identify the minimum truncation indices $k_A$ and $k_B$ required to capture 90% of the total variance:
$$k_A = \min\{k \mid \text{CumVar}_A(k) \ge 0.90\}, \quad k_B = \min\{k \mid \text{CumVar}_B(k) \ge 0.90\}$$

This constructs two non-square basis matrices:
$$Q_A = V_A^T [1 \dots k_A, :] \in \mathbb{R}^{k_A \times D}, \quad Q_B = V_B^T [1 \dots k_B, :] \in \mathbb{R}^{k_B \times D}$$

The transition matrix $M \in \mathbb{R}^{k_A \times k_B}$ is:
$$M = Q_A Q_B^T$$

Computing the SVD of the non-square matrix $M$:
$$M = U_M \Sigma_M V_M^T$$
where $\Sigma_M = \text{diag}(\cos \theta_1, \dots, \cos \theta_m)$ and $m = \min(k_A, k_B)$. The principal angles $\theta_i$ represent the geometric alignment between the two spaces. The Subspace Alignment Index is measured as the average cosine:
$$\text{Alignment}(A, B) = \frac{1}{\min(k_A, k_B)} \sum_{i=1}^{\min(k_A, k_B)} \cos \theta_i$$

### 5.2 Composite Structural Similarity Index (CSSI)
To quantify the topological similarity between two significance grids $\text{Grid}_A, \text{Grid}_B \in \mathbb{R}^{11 \times 26}$, we construct a composite metric. First, we discretize the significance values using a threshold of $\tau = 1.301$ ($p < 0.05$) to form ternary matrices $S \in \{-1, 0, 1\}^{11 \times 26}$:"
$$S_{i,j} = \begin{cases} 
   +1, & \text{Grid}_{i,j} \ge 1.301 \\\\
   -1, & \text{Grid}_{i,j} \le -1.301 \\\\
   0, & -1.301 < \text{Grid}_{i,j} < 1.301
\end{cases}$$

The Cosine Similarity of Significant Signs (CSPS) measures the alignment of positive and negative significance regions:
$$PS(A, B) = \frac{\sum_{i=1}^{11} \sum_{j=1}^{26} S_{A,i,j} \cdot S_{B,i,j}}{\sqrt{\sum_{i,j} S_{A,i,j}^2} \sqrt{\sum_{i,j} S_{B,i,j}^2}}$$

We combine CSPS with standard structural metrics (HOG, SSIM, and NMI) to calculate the Composite Structural Similarity Index (CSSI):
$$\text{CSSI}(A, B) = 0.40 \cdot \max(0, PS(A, B)) + 0.30 \cdot \text{HOG}(A, B) + 0.15 \cdot \text{SSIM}(A, B) + 0.15 \cdot \text{NMI}(A, B)$$

---

## 6. Empirical Observations and Discussion

Our measurements were conducted on a single NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM) using hidden state activations collected from `Gemma-2-2B-IT` over a corpus of $\sim 11,000$ prompts spanning 11 task domains.

### 6.1 Out-of-Distribution (OOD) Reference Scores
For contextual reference, the baseline classification accuracies of the studied `Gemma-2-2B-IT` model across selected subsets of tasks are summarized in Table 1.

#### Table 1: Task Subsegment Accuracy Reference Metrics
| Task Dataset | Sample Size | Correct (Score=1) | Incorrect (Score=0) | Refused/Filtered |
| :--- | :---: | :---: | :---: | :---: |
| **MultiArith** | 500 | 91.80% | 7.60% | 0.60% |
| **MMLU_ElemMath** | 500 | 43.80% | 54.20% | 2.00% |
| **TriviaQA** | 500 | 55.62% | 43.57% | 0.80% |
| **MMLU_CollBio** | 500 | 71.20% | 28.40% | 0.40% |

### 6.2 Geometric and Phase Space Alignment Observations
Comparing the representation spaces across SVD Cosine Subspaces and Entropy Phase CSSI grids suggests several interesting trends:
* **Trend A (Static Coordinate Semantic Alignment)**: Under static SVD subspace alignment, we observe that the coordinate spaces organize primarily around semantic categories. The semantic clustering quality index reached **2.2475**, compared to **1.5012** for format-based clustering, suggesting that static representation coordinates are largely anchored to semantic domains.
* **Trend B (Formatting Constraints in Entropy Transitions)**: When comparing the dynamic phase grids using CSSI, we observe a different alignment profile. The format-based clustering quality index was **1.0877**, whereas the semantic clustering index measured **0.9715**. Format-based clustering also showed an intra-group cohesion of **0.3746** (representing a modest **11.96%** variation over semantic grouping at **0.3471**). This suggests that downstream structural templates may influence the entropic trajectories of hidden layers.
* **Trend C (Observation of the Geometric Invariant Gap)**: By tracking the cosine subspace alignment between tasks with matching semantics but differing formats (SSSF vs. SSDF), we note a representation decoupling beginning in early layers (Layer 2, $\Delta = +0.0994$). This spatial difference maintains a relatively stable margin—which we label the Formatting Difference Gap ($\Delta \approx 0.115$)—across intermediate layers (Layer 10 to Layer 25), as presented in Table 2.

#### Table 2: Subspace Cosine Alignment and the Formatting Difference Gap ($\Delta$)
| Layer Level | Same Semantic Same Format (SSSF) | Same Semantic Diff Format (SSDF) | Formatting Difference Gap ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Layer 01** | 0.8501 | 0.7660 | +0.0841 |
| **Layer 02 (Early Decoupling)** | 0.8019 | 0.7025 | **+0.0994** |
| **Layer 05** | 0.7613 | 0.6812 | +0.0801 |
| **Layer 10 (Mid-Layer Transition)** | 0.7044 | 0.5846 | **+0.1198** |
| **Layer 15** | 0.6857 | 0.5923 | +0.0934 |
| **Layer 20 (Deep-Layer Stability)** | 0.6579 | 0.5435 | **+0.1144** |
| **Layer 25** | 0.6505 | 0.5321 | **+0.1184** |
| **Layer 26** | 0.6779 | 0.5674 | +0.1105 |

These empirical observations point to a possible dual nature of LLM latent paths: static knowledge mapping behaves primarily as a semantic landscape, while structural templates shape the layer-by-layer dynamic computation paths.

---

## 7. Discussion on Potential Practical Insights and Limitations

We discuss the possible implications of this analytical perspective, as well as the important constraints of our exploratory study:
1. **Potential Diagnostic Auxiliary Tool**: Using FWM spectral entropy as a diagnostic lens provides a non-invasive view of latent space configurations. By focusing on intermediate hidden states, it offers a diagnostic perspective that does not require training classifiers or adjusting active model parameters.
2. **Implications for Supervised Fine-Tuning and Preference Modeling**: The observation of a stable representation margin ($\Delta \approx 0.115$) associated with formatting styles suggests that instruction formats act as an active representational constraint. For researchers designing SFT or RLHF datasets, this highlights the import of balancing structural prompt templates to avoid overfitting models to specific response formats rather than the underlying reasoning logic.
3. **Scale and Scope Limitations**: We emphasize that these observations were recorded on a relatively small-scale model (`Gemma-2-2B-IT`) within a localized computing environment. The extent to which these geometric properties—such as the Formatting Difference Gap—scale to larger, multi-billion parameter architectures or generalize across diverse tokenizer designs remains an open question requiring broader systematic investigation.

---

## References
* **[1]** *NerVE: Nonlinear Eigenspectrum Dynamics in LLM Feed-Forward Networks.* arXiv preprint arXiv:2603.06922, 2026.
* **[2]** *Large Language Models Encode Semantics and Alignment in Linearly Separable Representations.* In Proceedings of the International Conference on Learning Representations (ICLR), 2025.
* **[3]** *The Geometry of Truth in LLM Representations.* In Proceedings of the Neural Information Processing Systems (NeurIPS), Mechanistic Interpretability Workshop, 2024.
* **[4]** *SSAM: Singular Subspace Alignment for Merging Multimodal Large Language Models.* In Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL), 2026.
* **[5]** *Subspace Alignment and Representation Drift in Instruction-tuned Transformers.* LessWrong Research Post, December 2025.
