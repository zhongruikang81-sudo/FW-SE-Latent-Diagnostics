# A Fractional Whitening Lens for LLM Latent Geometry

<p align="center">
  <img src="assets/composite_comparison_grid.png" width="100%" alt="FW-SE Latent Geometry Cover" />
</p>

[![Language](https://img.shields.io/badge/Language-Python-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Interpretability](https://img.shields.io/badge/Domain-Mechanistic%20Interpretability-red.svg)]()
[![Hardware](https://img.shields.io/badge/Hardware-RTX%204060%20(8GB%20VRAM)-brightgreen.svg)]()

I wanted to share a frequency-selective lens for looking at LLM latent spaces. This isn't a definitive theory on downstream tasks, just a methodological tool to see how semantic representations and structural prompt templates couple in the activations.

The core idea is straightforward: center the activations, run a thin SVD, apply fractional whitening, calculate spectral entropy, and map the significance:

$$\text{Zero-Mean Centering} \rightarrow \text{Thin SVD} \rightarrow \text{Fractional Mahalanobis Whitening (FWM)} \rightarrow \text{Spectral Entropy (FW-SE)} \rightarrow \text{Significance Mapping}$$

A quick caveat: these experiments were run on Gemma-2-2B-IT using a single laptop 4060 GPU. It's a Proof of Concept (PoC) to show that the pipeline works. Because LLM latent spaces are high-rank and complex, I might have overlooked some details. I'd love to hear critiques, code audits, or refutations from the mechanistic interpretability community regarding the geometric derivations of full-whitening decay and LOO-CV error inflation.

## Why standard Shannon entropy fails us

How do we check if an LLM is hallucinating or just unsure? The standard move is to measure the Shannon entropy of its output logits:

$$H(Y) = -\sum_{i=1}^{|V|} p_{i} \ln p_{i}$$

Ideally, if the model knows the answer, the next-token probability $P(y|X)$ spikes. If it's guessing, the distribution flattens out, and entropy climbs.

But aligned models (SFT/RLHF) break this simple heuristic. Why? Because they are heavily conditioned to spit out stereotyped formatting templates. You've seen them a thousand times: *"Sure, here is the answer..."* or *"As an AI language model..."*.

For these first few template tokens, the probability distribution spikes on the template words with near-absolute certainty: $P(y_{\text{template}}|X) \to 1$. Naturally, the initial Shannon entropy collapses to zero:

$$H(Y_{0}) \approx -1 \cdot \ln(1) - 0 = 0$$

This surface-level confidence is a complete illusion. Under the hood, the model might be totally clueless, but the template masks the uncertainty. If we want the real, uncorrupted signal, we have to bypass the logits entirely. We need to probe the high-dimensional hidden states $h \in \mathbb{R}^{D}$ directly.

But hidden layer activations—like FFN outputs $h' = \text{GeLU}(W_1 h)$—are a polysemantic mess because of superposition. To track the computation energy, we need a clean, orthogonal coordinate system ordered by variance. SVD is the classic tool for this.

## Where does this fit in?

Probing the spectral properties of latent representations has become popular in mechanistic interpretability. For instance, NerVE [1] analyzed the spectral entropy of normalized eigenvalues to study feature utilization during optimization. My goal here is different: I want to see if spectral dynamics can diagnose factual confidence and errors. But raw spectral properties are easily drowned out by low-frequency, macroscopic features. I solve this by introducing Fractional Mahalanobis Whitening (FWM) to act as a bandpass filter, letting us inspect the quieter, high-frequency components.

We also know that LLMs separate semantic concepts and instruction templates into orthogonal subspaces [2, 3]. But most studies treat these spaces as static. I wanted to look at their dynamic coupling. What I found is that formatting constraints drive the activation path through intermediate layers, creating a representational gap between different formats that persists across multiple layers.

Lastly, Mahalanobis whitening ($\Sigma^{-1/2}$) and principal angles are standard tools to correct anisotropy and compare high-dimensional subspaces [3–5]. But applying them directly to noisy LLM activations is tricky. I show why full whitening ($\alpha=1$) collapses into isotropic noise, and why Leave-One-Out Cross-Validation (LOO-CV) is mathematically unstable in these high-rank spaces. I solve these issues using fractional whitening $\alpha \in (0,1)$ combined with an adaptive 90% variance threshold.

## Building the frequency-selective lens

Let $H \in \mathbb{R}^{N \times D}$ be the activation matrix from the final token of the hidden states at layer $l$, where $N$ is the number of sample queries and $D$ is the hidden dimension ($D = 2048$ for Gemma-2-2B-IT). First, we get the mean activation vector $\mu_H \in \mathbb{R}^{1 \times D}$:

$$\mu_H = \frac{1}{N} \sum_{j=1}^N H_{j, \cdot}$$

And center the activations to $H_c$:

$$H_c = H - \mathbf{1}_N \mu_H$$

where $\mathbf{1}_N \in \mathbb{R}^{N \times 1}$ is a column vector of ones. Next, we apply thin SVD to the centered matrix $H_c$:

$$H_c = U S V^T$$

Here, $U \in \mathbb{R}^{N \times r}$ is the left singular matrix, $S = \text{diag}(\sigma_1, \sigma_2, \dots, \sigma_r) \in \mathbb{R}^{r \times r}$ is the singular value diagonal matrix (where $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 10^{-7}$), and $V \in \mathbb{R}^{D \times r}$ is the right singular matrix. The coordinates on these principal axes are $P \in \mathbb{R}^{N \times r}$:

$$P = H_c V = U S$$

What happens if we project onto raw, unwhitened coordinates? If we set our coordinate matrix to $W_{0} = P = U S$, the coordinate of sample $j$ along component $i$ is scaled by its raw singular value: $w_{j,i}^{(0)} = u_{j,i} \sigma_i$. But LLM activation spaces are notoriously anisotropic. Their singular values decay incredibly fast ($\sigma_1 \gg \sigma_r$). Consequently, formatting templates and basic grammar in the first few dominant dimensions completely overwhelm the energy spectrum:

$$E_{j,i} \propto u_{j,i}^2 \sigma_i^2$$

This raw state artificially compresses the spectral entropy. The factual uncertainty signals, which live in the quieter tail components, get completely washed out by the loud low-frequency templates. The statistical difference between correct and incorrect groups simply disappears.

But what if we go to the other extreme and apply full Mahalanobis whitening ($\alpha = 1$)? We scale the coordinates by $S^{-1}$:

$$W_1 = P S^{-1} = (U S) S^{-1} = U$$

This throws our coordinates straight into the row vectors of the left singular matrix $U$:

$$w_{j}^{(1)} = u_{j} = [u_{j,1}, u_{j,2}, \dots, u_{j,r}]$$

Once we trim the numerical null space, $U$ is just a square orthogonal matrix ($U \in \mathbb{R}^{r \times r}$, where $r \le N$). By SVD properties, its columns are orthonormal. But because it's square, its row vectors $u_j$ must also form a standard orthonormal basis of $\mathbb{R}^r$.

And here lies the catch—a massive geometric trap. By stripping away the singular value matrix $S$, we lose all variance weights. The row vector $u_j$ collapses into a unit vector sitting on a sphere:

$$\sum_{k=1}^r u_{j,k}^2 = \| u_j \|_2^2 = 1$$

Now the energy projected onto any coordinate is completely symmetric. By the law of total probability, the expected normalized energy density $\mathcal{E}_{j,i}$ flattens out into a uniform state:

$$\mathbb{E}[\mathcal{E}_{j,i}] = \mathbb{E} \left[ \frac{u_{j,i}^2}{\| u_j \|_2^2} \right] = \frac{1}{r} \sum_{k=1}^r \mathbb{E}[u_{j,k}^2] = \frac{1}{r} \mathbb{E}[1] = \frac{1}{r}$$

The energy distribution collapses to uniform noise. Spectral entropy for every single sample converges to the theoretical maximum:

$$H_{\text{spec}} \to \ln(r)$$

The diagnostic signal is dead, buried under isotropic noise.

The solution is to walk the middle ground. We introduce a fractional whitening factor $\alpha \in [0, 1]$:

$$W_{\alpha} = P S^{-\alpha} = U S^{1-\alpha}$$

Choosing $\alpha \in (0, 1)$ lets us dim the loud formatting templates (by partially dampening the top singular values) without letting the tail explode into white noise. It's a goldilocks zone for diagnostic signals.

For the $j$-th sample, the squared coordinate along the $i$-th principal component is:

$$W_{\text{sq}, j, i} = \left( W_{\alpha, j, i} \right)^2$$

Normalizing this yields the energy probability density $e_{j, i}$:

$$e_{j, i} = \frac{W_{\text{sq}, j, i}}{\sum_{k=1}^r W_{\text{sq}, j, k} + \epsilon}$$

where $\epsilon = 10^{-12}$ is a regularization constant. The Fractional Whitened Spectral Entropy of the sample is:

$$H_{\text{spec}, j} = - \sum_{i=1}^r e_{j, i} \ln(e_{j, i} + \epsilon)$$

To track this signal across the network, we scan layers $l$ and whitening factors $\alpha$. At each grid point, we split the spectral entropy distributions by correctness $y_j \in \{0, 1\}$. We split them into an incorrect group $X_0$ (with size $n_0$, mean $\bar{X}_0$, and variance $s_0^2$) and a correct group $X_1$ (with size $n_1$, mean $\bar{X}_1$, and variance $s_1^2$). We then run Welch's t-test:

$$t = \frac{\bar{X}_0 - \bar{X}_1}{\sqrt{\frac{s_0^2}{n_0} + \frac{s_1^2}{n_1}}}$$

The effective degrees of freedom $\nu$ are approximated using the Welch-Satterthwaite equation:

$$\nu \approx \frac{\left( \frac{s_0^2}{n_0} + \frac{s_1^2}{n_1} \right)^2}{\frac{1}{n_0-1}\left(\frac{s_0^2}{n_0}\right)^2 + \frac{1}{n_1-1}\left(\frac{s_1^2}{n_1}\right)^2}$$

And we map the significance intensity at the grid coordinate $(l, \alpha)$ using:

$$\text{Grid}(l, \alpha) = -\log_{10}(p) \cdot \text{sign}(t)$$

This maps the statistical difference across all layers and whitening factors, yielding the diagnostic grids.

![Composite Comparison Grid](https://raw.githubusercontent.com/zhongruikang81-sudo/FW-SE-Latent-Diagnostics/main/assets/composite_comparison_grid.png)

## The cross-validation trap in high dimensions

Run Leave-One-Out Cross-Validation (LOO-CV) on these unsupervised subspaces to "avoid data leakage," and your statistical significance will vanish instantly. The $p$-values flatline into a useless uniform distribution. Why?

First, LLM activations are high-rank. In Gemma-2-2B-IT's deep layers, you need an effective rank of $r_{\text{eff}} = 381$ just to cover 90% of the variance. They don't sit on a neat low-dimensional manifold. For an evaluation batch of $N = 2000$ samples, this means activations occupy a complex space where individual queries contain highly specific, mutually orthogonal semantic components.

Here is the math of why LOO fails. Let $C_{\text{full}} = H_c^T H_c \in \mathbb{R}^{D \times D}$ be the covariance matrix of the full dataset. If we exclude a single test sample $x_{\text{test}} \in \mathbb{R}^{1 \times D}$, the training covariance is $C_{\text{train}} = C_{\text{full}} - x_{\text{test}}^T x_{\text{test}}$. Because the space is high-rank, the held-out sample $x_{\text{test}}$ almost certainly contains a unique orthogonal component $v_{\text{unique}}$ that the training set has never seen:

$$v_{\text{unique}} \perp \text{Span}(V_{\text{train}})$$

The training basis $V_{\text{train}} \in \mathbb{R}^{D \times r}$ literally cannot represent the direction of $v_{\text{unique}}$.

So when we project $x_{\text{test}}$ onto this mismatched basis:

$$p_{\text{wrong}} = x_{\text{test}} V_{\text{train}}$$

the unrepresented component leaks into the tail dimensions of the training basis:

$$p_{\text{wrong}, j} \approx \delta > 0 \quad \text{for } j \to r_{\text{eff}}$$

And when we apply whitening, those tiny training singular values in the denominator cause a massive, non-linear explosion:

$$\tilde{p}_{j} = \frac{p_{\text{wrong}, j}}{(\sigma_{j}^{(\text{train})})^{\alpha}} = \frac{\delta}{(\sigma_{j}^{(\text{train})})^{\alpha}} \gg 1 \quad \text{as } \sigma_j \to \epsilon$$

The tail coordinates blow up, flattening the normalized energy distribution. The spectral entropy of the test sample artificial explodes:

$$H_{\text{spec}} \to \ln(r)$$

The real uncertainty signal gets buried under projection noise, destroying any statistical differences between groups.

To deploy this without running SVDs on-the-fly, we can leverage subspace saturation. We check the Reconstruction Error $\mathcal{E}_{\text{recon}}$ of online query $x_{\text{new}}$ on a pre-computed static reference basis $V_{\text{ref}}$:

$$\mathcal{E}_{\text{recon}}(x_{\text{new}}) = \| x_{\text{new}} - x_{\text{new}} V_{\text{ref}} V_{\text{ref}}^T \|_2^2$$

While the effective rank $r_{\text{eff}}(N)$ increases with reference corpus size $N$, it is bounded by the hidden dimension $D$. Due to latent anisotropy, the actual spanned subspace is much smaller than $D$. By using a sufficiently large, diverse offline Anchor Corpus, the reference covariance matrix achieves subspace saturation, covering the entire representational capability of the model:

$$\lim_{N \to \text{large}} \mathcal{E}_{\text{recon}}(x_{\text{new}}) \to \epsilon \approx 0$$

Once saturated, we freeze $V_{\text{ref}}$ and $S_{\text{ref}}$. The online complexity collapses to a fast static projection of $O(D^2)$.

![SVD Spectrum Analysis](https://raw.githubusercontent.com/zhongruikang81-sudo/FW-SE-Latent-Diagnostics/main/assets/svd_spectrum_analysis.png)

## Tuning the whitening factor without labels

We don't need task labels to find the optimal $\alpha$. Since singular values follow a power-law decay ($\sigma_i \propto i^{-\beta}$), we find $\alpha_{\text{opt}}$ by maximizing coordinate variance over the anchor corpus:

$$\alpha_{\text{opt}} = \arg\max_{\alpha} \text{Var}\left( \frac{\sigma_i}{\sigma_i^\alpha} \right) \quad \text{subject to } i \le r_{\text{eff}}$$

This makes the framework highly generalizable without task-specific labels.

## Measuring task similarity through principal angles

To verify the task classifications identified by our spectral lens, we use principal angles and a Composite Structural Similarity Index (CSSI) to compare task manifolds.

For tasks $A$ and $B$, SVD yields singular values and right singular matrices $V_A, V_B \in \mathbb{R}^{D \times r}$. We track cumulative variance ratios:

$$\text{CumVar}_A(k) = \frac{\sum_{i=1}^k \sigma_{A, i}^2}{\sum_{j=1}^{r_A} \sigma_{A, j}^2}, \quad \text{CumVar}_B(k) = \frac{\sum_{i=1}^k \sigma_{B, i}^2}{\sum_{j=1}^{r_B} \sigma_{B, j}^2}$$

We find the truncation indices $k_A, k_B$ for 90% variance:

$$k_A = \min \{ k \mid \text{CumVar}_A(k) \ge 0.90 \}, \quad k_B = \min \{ k \mid \text{CumVar}_B(k) \ge 0.90 \}$$

This constructs two non-square basis matrices $Q_A = V_A^T [1 \dots k_A, :] \in \mathbb{R}^{k_A \times D}$ and $Q_B = V_B^T [1 \dots k_B, :] \in \mathbb{R}^{k_B \times D}$. The transition matrix is $M = Q_A Q_B^T \in \mathbb{R}^{k_A \times k_B}$. We calculate the SVD of $M$:

$$M = U_M \Sigma_M V_M^T$$

where $\Sigma_M = \text{diag}(\cos \theta_1, \dots, \cos \theta_m)$ and $m = \min(k_A, k_B)$. The principal angles $\theta_i$ represent the geometric alignment between the two spaces. The Subspace Alignment Index is the average cosine:

$$\text{Alignment}(A, B) = \frac{1}{\min(k_A, k_B)} \sum_{i=1}^{\min(k_A, k_B)} \cos \theta_i$$

To compare two significance grids $\text{Grid}_A, \text{Grid}_B \in \mathbb{R}^{11 \times 26}$, we discretize values using a threshold of $\tau = 1.301$ ($p < 0.05$) to form ternary matrices $S \in \{-1, 0, 1\}^{11 \times 26}$:

$$S_{i, j} = \begin{cases} 
+1, & \text{Grid}_{i, j} \ge 1.301 \\
-1, & \text{Grid}_{i, j} \le -1.301 \\
0, & -1.301 < \text{Grid}_{i, j} \dots 1.301
\end{cases}$$

The Cosine Similarity of Significant Signs (CSPS) is:

$$PS(A, B) = \frac{\sum_{i=1}^{11} \sum_{j=1}^{26} S_{A, i, j} \cdot S_{B, i, j}}{\sqrt{\sum_{i,j} S_{A, i, j}^2} \sqrt{\sum_{i,j} S_{B, i, j}^2}}$$

We combine CSPS with HOG, SSIM, and NMI to calculate CSSI:

$$CSSI(A, B) = 0.40 \cdot \max(0, PS(A, B)) + 0.30 \cdot HOG(A, B) + 0.15 \cdot SSIM(A, B) + 0.15 \cdot NMI(A, B)$$

To make sure our diagnostic lens is stable and generalizable, we also ran a stratified split-half reliability test (50% random split) across all evaluated task domains. The split-half subsets show near-identical phase transition boundaries and statistical significance structures compared to the full datasets. The complete set of generated validation heatmaps is stored in the `assets/split_half` directory of the GitHub repository, serving as empirical proof of the framework's mathematical stability.

## What the Gemma-2 activations actually show

For verification, we provide observational results recorded using Gemma-2-2B-IT over a corpus of $\sim$11,000 prompts spanning 11 task domains.

Baseline accuracies across selected subsets of tasks are summarized in Table 1.

| Task Dataset | Sample Size | Correct (Score=1) | Incorrect (Score=0) | Refused/Filtered |
| :--- | :---: | :---: | :---: | :---: |
| **MultiArith** | 500 | 91.80% | 7.60% | 0.60% |
| **MMLU_ElemMath** | 500 | 43.80% | 54.20% | 2.00% |
| **TriviaQA** | 500 | 55.62% | 43.57% | 0.80% |
| **MMLU_CollBio** | 500 | 71.20% | 28.40% | 0.40% |

*Table 1: Task Subsegment Accuracy Reference Metrics*

Using this lens, we observe three core representational behaviors.

First, incorrect responses in Factual QA consistently show elevated spectral entropy under centered-whitened projections ($\alpha \approx 0.6$ in Layer 18-21, $p < 1e-10$), signaling high-dimensional energy leakage (OOD behavior).

Second, mathematical and logic tasks show a distinct polarization under the whitening parameter $\alpha$. In the low-$\alpha$ band, correct responses exhibit lower entropy, suggesting highly structured template execution in macroscopic low-frequency dimensions. But in the high-$\alpha$ band, correct responses exhibit higher entropy, representing active tracking of multiple mathematical operators in high-frequency dimensions. Conversely, incorrect reasoning falls back to simple heuristics (e.g., repeating specific figures), compressing high-frequency energy.

Third, we notice that static coordinates reflect semantic fields (Semantic Alignment = 2.2475 vs. Format Alignment = 1.5012). However, dynamic CSSI phase transitions are heavily shaped by structural templates (Format CSSI = 1.0877 vs. Semantic CSSI = 0.9715), exposing a representational gap (manifested as the formatting difference gap $\Delta$ from Layer 10 to 25) as detailed in Table 2.

![Layerwise Subspace Alignment Curves](https://raw.githubusercontent.com/zhongruikang81-sudo/FW-SE-Latent-Diagnostics/main/assets/layerwise_subspace_alignment_curves.png)

| Layer Level | Same Semantic Same Format (SSSF) | Same Semantic Diff Format (SSDF) | Formatting Difference Gap ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Layer 01** | 0.8501 | 0.7660 | +0.0841 |
| **Layer 02 (Early Decoupling)** | 0.8019 | 0.7025 | +0.0994 |
| **Layer 05** | 0.7613 | 0.6812 | +0.0801 |
| **Layer 10 (Mid-Layer Transition)** | 0.7044 | 0.5846 | +0.1198 |
| **Layer 15** | 0.6857 | 0.5923 | +0.0934 |
| **Layer 20 (Deep-Layer Stability)** | 0.6579 | 0.5435 | +0.1144 |
| **Layer 25** | 0.6505 | 0.5321 | +0.1184 |
| **Layer 26** | 0.6779 | 0.5674 | +0.1105 |

*Table 2: Subspace Cosine Alignment and the Formatting Difference Gap ($\Delta$)*

## Looking forward

The fractional whitening spectrum lens acts as a diagnostic tool. Probing intermediate hidden representations avoids parameter updates or active intervention. It gives us a way to track internal network uncertainty directly.

These findings also have implications for dataset design. The formatting divergence gap highlights that template boundaries influence computation paths. This points to the importance of balancing instruction formatting during SFT or preference tuning to avoid model over-conditioning.

Of course, these explorations are recorded on Gemma-2-2B-IT in a single laptop environment. Establishing whether these mathematical properties scale consistently to larger parameter regimes remains a valuable direction for collaborative exploration.

## References

[1] *NerVE: Nonlinear Eigenspectrum Dynamics in LLM Feed-Forward Networks.* arXiv preprint arXiv:2603.06922, 2026.  
[2] *Large Language Models Encode Semantics and Alignment in Linearly Separable Representations.* ICLR, 2025.  
[3] *The Geometry of Truth in LLM Representations.* NeurIPS, Mechanistic Interpretability Workshop, 2024.  
[4] *SSAM: Singular Subspace Alignment for Merging Multimodal Large Language Models.* ACL, 2026.  
[5] *Subspace Alignment and Representation Drift in Instruction-tuned Transformers.* LessWrong Research Post, December 2025.
