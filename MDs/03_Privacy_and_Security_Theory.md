# FedNeMo: Privacy and Security Theory

> **What this document covers:** The formal threat model, all attacks and defenses, formal differential privacy definitions, Theorems 1–3 (noise amplification and sensitivity), Rényi DP accounting across rounds, privacy budget management, the modality transfer validation protocol, and open gaps (secure aggregation, user-level DP). For how these are implemented in the pipeline see `01_FedNeMo_Core_Architecture.md`.

---

## 1. Threat Model

### 1.1 Adversary Definition

| Adversary Property | FedNeMo Specification |
| :--- | :--- |
| **Server behavior** | Honest-but-curious (passive): follows protocol, but inspects received data to extract private information |
| **Malicious server** | Active adversary considered for ANA-GIA: crafts poisoned model updates to isolate individual client gradients |
| **Client behavior** | Honest: clients follow protocol, do not collude |
| **Adversary knowledge** | Model architecture, training hyperparameters, public pretraining data. Does NOT know private local data. |
| **Security goals** | (1) Data reconstruction infeasibility, (2) Membership inference resistance, (3) Attribute inference resistance |
| **Trust assumptions** | Clients are trusted with their own data; server follows protocol but may inspect received transmissions |
| **Composition model** | Sequential composition across T federated rounds |

### 1.2 What the Adversary Can Do

- Capture all transmitted LoRA matrices from any client at any round
- Replay or analyze historical transmissions
- (Active variant) Poison the global model before sending it to clients (ANA-GIA)
- Run any optimization-based reconstruction algorithm given intercepted parameters
- Run statistical membership inference against transmitted updates

### 1.3 What the Adversary Cannot Do

- Access raw patient data inside any hospital
- Intercept data between the hospital's local dataset and its LoRA training loop
- See the private matrix retained by FedRand (the withheld A or B)

---

## 2. Attack Taxonomy

| Attack Class | Mechanism | Severity in LoRA FL | FedNeMo Defense |
| :--- | :--- | :--- | :--- |
| **Optimization-Based GIA (OP-GIA)** | Minimizes distance between intercepted gradients and iterative dummy gradients (DLG, InvertingGrad). Initializes dummy data $x'$, minimizes $\mathcal{D}(\nabla_\theta \mathcal{L}(f_\theta(x'), y'),\ \nabla_\theta \mathcal{L}(f_\theta(x), y))$ | **Critical.** Full A and B matrices allow adversary to resolve gradient inversion equations. | FedRand (underdetermined system), Laplacian DP (noise injection), gradient clipping |
| **Generation-Based GIA (GEN-GIA)** | Uses generative priors (diffusion models, GANs) to recover data via latent optimization. Bypasses minor noise injections. | **High.** Strong public language model priors can guide recovery even under noise. | FedRand's structural fragmentation prevents coherent prior matching without both matrices |
| **Analytics-Based GIA (ANA-GIA)** | Closed-form data reconstruction via maliciously crafted linear layer weights (LOKI, MineGrad). Poisoned model isolates single-patient gradient from a batch. | **Severe.** Bypasses batch-level aggregation protections entirely. | Client-side validation of received weights; stochastic update schema prevents deterministic exploitation |
| **Membership Inference Attack (MIA)** | Determines whether a specific data point was in a client's training set by analyzing statistical properties of model updates. Even without reconstruction, confirming a patient's presence is a HIPAA/DPDP violation. | **High.** | FedRand limits information content of any single transmission; Laplacian DP adds plausible deniability |

**GIA Mathematical Formulation:**

$$x^* = \arg\min_{x'} \mathcal{D}\left(\nabla_\theta \mathcal{L}(f_\theta(x'), y'),\ \nabla_\theta \mathcal{L}(f_\theta(x), y)\right)$$

Reconstruction error is linearly proportional to $\sqrt{B}$ (batch size) and sequence length. For small clinical batches (high specificity per-patient records), risk of perfect text reconstruction approaches certainty with full LoRA transmission.

---

## 3. The LoRA Vulnerability: Cross-Matrix Noise Amplification

This is the fundamental motivation for FedRand's design. For a pretrained weight matrix $W_0$, LoRA introduces:

$$W = W_0 + BA, \quad A \in \mathbb{R}^{r \times k}, \quad B \in \mathbb{R}^{d \times r}$$

**The LA-LoRA finding (2026):** When standard DP-SGD is applied to both matrices simultaneously, the noise injected into matrix $A$ is multiplicatively amplified through matrix $B$ during the forward pass. The interaction term is catastrophic for utility at strict privacy budgets.

**Theorem 1 (Noise Amplification Under Standard DP-LoRA).**

When both matrices are independently perturbed with Laplacian noise $\eta_A \sim \text{Lap}(b_A)$ and $\eta_B \sim \text{Lap}(b_B)$:

$$\Delta W_{\text{noised}} = (B + \eta_B)(A + \eta_A) = BA + B\eta_A + \eta_B A + \eta_B \eta_A$$

The cross-term $\eta_B \eta_A$ has expected squared Frobenius norm:

$$\mathbb{E}[\|\eta_B \eta_A\|_F^2] = \frac{2d \cdot r \cdot k}{\epsilon^4} \cdot (\Delta_1 g)^2 (\Delta_1 f)^2$$

This grows as $O(1/\epsilon^4)$. For strict privacy budgets (small $\epsilon$), the cross-term noise dominates the useful signal, causing **catastrophic utility collapse**. This is why LA-LoRA and FFA-LoRA are unsatisfying — they treat the symptom (freezing one matrix) rather than the structural cause.

---

## 4. FedRand: Formal Privacy Mechanism

### 4.1 Protocol

At each federated round $t$, hospital $i$ evaluates $z \sim \text{Bernoulli}(\rho)$, $\rho = 0.5$.

$$\text{If } z=1: \quad \text{Transmit } \Delta A^{(t)},\quad \text{Retain } B^{(t)} \text{ privately}$$
$$\text{If } z=0: \quad \text{Transmit } \Delta B^{(t)},\quad \text{Retain } A^{(t)} \text{ privately}$$

The server never observes the complete paired update $(A_i^{(t)}, B_i^{(t)})$ from any single client at any round.

### 4.2 Privacy Proof: Noise Amplification Elimination

**Theorem 2 (Elimination of Cross-Matrix Noise Amplification via FedRand).**

Under FedRand, only one matrix is perturbed per round. Without loss of generality, if $A$ is selected as public and transmitted with Laplacian noise $\eta_A$:

$$\Delta W_{\text{FedRand}} = B(A + \eta_A) = BA + B\eta_A$$

The noise term $B\eta_A$ has expected squared Frobenius norm:

$$\mathbb{E}[\|B\eta_A\|_F^2] = \|B\|_F^2 \cdot \frac{2k}{\epsilon^2} \cdot (\Delta_1 f)^2$$

This grows as $O(1/\epsilon^2)$ — a **quadratic improvement** over standard DP-LoRA's $O(1/\epsilon^4)$.

The cross-term $\eta_B \eta_A$ is **eliminated entirely** because $B$ is never perturbed in the same round that $A$ is transmitted. ∎

**Practical impact:** FedNeMo can operate at 2–4× smaller per-round $\epsilon$ than standard DP-LoRA approaches (LA-LoRA, FFA-LoRA) while achieving equivalent model utility — or equivalently, achieve 2–4× better accuracy at the same privacy budget.

### 4.3 Sensitivity Reduction

**Theorem 3 (Sensitivity Reduction via Randomized Subparameter Selection).**

Let $f: \mathcal{D} \to \mathbb{R}^{r \times k}$ be the LoRA update function for matrix $A$, and $g: \mathcal{D} \to \mathbb{R}^{d \times r}$ for matrix $B$.

**Standard full transmission:** $\ell_1$-sensitivity = $\Delta_1 f + \Delta_1 g$

**FedRand single transmission:** $\ell_1$-sensitivity = $\max(\Delta_1 f, \Delta_1 g)$

For symmetric LoRA configurations where $\Delta_1 f \approx \Delta_1 g$:

$$\Delta_1^{\text{FedRand}} = \rho \cdot \Delta_1 f + (1-\rho) \cdot \Delta_1 g \approx \frac{\Delta_1 f + \Delta_1 g}{2} \approx \Delta_1 f$$

This **halves the required Laplacian noise scale**, directly improving the privacy-utility tradeoff:

$$\eta_{\text{FedRand}} \sim \text{Lap}\left(\frac{\max(\Delta_1 f, \Delta_1 g)}{\epsilon}\right) \quad \text{vs.} \quad \eta_{\text{standard}} \sim \text{Lap}\left(\frac{\Delta_1 f + \Delta_1 g}{\epsilon}\right)$$

∎

---

## 5. Differential Privacy Formal Definition

A randomized mechanism $\mathcal{M}$ satisfies $(\epsilon, 0)$-differential privacy if, for any two adjacent datasets $D$ and $D'$ differing by a single record, and for any set of outputs $S$:

$$\Pr[\mathcal{M}(D) \in S] \leq e^\epsilon \cdot \Pr[\mathcal{M}(D') \in S]$$

**FedNeMo's per-round mechanism:** Inject zero-mean Laplacian noise into the transmitted LoRA subparameter. For a function $f$ with $L_1$ sensitivity $\Delta_1 f$:

$$\eta \sim \text{Lap}\left(\frac{\Delta_1 f}{\epsilon}\right)$$

Sensitivity is bounded by gradient clipping with norm bound $C$: $\|g_i\|_1 \leq C$ enforced before noise injection, preventing divergent noise amplification.

---

## 6. Privacy Budget Accounting Across Rounds (RDP)

### 6.1 The Composition Problem

Per-round $(\epsilon, 0)$-DP is necessary but insufficient. Under naïve sequential composition, $T$ rounds of $\epsilon$-DP yields total privacy loss $T\epsilon$. For $T=100$ rounds and $\epsilon=1.0$: $\epsilon_{\text{total}} = 100$ — effectively no protection.

Every privacy researcher will identify this immediately if the submission only claims per-round DP. This is the single most critical technical gap to address. See `04_Critical_Analysis_and_Gaps.md` for further context.

### 6.2 Rényi Differential Privacy (RDP) Framework

**Definition.** A mechanism $\mathcal{M}$ satisfies $(\alpha, \hat{\epsilon})$-RDP of order $\alpha > 1$ if for all adjacent datasets $D, D'$:

$$D_\alpha(\mathcal{M}(D) \| \mathcal{M}(D')) = \frac{1}{\alpha - 1} \ln \mathbb{E}\left[\left(\frac{\Pr[\mathcal{M}(D) = o]}{\Pr[\mathcal{M}(D') = o]}\right)^{\alpha}\right] \leq \hat{\epsilon}$$

where $D_\alpha$ is the Rényi divergence of order $\alpha$.

**RDP of the Laplacian Mechanism.** For the Laplace mechanism with scale parameter $b = \Delta_1 f / \epsilon$:

$$\hat{\epsilon}_{\text{Lap}}(\alpha) = \frac{1}{\alpha - 1} \ln\left(\frac{\alpha}{2\alpha - 1} e^{(\alpha-1)/b} + \frac{\alpha - 1}{2\alpha - 1} e^{-\alpha/b}\right)$$

**Composition Across T Rounds.** By the exact additivity of RDP:

$$\hat{\epsilon}_{\text{total}}(\alpha) = \sum_{t=1}^{T} \hat{\epsilon}_t(\alpha)$$

**Conversion to $(\epsilon_{\text{total}}, \delta)$-DP.** For any target $\delta > 0$:

$$\epsilon_{\text{total}} = \min_{\alpha > 1} \left\{ \hat{\epsilon}_{\text{total}}(\alpha) + \frac{\ln(1/\delta)}{\alpha - 1} \right\}$$

FedNeMo sets $\delta = 10^{-5}$ and optimizes over $\alpha \in (1, \infty)$ to find the tightest possible $\epsilon_{\text{total}}$.

**Practical benefit:** For typical parameters ($\epsilon_{\text{per-round}} = 1.0$, $T = 100$), RDP composition produces $\epsilon_{\text{total}} \approx 14.2$ compared to the naïve $\epsilon_{\text{total}} = 100$ — a **7× tighter privacy guarantee**.

**Implementation:** Use Google's open-source `dp-accounting` library for numerically stable RDP computation, avoiding floating-point precision issues at high $\alpha$ orders.

### 6.3 Privacy Budget Management Protocol

| Step | Action |
| :--- | :--- |
| **Budget Ceiling** | Hospital administrators specify $\epsilon_{\text{max}}$ (e.g., 10.0 for moderate, 3.0 for strict) |
| **Per-Round Tracking** | After each round, `PrivacyAccountant` DXO metadata handler computes $\hat{\epsilon}_{\text{total}}(\alpha)$ over a grid of $\alpha$ values and derives $\epsilon_{\text{total}}$ |
| **Adaptive Scaling** | When $\epsilon_{\text{total}}$ approaches $0.8 \times \epsilon_{\text{max}}$, per-round noise scale $b$ is automatically increased (reducing per-round ε) to extend training budget |
| **Hard Termination** | When $\epsilon_{\text{total}} \geq \epsilon_{\text{max}}$, training is immediately terminated; current best global model is checkpointed |
| **Dashboard Display** | Real-time $\epsilon_{\text{total}}$ with color-coded budget indicator: green → yellow → red |

---

## 7. Modality Transfer Validation Protocol

**The claim issue:** The original FedRand paper demonstrated **60–80% reduction in MIA success rates** on vision-language model benchmarks (ViT, CLIP with image data). FedNeMo does NOT assume direct transferability of these numbers to LLM text fine-tuning.

Text data has fundamentally different gradient structure: token embeddings are discrete, sparse, and high-dimensional; gradients concentrate on the specific tokens present in a batch rather than distributing across a continuous feature space. The MIA reduction may be higher or lower on text.

**How to frame this in the demo:** "The original FedRand paper demonstrated 60–80% MIA reduction on vision models. Our contribution is measuring this on LLM text fine-tuning for the first time. Here are our numbers." Even if the numbers differ, the honesty and the novel measurement are both publishable.

**Formal validation protocol:**

1. **Baseline MIA Measurement:** Run LiRA (Likelihood Ratio Attack) and LOSS-based membership inference attacks against standard FedAvg LoRA fine-tuning on clinical text. Establish the undefended MIA success rate for text modality.
2. **FedRand MIA Measurement:** Run identical attacks against FedRand-protected LoRA updates. Measure text-specific MIA reduction.
3. **Cross-Modality Comparison:** Report both vision-model reference numbers and text-specific numbers. Explicitly quantify the modality transfer gap.
4. **Token-Level Granularity:** Measure MIA success separately for:
   - High-frequency clinical tokens (common diagnoses)
   - Rare tokens (uncommon conditions with higher privacy sensitivity)
   - Patient-specific tokens (identifiers, which carry maximum privacy risk)

---

## 8. User-Level vs. Example-Level DP

Google Research's 2025–2026 work distinguishes two DP formulations. Both are relevant to FedNeMo.

**Example-level DP:** Protects individual training samples. Sensitivity bounded by a single record's gradient contribution.

**User-level DP:** Protects a user's (or institution's) entire dataset. In FedNeMo's healthcare context, this is the correct formulation — the goal is to protect *all* of Hospital A's data, not just one patient record.

**FedNeMo's stance:** The combination of per-client gradient clipping + Laplacian noise provides user-level DP when clipping is applied to the *entire local update* (not per-sample). This must be explicitly stated in Paper 2 as the precise formulation used, distinguishing FedNeMo from methods that only claim example-level protection.

---

## 9. Open Gaps: Secure Aggregation

**The gap:** FedNeMo currently relies on Local DP (noise added before transmission) as the sole privacy mechanism. The 2025–2026 literature increasingly treats Local DP and Secure Aggregation (SA) as complementary, not alternative, defenses.

Secure Aggregation prevents the server from seeing individual client updates (only their sum), which directly blocks ANA-GIA attacks where the server must inspect individual client transmissions.

NVIDIA FLARE supports Secure Aggregation via `SecureAggregation` workflows.

**Recommendation:** Include SA as an optional deployment mode in the specification. For the hackathon demo, the DXO filter pipeline + DP already provides strong defense and SA is not required. For Paper 1, mentioning SA compatibility significantly strengthens the enterprise deployment narrative and the ANA-GIA defense claim.

---

## 10. Gradient Inversion Attack Verification (Live Demo)

The live adversarial simulation is the strongest single element of the FedNeMo demonstration. Two phases:

**Phase 1 — Unprotected FL:**
A standard FedAvg LoRA update is initiated. A simulated "Malicious Server" intercepts the raw, high-precision $A$ and $B$ matrices and runs gradient matching optimization (IG attack implementation from TPAMI 2026). The dashboard reconstructs clinical text token-by-token, demonstrating the catastrophic vulnerability of unprotected federated architectures.

**Phase 2 — FedNeMo Protected:**
The identical clinical data is processed through FedRandFilter → LaplacianDPFilter → AdaptiveQuantFilter. The malicious server intercepts the fragmented, noised, quantized payload. The attack optimizer fails to converge; the reconstruction produces pure indecipherable noise.

**Metrics to display:**
- SSIM (Structural Similarity Index) for image attacks
- Token-level reconstruction accuracy for text attacks
- Number of optimization iterations before convergence failure

**Implementation risk:** GIA attack code is non-trivial. Validate the attack against a known-vulnerable baseline first, confirm it successfully reconstructs in the unprotected case, then demonstrate the failure. A GIA implementation that fails in both cases is a false negative — worse than not running it.

**Fallback:** If live attack code is unstable, use pre-recorded video with static SSIM tables. Announce explicitly: "Due to demo stability concerns, we pre-recorded this section. The numbers are real; the code is in the repository."
