# FedNeMo: Core Architecture and Algorithmic Pipeline

> **What this document covers:** The full system architecture, all algorithmic stages, module-level detail, code structure, and the NVFlare DXO filter pipeline. For Nemotron/NeMo infrastructure specifics see `02_Nemotron_and_NeMo_Stack.md`. For privacy theory and formal proofs see `03_Privacy_and_Security_Theory.md`.

---

## 1. Project Objective

FedNeMo enables multiple isolated healthcare institutions to collaboratively fine-tune an NVIDIA Nemotron LLM without ever sharing raw patient data. The system must simultaneously address three structural problems:

1. **Privacy leakage** — LoRA parameter transmission exposes private training data to Gradient Inversion Attacks (GIA) and Membership Inference Attacks (MIA).
2. **Communication overhead** — Full-precision adapter transmission across 100+ federation rounds is operationally prohibitive on constrained institutional networks.
3. **Statistical heterogeneity (Non-IID drift)** — A tertiary oncology center generates fundamentally different data distributions than a rural primary health center; naive averaging of LoRA updates degrades the global model for every participant.

FedNeMo resolves all three simultaneously. Raw data never crosses the network at any point. Only processed mathematical updates flow between clients and the server.

---

## 2. System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FedNeMo ORCHESTRATOR                         │
│                    (Built on NVIDIA FLARE Server)                    │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ FedPrep      │  │ Adaptive     │  │ ModelController          │  │
│  │ Aggregator   │  │ Global       │  │ (FedAvg Aggregation)     │  │
│  │ (Stage 2)    │  │ Scheduler    │  │ (Stage 5)                │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ GIAShield    │  │ Privacy      │  │ NeMo Model Registry      │  │
│  │ Monitor      │  │ Accountant   │  │ (Nemotron checkpoints)   │  │
│  │ (Attack Sim) │  │ (RDP Tracker)│  │                          │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
└─────────────────────────┬───────────────────────────────────────────┘
                          │ Encrypted Channel │ Quantized Updates
          ┌───────────────┼───────────────────┼───────────────────┐
          ▼               ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  (...)
│   HOSPITAL A    │  │   HOSPITAL B    │  │   HOSPITAL C    │
│  (Metro Tertiary)│  │  (Community)   │  │  (Research)     │
│                 │  │                 │  │                 │
│ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌─────────────┐ │
│ │ MTAE+OCSVM  │ │  │ │ MTAE+OCSVM  │ │  │ │ MTAE+OCSVM  │ │
│ │ DataQuality │ │  │ │ DataQuality │ │  │ │ DataQuality │ │
│ │ (Stage 1)   │ │  │ │ (Stage 1)   │ │  │ │ (Stage 1)   │ │
│ └──────┬──────┘ │  │ └──────┬──────┘ │  │ └──────┬──────┘ │
│        ▼        │  │        ▼        │  │        ▼        │
│ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌─────────────┐ │
│ │ FedPrep     │ │  │ │ FedPrep     │ │  │ │ FedPrep     │ │
│ │ Local Stats │ │  │ │ Local Stats │ │  │ │ Local Stats │ │
│ │ (Stage 2)   │ │  │ │ (Stage 2)   │ │  │ │ (Stage 2)   │ │
│ └──────┬──────┘ │  │ └──────┬──────┘ │  │ └──────┬──────┘ │
│        ▼        │  │        ▼        │  │        ▼        │
│ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌─────────────┐ │
│ │ Nemotron    │ │  │ │ Nemotron    │ │  │ │ Nemotron    │ │
│ │ + NeMo PEFT │ │  │ │ + NeMo PEFT │ │  │ │ + NeMo PEFT │ │
│ │ LoRA Engine │ │  │ │ LoRA Engine │ │  │ │ LoRA Engine │ │
│ │ (Stage 3)   │ │  │ │ (Stage 3)   │ │  │ │ (Stage 3)   │ │
│ └──────┬──────┘ │  │ └──────┬──────┘ │  │ └──────┬──────┘ │
│        ▼        │  │        ▼        │  │        ▼        │
│ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌─────────────┐ │
│ │StochasticLoRA│ │  │ │StochasticLoRA│ │  │ │StochasticLoRA││
│ │ FedRand     │ │  │ │ FedRand     │ │  │ │ FedRand     │ │
│ │ (Stage 4a)  │ │  │ │ (Stage 4a)  │ │  │ │ (Stage 4a)  │ │
│ └──────┬──────┘ │  │ └──────┬──────┘ │  │ └──────┬──────┘ │
│        ▼        │  │        ▼        │  │        ▼        │
│ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌─────────────┐ │
│ │ AdaptQuant  │ │  │ │ AdaptQuant  │ │  │ │ AdaptQuant  │ │
│ │ + Laplacian │ │  │ │ + Laplacian │ │  │ │ + Laplacian │ │
│ │ DP Encoder  │ │  │ │ DP Encoder  │ │  │ │ DP Encoder  │ │
│ │ (Stage 4b)  │ │  │ │ (Stage 4b)  │ │  │ │ (Stage 4b)  │ │
│ └─────────────┘ │  │ └─────────────┘ │  │ └─────────────┘ │
│ 📊 Private EHR  │  │ 📊 Private EHR  │  │ 📊 Private EHR  │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

---

## 3. The Five-Stage Pipeline

### Stage 1 — Unsupervised Outlier Detection (MTAE + OCSVM)

**Purpose:** Sanitize each client's dataset before any training begins. Prevents poisoned, corrupted, or adversarially injected data from influencing LoRA updates.

**Why Nemotron does not handle this:** If you feed poisoned data to an LLM, it learns the poisoned data. This is the "garbage in, garbage out" problem. The detection must happen upstream.

**Why not an LSTM:** LSTMs are computationally heavy sequential models outdated for general structural anomaly detection. The MTAE+OCSVM pipeline is purpose-built, lightweight, and operates before the main training loop.

#### 3.1.1 Multi-Task Autoencoder (MTAE)

A lightweight autoencoder runs on the client's local data. It simultaneously performs reconstruction and classification, minimizing a dual-objective loss:

$$\mathcal{L}_{\text{MTAE}} = \alpha \cdot \underbrace{\frac{1}{N}\sum_{i=1}^{N} \|x_i - \hat{x}_i\|^2}_{\text{Reconstruction Loss (MSE)}} + (1-\alpha) \cdot \underbrace{\left(-\frac{1}{N}\sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})\right)}_{\text{Classification Loss (Cross-Entropy)}}$$

This produces per-sample loss pairs $(l_{\text{recon},j},\ l_{\text{class},j})$ that characterize data quality:

- **High reconstruction loss** → structural abnormality: corrupted text encodings, garbled OCR outputs, malformed JSON records.
- **High classification loss** → semantic anomaly: mislabeled ICD-10 codes, incorrectly attributed clinical notes, adversarially poisoned samples.

#### 3.1.2 Clinical Text MTAE Architecture

Standard MTAE operates on continuous pixel values. Clinical text is discrete, requiring an adapted approach:

**Embedding-Space Autoencoder (Recommended — Approach A):**
1. Feed clinical text through the frozen Nemotron backbone to extract hidden-state embeddings $h_i \in \mathbb{R}^{d_{\text{model}}}$.
2. Train a lightweight autoencoder on these embeddings (not raw tokens):
   - Encoder: $z_i = f_{\text{enc}}(h_i) \in \mathbb{R}^{d_z}$
   - Decoder: $\hat{h}_i = f_{\text{dec}}(z_i) \in \mathbb{R}^{d_{\text{model}}}$
   - Classifier: $\hat{y}_i = f_{\text{cls}}(z_i) \in \mathbb{R}^C$
3. Loss: $\mathcal{L} = \alpha \cdot \text{MSE}(h_i, \hat{h}_i) + (1-\alpha) \cdot \text{CE}(y_i, \hat{y}_i)$

This operates in a continuous embedding space, making it compatible with the downstream OCSVM.

#### 3.1.3 Federated OCSVM Coordination

Filtering is not purely local — it is coordinated globally to establish consistent quality boundaries:

1. Each hospital transmits only the *distribution statistics* of its MTAE loss pairs to the server: `{mean, variance, 10th/25th/50th/75th/90th percentiles, count}`. Raw embeddings or samples are never transmitted.
2. The FLARE server fits a global One-Class Support Vector Machine (OCSVM) with an RBF kernel on the aggregated loss statistics.
3. The server broadcasts the OCSVM decision boundary back to all hospitals.
4. Each hospital locally filters samples that fall outside the decision boundary, pruning them from the training pool before any LoRA round begins.

**Outcome:** Achieves up to **7% accuracy improvement** on non-IID data (per original MTAE study). The Nemotron model only ever trains on verified, high-quality clinical data.

---

### Stage 2 — Federated Preprocessing via Aggregated Statistics (FedPS)

**Purpose:** Harmonize data representations across hospitals without exposing raw data. Eliminates the non-IID preprocessing drift that causes federated models to silently diverge.

**The problem it solves:** If `blood_glucose` is normalized in mg/dL at Hospital A and mmol/L at Hospital B, no aggregation algorithm can compensate — the model receives mathematically incomparable inputs.

#### 3.2.1 Scope Clarification: Structured vs. Unstructured Clinical Data

FedPS applies to two distinct components of Electronic Health Records differently:

**Structured Metadata** (FedPS directly applies):
- Laboratory values (blood glucose, HbA1c, creatinine, vital signs)
- ICD-10 diagnostic codes and CPT procedure codes
- Medication lists (NDC codes), demographic fields (age, weight, BMI)
- Billing records and coded clinical observations

**Unstructured Clinical Text** (handled by Federated Tokenizer Consistency Protocol — see Section 3.2.3):
- Discharge summaries, progress notes, radiology reports, pathology reports
- Free-text clinical narratives constituting the majority of EHR data by volume
- FedPS's numerical scaling and categorical encoding do NOT operate on free text

#### 3.2.2 FedPS Workflow for Structured Metadata

The workflow proceeds in five deterministic steps per training initialization:

1. **Local Statistics Computation** — Each hospital computes compact statistical summaries of its structured data.
2. **Secure Aggregation** — Only these summaries (never raw data) are transmitted to the server.
3. **Global Parameter Derivation** — The server aggregates summaries into globally consistent preprocessing parameters.
4. **Parameter Broadcast** — Global parameters are sent back to all hospitals.
5. **Local Transformation** — Every hospital applies identical preprocessing using the global parameters.

| Preprocessing Function | Client Transmits | Server Aggregates | Clinical Impact |
| :--- | :--- | :--- | :--- |
| **Numerical Feature Scaling** | Local min, max, $\sum x_i$, $\sum x_i^2$ | Global mean $\mu$, variance $\sigma^2$ | Standardizes lab values across unit systems (mg/dL vs. mmol/L) |
| **Categorical Encoding** | Count-Min Sketch, local unique categorical sets | Global set union and frequency mapping | Harmonizes ICD-10 coding variants and departmental acronyms |
| **Target Discretization** | KLL sketches (Karnin-Lang-Liberty quantile summaries) | Global quantile bin boundaries with relative error bounds | Converts continuous measurements into robust discrete tokens |
| **Missing-Value Imputation** | Covariance matrices $X^\top X$, cross-covariance $X^\top y$, local eigenvalue decompositions | Federated Bayesian Linear Regression: $\hat{\beta} = (\sum_k X_k^\top X_k + \lambda I)^{-1} \sum_k X_k^\top y_k$ | Predicts absent metadata (BMI, allergies) without data leaving institutions |
| **Power Transforms** | Local log-likelihood statistics for Box-Cox $\lambda$ | Global $\lambda$ via Brent's method | Normalizes skewed distributions (billing amounts, rare biomarkers) |

#### 3.2.3 Federated Tokenizer Consistency Protocol (for Free Text)

FedPS cannot harmonize free-text clinical narratives. A complementary protocol handles this:

1. **Shared Tokenizer Enforcement:** All hospitals use the identical Nemotron-3 SentencePiece tokenizer (256,000-token vocabulary). Eliminates tokenizer drift where different preprocessing pipelines produce different tokenizations of identical phrases.

2. **Federated Prompt Template Standardization:** FedNeMo defines a canonical prompt format that embeds structured metadata and unstructured text together:

   ```
   [PATIENT_RECORD]
   Demographics: {age} years, {sex}, BMI: {bmi_normalized}
   Lab Values: {lab_panel_standardized}
   ICD-10 Codes: {icd_codes_harmonized}
   Medications: {medication_list_unified}
   
   Clinical Note:
   {free_text_narrative}
   
   Task: {clinical_task_instruction}
   [/PATIENT_RECORD]
   ```
   
   Fields in `{}` brackets are produced by FedPS-harmonized structured data. The `free_text_narrative` is passed through unchanged after tokenization.

3. **Federated Vocabulary Frequency Analysis:** Using Count-Min Sketch summaries from the FedPS categorical encoding stage, FedNeMo aggregates token frequency distributions across hospitals to identify domain-specific medical terms that appear frequently across the federation but may be underrepresented in the base tokenizer vocabulary. These are flagged for potential vocabulary extension or special-token registration.

---

### Stage 3 — Local Fine-Tuning via LoRA

**Purpose:** Each hospital independently fine-tunes Nemotron on its clean, harmonized local data.

**Mechanism:** Low-Rank Adaptation (LoRA) is used. For a frozen pretrained weight matrix $W_0 \in \mathbb{R}^{d \times k}$, LoRA introduces trainable decomposition matrices:

$$W = W_0 + \Delta W = W_0 + B \cdot A$$

where $A \in \mathbb{R}^{r \times k}$ is the down-projection matrix, $B \in \mathbb{R}^{d \times r}$ is the up-projection matrix, and $r \ll \min(d,k)$ is the rank hyperparameter.

**Key constraint:** Only the LoRA matrices $A$ and $B$ are updated. The full 31.6B parameter Nemotron backbone remains frozen. This makes local training feasible on hospital hardware and limits what can be extracted via gradient inversion.

**LoRA rank:** $r = 32$ for the hackathon demo. This constrains the hypothesis space of modifications, preventing overwriting of pretrained representations (structural resistance to catastrophic forgetting).

For LoRA target module selection and the Mamba-2 fused kernel problem, see `02_Nemotron_and_NeMo_Stack.md`.

**Handling all departments from a single model:** Nemotron-3 Nano's Latent Mixture-of-Experts (MoE) architecture with 128 routed experts acts as an internal router. When a Cardiology report is processed, it activates a different subset of experts than when an Oncology report is processed. A single model instance handles all hospital departments simultaneously without cross-contamination — no separate model instances per department are needed.

---

### Stage 4a — Randomized LoRA Subparameter Updates (FedRand / StochasticLoRA)

**Purpose:** Prevent Gradient Inversion Attacks by ensuring the server never sees a complete LoRA update from any single client at any round.

**Mechanism:** Each hospital flips a Bernoulli coin with probability $\rho = 0.5$ before transmitting.

**Case 1 — Matrix $A$ selected as public ($z = 1$):**
$$A_{\text{local}}^{(t)} \leftarrow A_{\text{global}}^{(t)}, \quad B_{\text{local}}^{(t)} \leftarrow B_{\text{private}}^{(t-1)}$$
Hospital sends $\Delta A$ to server; $\Delta B$ stays private.

**Case 2 — Matrix $B$ selected as public ($z = 0$):**
$$B_{\text{local}}^{(t)} \leftarrow B_{\text{global}}^{(t)}, \quad A_{\text{local}}^{(t)} \leftarrow A_{\text{private}}^{(t-1)}$$
Hospital sends $\Delta B$ to server; $\Delta A$ stays private.

**Privacy effect:** The server never simultaneously observes the paired $(A_i^{(t)}, B_i^{(t)})$ of any client at any round. This makes the optimization equations for GIA perpetually underdetermined. See Theorems 1–3 in `03_Privacy_and_Security_Theory.md` for formal proof.

**Communication effect:** Only half the LoRA matrices per layer are transmitted each round — immediately halving upstream bandwidth.

**Novel extension:** FedRand was originally validated only on vision-language models (ViT, CLIP). FedNeMo extends it to Nemotron's Mamba-2 SSM layers, targeting `in_proj`, `out_proj`, `x_proj`, and `dt_proj` in addition to standard Transformer projections. This extension to hybrid Mamba-Transformer architectures has no prior published work.

**Implementation:**

```python
class StochasticLoRAPartitioner:
    def __init__(self, share_probability=0.5):
        self.rho = share_probability
    
    def partition(self, lora_A, lora_B, round_num):
        # Bernoulli sampling per layer
        mask = torch.bernoulli(torch.full((len(lora_A),), self.rho))
        
        shared_params = {}
        private_params = {}
        
        for layer_idx, share_A in enumerate(mask):
            if share_A:  # Share A, keep B private
                shared_params[layer_idx] = {'A': lora_A[layer_idx]}
                private_params[layer_idx] = {'B': lora_B[layer_idx]}
            else:        # Share B, keep A private
                shared_params[layer_idx] = {'B': lora_B[layer_idx]}
                private_params[layer_idx] = {'A': lora_A[layer_idx]}
        
        return shared_params, private_params
```

---

### Stage 4b — Adaptive Differential-Private Quantization

**Purpose:** Apply formal differential privacy guarantees and compress the already-fragmented LoRA parameters before transmission.

#### 3.4.1 Laplacian DP Noise Injection

Zero-mean Laplacian noise is injected into the selected (public) matrix:

$$\eta \sim \text{Lap}\left(\frac{C}{\epsilon}\right)$$

where $C$ is the gradient clipping norm bound (enforced before noise injection to bound $L_1$ sensitivity) and $\epsilon$ is the per-round privacy budget. For formal definition, composition, and RDP accounting across rounds, see `03_Privacy_and_Security_Theory.md`.

**Key design choice — Laplacian over Gaussian:** Laplacian DP provides formal $(\epsilon, 0)$-DP with tighter $\ell_1$ bounds for bounded LoRA updates. Gaussian DP requires $(\epsilon, \delta)$-DP formulation from the outset and has heavier practical implementation overhead.

**Unbiasedness guarantee:** The consecutive application of zero-mean Laplacian noise followed by stochastic uniform quantization preserves unbiasedness in expectation: $\mathbb{E}[Q(\theta + \eta)] = \theta$.

#### 3.4.2 Dual-Stage Adaptive Quantization

**Downlink (Server → Client): Cosine Annealing Bit-Length Scheduler**

The bit-length for the global model broadcast follows a cosine annealing schedule:

$$b^{(t)}_{\text{down}} = b_{\min} + \frac{1}{2}(b_{\max} - b_{\min})\left(1 + \cos\left(\frac{\pi \cdot t}{T}\right)\right)$$

where $b_{\max} = 16$ bits initially and $b_{\min} = 4$ bits at convergence. High precision early in training when the loss landscape is steep; aggressive compression as the model stabilizes.

**Uplink (Client → Server): Shannon Entropy Client Importance Weighting**

Each hospital's transmission precision is determined by its dataset's informational diversity:

$$\nu_i = \lambda_h \cdot \frac{H(D_i)}{H_{\max}} + (1 - \lambda_h) \cdot \frac{|D_i|}{N_{\max}}$$

where:
$$H(D_i) = -\sum_{k=1}^{K} p_k \log_2(p_k), \quad H_{\max} = \log_2(K)$$

$p_k$ is the proportion of samples in clinical class $k$, $|D_i|$ is dataset size, $N_{\max}$ is maximum size across clients, and $\lambda_h \in [0,1]$ is a balancing hyperparameter.

| Hospital Type | Shannon Entropy Profile | Assigned Precision |
| :--- | :--- | :--- |
| Metro Tertiary (diverse, large) | High $H(D_i)$ | INT8 |
| Community Clinic (common conditions) | Medium $H(D_i)$ | INT6 |
| Research Hospital (oncology + cardiology) | Medium-High $H(D_i)$ | INT8 |
| Rural PHC (specialized, small) | Low $H(D_i)$ | INT4 |
| Specialty Clinic (single domain) | Very Low $H(D_i)$ | INT2–4 |

Low-entropy clients (specialized clinics) transmit aggressive compression without sacrificing global accuracy — their narrow contributions are not informationally critical at high precision.

**Implementation:**

```python
class AdaptiveQuantDP:
    def __init__(self, initial_bits=8, epsilon=1.0):
        self.global_scheduler = CosineAnnealingScheduler(initial_bits, min_bits=2)
        self.epsilon = epsilon
    
    def compress_and_protect(self, shared_params, client_entropy, round_num):
        # 1. Global bit-length from cosine schedule
        global_bits = self.global_scheduler.get_bits(round_num)
        
        # 2. Client-specific adjustment based on dataset entropy
        client_bits = max(2, global_bits - entropy_penalty(client_entropy))
        
        # 3. Quantize
        quantized = adaptive_quantize(shared_params, client_bits)
        
        # 4. Add Laplacian noise (tighter bounds than Gaussian for bounded updates)
        sensitivity = compute_sensitivity(quantized, client_bits)
        noise = np.random.laplace(0, sensitivity / self.epsilon, quantized.shape)
        
        return quantized + noise
```

**Combined effect:** FedRand reduces upstream payload by ~50%. Adaptive quantization reduces it by a further ~45–52%. Total communication reduction exceeds **75% vs. standard 32-bit FedAvg**, validated on federations of up to 1,000 clients.

---

### Stage 5 — Global Aggregation and Broadcast

**Server actions:**
1. De-quantizes incoming payloads from all hospitals.
2. Aggregates the fragmented, noised, entropy-weighted LoRA updates via weighted FedAvg into a coherent global model update.
3. Broadcasts the updated global model back to all hospitals.
4. Increments the RDP privacy accountant (see `03_Privacy_and_Security_Theory.md`).
5. Checks privacy budget ceiling; terminates training if $\epsilon_{\text{total}} \geq \epsilon_{\text{max}}$.

**Shannon entropy weighting in aggregation:** The global aggregation uses entropy-weighted averaging, not uniform FedAvg:

$$\min_{\Delta A, \Delta B} F(\Delta A, \Delta B) = \sum_{i=1}^{N} \frac{\nu_i}{\sum_j \nu_j} F_i(\Delta A, \Delta B)$$

This formally increases $N_{\text{eff}} = \frac{(\sum_i \nu_i)^2}{\sum_i \nu_i^2}$ under non-IID conditions, accelerating convergence relative to uniform weighting. (Proof in `07_Publication_Roadmap.md` — Paper 3.)

---

## 4. NVFlare DXO Filter Pipeline

All FedNeMo algorithmic components are implemented as **Data Exchange Object (DXO) Filters** in NVIDIA FLARE — not as standalone scripts. This is the key architectural decision that makes FedNeMo a reusable production framework rather than a research prototype.

| Custom NVFlare Component | Execution Context | Operation | Output |
| :--- | :--- | :--- | :--- |
| **Lightning Client API** | Local Hospital Trainer | Extracts high-precision LoRA matrix differentials $\Delta A$, $\Delta B$ from NeMo optimizer | Dense `WEIGHT_DIFF` DXO |
| **FedRandFilter** | DXO Outbound Chain | Evaluates $z \sim \text{Bernoulli}(\rho)$; zeroes out either $\Delta A$ or $\Delta B$ based on $z$ | Fragmented Subparameter DXO |
| **LaplacianDPFilter** | DXO Outbound Chain | Clips gradients to norm $C$; injects $\eta \sim \text{Lap}(C/\epsilon)$ noise | Privacy-Preserved DXO |
| **AdaptiveQuantFilter** | DXO Outbound Chain | Computes $H(D_i)$; determines $\nu_i$; truncates to INT8/INT4/INT2 dynamically | Compressed, Noised DXO |
| **ModelController** | Central Server Inbound | De-quantizes; aggregates via entropy-weighted FedAvg; updates global model | Updated Global Nemotron State |
| **PrivacyAccountant** | Central Server — DXO Metadata | Tracks cumulative RDP $\hat{\epsilon}_{\text{total}}(\alpha)$; enforces budget ceiling | Per-round ε metrics + termination signal |

All custom filters inherit from `nvflare.apis.filter.Filter` and are injected into `task_result_filters` in the FLARE configuration chain. This modular architecture allows enterprise security teams to independently audit each privacy layer without touching the core training loop.

---

## 5. Catastrophic Forgetting Mitigation

A standard critique of domain-specific LLM fine-tuning is catastrophic forgetting — the model forgets general knowledge while learning clinical knowledge.

FedNeMo addresses this through three structural mechanisms:

1. **Intrinsic MoE Sparsity:** Nemotron-3's 128-expert architecture means clinical fine-tuning predominantly activates and modifies a subset of experts, leaving the majority of the network's foundational knowledge intact.

2. **Federated Averaging as Regularization:** Periodic aggregation of LoRA updates across hospitals with diverse data distributions acts as an implicit regularizer, preventing any single client's narrow specialization from dominating the global model.

3. **LoRA's Intrinsic Rank Constraint:** With $r = 32$ out of $d = 4096$, the hypothesis space of possible modifications is structurally bounded, preventing wholesale overwriting of pretrained representations.

**Evaluation protocol:** Benchmark simultaneously on clinical tasks (PubMedQA, MedQA, ICD-10 prediction) and foundational reasoning tasks (PIQA, ARC-Challenge) at each federated round. Retention of general reasoning while clinical accuracy improves is the empirical proof of forgetting resistance.

---

## 6. Research Foundation

| # | Paper | Core Idea | FedNeMo Module |
|---|---|---|---|
| 1 | Ardıç & Genç — Adaptive Quantization + DP | Dual-tier bit-length scheduling with Laplacian DP | AdaptiveQuantFilter + LaplacianDPFilter |
| 2 | Guo et al., TPAMI 2026 — Gradient Inversion Attacks | Taxonomy of OP-GIA, GEN-GIA, ANA-GIA | GIAShield attack simulation + defense pipeline |
| 3 | Xu et al. — FedPS | Federated preprocessing via data sketches | FedPrep Aggregator + Local Stats module |
| 4 | Park et al. 2025 — FedRand | Randomized LoRA A/B matrix selection per round | FedRandFilter (StochasticLoRA) |
| 5 | Ardıç & Genç — MTAE sample selection | Multi-task autoencoders for outlier filtering | DataQuality module (MTAE + OCSVM) |
| 6 | Zhang et al., KDD 2025 — HtFLlib | Benchmark for heterogeneous FL | Evaluation framework reference |

Full citations in `09_References.md`.

> **Note on FedRE (Yao et al., CVPR 2026):** Representation entanglement for model-heterogeneous FL was considered but is explicitly deprioritized for the hackathon build. It adds an entirely separate training loop and is not required for the core privacy + communication efficiency contribution. Include as future work in Paper 1. See `04_Critical_Analysis_and_Gaps.md` for rationale.
