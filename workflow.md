# FedNeMo: Algorithmic Workflow and Procedural Blueprint

This document specifies the step-by-step execution algorithm of the **FedNeMo** federated fine-tuning framework, designed for privacy-preserving, communication-efficient collaborative training of hybrid Mamba-Transformer models across hospital datasets. 

Analogous to a procedural recipe (like preparing a tea, where each step leads deterministically to the next), this workflow traces a data sample from its raw entry at a local hospital node, through preprocessing, filtering, local fine-tuning, parameter obfuscation, compression, transmission, global aggregation, and validation.

---

## 1. System Models and Roles

To execute the FedNeMo pipeline, the system utilizes the following specific models:

1. **Foundational Client/Server LLM (Production Target):** `NVIDIA Nemotron-3 Nano (30B-A3B)`. A hybrid Mamba-2/Transformer Mixture-of-Experts (MoE) model.
   - **Total parameters:** 31.6 Billion.
   - **Active parameters per token:** ~3.5 Billion (3.2B base + embeddings).
   - **Layers:** 29 total (23 interleaved Mamba-2 SSM layers + 6 self-attention layers).
   - **Experts:** 128 routed experts + 1 shared expert per MoE layer (activates 5–6 experts per token dynamically using Latent MoE compression).
   - **Context Window:** 262,144 (256K) tokens.
2. **Foundational Client/Server LLM (Demo/Simulation Target):** `Nemotron-Mini-4B`. A standard dense Transformer model (~4 Billion parameters) used for client-side VRAM resource feasibility on standard GPU setups during simulation.
3. **Data Quality Model (Local Client-side):** Multi-Task Autoencoder (MTAE) paired with a One-Class Support Vector Machine (OCSVM) containing a Radial Basis Function (RBF) kernel.
4. **Local Data Preprocessing Models (FedPS):** Count-Min Sketches, Karnin-Lang-Liberty (KLL) quantile sketches, and Federated Bayesian Linear Regression (BLR).

---

## 2. Input Data Types, Formats, and Schema

The input clinical data at each hospital node is divided into two distinct modalities: **Structured EHR Metadata** and **Unstructured Clinical Narratives**. 

### 2.1 Tabular Structured EHR Metadata (Input to Stage 2 - FedPS)
This data is stored locally as tabular records (e.g., CSV, SQL tables, or FHIR resources) containing numerical values, clinical codes, and demographics.

```json
{
  "patient_id": "PT-908127",
  "demographics": {
    "age": 58,
    "sex": "Female",
    "bmi": 28.4
  },
  "labs": {
    "blood_glucose": 142.0,    // Hospital A units: mg/dL
    "hba1c": 7.1,
    "creatinine": 0.9
  },
  "diagnoses": ["ICD10:E11.9", "ICD10:I10"], // Type 2 Diabetes, Essential Hypertension
  "medications": ["NDC:0173-0862-01"]       // Metformin 500mg
}
```

### 2.2 Unstructured Clinical Narratives (Input to Tokenization & Prompts)
Free-text discharge notes, nursing assessments, or physician observations.

```text
Patient is a 58-year-old female presenting for a follow-up on her type 2 diabetes. 
Reports compliance with Metformin but notes intermittent morning hyperglycemia. 
Vitals stable. Cardiac examination reveals normal S1/S2 without murmurs.
```

### 2.3 Canonically Standardized Prompt Template (Output of Preprocessing, Input to LLM)
FedNeMo merges both structured (globally harmonized via FedPS) and unstructured text into a standard template:

```
[PATIENT_RECORD]
Demographics: 58 years, Female, BMI: 0.62 (normalized)
Lab Values: blood_glucose: 7.88 mmol/L (standardized), hba1c: 7.1%, creatinine: 79.6 umol/L
ICD-10 Codes: E11.9, I10
Medications: Metformin HCl 500mg

Clinical Note:
Patient is a 58-year-old female presenting for a follow-up on her type 2 diabetes. 
Reports compliance with Metformin but notes intermittent morning hyperglycemia. 
Vitals stable. Cardiac examination reveals normal S1/S2 without murmurs.

Task: Extract clinical risk factors and generate a patient instruction summary.
[/PATIENT_RECORD]
```

---

## 3. Step-by-Step Execution Workflow

The FedNeMo workflow operates in five deterministic phases, followed by real-time validation checks.

```
+---------------------------------------------------------------------------------+
|                                 INIT ROUND t                                    |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
| Stage 1: Data Quality Check                                                     |
| (MTAE Extraction -> Loss Computation -> OCSVM Outlier Pruning)                  |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
| Stage 2: Preprocessing (FedPS)                                                  |
| (Local Sketches -> Global Statistics Aggregation -> Local Harmonization)       |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
| Stage 3: Local LoRA Training                                                     |
| (Targeting Attention & Mamba-2 SSM Layers with Fused CUDA Kernel Hooks)         |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
| Stage 4: Secure Parameter Exchange                                              |
| (Stochastic FedRand A/B Split -> Laplacian LDP -> Shannon Entropy Quantization) |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
| Stage 5: Global Server Aggregation                                              |
| (De-quantization -> Entropy-Weighted FedAvg -> RDP Accounting & Termination)    |
+---------------------------------------------------------------------------------+
```

---

### STAGE 1: Unsupervised Outlier Detection and Quality Assurance (Local Client-Side)

Before a dataset is tokenized or trained, each hospital node executes an unsupervised data sanitation loop using a **Multi-Task Autoencoder (MTAE)** to prevent data poisoning and isolate corrupted EHR records.

#### 1.1 Local Embedding-Space MTAE Architecture
Standard autoencoders reconstruct discrete text tokens poorly. FedNeMo extracts the dense hidden embeddings $h_i \in \mathbb{R}^{d_{\text{model}}}$ of the input text sequence from the frozen layer-0 projection of the Nemotron backbone. The MTAE then processes these embeddings:
- **Encoder Network ($f_{\text{enc}}$):** Compresses $h_i$ into a lower-dimensional latent representation $z_i \in \mathbb{R}^{d_z}$.
  $$z_i = \text{ReLU}(W_{\text{enc}} h_i + b_{\text{enc}})$$
- **Decoder Network ($f_{\text{dec}}$):** Reconstructs the original embedding from the latent vector.
  $$\hat{h}_i = W_{\text{dec}} z_i + b_{\text{dec}}$$
- **Classifier Network ($f_{\text{cls}}$):** Predicts the clinical diagnostic class from the latent vector.
  $$\hat{y}_i = \text{Softmax}(W_{\text{cls}} z_i + b_{\text{cls}})$$

#### 1.2 The Dual-Loss Objective
The MTAE minimizes the combined reconstruction error and categorical classification error:
$$\mathcal{L}_{\text{MTAE}} = \alpha \cdot \underbrace{\frac{1}{N}\sum_{i=1}^{N} \|h_i - \hat{h}_i\|^2}_{\text{Reconstruction Loss (MSE)}} + (1 - \alpha) \cdot \underbrace{\left(-\frac{1}{N}\sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})\right)}_{\text{Classification Loss (Cross-Entropy)}}$$
where $\alpha \in [0,1]$ is a task-balancing hyperparameter (default $\alpha = 0.5$).

#### 1.3 Local Autoencoder/Encoder Verification Procedure
The autoencoder verification is executed at the client node to identify anomalous samples. It flags two distinct classes of outliers based on their loss profile:
1. **Structural Anomalies:** Samples yielding a reconstruction loss $l_{\text{recon}}$ higher than a threshold $\tau_{\text{recon}}$ indicate structural corruptions (e.g., malformed OCR scans, garbled ASCII values, character encoding errors).
2. **Semantic/Poisoning Anomalies:** Samples yielding a classification loss $l_{\text{class}}$ higher than a threshold $\tau_{\text{class}}$ indicate semantic anomalies (e.g., mismatched diagnosis/treatment pairs, label noise, or active adversarial data poisoning).

#### 1.4 Global OCSVM Boundary Coordination
To avoid setting static thresholds manually at each hospital:
1. Each client hospital extracts its per-sample loss vectors $v_j = [l_{\text{recon},j},\ l_{\text{class},j}]^\top \in \mathbb{R}^2$ for its dataset.
2. Clients calculate local distribution statistics `{mean, variance, percentiles}` and send them to the central NVIDIA FLARE server (no raw data or embeddings are shared).
3. The server fits a global One-Class Support Vector Machine (OCSVM) using a Radial Basis Function (RBF) kernel:
   $$K(u, v) = \exp(-\gamma \|u - v\|^2)$$
   The boundary maximizes the margin separating normal data from anomalous regions.
4. The server broadcasts the trained global OCSVM boundary coefficients back to all clients.
5. The local client filters all samples, checking if the sample's loss pair $v_j$ is classified as an outlier:
   $$\text{Decision}(v_j) = \text{sign}\left(\sum_{i=1}^{M} \alpha_i K(v_i, v_j) - \rho\right)$$
   If $\text{Decision}(v_j) < 0$, the record is discarded from the training dataset.

#### 1.5 Local Sanitation Pseudocode
```python
def sanitize_local_dataset(raw_dataset, frozen_backbone, mtae_model, ocsvm_boundary):
    clean_dataset = []
    
    for record in raw_dataset:
        # Step 1: Extract embedding representation
        with torch.no_grad():
            h_i = frozen_backbone.get_embeddings(record.text)
            
        # Step 2: Forward pass through MTAE
        h_recon, y_pred, z_latent = mtae_model(h_i)
        
        # Step 3: Compute individual losses
        l_recon = torch.mean((h_i - h_recon) ** 2).item()
        l_class = cross_entropy(y_pred, record.true_label).item()
        
        # Step 4: Evaluate against global OCSVM boundary
        loss_pair = np.array([[l_recon, l_class]])
        is_normal = ocsvm_boundary.predict(loss_pair) # Returns 1 if normal, -1 if outlier
        
        if is_normal == 1:
            clean_dataset.append(record)
        else:
            logger.warning(f"Discarding poisoned/anomalous record: Recon loss={l_recon:.4f}, Class loss={l_class:.4f}")
            
    return clean_dataset
```

---

### STAGE 2: Federated Preprocessing via Aggregated Statistics (FedPS)

Once data is sanitized, the client nodes participate in a federated preprocessing phase to resolve data heterogeneity (Non-IID domain drift) without transmitting raw patient values.

#### 2.1 Preprocessing Workflow
1. **Local Sketching:** Each client processes its structured tabular fields to generate compact, cryptographic mathematical sketches:
   - Numerical values: Minimum ($x_{\min}$), maximum ($x_{\max}$), sum ($\sum x_i$), and sum of squares ($\sum x_i^2$).
   - Categorical codes (ICD-10, billing codes): Unique sets and Count-Min Sketches.
   - Target distributions: Karnin-Lang-Liberty (KLL) quantile summaries.
2. **Aggregation:** Sketches are sent to the central FLARE server. The server sums the moments and unions the categorical sets:
   $$\mu_{\text{global}} = \frac{\sum_k \sum x_{i,k}}{\sum_k N_k}, \quad \sigma^2_{\text{global}} = \frac{\sum_k \sum x_{i,k}^2}{\sum_k N_k} - \mu_{\text{global}}^2$$
3. **Broadcast & Transform:** The server derives global scaling factors, bin boundaries, and imputation parameters (via Federated Bayesian Linear Regression parameters $\hat{\beta}$) and broadcasts them. Clients apply Z-score scaling and discretization to their local records.

---

### STAGE 3: Local Model Configuration and PEFT Fine-Tuning

With clean, harmonized data, clients load the foundational model and configure Low-Rank Adaptation (LoRA).

#### 3.1 LoRA Target Modules
Unlike standard configurations that target only the self-attention projections, FedNeMo targets both Transformer and Mamba-2 SSM layers to ensure gradient flow across the 3.5B active parameters:
- **Transformer Layers:** `linear_qkv` (Fused Query-Key-Value projection), `linear_proj` (Attention output projection).
- **Mamba-2 Layers:** `in_proj` / `x_proj` (SSM input projections), `out_proj` (SSM output projection), `dt_proj` (SSM discretization projection).

#### 3.2 CUDA Fused Kernel Compatibility Hook
Because Mamba-2 uses highly optimized fused CUDA kernels (`selective_scan_cuda` and `causal_conv1d_cuda`), standard Python-level weight patching is bypassed, resulting in zero weight updates on SSM layers. FedNeMo resolves this using a PyTorch hook-based weight materialization script:

```python
def mamba_lora_pre_hook(module, input):
    """Intercepts input before fused CUDA kernel and adds LoRA perturbation to base weight."""
    if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
        # Materialize LoRA delta in-place: W_modified = W_0 + (B * A) * scaling
        delta_w = module.lora_B.weight @ module.lora_A.weight
        module.weight.data.add_(module.scaling * delta_w)
        module._lora_applied = True

def mamba_lora_post_hook(module, input, output):
    """Subtracts the materialized perturbation after CUDA execution to preserve gradient calculation."""
    if getattr(module, '_lora_applied', False):
        delta_w = module.lora_B.weight @ module.lora_A.weight
        module.weight.data.sub_(module.scaling * delta_w)
        module._lora_applied = False
```

Clients register these hooks during trainer initialization. An automated validation pass executes one dummy step to verify that the gradient norms of Mamba-2 LoRA variables are non-zero:
$$\|\nabla_{\theta_{\text{lora}}} \mathcal{L}\|_2 > 10^{-10}$$

---

### STAGE 4: Secure Parameter Exchange

At the end of each local epoch, the computed LoRA weight updates are prepared for network transmission through the NVFlare DXO filter pipeline.

#### 4.1 Step 4a: Randomized LoRA Subparameter Partitioning (FedRand)
To prevent Gradient Inversion Attacks (which require paired access to both LoRA matrices $A$ and $B$ to optimize input reconstruction), the client fragments the parameter updates:
1. Draw a binary choice variable $z \sim \text{Bernoulli}(0.5)$.
2. **If $z=1$ (Share Matrix A):**
   - Transmit updated down-projection update: $\Delta A_i^{(t)} = A_{i, \text{local}}^{(t)} - A_{\text{global}}^{(t)}$.
   - Keep up-projection update $B_{i, \text{local}}^{(t)}$ stored privately in local memory.
3. **If $z=0$ (Share Matrix B):**
   - Transmit updated up-projection update: $\Delta B_i^{(t)} = B_{i, \text{local}}^{(t)} - B_{\text{global}}^{(t)}$.
   - Keep down-projection update $A_{i, \text{local}}^{(t)}$ stored privately in local memory.

This eliminates the cross-matrix noise amplification effect. The Frobenius norm of the remaining noise grows as $O(1/\epsilon^2)$ rather than the standard DP-LoRA complexity of $O(1/\epsilon^4)$, allowing tighter privacy guarantees.

#### 4.2 Step 4b: Local Differential Privacy & Adaptive Quantization
Before sending the chosen parameter update matrix $\Delta \Theta \in \{\Delta A_i^{(t)}, \Delta B_i^{(t)}\}$, the client executes the following steps:
1. **LDP Noise Injection:** The selected matrix updates are clipped to a norm bound $C$ to restrict sensitivity, and Laplacian noise is added to satisfy $(\epsilon, 0)$-DP:
   $$\Delta \Theta^* = \Delta \Theta + \text{Lap}\left(0, \frac{C}{\epsilon}\right)$$
2. **Shannon Entropy Importance Quantification:** The client computes its statistical importance score $\nu_i$ based on the normalized Shannon Entropy of its class labels:
   $$\nu_i = \lambda_h \cdot \frac{H(D_i)}{H_{\max}} + (1 - \lambda_h) \cdot \frac{|D_i|}{N_{\max}}$$
   where:
   $$H(D_i) = -\sum_{k=1}^{K} p_k \log_2(p_k), \quad H_{\max} = \log_2(K)$$
3. **Stochastic Quantization:** 
   - High-entropy clients (dense clinical diversity) encode updates to INT8 precision.
   - Low-entropy clients (homogeneous diagnostics) quantize aggressively down to INT4 or INT2 precision to reduce upstream bandwidth.

---

### STAGE 5: Global Server Aggregation and Distribution

The central NVIDIA FLARE server receives the processed payloads from the clients.

#### 5.1 Step 5a: Aggregation
The server de-quantizes the incoming parameter updates and aggregates them using entropy-weighted averaging:
$$\Delta \Theta_{\text{global}}^{(t)} = \sum_{i=1}^{M} \frac{\nu_i}{\sum_j \nu_j} \Delta \Theta_i^{*(t)}$$
The aggregated update is applied to the global model checkpoint.

#### 5.2 Step 5b: Rényi Differential Privacy Accounting
The server updates the cumulative privacy accountant to track the total privacy budget spent:
1. For each Laplace step of scale parameter $b$, calculate RDP at order $\alpha$:
   $$\hat{\epsilon}_{\text{Lap}}(\alpha) = \frac{1}{\alpha - 1} \ln\left(\frac{\alpha}{2\alpha - 1} e^{(\alpha-1)/b} + \frac{\alpha - 1}{2\alpha - 1} e^{-\alpha/b}\right)$$
2. Sum RDP values across all rounds:
   $$\hat{\epsilon}_{\text{total}}(\alpha) = \sum_{t=1}^{T} \hat{\epsilon}_t(\alpha)$$
3. Convert the accumulated RDP parameter to standard $(\epsilon_{\text{total}}, \delta)$-DP for target failure probability $\delta = 10^{-5}$:
   $$\epsilon_{\text{total}} = \min_{\alpha > 1} \left\{ \hat{\epsilon}_{\text{total}}(\alpha) + \frac{\ln(1/\delta)}{\alpha - 1} \right\}$$
4. **Termination Check:** If $\epsilon_{\text{total}} \geq \epsilon_{\text{max}}$, training halts, and the model is checkpointed. Otherwise, the global updates are broadcast down to clients using the Cosine Annealing bit-length schedule to begin round $t+1$.

---

## 4. Adversarial Attack Simulation & Verification Strategy

To verify the mathematical protection against Gradient Inversion Attacks, the orchestrator runs a parallel verification step:

```
                  [ Transmitted LoRA Update ]
                               |
              +----------------+----------------+
              |                                 |
              v                                 v
      [ UNPROTECTED PATH ]               [ PROTECTED PATH ]
 (Raw A and B transmitted)          (FedRand + DP + Quantization)
              |                                 |
              v                                 v
  [ Optimization-Based GIA ]        [ Optimization-Based GIA ]
    - Cosine distance matching        - Cosine distance matching
    - Dummy sequence iterations       - Dummy sequence iterations
              |                                 |
              v                                 v
 [ Reconstructed Clinical Text ]       [ Indecipherable Noise ]
   (Tokens successfully matched)         (Optimizer fails to converge)
```

1. **Unprotected Simulation:** The orchestrator runs a simulated training round with the FedRand and DP filters disabled. A simulated malicious server intercepts both $A$ and $B$ update matrices. It executes an optimization-based gradient matching algorithm (such as DLG or InvertingGrad) to reconstruct the clinical prompt tokens. The reconstructed text is displayed on the telemetry dashboard, verifying that unprotected federated setups leak patient records.
2. **Protected Simulation:** The orchestrator runs the identical training round with FedNeMo's filters enabled. The malicious server intercepts only one of the matrices ($\Delta A$ or $\Delta B$) perturbed with Laplacian noise and quantized. The reconstruction optimization loop is run again. The optimizer fails to minimize the distance metric, yielding only random characters and confirming the privacy guarantees.
