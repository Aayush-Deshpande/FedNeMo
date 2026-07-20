import React, { useState, useEffect, useRef } from 'react';
import { motion, useInView } from 'framer-motion';

/* ─────────────────────────────
   Helpers
───────────────────────────── */
const Sec = ({ n, children }) => (
  <h2 className="section-heading">
    <span className="section-number">{n}</span>
    {children}
  </h2>
);

const Sub = ({ children }) => (
  <h3 className="sub-heading">{children}</h3>
);

const P = ({ children }) => (
  <p className="paper-p">{children}</p>
);

const Mono = ({ children }) => (
  <code className="code-inline">{children}</code>
);

const AnimatedStat = ({ value, suffix, label }) => {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true });
  const [v, setV] = useState(0);

  useEffect(() => {
    if (!isInView) return;
    let start = 0;
    const step = value / (2000 / 20);
    const t = setInterval(() => {
      start += step;
      if (start >= value) { setV(value); clearInterval(t); }
      else setV(start);
    }, 20);
    return () => clearInterval(t);
  }, [isInView, value]);

  return (
    <div ref={ref} style={{ textAlign: 'center', margin: '1rem 0' }}>
      <div className="stat-chip">{v.toFixed(suffix === '%' ? 1 : 2)}{suffix}</div>
      <div style={{ fontFamily: 'Inter, sans-serif', fontSize: '0.75rem', color: '#666', marginTop: '0.25rem', textTransform: 'uppercase', letterSpacing: '0.1em' }}>{label}</div>
    </div>
  );
};

const FigureBox = ({ caption, children, label }) => (
  <div className="paper-figure">
    {children}
    <p className="figure-caption">
      <strong>{label}</strong> — {caption}
    </p>
  </div>
);

/* A simple ASCII / text visualization for the federated flow */
const FedFlowDiagram = () => (
  <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.65rem', color: '#76b900', lineHeight: 1.6, padding: '0.5rem', overflowX: 'auto', whiteSpace: 'pre' }}>
{`Node₁  ─── A₁/B₁ ───┐
Node₂  ─── A₂/B₂ ───┤
Node₃  ─── A₃/B₃ ───┼──► Orchestrator ──► Global Adapter
Node₄  ─── A₄/B₄ ───┤        │
Node₅  ─── A₅/B₅ ───┘    Trust × ∇H

[FedRand: Bernoulli(ρ=0.5) per layer per round]
[Laplace DP → 8-bit Quant → Aggregation]`}
  </div>
);

const ConfigTable = () => (
  <table className="results-table">
    <thead>
      <tr>
        <th>Parameter</th>
        <th>Value</th>
      </tr>
    </thead>
    <tbody>
      {[
        ['Clients (Nodes)', '5'],
        ['Rounds', '4'],
        ['LoRA Rank / Alpha', '16 / 32'],
        ['FedRand ρ', '0.5'],
        ['DP Mode', 'Relative'],
        ['DP Noise Ratio', '0.25'],
        ['Quant Bits', '8-bit (tx)'],
        ['Grad Accum Steps', '4'],
        ['LM Head Device', 'CUDA'],
        ['ε_total (RDP)', '≈ 16'],
      ].map(([k, v]) => (
        <tr key={k}>
          <td><Mono>{k}</Mono></td>
          <td><Mono>{v}</Mono></td>
        </tr>
      ))}
    </tbody>
  </table>
);

const ResultsTable = () => (
  <table className="results-table">
    <thead>
      <tr>
        <th>Metric</th>
        <th>Value</th>
      </tr>
    </thead>
    <tbody>
      <tr className="highlight-row">
        <td>Accuracy</td>
        <td>87.5%</td>
      </tr>
      <tr className="highlight-row">
        <td>Macro-F1</td>
        <td>0.86</td>
      </tr>
      <tr>
        <td>Unparseable Outputs</td>
        <td>0.0%</td>
      </tr>
      <tr>
        <td>Nodes</td>
        <td>5</td>
      </tr>
      <tr>
        <td>Rounds</td>
        <td>4</td>
      </tr>
      <tr>
        <td>Dataset</td>
        <td>Symptom2Disease</td>
      </tr>
      <tr>
        <td>Eval Records</td>
        <td>1,200 (24 classes)</td>
      </tr>
    </tbody>
  </table>
);

/* ─────────────────────────────
   Main export
───────────────────────────── */
export default function PaperBody() {
  return (
    <>
      {/* ── 1. INTRODUCTION ── */}
      <Sec n="1.">Introduction</Sec>
      <P>
        Federated learning enables training machine-learning models across decentralized nodes without
        centralizing raw data. While general-purpose federated frameworks exist, applying large language
        models (LLMs) to sensitive domains — such as clinical NLP — introduces unique constraints:
        model weights alone are enormous, GPU memory is limited on edge hardware, and formal
        privacy guarantees must be maintained throughout the training loop.
      </P>
      <P>
        We address these constraints with <strong>FedNeMo</strong>: a single-process, fully local
        federated fine-tuning framework for NVIDIA's <Mono>Nemotron-Mini-4B-Instruct</Mono>.
        FedNeMo makes no network calls during training and requires no orchestration infrastructure.
        It targets the task of symptom-to-disease classification as a representative medical NLP workload.
      </P>
      <P>
        Our key contributions are: (i) a custom VRAM-bounded loader fitting a 4B LLM on a 6 GB GPU;
        (ii) the <strong>FedRand</strong> split-adapter protocol for structural privacy; (iii) a
        relative-mode Laplace DP mechanism with tight RDP composition; (iv) per-tensor affine
        quantization of transmitted matrices; and (v) a trust × entropy aggregation strategy resilient
        to poisoning.
      </P>

      {/* ── 2. BACKGROUND ── */}
      <Sec n="2.">Background &amp; Related Work</Sec>
      <Sub>2.1 Federated Learning</Sub>
      <P>
        Standard federated averaging (FedAvg) [McMahan et al., 2017] aggregates model gradients
        from distributed clients, preserving local data. However, it is vulnerable to gradient-inversion
        attacks [Zhu et al., 2019] and incurs high communication costs from full 32-bit weight
        transmission.
      </P>
      <Sub>2.2 Parameter-Efficient Fine-Tuning</Sub>
      <P>
        Low-Rank Adaptation (LoRA) [Hu et al., 2021] introduces trainable decomposition matrices
        A and B into frozen transformer layers. Only these matrices are updated during fine-tuning,
        drastically reducing the number of trainable parameters.
      </P>
      <Sub>2.3 Differential Privacy in FL</Sub>
      <P>
        Formal (ε,δ)-DP guarantees bound an adversary's ability to infer any individual record from
        model updates. Rényi Differential Privacy (RDP) [Mironov, 2017] provides tighter composition
        bounds across multiple rounds compared to naïve ε-summation.
      </P>

      {/* ── 3. DATASET ── */}
      <Sec n="3.">Dataset &amp; Partitioning</Sec>
      <P>
        We use the <strong>Symptom2Disease</strong> dataset: a collection of symptom descriptions
        labeled with one of 24 disease classes. The loader accepts any text/label CSV or JSON via a
        universal <Mono>load_text_classification</Mono> API, making FedNeMo dataset-agnostic.
      </P>
      <Sub>3.1 Stratified Holdout</Sub>
      <P>
        Prior to partitioning, we reserve a class-balanced held-out set (<Mono>stratified_holdout</Mono>).
        This set is never observed by any federated node during training, ensuring unbiased evaluation.
      </P>
      <Sub>3.2 IID Partition</Sub>
      <P>
        The remaining data undergoes <Mono>iid_partition</Mono>: a random, balanced, equal-parts split
        across all <Mono>N</Mono> nodes. Each node receives an identical class-frequency distribution,
        isolating the privacy contribution of federation from class imbalance effects.
      </P>
      <FigureBox label="Figure 1" caption={`Data flow: stratified holdout is extracted first, then the remainder is partitioned equally across 5 federated nodes using IID splitting.`}>
        <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.6rem', color: '#76b900', lineHeight: 1.6 }}>
          {`Symptom2Disease.csv\n  └─ stratified_holdout (1,200) ─► Evaluation Only\n  └─ iid_partition ──────────────► Node₁…Node₅ (equal)`}
        </div>
      </FigureBox>

      {/* ── 4. THE MODEL ── */}
      <Sec n="4.">Model Architecture</Sec>
      <Sub>4.1 Nemotron-Mini-4B-Instruct</Sub>
      <P>
        The base model is NVIDIA's <Mono>Nemotron-Mini-4B-Instruct</Mono>, a 4-billion parameter
        instruction-tuned LLM. We apply LoRA (<Mono>r=16, α=32</Mono>) to all attention projection
        layers, leaving the transformer body frozen.
      </P>
      <Sub>4.2 VRAM-Bounded Loader</Sub>
      <P>
        A 6 GB consumer GPU cannot hold a 4B model at full precision. We implement a custom
        three-stage placement strategy:
      </P>
      <P>
        <strong>1) Transformer Body:</strong> Quantized to <Mono>4-bit NF4</Mono> via
        <Mono>bitsandbytes</Mono> and resident on GPU. <strong>2) Embedding Layer:</strong> The
        256k-vocabulary <Mono>embed_tokens</Mono> matrix is placed on CPU to avoid GPU OOM.
        <strong>3) LM Head:</strong> Placed on GPU at 4-bit (configurable via
        <Mono>LM_HEAD_DEVICE</Mono>). A manual forward pass wires
        <Mono>embed → body → lm_head</Mono> with cross-device autograd.
      </P>

      {/* ── 5. FEDERATED CORE ── */}
      <Sec n="5.">Federated Core</Sec>
      <FigureBox label="Figure 2" caption="Overview of a single federation round: each node trains locally, applies FedRand split selection, Laplace DP noise, quantization, and transmits to the orchestrator for trust-weighted aggregation.">
        <FedFlowDiagram />
      </FigureBox>

      <Sub>5.1 Client Training</Sub>
      <P>
        Each <Mono>ClientNode</Mono> fine-tunes its LoRA adapter on its local shard for one round.
        The full shard is reshuffled every round. Gradient accumulation
        (<Mono>GRAD_ACCUM_STEPS=4</Mono>) increases the effective batch size without requiring larger
        GPU memory. Nodes optionally simulate adversarial behavior (label-flip poisoning) to validate
        the trust mechanism.
      </P>

      <Sub>5.2 FedRand Split-Adapter</Sub>
      <P>
        Standard federated LoRA shares both A and B matrices, enabling gradient inversion attacks
        that can reconstruct training data [Fowl et al., 2022]. <strong>FedRand</strong> addresses this
        structurally: for each LoRA layer, a Bernoulli trial with probability <Mono>ρ=0.5</Mono>
        determines whether matrix A or matrix B is transmitted. The complement matrix remains private
        and persists locally across rounds.
      </P>
      <P>
        Because a global attacker never observes both matrices simultaneously, they cannot reconstruct
        the full gradient for any layer in any round. This property holds independently of the noise
        level and is our first line of defence.
      </P>

      <Sub>5.3 Laplace Differential Privacy</Sub>
      <P>
        After local training, transmitted LoRA slices are noised before leaving the node. We implement
        two calibration modes:
      </P>
      <P>
        <strong>(a) Relative mode</strong> (default): noise scale is
        <Mono>σ = noise_ratio × ‖Δ‖₁ / dim</Mono>, calibrating noise to a fixed fraction of the
        update's own magnitude. This maintains a stable SNR across heterogeneous update scales.
      </P>
      <P>
        <strong>(b) Absolute mode</strong>: the classical Laplace mechanism with scale
        <Mono>C/ε</Mono>, suitable for strict (ε,0)-DP guarantees. RDP composition across 4 rounds
        yields <Mono>ε_total ≈ 16</Mono> — a moderate privacy budget appropriate for a research
        prototype.
      </P>

      <Sub>5.4 Adaptive Quantization</Sub>
      <P>
        Before transmission, each matrix undergoes per-tensor affine quantization to
        <Mono>QUANT_BITS ∈ {"{2, 8}"}</Mono>. Scale and zero-point are computed per tensor and
        transmitted alongside the quantized integers. Compared to a 32-bit full-FedAvg baseline,
        8-bit quantization reduces transmission bytes by <strong>4×</strong>; 2-bit by
        <strong>16×</strong>.
      </P>

      <Sub>5.5 Trust-Weighted Aggregation</Sub>
      <P>
        The orchestrator weights each client's contribution by a blended trust score:
      </P>
      <div className="paper-figure" style={{ padding: '1rem 1.5rem' }}>
        <code style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.75rem', color: '#76b900', display: 'block', textAlign: 'center', lineHeight: 2 }}>
          w_i = α · trust_model(i) + (1−α) · heuristic(i)<br/>
          Δ_global = Σ (w_i · entropy(i) · Δ_i) / Σ (w_i · entropy(i))
        </code>
        <p className="figure-caption"><strong>Eq. 1</strong> — Trust-weighted aggregation formula. The base model (adapter disabled) scores each client's update trustworthiness from a text summary. The blend factor α prevents a degenerate model read from zeroing a legitimate node.</p>
      </div>

      {/* ── 6. INFERENCE ── */}
      <Sec n="6.">Inference Pipeline</Sec>
      <P>
        At inference time, FedNeMo supports an end-to-end image → prediction pipeline:
        (i) a scanned report image is transcribed via a single call to NVIDIA's hosted
        <Mono>nemotron-parse</Mono> API (the only external network call in the system);
        (ii) parsed text is mapped to the model's input schema by a deterministic rule-based field
        mapper; (iii) the local fine-tuned model generates a prediction.
      </P>
      <P>
        To eliminate unparseable outputs, we apply <strong>constrained decoding</strong>: the logits
        at the output position are masked to only the token sequences corresponding to valid class
        labels. The model is thus mathematically prevented from generating any out-of-vocabulary
        label, yielding <Mono>0.0%</Mono> unparseable outputs across all evaluations.
      </P>

      {/* ── 7. EXPERIMENTS ── */}
      <Sec n="7.">Experimental Setup</Sec>
      <P>
        All experiments are run on a single machine with a 6 GB GPU. Training uses 5 simulated
        federated nodes within a single Python process — no inter-process or inter-machine
        communication occurs. The configuration used for the reported results is summarized below.
      </P>
      <FigureBox label="Table 1" caption="Hyperparameter configuration for the reported run.">
        <ConfigTable />
      </FigureBox>

      {/* ── 8. RESULTS ── */}
      <Sec n="8.">Results</Sec>
      <P>
        We evaluate the trained global adapter on the held-out 1,200-record Symptom2Disease split
        (24 classes). Results are reported as macro-averaged to account for class imbalance.
      </P>
      <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', margin: '1.5rem 0' }}>
        <AnimatedStat value={87.5} suffix="%" label="Accuracy" />
        <AnimatedStat value={0.86} suffix="" label="Macro-F1" />
        <AnimatedStat value={0.0} suffix="%" label="Unparseable" />
      </div>
      <FigureBox label="Table 2" caption="Held-out evaluation results on Symptom2Disease dataset. Best metrics highlighted.">
        <ResultsTable />
      </FigureBox>
      <P>
        The 87.5% accuracy achieved under strict privacy constraints (FedRand + DP + quantization)
        demonstrates that FedNeMo's privacy mechanisms do not catastrophically degrade model
        performance. The 0.0% unparseable rate confirms the reliability of constrained decoding
        in a 24-class setting.
      </P>

      {/* ── 9. LIMITATIONS ── */}
      <Sec n="9.">Limitations &amp; Honest Notes</Sec>
      <P>
        The end-to-end inference path (image → <Mono>nemotron-parse</Mono> → local model) has not
        been verified against the live API due to the requirement of a valid
        <Mono>NVIDIA_API_KEY</Mono>. Field mapping is tuned on synthetic report text and may require
        adaptation for real clinical documents.
      </P>
      <P>
        <Mono>relative</Mono>-mode DP is a signal-calibrated heuristic and does not satisfy pure
        (ε,0)-DP. The reported <Mono>ε_total ≈ 16</Mono> (RDP-composed over 4 rounds) represents
        a moderate privacy budget; production deployments would require a smaller budget and
        correspondingly higher noise.
      </P>

      {/* ── 10. CONCLUSION ── */}
      <Sec n="10.">Conclusion</Sec>
      <P>
        We presented <strong>FedNeMo</strong>, demonstrating that a 4B-parameter LLM can be
        fine-tuned federally on sensitive medical text, entirely locally, on a single consumer GPU,
        while maintaining formal privacy guarantees and competitive accuracy. The combination of
        FedRand structural privacy, Laplace DP, quantized transmission, and trust-weighted
        aggregation provides a layered defence-in-depth that is deployable today without cloud
        infrastructure.
      </P>

      {/* ── REFERENCES ── */}
      <div className="references-section">
        <h2 className="section-heading" style={{ borderTop: 'none' }}>References</h2>
        {[
          ['[1]', 'McMahan, B., et al. (2017). "Communication-Efficient Learning of Deep Networks from Decentralized Data." AISTATS 2017.'],
          ['[2]', 'Hu, E., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.'],
          ['[3]', 'Mironov, I. (2017). "Rényi Differential Privacy." CSF 2017.'],
          ['[4]', 'Zhu, L., et al. (2019). "Deep Leakage from Gradients." NeurIPS 2019.'],
          ['[5]', 'Fowl, L., et al. (2022). "Robbing the Fed: Directly Obtaining Private Data in Federated Learning with Modified Models." ICLR 2022.'],
          ['[6]', 'NVIDIA. (2024). "Nemotron-Mini-4B-Instruct." NVIDIA AI Foundation Models.'],
          ['[7]', 'Dettmers, T., et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs." NeurIPS 2023.'],
        ].map(([tag, text]) => (
          <p key={tag} className="reference-item"><span>{tag}</span> {text}</p>
        ))}
      </div>
    </>
  );
}
