import React, { useRef, useState, useEffect } from 'react';
import { motion, useInView, useScroll, useSpring } from 'framer-motion';

/* ─── UTILS ─── */
const FadeUp = ({ children, delay = 0, className = '' }) => (
  <motion.div
    className={className}
    initial={{ opacity: 0, y: 32, filter: 'blur(6px)' }}
    whileInView={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
    viewport={{ once: true, margin: '-80px' }}
    transition={{ duration: 0.9, delay, ease: [0.16, 1, 0.3, 1] }}
  >
    {children}
  </motion.div>
);

const FadeIn = ({ children, delay = 0 }) => (
  <motion.div
    initial={{ opacity: 0, filter: 'blur(8px)' }}
    whileInView={{ opacity: 1, filter: 'blur(0px)' }}
    viewport={{ once: true, margin: '-60px' }}
    transition={{ duration: 1, delay, ease: 'easeOut' }}
  >
    {children}
  </motion.div>
);

const AnimCounter = ({ target, suffix = '', decimals = 1, delay = 0 }) => {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true });
  const [val, setVal] = useState(0);
  useEffect(() => {
    if (!inView) return;
    let s = 0;
    const step = target / (2000 / 16);
    const t = setInterval(() => {
      s += step;
      if (s >= target) { setVal(target); clearInterval(t); }
      else setVal(s);
    }, 16);
    return () => clearInterval(t);
  }, [inView, target]);
  return (
    <span ref={ref}>
      {val.toFixed(decimals)}{suffix}
    </span>
  );
};

/* ─── NAV ─── */
export function NavPill() {
  const sections = ['abstract','dataset','model','federated','results','conclusion'];
  return (
    <motion.nav className="nav-pill"
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 1.2, duration: 0.8 }}
    >
      {sections.map(s => (
        <a key={s} href={`#${s}`}>{s}</a>
      ))}
    </motion.nav>
  );
}

/* ─── HERO ─── */
export function HeroSection() {
  return (
    <section className="sec sec-center" id="hero" style={{ minHeight: '100vh', gap: '1.5rem' }}>
      <motion.div
        initial={{ opacity: 0, scale: 0.9, filter: 'blur(12px)' }}
        animate={{ opacity: 1, scale: 1, filter: 'blur(0px)' }}
        transition={{ duration: 1.5, ease: [0.16, 1, 0.3, 1] }}
        style={{ textAlign: 'center' }}
      >
        <div className="label" style={{ justifyContent: 'center', marginBottom: '2rem' }}>
          Privacy-Preserving Federated LLM Fine-Tuning
        </div>
        <h1 className="display" style={{ marginBottom: '1.5rem' }}>
          <span className="green">FedNeMo</span>
        </h1>
        <p style={{ fontSize: '1.5rem', fontWeight: 300, color: '#ccc', maxWidth: '700px', margin: '0 auto 2rem', lineHeight: 1.5 }}>
          Local, in-process federated fine-tuning of Nemotron-Mini-4B for medical text classification. No servers. No cloud. 100% private.
        </p>
        <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', flexWrap: 'wrap' }}>
          {['87.5% Accuracy', '5 Nodes', '4 Rounds', 'Laplace DP', 'FedRand', '6GB GPU'].map(t => (
            <span key={t} className="chip" style={{ fontSize: '0.8rem' }}>{t}</span>
          ))}
        </div>
      </motion.div>

      <motion.div
        animate={{ y: [0, 12, 0] }}
        transition={{ repeat: Infinity, duration: 2.5, ease: 'easeInOut' }}
        style={{ position: 'absolute', bottom: '2.5rem', color: '#333', fontSize: '1.5rem' }}
      >↓</motion.div>
    </section>
  );
}

/* ─── ABSTRACT ─── */
export function AbstractSection() {
  return (
    <section className="sec" id="abstract" style={{ background: 'rgba(4,4,4,0.6)' }}>
      <FadeUp>
        <div className="label">Abstract</div>
        <h2 className="heading" style={{ marginBottom: '2rem' }}>The Research</h2>
      </FadeUp>
      <FadeUp delay={0.2}>
        <p className="abstract-text">
          We present <strong>FedNeMo</strong>, a fully local, in-process federated fine-tuning framework for NVIDIA's{' '}
          <code style={{ fontFamily: 'var(--mono)', color: '#76b900', fontSize: '0.9em' }}>Nemotron-Mini-4B-Instruct</code> model,
          applied to the task of medical text classification (symptom description → disease). FedNeMo operates without any
          servers, cloud orchestration, or outbound training traffic—protecting sensitive patient data entirely within
          the local execution environment. We introduce a comprehensive privacy-preserving stack comprising:{' '}
          <strong>(i) FedRand</strong> probabilistic split-adapter sharing to structurally defeat gradient inversion attacks;{' '}
          <strong>(ii) Laplace differential privacy</strong> calibrated relative to update magnitude;{' '}
          <strong>(iii) adaptive per-tensor quantization</strong> at 2-bit or 8-bit precision; and{' '}
          <strong>(iv) trust × entropy-weighted aggregation</strong> to deter poisoning. Our system achieves{' '}
          <strong>87.5% accuracy</strong> and <strong>0.86 Macro-F1</strong> on the held-out Symptom2Disease benchmark
          (1,200 records, 24 classes) with 0.0% unparseable outputs.
        </p>
      </FadeUp>
      <FadeUp delay={0.4} style={{ marginTop: '2rem' }}>
        <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', marginTop: '2rem' }}>
          {['Federated Learning','Differential Privacy','Quantization','LoRA','Medical NLP','NVIDIA Nemotron','FedRand','Trust Aggregation'].map(k => (
            <span key={k} className="chip">{k}</span>
          ))}
        </div>
      </FadeUp>
    </section>
  );
}

/* ─── PROBLEM ─── */
export function ProblemSection() {
  return (
    <section className="sec" id="problem">
      <div style={{ maxWidth: '1100px', width: '100%' }}>
        <FadeUp>
          <div className="label">§ 1 — Introduction</div>
          <h2 className="heading">Why Federated?</h2>
        </FadeUp>
        <div className="grid-2" style={{ marginTop: '3rem', alignItems: 'start' }}>
          <FadeUp delay={0.1}>
            <p className="body-text">
              When fine-tuning large language models on sensitive clinical data—like mapping symptom descriptions
              to diagnoses—centralizing that data in a cloud server is a non-starter. Patient records are legally
              and ethically protected. Standard centralized fine-tuning pipelines violate this boundary.
            </p>
            <p className="body-text" style={{ marginTop: '1rem' }}>
              <strong>Federated learning</strong> addresses this by training models across distributed nodes
              without moving raw data. But standard FedAvg [McMahan et al., 2017] still exposes gradient updates
              to inversion attacks, incurs massive communication costs, and assumes unlimited GPU resources.
            </p>
          </FadeUp>
          <FadeUp delay={0.25}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              {[
                { n: '01', t: 'No Data Centralization', d: 'Training data never leaves the local node. No cloud API call occurs during training.' },
                { n: '02', t: 'Structural Privacy', d: 'FedRand prevents gradient inversion by design—not just by adding noise.' },
                { n: '03', t: 'Edge Hardware', d: 'Entire 4B-parameter model pipeline fits on a single 6 GB consumer GPU.' },
                { n: '04', t: 'Formal DP Bounds', d: 'RDP composition yields ε_total ≈ 16 across 4 rounds—provably bounding information leakage.' },
              ].map(item => (
                <motion.div key={item.n} className="panel" whileHover={{ scale: 1.02 }}>
                  <div className="panel-title">
                    <span style={{ fontFamily: 'var(--mono)', color: 'var(--nvidia)', fontSize: '0.7rem' }}>{item.n}</span>
                    {item.t}
                  </div>
                  <p className="panel-body">{item.d}</p>
                </motion.div>
              ))}
            </div>
          </FadeUp>
        </div>
      </div>
    </section>
  );
}

/* ─── DATASET ─── */
export function DatasetSection() {
  return (
    <section className="sec sec-center" id="dataset">
      <FadeUp>
        <div className="label" style={{ justifyContent: 'center' }}>§ 2 — Dataset &amp; Partitioning</div>
        <h2 className="heading">Symptom2Disease</h2>
        <p className="body-text" style={{ margin: '0 auto 3rem', textAlign: 'center' }}>
          A text classification dataset mapping free-form symptom descriptions to one of{' '}
          <strong>24 disease classes</strong>. FedNeMo uses a dataset-agnostic loader: any
          text/label CSV or JSON works out of the box.
        </p>
      </FadeUp>

      {/* Animated data flow */}
      <FadeUp delay={0.2}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap', justifyContent: 'center' }}>
          {[
            { label: 'Full Dataset', val: '100%', color: '#555' },
            { label: '→' },
            { label: 'Stratified Holdout', val: '~17%', color: '#76b900' },
            { label: '+', extra: true },
            { label: 'IID Partition × 5 Nodes', val: '~83%', color: '#76b900' },
          ].map((item, i) => (
            item.label === '→' || item.label === '+' ? (
              <span key={i} style={{ color: '#444', fontSize: '1.5rem', fontWeight: 700 }}>{item.label}</span>
            ) : (
              <div key={i} className="panel" style={{ padding: '1.5rem 2rem', textAlign: 'center' }}>
                <div style={{ fontFamily: 'var(--mono)', fontSize: '1.5rem', fontWeight: 700, color: item.color }}>{item.val}</div>
                <div style={{ fontSize: '0.8rem', color: '#666', marginTop: '0.25rem' }}>{item.label}</div>
              </div>
            )
          ))}
        </div>
      </FadeUp>

      <div className="grid-2" style={{ maxWidth: '1000px', marginTop: '3rem', textAlign: 'left' }}>
        <FadeUp delay={0.3}>
          <h3 className="subheading" style={{ marginTop: 0 }}>Stratified Holdout</h3>
          <p className="body-text">
            Before any partition, a class-balanced held-out set is reserved via{' '}
            <code className="chip">stratified_holdout()</code>. This set is <em>never</em> observed
            by any federated node during training, ensuring unbiased evaluation.
          </p>
        </FadeUp>
        <FadeUp delay={0.4}>
          <h3 className="subheading" style={{ marginTop: 0 }}>IID Partition</h3>
          <p className="body-text">
            The remainder is split via <code className="chip">iid_partition()</code>: a random,
            balanced, equal-parts split across N nodes. Each node receives an identical
            class-frequency distribution, isolating the privacy contribution of federation.
          </p>
        </FadeUp>
      </div>
    </section>
  );
}

/* ─── MODEL ─── */
export function ModelSection() {
  return (
    <section className="sec" id="model">
      <div style={{ maxWidth: '1100px', width: '100%' }}>
        <FadeUp>
          <div className="label">§ 3 — Model Architecture</div>
          <h2 className="heading">Nemotron-Mini-4B-Instruct</h2>
          <p className="body-text">
            NVIDIA's 4-billion parameter instruction-tuned LLM, fine-tuned via LoRA
            (<code className="chip">r=16, α=32</code>) on all attention projection layers.
            The transformer body remains frozen—only the LoRA matrices are trained.
          </p>
        </FadeUp>

        <div className="grid-3" style={{ marginTop: '3rem' }}>
          {[
            { title: '4-bit NF4 Body', desc: 'Transformer body is quantized to 4-bit NF4 and resident on GPU. Enables loading a 4B model on 6GB VRAM.', chip: 'GPU · bitsandbytes' },
            { title: 'CPU Embeddings', desc: '256k-vocabulary embed_tokens placed on CPU to avoid GPU OOM. Cross-device autograd connects the two stages seamlessly.', chip: 'CPU · embed_tokens' },
            { title: 'LM Head on CUDA', desc: '4-bit LM Head stays on GPU. Manual forward pass wires embed→body→lm_head with full cross-device gradient flow.', chip: 'CUDA · lm_head' },
          ].map((item, i) => (
            <FadeUp key={item.title} delay={i * 0.15}>
              <motion.div className="panel" style={{ height: '100%' }} whileHover={{ scale: 1.03 }}>
                <code className="chip" style={{ marginBottom: '1rem', display: 'inline-block' }}>{item.chip}</code>
                <div className="panel-title">{item.title}</div>
                <p className="panel-body">{item.desc}</p>
              </motion.div>
            </FadeUp>
          ))}
        </div>

        <FadeIn delay={0.5}>
          <div className="formula" style={{ marginTop: '3rem' }}>
            Forward: embed_tokens (CPU) → 4-bit body (GPU) → lm_head (GPU) → logits<br />
            LoRA: W = W₀ + α/r · B·A &nbsp;|&nbsp; r=16, α=32 &nbsp;|&nbsp; Frozen W₀
          </div>
        </FadeIn>
      </div>
    </section>
  );
}

/* ─── FEDERATED CORE ─── */
export function FederatedSection() {
  const steps = [
    {
      n: 'FS-01', t: 'Client Training',
      d: 'Each ClientNode fine-tunes its LoRA adapter on its local shard for one round. The full shard is reshuffled every round. Gradient accumulation (GRAD_ACCUM_STEPS=4) increases the effective batch size without larger GPU memory. Nodes can optionally simulate label-flip poisoning to validate the trust mechanism.',
    },
    {
      n: 'FS-02', t: 'FedRand Split-Adapter',
      d: 'For each LoRA layer, a Bernoulli trial with probability ρ=0.5 decides whether matrix A or B is transmitted. The complement stays private and persists locally. Because a global attacker never observes both matrices simultaneously for any layer in any round, gradient inversion is structurally defeated.',
    },
    {
      n: 'FS-03', t: 'Laplace Differential Privacy',
      d: 'Transmitted slices are noised before leaving the node. Relative mode: σ = noise_ratio × ‖Δ‖₁/dim. Absolute mode: classical C/ε Laplace. RDP composition across 4 rounds yields ε_total ≈ 16—a moderate budget for a research prototype.',
    },
    {
      n: 'FS-04', t: 'Adaptive Quantization',
      d: 'Matrices are quantized to QUANT_BITS ∈ {2, 8} using per-tensor affine quantization (scale + zero-point transmitted alongside quantized integers). Versus 32-bit FedAvg: 8-bit = 4× reduction, 2-bit = 16× reduction in transmission bytes.',
    },
    {
      n: 'FS-05', t: 'Trust-Weighted Aggregation',
      d: 'The base model (adapter disabled) scores each client update from a text summary. Blended with a deterministic heuristic to prevent a bad model read from zeroing a legitimate node. Final weight: w_i = α·trust_model(i) + (1−α)·heuristic(i). Aggregation: Δ_global = Σ(w_i · H(i) · Δ_i) / Σ(w_i · H(i)) where H(i) is the entropy importance of node i.',
    },
  ];

  return (
    <section className="sec" id="federated">
      <div style={{ maxWidth: '1100px', width: '100%' }}>
        <FadeUp>
          <div className="label">§ 4 — Federated Core</div>
          <h2 className="heading">The Privacy Stack</h2>
          <p className="body-text">
            FedNeMo's privacy guarantees come from four layered mechanisms applied in sequence
            during each federation round.
          </p>
        </FadeUp>

        <div className="grid-2" style={{ alignItems: 'start' }}>
          <FadeUp delay={0.15}>
            <div className="flow">
              {steps.map((step, i) => (
                <motion.div
                  key={step.n}
                  className="flow-item"
                  initial={{ opacity: 0, x: -20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true, margin: '-40px' }}
                  transition={{ duration: 0.7, delay: i * 0.12 }}
                >
                  <div className="flow-dot">{step.n.split('-')[1]}</div>
                  <div className="flow-content">
                    <div className="flow-title">{step.t}</div>
                    <div className="flow-desc">{step.d}</div>
                  </div>
                </motion.div>
              ))}
            </div>
          </FadeUp>

          <FadeUp delay={0.3}>
            <div style={{ position: 'sticky', top: '20vh', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
              <div className="panel">
                <div className="panel-title">Round Architecture</div>
                <div style={{ fontFamily: 'var(--mono)', fontSize: '0.65rem', color: '#76b900', lineHeight: 2, marginTop: '0.5rem' }}>
                  {`Node₁ ──[FedRand]──[DP]──[Quant]──┐\nNode₂ ──[FedRand]──[DP]──[Quant]──┤\nNode₃ ──[FedRand]──[DP]──[Quant]──┼──► Orchestrator\nNode₄ ──[FedRand]──[DP]──[Quant]──┤        │\nNode₅ ──[FedRand]──[DP]──[Quant]──┘   Trust×∇H\n                                         │\n                                    Global Δ`}
                </div>
              </div>

              <div className="formula">
                Bernoulli(ρ=0.5) → share A or B<br/>
                σ_Laplace = η × ‖Δ‖₁/dim<br/>
                Q(Δ) = round(Δ/s + z) @ {`{2,8}`}-bit<br/>
                w_i = α·TM(i) + (1-α)·H(i)
              </div>

              <div className="panel">
                <div className="panel-title">Configuration</div>
                <table className="data-table" style={{ marginTop: '0.5rem' }}>
                  <tbody>
                    {[
                      ['FedRand ρ','0.5'],['DP Mode','Relative'],['Noise Ratio','0.25'],
                      ['Quant Bits','8-bit (tx)'],['ε_total (RDP)','≈ 16'],['LoRA r / α','16 / 32'],
                    ].map(([k,v]) => (
                      <tr key={k}>
                        <td><code className="chip">{k}</code></td>
                        <td style={{ color: '#76b900', fontFamily: 'var(--mono)', fontSize: '0.8rem' }}>{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </FadeUp>
        </div>
      </div>
    </section>
  );
}

/* ─── RESULTS ─── */
export function ResultsSection() {
  return (
    <section className="sec sec-center" id="results">
      <div style={{ maxWidth: '1100px', width: '100%' }}>
        <FadeUp>
          <div className="label" style={{ justifyContent: 'center' }}>§ 5 — Results</div>
          <h2 className="heading" style={{ textAlign: 'center' }}>Performance Metrics</h2>
          <p className="body-text" style={{ margin: '0 auto 3rem', textAlign: 'center' }}>
            Evaluated on 1,200 held-out records across 24 disease classes — never seen by any node during training.
          </p>
        </FadeUp>

        {/* Big animated stats */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: '2rem', marginBottom: '4rem' }}>
          {[
            { target: 87.5, suffix: '%', label: 'Accuracy', dec: 1 },
            { target: 0.86, suffix: '', label: 'Macro-F1', dec: 2 },
            { target: 0.0, suffix: '%', label: 'Unparseable', dec: 1 },
          ].map(stat => (
            <FadeUp key={stat.label}>
              <div>
                <div className="big-stat">
                  <AnimCounter target={stat.target} suffix={stat.suffix} decimals={stat.dec} />
                </div>
                <div className="stat-label">{stat.label}</div>
              </div>
            </FadeUp>
          ))}
        </div>

        <FadeIn delay={0.3}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', textAlign: 'left' }}>
            <div className="panel">
              <div className="panel-title">Evaluation Config</div>
              <table className="data-table">
                <thead><tr><th>Setting</th><th>Value</th></tr></thead>
                <tbody>
                  {[
                    ['Dataset','Symptom2Disease'],['Holdout Records','1,200'],['Classes','24'],
                    ['Clients','5'],['Rounds','4'],['DP On','✓'],['Quant','8-bit'],['LoRA r','16'],
                  ].map(([k,v]) => (
                    <tr key={k}>
                      <td>{k}</td>
                      <td style={{ color: '#76b900', fontFamily: 'var(--mono)', fontSize: '0.8rem' }}>{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="panel">
              <div className="panel-title">Key Results</div>
              <table className="data-table">
                <thead><tr><th>Metric</th><th>Value</th></tr></thead>
                <tbody>
                  <tr className="hero-row">
                    <td>Accuracy</td>
                    <td>87.5%</td>
                  </tr>
                  <tr className="hero-row">
                    <td>Macro-F1</td>
                    <td>0.86</td>
                  </tr>
                  <tr>
                    <td>Unparseable Outputs</td>
                    <td>0.0%</td>
                  </tr>
                  <tr>
                    <td>Comm. Savings vs FedAvg-32</td>
                    <td>4× (8-bit) / 16× (2-bit)</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </FadeIn>
      </div>
    </section>
  );
}

/* ─── CONCLUSION ─── */
export function ConclusionSection() {
  return (
    <section className="sec sec-center" id="conclusion" style={{ paddingBottom: '15vh' }}>
      <div style={{ maxWidth: '900px' }}>
        <FadeUp>
          <div className="label" style={{ justifyContent: 'center' }}>§ 6 — Conclusion</div>
          <h2 className="heading" style={{ textAlign: 'center', marginBottom: '2rem' }}>
            Privacy Without Compromise
          </h2>
        </FadeUp>
        <FadeUp delay={0.2}>
          <p className="body-text" style={{ margin: '0 auto 1.5rem', textAlign: 'center', color: '#ccc', fontSize: '1.3rem', fontStyle: 'italic' }}>
            FedNeMo demonstrates that a 4B-parameter LLM can be fine-tuned federally on sensitive medical text—entirely locally, on a single consumer GPU—while maintaining formal privacy guarantees and competitive accuracy.
          </p>
        </FadeUp>
        <FadeUp delay={0.35}>
          <p className="body-text" style={{ margin: '0 auto 3rem', textAlign: 'center' }}>
            The combination of FedRand structural privacy, Laplace DP, quantized transmission, and trust-weighted aggregation provides layered defence-in-depth deployable today without cloud infrastructure.
          </p>
        </FadeUp>

        {/* References */}
        <FadeIn delay={0.5}>
          <div className="divider" />
          <div className="panel-title" style={{ marginBottom: '1rem' }}>References</div>
          {[
            ['[1]','McMahan, B., et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS 2017.'],
            ['[2]','Hu, E., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.'],
            ['[3]','Mironov, I. (2017). Rényi Differential Privacy. CSF 2017.'],
            ['[4]','Zhu, L., et al. (2019). Deep Leakage from Gradients. NeurIPS 2019.'],
            ['[5]','Fowl, L., et al. (2022). Robbing the Fed: Directly Obtaining Private Data in Federated Learning. ICLR 2022.'],
            ['[6]','NVIDIA. (2024). Nemotron-Mini-4B-Instruct. NVIDIA AI Foundation Models.'],
            ['[7]','Dettmers, T., et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. NeurIPS 2023.'],
          ].map(([tag, text]) => (
            <div key={tag} className="ref-item">
              <span className="ref-tag">{tag}</span>
              <span>{text}</span>
            </div>
          ))}
        </FadeIn>
      </div>
    </section>
  );
}
