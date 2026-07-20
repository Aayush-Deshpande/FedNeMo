import React, { useState, useEffect } from 'react';
import { motion, useInView } from 'framer-motion';
import { Cpu, Shield, Network, Zap, Lock, Database, GitMerge, FileArchive, Activity, CheckCircle2 } from 'lucide-react';

const FluidStat = ({ label, targetValue, suffix, delay }) => {
  const ref = React.useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-10%" });
  const [value, setValue] = useState(0);

  useEffect(() => {
    if (isInView) {
      let start = 0;
      const duration = 2000;
      const incrementTime = 20;
      const step = (targetValue / (duration / incrementTime));

      const timer = setInterval(() => {
        start += step;
        if (start >= targetValue) {
          setValue(targetValue);
          clearInterval(timer);
        } else {
          setValue(start);
        }
      }, incrementTime);
      return () => clearInterval(timer);
    }
  }, [isInView, targetValue]);

  return (
    <motion.div 
      ref={ref}
      className="fluid-stat"
      initial={{ opacity: 0, y: 30, filter: 'blur(10px)' }}
      animate={isInView ? { opacity: 1, y: 0, filter: 'blur(0px)' } : {}}
      transition={{ duration: 1.5, delay, ease: "easeOut" }}
    >
      <div className="stat-value">
        {value.toFixed(suffix === '%' ? 1 : 2)}{suffix}
      </div>
      <div className="stat-label">{label}</div>
    </motion.div>
  );
};

const SectionHeader = ({ title, subtitle }) => (
  <div className="tree-branch" style={{ marginTop: '15vh', marginBottom: '5vh' }}>
    <motion.div style={{ textAlign: 'center' }}
      initial={{ opacity: 0, filter: 'blur(10px)' }}
      whileInView={{ opacity: 1, filter: 'blur(0px)' }}
      viewport={{ once: true, margin: "-10%" }}
      transition={{ duration: 1.5 }}
    >
      <span className="monospaced">{subtitle}</span>
      <h2 className="gradient-text" style={{ margin: '0 auto' }}>{title}</h2>
    </motion.div>
  </div>
);

const Branch = ({ side, icon, title, mono, desc, list }) => (
  <div className="tree-branch">
    <motion.div
      className={`branch-content branch-${side}`}
      initial={{ opacity: 0, x: side === 'left' ? -50 : 50, filter: 'blur(10px)' }}
      whileInView={{ opacity: 1, x: 0, filter: 'blur(0px)' }}
      viewport={{ once: true, margin: "-10%" }}
      transition={{ duration: 1.2, ease: "easeOut" }}
    >
      <div className="branch-icon" style={{ justifyContent: side === 'left' ? 'flex-end' : 'flex-start' }}>
        {icon}
      </div>
      <span className="monospaced" style={{ textAlign: side === 'left' ? 'right' : 'left' }}>{mono}</span>
      <h3>{title}</h3>
      <p style={{ marginLeft: side === 'left' ? 'auto' : 0 }}>{desc}</p>
      {list && (
        <ul style={{ 
          marginTop: '1rem', 
          marginLeft: side === 'left' ? 'auto' : 0, 
          textAlign: side === 'left' ? 'right' : 'left',
          listStyle: 'none',
          color: '#a0a0a0',
          fontSize: '1.1rem',
          lineHeight: '1.8'
        }}>
          {list.map((item, i) => <li key={i}>{item}</li>)}
        </ul>
      )}
    </motion.div>
  </div>
);

export default function ResearchSections() {
  return (
    <>
      <SectionHeader subtitle="01 / The Problem" title="Medical Privacy in AI" />
      <Branch 
        side="left"
        icon={<Database size={40} color="#76b900" />}
        mono="Symptom2Disease Mapping"
        title="100% Local Execution"
        desc="When mapping highly sensitive symptom descriptions to diseases, sending data to centralized cloud servers exposes patient records to unacceptable risks. FedNeMo solves this by bringing the training entirely to the local node."
        list={["No servers.", "No cloud orchestration.", "Local Python single-process."]}
      />

      <SectionHeader subtitle="02 / The Dataset" title="Data & Partitioning" />
      <Branch 
        side="right"
        icon={<GitMerge size={40} color="#76b900" />}
        mono="IID & Holdout"
        title="Balanced Stratification"
        desc="To ensure robust generalization, we reserve a class-balanced held-out set before partitioning. The remaining data undergoes an IID partition, creating a random, balanced, equal-parts split across 5 federated nodes."
      />

      <SectionHeader subtitle="03 / The Engine" title="Nemotron-Mini-4B-Instruct" />
      <Branch 
        side="left"
        icon={<Cpu size={40} color="#76b900" />}
        mono="6GB VRAM Bound"
        title="Custom 4-Bit Pipeline"
        desc="Powered by NVIDIA's 4B-parameter foundation model. We utilize a custom VRAM-bounded loader for a 6GB GPU."
        list={[
          "4-bit NF4 transformer body on GPU.",
          "256k-vocab embed_tokens on CPU.",
          "Cross-device autograd maximizes speed without OOM errors."
        ]}
      />

      <SectionHeader subtitle="04 / Federated Core" title="Privacy-Preserving Stack" />
      <Branch 
        side="right"
        icon={<Lock size={40} color="#76b900" />}
        mono="Bernoulli(ρ) = 0.5"
        title="FedRand Split-Adapter"
        desc="Structurally preventing global inversion attacks. Per layer per round, nodes probabilistically share either adapter A or B. The unshared matrix stays strictly private and persists locally across rounds."
      />
      <Branch 
        side="left"
        icon={<Shield size={40} color="#76b900" />}
        mono="ε_total ≈ 16"
        title="Laplace Differential Privacy"
        desc="Updates are noised using a relative Laplace mechanism. Noise is calibrated to a fraction of the update's own magnitude, maintaining high Signal-to-Noise Ratio (SNR) while satisfying formal DP bounds across rounds."
      />
      <Branch 
        side="right"
        icon={<FileArchive size={40} color="#76b900" />}
        mono="8-bit / 2-bit"
        title="Adaptive Quantization"
        desc="To drastically slash communication overhead compared to a 32-bit full-FedAvg baseline, transmitted matrices undergo adaptive per-tensor affine quantization with scale and zero-point parameters."
      />
      <Branch 
        side="left"
        icon={<Activity size={40} color="#76b900" />}
        mono="Deterring Poisoning"
        title="Trust-Weighted Aggregation"
        desc="The base model acts as a trust agent, scoring each update's trustworthiness from a text summary. The orchestrator combines per-slot updates weighted by this trust factor multiplied by entropy importance."
      />

      <SectionHeader subtitle="05 / Inference" title="Constrained Decoding" />
      <Branch 
        side="right"
        icon={<CheckCircle2 size={40} color="#76b900" />}
        mono="Deterministic Output"
        title="Rule-Based Validation"
        desc="During inference, the model output is mathematically snapped to a valid dataset class. This guarantees that evaluation results in precisely 0.0% unparseable outputs."
      />

      <SectionHeader subtitle="06 / Results" title="Performance Metrics" />
      <div className="tree-branch" style={{ flexDirection: 'column', alignItems: 'center', marginBottom: '30vh' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '3rem', marginTop: '2rem' }}>
          <FluidStat label="Accuracy (Held-out)" targetValue={87.5} suffix="%" delay={0.2} />
          <FluidStat label="Macro-F1 Score" targetValue={0.86} suffix="" delay={0.4} />
          <FluidStat label="Unparseable Outputs" targetValue={0.0} suffix="%" delay={0.6} />
        </div>
      </div>
    </>
  );
}
