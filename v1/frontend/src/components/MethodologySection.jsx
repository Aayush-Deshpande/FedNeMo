import React from 'react';
import { motion } from 'framer-motion';
import { Lock, GitMerge, FileArchive, Shield } from 'lucide-react';

const methodologySteps = [
  {
    icon: <Lock size={40} color="#fff" />,
    title: 'FedRand Split-Adapter',
    desc: 'Per layer per round, nodes probabilistically share either adapter A or B (Bernoulli ρ). The unshared layer stays strictly private, structurally preventing global inversion attacks.',
    side: 'left'
  },
  {
    icon: <Shield size={40} color="#fff" />,
    title: 'Laplace Differential Privacy',
    desc: 'Before transmission, updates are noised using a relative Laplace mechanism calibrated to the update\'s own magnitude, maintaining high Signal-to-Noise Ratio while satisfying formal DP bounds (ε_total ≈ 16).',
    side: 'right'
  },
  {
    icon: <FileArchive size={40} color="#fff" />,
    title: 'Quantized Transmission',
    desc: 'Transmitted matrices undergo adaptive per-tensor affine quantization down to 2-bit or 8-bit precision, drastically slashing communication costs compared to standard 32-bit FedAvg.',
    side: 'left'
  },
  {
    icon: <GitMerge size={40} color="#fff" />,
    title: 'Trust-Weighted Aggregation',
    desc: 'The orchestrator combines updates weighted by a unique "trust × entropy" score, computed locally by the base model acting as a trust agent to deter label-flip or poisoning attacks.',
    side: 'right'
  }
];

export default function MethodologySection() {
  return (
    <>
      <div className="tree-branch" style={{ marginTop: '20vh', marginBottom: 0 }}>
        <h2 className="gradient-text" style={{ textAlign: 'center', margin: '0 auto' }}>The Methodology</h2>
      </div>
      
      {methodologySteps.map((step, idx) => (
        <div className="tree-branch" key={idx}>
          <motion.div
            className={`branch-content branch-${step.side}`}
            initial={{ opacity: 0, x: step.side === 'left' ? -50 : 50, filter: 'blur(10px)' }}
            whileInView={{ opacity: 1, x: 0, filter: 'blur(0px)' }}
            viewport={{ once: true, margin: "-10%" }}
            transition={{ duration: 1.2, ease: "easeOut" }}
          >
            <div className="branch-icon" style={{ justifyContent: step.side === 'left' ? 'flex-end' : 'flex-start' }}>
              {step.icon}
            </div>
            <h3>{step.title}</h3>
            <p style={{ marginLeft: step.side === 'left' ? 'auto' : 0 }}>{step.desc}</p>
          </motion.div>
        </div>
      ))}
    </>
  );
}
