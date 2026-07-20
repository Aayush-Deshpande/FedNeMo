import React from 'react';
import { motion } from 'framer-motion';

const featureData = [
  {
    title: 'High Accuracy',
    stat: '87.5%',
    desc: 'Uncompromised performance. Achieves 87.5% Accuracy and 0.86 Macro-F1 on the Symptom2Disease dataset.',
  },
  {
    title: 'Communication Efficiency',
    stat: 'Reduced Cost',
    desc: 'Adaptive 2-bit or 8-bit quantized transmission minimizes communication overhead while maintaining model fidelity.',
  },
  {
    title: 'Privacy-Preserving',
    stat: 'Laplace DP',
    desc: 'Powered by FedRand split-adapter and Laplace differential privacy, ensuring your data never leaves your device.',
  },
  {
    title: 'Resource Efficient',
    stat: '6GB VRAM',
    desc: 'VRAM-bounded execution means you can run the entire training stack on a single consumer GPU.',
  }
];

export default function FeaturesSection() {
  return (
    <section className="section">
      <div className="features">
        {featureData.map((feature, index) => (
          <motion.div
            key={index}
            className="feature-card"
            initial={{ opacity: 0, y: 50 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.8, delay: index * 0.2 }}
          >
            <div className="stat-highlight">{feature.stat}</div>
            <h2>{feature.title}</h2>
            <p>{feature.desc}</p>
          </motion.div>
        ))}
      </div>
    </section>
  );
}
