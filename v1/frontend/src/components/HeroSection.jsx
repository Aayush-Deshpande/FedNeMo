import React from 'react';
import { motion } from 'framer-motion';

export default function HeroSection() {
  return (
    <div className="section section-center" style={{ minHeight: '100vh', justifyContent: 'center' }}>
      <motion.div
        initial={{ opacity: 0, filter: 'blur(10px)', y: 30 }}
        animate={{ opacity: 1, filter: 'blur(0px)', y: 0 }}
        transition={{ duration: 1.5, ease: 'easeOut' }}
      >
        <span className="monospaced" style={{ fontSize: '1.2rem', marginBottom: '1rem' }}>
          Research Paper // Architecture Overview
        </span>
        <h1 className="gradient-text">FedNeMo</h1>
        <p style={{ fontSize: '1.8rem', fontWeight: '600', color: '#fff', margin: '0 auto 1rem auto' }}>
          Local, In-Process Federated Fine-Tuning.
        </p>
        <p style={{ margin: '0 auto', fontSize: '1.4rem' }}>
          Privacy-preserving fine-tuning of Nemotron-Mini-4B.
        </p>
      </motion.div>
    </div>
  );
}
