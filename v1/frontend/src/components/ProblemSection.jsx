import React from 'react';
import { motion } from 'framer-motion';

export default function ProblemSection() {
  return (
    <div className="tree-branch">
      <motion.div
        className="branch-content branch-left"
        initial={{ opacity: 0, x: -50, filter: 'blur(10px)' }}
        whileInView={{ opacity: 1, x: 0, filter: 'blur(0px)' }}
        viewport={{ once: true, margin: "-10%" }}
        transition={{ duration: 1.2, ease: "easeOut" }}
      >
        <h2 className="gradient-text">The Problem</h2>
        <h3>Medical Privacy in AI</h3>
        <p style={{ marginBottom: '1.5rem', marginLeft: 'auto' }}>
          When fine-tuning AI models on medical data (like mapping symptom descriptions to diseases), privacy is paramount. Sending sensitive medical records to a centralized cloud server exposes data to unacceptable risks.
        </p>
        <p style={{ marginLeft: 'auto' }}>
          <strong>FedNeMo solves this.</strong> By bringing the training directly to the local node, data never leaves the hospital or device. No servers, no cloud orchestration—just 100% local, privacy-preserving federated fine-tuning.
        </p>
      </motion.div>
    </div>
  );
}
