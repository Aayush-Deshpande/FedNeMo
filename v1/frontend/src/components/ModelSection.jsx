import React from 'react';
import { motion } from 'framer-motion';
import { Cpu, Zap, Network } from 'lucide-react';

export default function ModelSection() {
  return (
    <div className="tree-branch">
      <motion.div
        className="branch-content branch-right"
        initial={{ opacity: 0, x: 50, filter: 'blur(10px)' }}
        whileInView={{ opacity: 1, x: 0, filter: 'blur(0px)' }}
        viewport={{ once: true, margin: "-10%" }}
        transition={{ duration: 1.2, ease: "easeOut" }}
      >
        <h2 className="gradient-text">The Engine</h2>
        <h3>Nemotron-Mini-4B-Instruct</h3>
        <p style={{ marginBottom: '2rem' }}>
          Powered by a lightweight yet incredibly capable 4-Billion parameter foundation model from NVIDIA, specifically tuned to understand complex medical syntax locally.
        </p>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
          <motion.div whileHover={{ scale: 1.05, x: 10 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '0.5rem' }}>
              <Cpu size={28} color="#fff" />
              <h3 style={{ margin: 0 }}>VRAM Bounded</h3>
            </div>
            <p>Custom 4-bit NF4 transformer body execution allows the entire pipeline to fit inside a single 6GB consumer GPU.</p>
          </motion.div>

          <motion.div whileHover={{ scale: 1.05, x: 10 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '0.5rem' }}>
              <Zap size={28} color="#fff" />
              <h3 style={{ margin: 0 }}>Cross-Device Autograd</h3>
            </div>
            <p>Embeddings run on CPU, while the 4-bit LM Head stays on the GPU, maximizing speed without hitting out-of-memory errors.</p>
          </motion.div>

          <motion.div whileHover={{ scale: 1.05, x: 10 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '0.5rem' }}>
              <Network size={28} color="#fff" />
              <h3 style={{ margin: 0 }}>Constrained Decoding</h3>
            </div>
            <p>Outputs are mathematically snapped to valid classes, ensuring absolutely 0.0% unparseable outputs during evaluation.</p>
          </motion.div>
        </div>
      </motion.div>
    </div>
  );
}
