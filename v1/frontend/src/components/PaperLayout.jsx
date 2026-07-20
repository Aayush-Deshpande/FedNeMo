import React from 'react';
import { motion } from 'framer-motion';

export default function PaperLayout({ children }) {
  return (
    <div className="paper-wrapper">
      <motion.div
        className="paper-document"
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1.4, ease: [0.16, 1, 0.3, 1] }}
      >
        {/* ── TITLE BLOCK ── */}
        <div className="paper-title-block">
          <span className="conference-tag">
            FedNeMo &nbsp;·&nbsp; Privacy-Preserving Federated Learning &nbsp;·&nbsp; 2025
          </span>

          <h1 className="paper-main-title">
            FedNeMo: Local, In-Process Federated Fine-Tuning<br />
            of Nemotron-Mini-4B for Medical Text Classification
          </h1>

          <p className="paper-subtitle">
            A Privacy-Preserving Stack with FedRand, Laplace DP, Quantized Transmission,<br />
            and Trust-Weighted Aggregation — Achieving 87.5% Accuracy on Symptom2Disease
          </p>

          <p className="paper-authors">
            <a href="#">Aayush Deshpande</a>
          </p>
          <p className="paper-affiliation">
            Local Research, India &nbsp;·&nbsp; Powered by NVIDIA Nemotron-Mini-4B-Instruct
          </p>
        </div>

        {/* ── ABSTRACT ── */}
        <div className="abstract-block">
          <span className="abstract-label">Abstract</span>
          <p className="abstract-text">
            We present <strong>FedNeMo</strong>, a fully local, in-process federated fine-tuning framework for 
            NVIDIA's Nemotron-Mini-4B-Instruct model, applied to the task of medical text classification 
            (symptom description → disease). FedNeMo operates without any servers, cloud orchestration, 
            or outbound training traffic, protecting sensitive patient data entirely within the local execution 
            environment. We introduce a comprehensive privacy-preserving stack comprising: 
            (i) <strong>FedRand</strong> probabilistic split-adapter sharing to structurally defeat gradient inversion attacks, 
            (ii) <strong>Laplace differential privacy</strong> calibrated relative to update magnitude, 
            (iii) <strong>adaptive per-tensor quantization</strong> at 2- or 8-bit precision for communication efficiency, and 
            (iv) <strong>trust × entropy-weighted aggregation</strong> to deter poisoning. 
            Our system achieves <strong>87.5% accuracy</strong> and <strong>0.86 Macro-F1</strong> on the held-out 
            Symptom2Disease benchmark (1,200 records, 24 classes), with 0.0% unparseable outputs across 
            5 federated nodes and 4 communication rounds. Training is entirely local; the single external 
            API call is limited to optional inference-time report transcription.
          </p>
          <p className="keywords">
            <strong>Keywords:</strong> Federated Learning, Differential Privacy, Quantization, 
            LoRA, LLM Fine-tuning, Medical NLP, NVIDIA Nemotron, FedRand, Trust-Weighted Aggregation
          </p>
        </div>

        {/* ── TWO-COLUMN BODY ── */}
        <div className="paper-columns">
          {children}
        </div>
      </motion.div>
    </div>
  );
}
