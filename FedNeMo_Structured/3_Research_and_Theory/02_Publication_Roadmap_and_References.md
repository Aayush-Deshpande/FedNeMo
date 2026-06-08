## 9. Publication Roadmap

This project naturally fragments into four publishable contributions:

| Paper | Title | Venue Target | Novel Contribution |
| :--- | :--- | :--- | :--- |
| **Paper 1** | FedNeMo: Communication-Efficient Privacy-Preserving Federated Fine-tuning of Large Language Models | ICLR / NeurIPS | Full system paper; first federated framework for hybrid Mamba-Transformer MoE models. |
| **Paper 2** | Privacy Accounting for Randomized Subparameter Selection with Laplacian DP in Federated LoRA | IEEE S&P / CCS | Formal privacy analysis of FedRand + DP composition; novel theoretical bounds. |
| **Paper 3** | Shannon Entropy Client Importance for Non-IID Federated LLM Fine-Tuning | EMNLP / ACL Findings | Transfer of entropy-based weighting from CNNs to LLM LoRA; convergence analysis. |
| **Paper 4** | Federated Clinical NLP Benchmark: Systematic Evaluation of Federated LoRA Strategies | ACL / ML4H | Ablation study and benchmark across clinical/financial NLP tasks. |

*Table 7: Publication Roadmap Derived from the FedNeMo Project.*

---

## 10. References and Resources

### 10.1 Core Research Papers (Local PDFs)

1. [Exploring the Vulnerabilities of Federated Learning: A Deep Dive into Gradient Inversion Attacks](file:///e:/FedNeMo/Exploring%20the%20Vulnerabilities%20of%20Federated%20Learning%20A%20Deep%20Dive%20into%20Gradient%20Inversion%20Attacks.pdf) — Guo et al., TPAMI 2026. Taxonomy of OP-GIA, GEN-GIA, ANA-GIA; 3-stage defense pipeline.
2. [FedRand: Enhancing Privacy in Federated Learning with Randomized LoRA Subparameter Updates](file:///e:/FedNeMo/FedRand%20Enhancing%20Privacy%20in%20Federated%20Learning%20with%20Randomized%20LoRa%20Subparameter%20Updates.pdf) — Park et al., 2025. Randomized A/B matrix partitioning per round.
3. [Enhanced Privacy and Communication Efficiency in Non-IID FL with Adaptive Quantization and DP](file:///e:/FedNeMo/Enhanced%20Privacy%20and%20Communication%20Efficiency%20in%20Non-IID%20Federated%20Learning%20with%20Adaptize%20Quantization%20and%20Differential%20Privacy.pdf) — Ardıç and Genç. Dual-tier bit-length scheduling with Laplacian DP.
4. [FedPS: Federated Data Preprocessing via Aggregated Statistics](file:///e:/FedNeMo/FedPS%20Federated%20data%20Preprocessing%20via%20aggregated%20Statistics.pdf) — Xu et al. Privacy-safe global data normalization via data sketches.
5. [FedRE: A Representation Entanglement Framework for Model-Heterogeneous FL](file:///e:/FedNeMo/FedRE%20A%20Representation%20Entanglement%20Framework%20for%20Model-Heterogeneous.pdf) — Yao et al., CVPR 2026. Entangled representations with randomized weights.
6. [Sample Selection Using Multi-Task Autoencoders in FL with Non-IID Data](file:///e:/FedNeMo/Sample%20Selection%20Using%20Multi-Task%20Autoencoders%20in%20Federated%20Learning%20with%20Non-IID%20Data.pdf) — Ardıç and Genç. MTAE-based contribution-aware sample filtering.
7. [HtFLlib: A Comprehensive Heterogeneous FL Library and Benchmark](file:///e:/FedNeMo/HtFLlib%20A%20Comprehensive%20Heterogeneous%20Federated%20Learning%20Library%20and%20Benchmark.pdf) — Zhang et al., KDD 2025. Standardized benchmarking with 40 architectures.

### 10.2 Project Strategy Documents

- [Claude's Directed Analysis](file:///e:/FedNeMo/claude_directed.md) — Problem statement synthesis, component breakdown, and publication roadmap.
- [Manus's Strategic Suggestions for FedNeMo](file:///e:/FedNeMo/Manus's%20Strategic%20Suggestions%20for%20FedNeMo.md) — Presentation strategy and live demo recommendations.
- [Hackathon Strategy Plan](file:///e:/FedNeMo/hackathon_strategy.md) — Full system architecture, module breakdown, and implementation timeline.

### 10.3 NVIDIA Documentation

- [Parameter Efficient Fine-Tuning (PEFT) — NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/peft.html)
- [NVIDIA Nemotron AI Models — Developer Portal](https://developer.nvidia.com/nemotron-models)
- [Introducing Nemotron-3 Super — NVIDIA Technical Blog](https://developer.nvidia.com/blog/introducing-nemotron-3-super/)
- [Turning ML to Federated Learning in Minutes with NVIDIA FLARE](https://developer.nvidia.com/blog/turning-machine-learning-to-federated-learning-in-minutes-with-nvidia-flare/)
- [Federated Learning Without Refactoring Overhead Using NVIDIA FLARE](https://developer.nvidia.com/blog/federated-learning-without-refactoring-overhead-nvidia-flare/)
- [Filters — NVIDIA FLARE Documentation](https://nvflare.readthedocs.io/en/main/user_guide/filters.html)
- [Scalable Federated Learning with NVIDIA FLARE for Enhanced LLM Performance](https://developer.nvidia.com/blog/scalable-federated-learning-nvidia-flare/)

### 10.4 External Research

- [MambaPEFT: Exploring Parameter-Efficient Fine-Tuning for Mamba](https://arxiv.org/abs/2403.11144) — arXiv, 2024.
- [Parameter-Efficient Fine-Tuning of State Space Models](https://icml.cc/) — ICML Poster, 2025.
- [Optimizing Federated Learning in the Era of LLMs: Message Quantization and Streaming](https://arxiv.org/abs/2402.13284) — arXiv, 2024.
- [Empowering Federated Learning for Massive Models with NVIDIA FLARE](https://arxiv.org/abs/2310.18342) — arXiv, 2023.
- [Federated Learning in Biomedical and Health Informatics: A Systematic Review](https://www.techrxiv.org/) — TechRxiv, 2025.
