# FedNeMo — PPT Contents Reference
# India Agentic AI Open Hackathon | Track B — Model Finetuning & Customisation
# Team: Midnight Ciphers
# File: FedNeMo_Final.pptx (14 slides)
# Last updated: June 19, 2026

---

## HOW TO READ THIS FILE

Each slide entry shows:
- STATUS: ✅ Already in the deck | 🔲 Not yet in the deck | ⚠️ In the deck but needs improvement
- CONTENT: Exact text / table rows / diagram elements that belong on the slide
- SOURCE: Where the content comes from (paper, doc, your input)
- NOTES: What a judge expects to see; what to watch out for

---

## SLIDE 1 — Title
**STATUS: ✅ Done**

**What's in the deck:**
- Headline: "FedNeMo: Privacy-Preserving Federated Fine-Tuning of Nemotron for Healthcare"
- Subline: "Midnight Ciphers   |   Track B — Model Finetuning & Customisation"

**What could still be added (optional):**
- A short one-line tagline below the subtitle, e.g.:
  "Five research-backed components. One privacy-hardening layer for federated LLM fine-tuning."
- Team member names (some teams add these on the title slide)

**Source:** Your input (team name, track choice)
**Judge expectation:** Immediately clear what the project is and which track it's for.

---

## SLIDE 2 — Contents + Instructions
**STATUS: ✅ Done (template-owned, do not edit)**

**What's in the deck:**
- Four sections: Team Introduction / AI Use-case / Tech Stack / Roadmap
- Template instructions retained (judges need to see you followed the format)

---

## SLIDE 3 — Team Introduction (Section Divider)
**STATUS: ✅ Done (template divider slide, do not edit)**

---

## SLIDE 4 — Team Introduction: Midnight Ciphers
**STATUS: ✅ Done**

**What's in the deck:**
- Role table (4 rows):
  | Member | Focus area | Responsibility on FedNeMo |
  | Aayush Deshpande | Federated systems / NVIDIA stack | FLARE integration, NeMo LoRA pipeline, system architecture |
  | Laukik Rathod | Privacy & security ML | FedRand filter, Laplacian DP mechanism, RDP accounting |
  | Harshal Yadav | Data engineering / FL pipelines | FedPS preprocessing, non-IID data partitioning, MTAE quality filter |
  | Yash Kalal | Evaluation & visualisation | GIA/MIA evaluation protocol, Streamlit telemetry dashboard |
- Three bullets:
  - Domain focus: ML, federated learning, privacy-preserving systems, LLM fine-tuning
  - Building on NVIDIA NeMo 2.0, NVIDIA FLARE, Nemotron
  - Motivated by ABDM data-silo problem and absence of a privacy layer for federated LLM fine-tuning

**What's NOT in the deck (template also asks for):**
🔲 Details of your organisation — you are a student team. Add: "Student engineering team, [your institution name if you want to include it]."
🔲 Current AI deployment setup (NIMs, OpenAI API, etc.) — if you use any AI tools currently, add a line. If not, say "developing locally using HuggingFace transformers and NeMo framework documentation."
🔲 Past hackathon experience (optional) — add if you have any.

**Source:** Your input
**Judge expectation:** Can this team execute? Do they know the NVIDIA stack?

---

## SLIDE 5 — AI Use Case (Section Divider)
**STATUS: ✅ Done (template divider, do not edit)**

---

## SLIDE 6 — AI Use Case: The Problem FedNeMo Solves
**STATUS: ✅ Done**

**What's in the deck:**
- Project title: FedNeMo — Communication-Efficient, Privacy-Preserving Federated Fine-Tuning of Nemotron LLMs
- Problem bullets:
  - ABDM: 900M health accounts, 1B+ linked records, siloed across hospitals
  - DPDP Act 2023: hospitals = Data Fiduciaries; penalties up to ₹250 crore
  - Standard FL transmits LoRA updates which can be gradient-inverted to reconstruct patient records (cite: Guo et al.)
  - Adding noise destroys quality. Cutting bandwidth hurts accuracy. No system solves all three for LLM fine-tuning.
  - Motivation: building FedNeMo as the missing privacy layer between FLARE and NeMo
- Trilemma diagram (right): Model quality / Data privacy / Communication efficiency — "FedNeMo resolves all three"
- TRACK: B chip

**What's NOT in the deck (template also asks for):**
🔲 Project motivation — "are you currently working on this or a similar project in your organisation?"
   Add: "We are actively developing FedNeMo as a framework solution. The Indian healthcare landscape under ABDM and DPDP is our primary deployment context."
🔲 Results achieved (optional) — not applicable at this stage. Leave blank or say "System design and prototype architecture complete; GPU training planned during hackathon."

**Source:** Structured/1/01 (India context), design.md, Guo et al. GIA paper
**Judge expectation:** Is this a real, important problem? Does the team understand the India-specific context?

---

## SLIDE 7 — AI Use Case: FedNeMo — How It Works
**STATUS: ✅ Done**

**What's in the deck:**
- Research component table (5 rows + header):
  | Component | Research basis | What it solves | Key result (from paper) |
  | MTAE + OCSVM Data quality | Ardıç & Genç (IEEE Access 2025) | Removes poisoned/noisy samples | Up to 7.02% accuracy gain on non-IID data (CIFAR-10) |
  | FedPS Preprocessing | Xu & Cormode (Feb 2026) | Harmonises lab values, ICD codes across hospitals | Consistent preprocessing across heterogeneous FL clients |
  | FedRand Privacy | Park et al. KAIST (2025) | Only one LoRA matrix (A or B) transmitted per round | Significant MIA resistance; accuracy maintained vs full LoRA |
  | Adaptive Quant + Laplacian DP | Ardıç & Genç (IEEE Access 2025) | Formal (ε,0)-DP guarantee; entropy-driven compression | 45–52% communication reduction; robust privacy on medical data |
  | NVIDIA FLARE Orchestration | NVIDIA (FLARE 2.5+) | Federated round management; DXO filter pipeline | Production-grade FL transport; no refactoring of training loop |
- System flow caption: "raw data never leaves the hospital — only processed LoRA deltas travel"
- Flow diagram (5 boxes): Local data + MTAE filter → FedPS harmonise → Nemotron LoRA fine-tune → FedRand → DP → Quant → FLARE orchestrator

**What could still be added:**
🔲 A "What makes this novel" callout — none of these papers combine all five components. None targets Nemotron. None is designed for Indian healthcare. That combination is the contribution. Consider adding a short sentence below the table.
🔲 A note about the FedRE paper (model-heterogeneous FL) — it's in your Papers/ folder but was cut from scope. You could add it as "future extension" to show you read more than what you're using.

**Source:** All 5 research papers (actual PDFs read)
**Judge expectation:** Is the solution grounded in real research? Can the team explain each component?

---

## SLIDE 8 — AI Use Case: Dataset Strategy & Evaluation Plan
**STATUS: ✅ Done**

**What's in the deck:**
- Dataset table (3 rows + header):
  | Dataset | Use | Why this dataset | License |
  | MedQA-USMLE | Primary training & evaluation | Medical QA; clean benchmark; simulates clinical reasoning | Research use (permissive) |
  | Synthetic clinical records | Privacy-safe training data | Self-generated; no PHI risk; fully controllable non-IID distribution | No licence required |
  | MIMIC-III / PubMedQA | Extended evaluation (planned) | Real EHR notes; rigorous clinical NLP benchmark | PhysioNet DUA / Apache 2.0 |
- Evaluation plan:
  - Non-IID setup: 5 hospitals, ~100× volume range, metro tertiary → rural PHC
  - Privacy metrics: GIA (DLG/InvertingGrad) + MIA (LiRA) before and after FedNeMo defences
  - Utility: MedQA accuracy; convergence vs centralised baseline; catastrophic-forgetting check (PIQA/ARC-Challenge)
  - Efficiency: bytes transmitted vs 32-bit FedAvg baseline over 50+ rounds

**What could still be added:**
🔲 Scale of datasets: how many rows, tokens, GB? Template asks for this. Add: MedQA-USMLE ~13K questions (US set); synthetic records generated to match ~50K–100K samples across 5 hospital shards.
🔲 Exact non-IID partition strategy: the 5-hospital table (H1 metro 50K notes, H2 community 15K, H3 research 30K, H4 rural PHC 3K, H5 specialty 8K). This is fully designed in your docs — worth adding.

**Source:** design.md §5.3, MedQA dataset documentation
**Judge expectation:** Do you have rights to the datasets? Do you know what "done" looks like?

---

## SLIDE 9 — Tech Stack (Section Divider)
**STATUS: ✅ Done (template divider, do not edit)**

---

## SLIDE 10 — Tech Stack: NVIDIA Integration
**STATUS: ✅ Done**

**What's in the deck:**
- "Fine-tuning (PEFT)? Yes — LoRA on Nemotron via NVIDIA NeMo 2.0."
- NVIDIA technology table (4 rows + header):
  | NVIDIA FLARE 2.5+ | DXO filter pipeline exact extension point | FedRandFilter, LaplacianDPFilter, AdaptiveQuantFilter as custom DXO Filter subclasses |
  | NVIDIA NeMo 2.0 | ModelTransform injects LoRA at Megatron-Core level | llm.peft.LoRA targeting attention + SSM projections; flare.patch(trainer) |
  | Nemotron-Mini-4B (demo) / Nemotron-3 (prod) | 256K context for patient histories; Hindi/multilingual; MoE routes specialties | Base model across 5 clients; Adapter targets: linear_qkv, in_proj, out_proj, dt_proj |
  | TensorRT-LLM / NIM | Post-training deployment | Inference optimisation (planned) |
- Prior experience note: team working with NeMo docs, FLARE Client API, HuggingFace transformers/PEFT
- Literature citations line

**What's NOT in the deck (template asks for):**
🔲 "What steps need to be / are done differently to develop and deploy your solution?" — Add: "Standard NeMo fine-tuning is single-site. FedNeMo adds a client-side DXO filter chain (FedRand→DP→Quant) that runs before each federated round, and replaces direct model sharing with fragmented, noised, compressed updates."
🔲 "Will you perform model training — Yes/No" — ✅ Already answered as "Yes"

**Source:** design.md §11, workflow.md, gemini.md §6
**Judge expectation:** Deep NVIDIA integration, not just name-dropping. Can the team explain WHY each SDK?

---

## SLIDE 11 — Tech Stack: Bottlenecks & How We Address Them
**STATUS: ✅ Done**

**What's in the deck:**
- 5-row bottleneck table:
  | DP noise destroys quality | Naive noise at strong ε collapses accuracy | FedRand eliminates cross-matrix amplification O(1/ε⁴)→O(1/ε²); entropy-weighted per-client ε |
  | Privacy budget grows unbounded | 100 rounds × ε=1.0 → ε_total=100 | RDP accounting with enforced ε_max ceiling; live dashboard |
  | Mamba-2 fused kernels bypass LoRA | selective_scan_cuda skips Python hooks | NeMo ModelTransform injects before kernel compilation; fallback hook; validation gate |
  | Non-IID drift across hospitals | Tertiary (50K notes) vs rural PHC (3K notes) | Shannon-entropy weighting; FedPS harmonises structured metadata |
  | FedPS scope: free text | FedPS = tabular only; discharge notes are unstructured | FedPS for structured EHR; shared tokenizer + canonical prompt for narratives |

**What's NOT in the deck (but template asks for):**
🔲 "Any similar work you encountered and took motivation from the literature?" — This is partially answered by Slide 7's table. Could add a one-liner here: "Most related: LA-LoRA (2026), FFA-LoRA (ICLR 2024), FedASK (NeurIPS 2025). FedNeMo differs by combining structural privacy defence (FedRand) with formal DP and targeting Nemotron's hybrid architecture."

**Source:** design.md §9, §18, judge-eval holes, all 5 papers
**Judge expectation:** Honest awareness of technical risks = trust. Shows you've thought beyond the happy path.

---

## SLIDE 12 — Road Map (Section Divider)
**STATUS: ✅ Done (template divider, do not edit)**

---

## SLIDE 13 — Roadmap: Build Plan & Mentor Support
**STATUS: ✅ Done**

**What's in the deck:**
- 4-phase build plan table:
  | Phase 1 (Weeks 1–2) | End-to-end federated LoRA on Nemotron-Mini-4B | FLARE simulation; NeMo LoRA on MedQA; FedAvg baseline |
  | Phase 2 (Weeks 3–4) | Privacy hardening | FedRand + DP + RDP as FLARE DXO filters; ε_total dashboard; GIA demo |
  | Phase 3 (Weeks 5–6) | Communication efficiency + data quality | Adaptive quant; FedPS; MTAE+OCSVM; non-IID evaluation |
  | Phase 4 (Week 7) | Evaluation + submission | Privacy-utility curve; MIA study; convergence comparison; Streamlit dashboard |
- Limitations (3 bullets):
  - FedRand MIA numbers from Park et al. are vision-domain → will measure on Nemotron text
  - MTAE accuracy gains (7%) from CIFAR-10 → will validate on clinical text
  - Nemotron-3 Nano (30B) = production target; demo uses Mini-4B
- Mentor support (3 asks):
  ① NeMo ModelTransform LoRA vs Mamba-2 kernel ordering
  ② FLARE Secure Aggregation integration
  ③ GPU access (A100/MI300X) for Nemotron fine-tuning

**What could still be added:**
🔲 Long-term goals (template asks for this): "Scale FedNeMo to 100+ hospital nodes. Publish as open-source NVIDIA FLARE extension. Submit to MLSys / NeurIPS. Apply to financial services (RBI data constraints) and legal sectors."
🔲 "Ideas to overcome limitations": brief statements beyond the 3 bullets — e.g., "For MIA on text, we will use the LiRA protocol with shadow models trained on MedQA subsets."

**Source:** design.md §14, judge-eval, MDs/01
**Judge expectation:** Specific plan = credible. Vague roadmap = rejected. Mentor asks show self-awareness.

---

## SLIDE 14 — Contact (Template-owned)
**STATUS: ✅ Done (do not edit)**
- "Need Help? Contact Us!! — Join the Slack Workspace here"

---

## SUMMARY: WHAT'S MISSING / NEEDS TO BE ADDED

### 🔲 Must-add before submission (template asks for it explicitly):
1. **Organisation details** (Slide 4): Institution name, student/academic/startup status
2. **Current AI deployment setup** (Slide 4): What tools you use today (even if basic)
3. **Dataset scale** (Slide 8): Exact row counts / GB / tokens for MedQA-USMLE and synthetic records
4. **Non-IID 5-hospital partition table** (Slide 8): H1–H5 with sizes and ICD profiles
5. **"What you do differently"** (Slide 10): One sentence on how this differs from standard NeMo fine-tuning
6. **Long-term goals** (Slide 13): Explicit statement about where FedNeMo goes after the hackathon
7. **Ideas to overcome limitations** (Slide 13): Brief per-limitation fix (template asks for this)

### ⚠️ Optional but strengthens the submission:
8. **"What makes this novel" callout** (Slide 7): No paper does all five components on Nemotron for Indian healthcare
9. **Related work one-liner** (Slide 11): LA-LoRA, FFA-LoRA, FedASK as what you differ from
10. **FedRE as future extension** (Slide 7 or 13): Shows you read more than you're building
11. **Past hackathon experience** (Slide 4): Add if any
12. **One-line tagline on title** (Slide 1): Optional but makes first impression stronger

### ✅ Everything else is in and correct:
- Project title and description ✅
- Problem framing with real context (ABDM, DPDP) ✅
- All 5 research papers cited with correct numbers ✅
- Architecture diagram and system flow ✅
- NVIDIA integration justified per technology ✅
- Privacy mechanism explained (FedRand + DP + RDP) ✅
- Dataset table with licences ✅
- Evaluation plan (GIA, MIA, utility, efficiency metrics) ✅
- TRACK: B marker ✅
- Bottlenecks with specific technical mitigations ✅
- 4-phase build plan ✅
- Mentor asks (3 specific) ✅
- Limitations acknowledged ✅
- 14 slides (within 15-slide cap) ✅
- Template structure preserved ✅

---

## PAPER CITATIONS IN THE DECK

| Slide | Claim | Paper | Verified from PDF |
|---|---|---|---|
| S7 | Up to 7.02% accuracy gain on non-IID (CIFAR-10) | Ardıç & Genç, MTAE paper | ✅ Yes — abstract states this exactly |
| S7 | 45–52% communication reduction | Ardıç & Genç, AdaptQ+DP paper | ✅ Yes — abstract: 52.64% MNIST, 45.06% CIFAR-10, 31–37% medical imaging |
| S7 | FedRand: MIA resistance, accuracy maintained | Park et al. KAIST 2025 | ✅ Yes — abstract confirms empirical validation |
| S7 | FedPS: consistent preprocessing for heterogeneous FL | Xu & Cormode Feb 2026 | ✅ Yes — abstract states this is the contribution |
| S7 | GIA taxonomy: OP-GIA, GEN-GIA, ANA-GIA | Guo et al. GIA paper | ✅ Yes — abstract categorises all three |
| S11 | O(1/ε⁴) → O(1/ε²) noise amplification | FedRand + DP composition (design.md Theorem 2) | Derived from papers, not directly stated |
| S13 | 7% accuracy gain from MTAE | Ardıç & Genç MTAE paper | ✅ Yes — abstract states "up to 7.02%" |
