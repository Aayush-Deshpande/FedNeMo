# FedNeMo: Project Scope and Architectural Design

This document expands on the complete project scope, detailing the architectural data flow, how algorithms are executed step-by-step, and addressing specific questions regarding model capability and data security.

---

## 1. Project Scope and Objective
The core objective of FedNeMo is to enable multiple isolated healthcare institutions to collaboratively train an NVIDIA Nemotron Large Language Model without ever sharing raw patient data. The system must natively handle wildly different data types (Non-IID drift), compress the massive communication payloads of LLM training, and guarantee mathematical resistance against Gradient Inversion Attacks and data poisoning.

---

## 2. Core Architecture and Data Flow

It is critical to understand that **raw data never flows across the network**. Only mathematical model updates (gradients) are transmitted.

### The Algorithm: Step-by-Step Execution
When a training round begins, the system executes the following deterministic steps:

1. **Stage 1: Outlier Detection (Data Poisoning Check)**
   - *Local Action:* Each hospital passes its raw data through a lightweight Multi-Task Autoencoder (MTAE). 
   - *Filtering:* A One-Class Support Vector Machine (OCSVM) analyzes the reconstruction loss. Any data point that looks anomalous, corrupted, or maliciously poisoned is strictly pruned and discarded.
2. **Stage 2: Federated Preprocessing (FedPS)**
   - *Local Action:* Hospitals compute abstract statistical summaries of their data (e.g., minimums, maximums, frequent item sketches for billing codes).
   - *Server Action:* The server aggregates these abstract statistics to create global transformation rules, and sends the rules back to the hospitals.
   - *Local Action:* Hospitals apply these rules so all data is formatted identically (harmonization).
   - *Why do this?* It solves the "Statistical Heterogeneity" problem. Hospitals use different measurement units (e.g., mg/dL vs mmol/L) and codes. Unstandardized data confuses the LLM. FedPS ensures the model receives mathematically uniform data from every hospital, without ever transmitting raw, private patient records.
3. **Stage 3: Local Fine-Tuning**
   - *Local Action:* The Nemotron model trains on the clean, harmonized local data using Low-Rank Adaptation (LoRA). Instead of updating all 30 billion parameters, it only updates tiny LoRA matrices ($A$ and $B$).
4. **Stage 4: Secure Parameter Transmission**
   - *FedRand Protocol:* Instead of sending both $A$ and $B$ matrices (which exposes the model to Gradient Inversion Attacks), the hospital flips a coin and randomly selects only *one* matrix to send.
   - *Differential Privacy:* Laplacian noise is mathematically injected into the chosen matrix to obfuscate any remaining individual patient traces.
   - *Adaptive Quantization:* The noised matrix is compressed to reduce bandwidth payload.
   - *Transmission:* The compressed matrix is sent to the central NVIDIA FLARE server.
5. **Stage 5: Global Aggregation**
   - *Server Action:* The central server decrypts and averages the incoming LoRA updates from all hospitals into a single, smarter global model update.
   - *Broadcast:* The server sends this updated global model back down to the hospitals, and the next round begins.

---

## 3. Addressing Single Models vs. Multiple Departments

**Question:** *Are Nemotron models capable of handling all departments (Oncology, Cardiology, Pediatrics) at the same time, or do we have to create an instance of a model for each department?*

**Answer:** You only need **one single instance** of the model per hospital to handle all departments. 

You do not need separate models for different departments because the Nemotron-3 Nano architecture utilizes a **Mixture-of-Experts (MoE)** design. 
- In standard models, every parameter is used for every word. 
- In an MoE model, the neural network acts as a router. When it reads a Cardiology report, it routes the text specifically to "Expert 1". When it reads an Oncology report, it routes it to "Expert 4". 
- Therefore, a single foundational Nemotron model has the internal capacity to learn and compartmentalize knowledge from entirely different hospital wings simultaneously. 

---

## 4. Addressing Data Poisoning and Anomalies

**Question:** *How are we going to handle the data poisoning section? Do we use Nemotron? Do we use an LSTM to check the nature of the data and isolate bad nodes?*

**Answer:** Data poisoning is handled **before** the Nemotron model ever sees the data. 

1. **Nemotron does not handle this natively.** If you feed poisoned data to an LLM, it will simply learn the poisoned data (commonly known as the "garbage in, garbage out" problem).
2. **We do not use an LSTM.** LSTMs are computationally heavy, sequential models that are outdated for general structural anomaly detection.
3. **The FedNeMo Solution:** As detailed in *Stage 1* of the pipeline above, the framework uses a **Multi-Task Autoencoder (MTAE)** combined with a **One-Class Support Vector Machine (OCSVM)**. 
   - The MTAE attempts to reconstruct the data.
   - Poisoned, corrupted, or wildly out-of-distribution data is mathematically difficult to reconstruct, resulting in a "high reconstruction loss."
   - The OCSVM looks at these loss scores. If a node (or a specific patient record) produces a bizarre loss score, the OCSVM immediately flags it as an anomaly and strictly isolates/prunes it from the training pool. This ensures the Nemotron model only ever trains on verified, high-quality clinical data.
