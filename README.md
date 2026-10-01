# Cross-Model Latent Memory Transfer (CMLMT)

**Author:** Arjun Vinod Patil  
**Status:** Active Research / Experimental Setup  
**License:** MIT  

---

## 📌 Overview

This project investigates whether task-relevant factual and compositional information stored inside a Small Language Model (SLM) can be extracted, compressed, and transferred to an independent, stateless target model **without providing the original text, conversation history, or context documents**.

### The Core Hypothesis
> Can a language model's internal representations act as an explicit, transferable, and compressed form of memory across model boundaries?

Rather than assuming cross-model latent transfer is seamless, this project experimentally evaluates what information survives latent extraction, the trade-offs of representational compression, and whether extracted internal memories can be decoded by different model architectures.

---

## 🏗️ Conceptual Architecture

```text
         ┌──────────────────────────────────────┐
         │         Source SLM (Model A)         │
         │   [Context / Prompt / Fact Ingestion] │
         └──────────────────┬───────────────────┘
                            │
                            ▼
              Hidden Representations (H)
                            │
                            ▼
               Memory Extraction Function
                        M = f(H)
                            │
                            ▼
                 Compact Latent Memory (M)
                            │
       ═════════════════════╧═════════════════════  <-- Context & State Erasure Boundary
                            │                           (Original text, KV cache, and
                            ▼                           prompt history are discarded)
         ┌──────────────────────────────────────┐
         │        Target Model (Model B)        │
         │     [Stateless Query Decoder]        │
         └──────────────────┬───────────────────┘
                            │
                Input: [Query] + [Memory M]
                            │
                            ▼
                      Target Answer
```

---

## 🔬 Research Questions

### Primary Question
How much task-relevant factual and compositional information can be extracted from one language model's internal activations and successfully decoded by a separate, stateless model via a compressed latent representation?

### Secondary Questions
* **Compression & Capacity:** What is the minimal dimensional boundary $M$ before catastrophic information loss occurs?
* **Layer Sensitivity:** Which internal layers (early, intermediate, or deep MLP/Attention blocks) yield the highest fidelity transferable memory?
* **Cross-Architecture Decodability:** Can a latent memory extracted from Model A (e.g., Pythia-410M) be decoded by Model B (e.g., Qwen2.5-0.5B or LLaMA-3.2-1B)?
* **Information Taxonomy:** How does memory retention vary across factual key-value tuples, relational graphs, compositional logic, and noisy or conflicting data?
* **Memory Interference:** Does concatenating or blending multiple latent memories $M_1, M_2, \dots, M_k$ cause representational overlap or crosstalk?

---

## 🧪 Experimental Pipeline

```text
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌──────────────────┐
│ 1. Source SLM   │ ──► │ 2. Controlled    │ ──► │ 3. Latent       │ ──► │ 4. Context       │
│    Selection    │     │    Dataset       │     │    Extraction   │     │    Erasure       │
└─────────────────┘     └──────────────────┘     └─────────────────┘     └──────────────────┘
                                                                                  │
                                                                                  ▼
                                                 ┌─────────────────┐     ┌──────────────────┐
                                                 │ 6. Quantitative │ ◄── │ 5. Target        │
                                                 │    Evaluation   │     │    Injection     │
                                                 └─────────────────┘     └──────────────────┘
```

### Step 1 — Source Model & Latent Hooking
Select lightweight, locally runnable open-weights SLMs (100M – 500M parameters) to inspect and extract internal activations without heavy infrastructure:
* Hidden layer states $H \in \mathbb{R}^{L \times T \times D}$
* Multi-Head Attention key/value matrices
* MLP intermediate activations

### Step 2 — Controlled Benchmark Dataset
To isolate memory retrieval performance from pre-training prior knowledge, evaluations utilize synthetic key-value environments alongside perturbed factual benchmarks:

| Dataset Component | Example / Description | Evaluation Focus |
| :--- | :--- | :--- |
| **Synthetic Entities** | `Person: Alice \| City: Berlin \| Age: 27 \| Preferred Language: Python` | Exact-match retrieval |
| **Compositional Chains** | `Alice works with Bob -> Bob lives in Tokyo` | Multi-hop relational inference |
| **Paraphrased Queries** | *"Where does Alice reside?"* vs *"What is Alice's city?"* | Invariance to semantic query variations |
| **Conflicting Facts** | `Fact T1: Alice lives in Berlin` vs `Fact T2: Alice moved to Kyoto` | Recency, overriding, and update dynamics |

### Step 3 — Baseline Comparisons
CMLMT is evaluated against four standard operational paradigms:

1. **Zero-Context Baseline:** $\text{Query} \rightarrow \text{Target Model} \rightarrow \text{Answer}$ *(Measures pre-training bias)*
2. **Full-Context Baseline:** $\text{Prompt } [\text{Context} + \text{Query}] \rightarrow \text{Target Model} \rightarrow \text{Answer}$ *(Upper-bound performance)*
3. **RAG Baseline:** $\text{Query} \rightarrow \text{Vector Database} \rightarrow \text{Retrieved Text} \rightarrow \text{Target Model}$ *(Textual retrieval baseline)*
4. **KV Cache Transfer:** Persistent state transfer across identical model instances *(Structural upper bound where feasible)*
5. **Latent Transfer (Ours):** $[\text{Query} + \text{Compressed Memory } M] \rightarrow \text{Target Model} \rightarrow \text{Answer}$ *(Experimental condition)*

### Step 4 — Memory Extraction & Compression Formulations
Let $H_l$ denote the activation tensor at layer $l$ produced by the source model when reading context $C$. The extracted latent memory $M$ is defined by the transformation:

$$M = f(H_l)$$

Multiple extraction functions $f(\cdot)$ are benchmarked:
* **Sequence Pooling:** $M = \text{MeanPool}(H_l)$ or $\text{MaxPool}(H_l)$ across token length $T$.
* **Dimensional Projections:** $M = W_p \cdot H_l + b_p$ (Linear projection to a lower-dimensional embedding).
* **Soft-Prompt Prefixing:** Mapping hidden states to soft prompt tokens inserted directly into the target model's input space.
* **Autoencoded Representations:** Bottlenecking activations via a trained Sparse Autoencoder (SAE).

### Step 5 — Context Isolation Protocol
Prior to evaluation on the target model, strict state erasure is enforced:
* Discard original input text context $C$.
* Flush all source model KV cache memory buffers.
* Execute the target model in a fresh, completely stateless runtime instance.

### Step 6 — Target Decoding & Metric Evaluation
The target model receives only the Query $Q$ and the compressed latent tensor $M$. Evaluation metrics include:
* **Exact Match (EM)** and **F1 Score** for precise factual lookup.
* **Information Retention Ratio ($\text{IRR}$):**

$$\text{IRR} = \frac{\text{Accuracy}_{\text{CMLMT}}}{\text{Accuracy}_{\text{Full Context}}}$$

* **Compression Ratio ($\text{CR}$):** Ratio of raw context token bytes to the byte footprint of compressed latent memory $M$:

$$\text{CR} = \frac{\text{Size of Raw Context Tokens}}{\text{Size of Latent Memory } M}$$

---

## 🗺️ Project Roadmap

- [x] **Phase 0:** Project scoping, theoretical formulation, and architecture design.
- [ ] **Phase 1:** Implementation of activation hooking mechanisms (PyTorch / HuggingFace hooks).
- [ ] **Phase 2:** Synthetic dataset generator & single-model benchmark pipeline.
- [ ] **Phase 3:** Memory extraction function experiments ($f(H)$ pooling vs. linear projection layers).
- [ ] **Phase 4:** Cross-model transfer evaluation (Source Model $\neq$ Target Model) & publish findings.

---

## 📄 Citation & Attribution

If you reference or build upon this research proposal, please cite this repository:

```bibtex
@misc{patil2026cmlmt,
  author       = {Arjun Vinod Patil},
  title        = {Cross-Model Latent Memory Transfer: Extracting and Transferring Internal Representations Across Stateless Language Models},
  year         = {2026},
  howpublished = {\url{https://github.com/arjun05-tf/cross-model-latent-memory}},
  note         = {GitHub repository}
}
```