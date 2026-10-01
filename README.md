# Cross-Model Latent Memory Transfer (CMLMT)

**Author:** Arjun Vinod Patil & Anuj Dalvi  
**Status:** Active Research / Experimental Setup  
**License:** MIT

---

## 📌 Overview

This project investigates whether task-relevant factual and compositional information stored inside a Small Language Model (SLM) can be extracted, compressed, and transferred to a separate, stateless model **without providing the original text, conversation history, or context documents**.

### The Core Hypothesis

> Can information stored in a language model's internal representations be extracted and used as a compact form of memory by another model?

The project tests how much information can be retained after extraction and compression, and whether the extracted representation can be used by a different model.

---

## 🏗️ Conceptual Architecture

```text
         ┌──────────────────────────────────────┐
         │         Source SLM (Model A)         │
         │   [Context / Prompt / Fact Input]    │
         └──────────────────┬───────────────────┘
                            │
                            ▼
                 Hidden Representations (H)
                            │
                            ▼
                  Memory Extraction
                         M = f(H)
                            │
                            ▼
                    Latent Memory (M)
                            │
       ═════════════════════╧═════════════════════
         Original text, KV cache, and history
                    are discarded
                            │
                            ▼
         ┌──────────────────────────────────────┐
         │        Target Model (Model B)        │
         │             [Stateless]              │
         └──────────────────┬───────────────────┘
                            │
                  Input: Query + Memory M
                            │
                            ▼
                       Target Answer
```

---

## 🔬 Research Questions

### Primary Question

How much task-relevant factual and compositional information can be extracted from one language model's internal activations and used by a separate, stateless model through a compressed representation?

### Secondary Questions

* **Compression & Capacity:** How small can the memory representation become before performance drops significantly?

* **Layer Sensitivity:** Which internal layers produce the most useful representations for transfer?

* **Cross-Architecture Decodability:** Can a representation extracted from Model A (e.g., Pythia-410M) be used by Model B (e.g., Qwen2.5-0.5B or LLaMA-3.2-1B)?

* **Information Types:** How does performance change for factual key-value pairs, relational information, compositional questions, and conflicting data?

* **Memory Interference:** Does combining multiple memories cause them to interfere with each other?

---

## 🧪 Experimental Pipeline

```text
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌──────────────────┐
│ 1. Source SLM   │ ──► │ 2. Controlled    │ ──► │ 3. Latent       │ ──► │ 4. Context       │
│    Selection    │     │    Dataset       │     │    Extraction   │     │    Removal       │
└─────────────────┘     └──────────────────┘     └─────────────────┘     └──────────────────┘
                                                                                 │
                                                                                 ▼
                                                   ┌─────────────────┐     ┌──────────────────┐
                                                   │ 6. Quantitative │ ◄── │ 5. Target       │
                                                   │    Evaluation   │     │    Injection     │
                                                   └─────────────────┘     └──────────────────┘
```

### Step 1: Source Model & Activation Extraction

Select lightweight, locally runnable open-weight SLMs (100M–500M parameters) so that internal activations can be inspected without heavy infrastructure:

* Hidden layer states $H \in \mathbb{R}^{L \times T \times D}$
* Multi-Head Attention key/value matrices
* MLP intermediate activations

### Step 2: Controlled Benchmark Dataset

To separate memory retrieval from information already present in the model, the evaluation will use synthetic key-value environments alongside modified factual benchmarks:

| Dataset Component        | Example / Description                                                    | Evaluation Focus                    |
| :----------------------- | :----------------------------------------------------------------------- | :---------------------------------- |
| **Synthetic Entities**   | `Person: Alice \| City: Berlin \| Age: 27 \| Preferred Language: Python` | Exact-match retrieval               |
| **Compositional Chains** | `Alice works with Bob -> Bob lives in Tokyo`                             | Multi-hop relational inference      |
| **Paraphrased Queries**  | *"Where does Alice reside?"* vs *"What is Alice's city?"*                | Effect of query wording             |
| **Conflicting Facts**    | `Fact T1: Alice lives in Berlin` vs `Fact T2: Alice moved to Kyoto`      | Updating and overriding information |

### Step 3: Baseline Comparisons

CMLMT is evaluated against five setups:

1. **Zero-Context Baseline:** $\text{Query} \rightarrow \text{Target Model} \rightarrow \text{Answer}$
   Measures how well the target model performs without additional information.

2. **Full-Context Baseline:** $\text{Prompt } [\text{Context} + \text{Query}] \rightarrow \text{Target Model} \rightarrow \text{Answer}$
   Provides the original information directly to the target model.

3. **RAG Baseline:** $\text{Query} \rightarrow \text{Vector Database} \rightarrow \text{Retrieved Text} \rightarrow \text{Target Model}$
   Provides a standard retrieval-based comparison.

4. **KV Cache Transfer:** Persistent state transfer across identical model instances, where feasible.

5. **Latent Transfer (Ours):** $[\text{Query} + \text{Compressed Memory } M] \rightarrow \text{Target Model} \rightarrow \text{Answer}$
   The main experimental setup.

### Step 4: Memory Extraction & Compression

Let $H_l$ denote the activation tensor at layer $l$ produced by the source model when reading context $C$.

The extracted memory $M$ is:

\(M = f(H_l)\)

Several extraction methods will be tested:

* **Sequence Pooling:** $M = \text{MeanPool}(H_l)$ or $M = \text{MaxPool}(H_l)$ across token length $T$.

* **Dimensional Projection:** $M = W_p \cdot H_l + b_p$ using a linear projection to reduce the representation size.

* **Soft-Prompt Prefixing:** Mapping hidden states to soft prompt tokens that can be passed to the target model.

* **Autoencoded Representations:** Compressing activations using a trained Sparse Autoencoder (SAE).

### Step 5: Context Isolation

Before evaluating the target model:

* Discard the original input text $C$.
* Clear the source model's KV cache.
* Run the target model without previous conversation state.

The target model should only have access to the query $Q$ and extracted memory $M$.

### Step 6: Target Decoding & Evaluation

The target model receives only the query $Q$ and compressed memory $M$.

Metrics include:

* **Exact Match (EM)** and **F1 Score** for factual retrieval.

* **Information Retention Ratio ($\text{IRR}$):**

\(\text{IRR} = \frac{\text{Accuracy}_{\text{CMLMT}}}{\text{Accuracy}_{\text{Full Context}}}\)

* **Compression Ratio ($\text{CR}$):**

\(\text{CR} = \frac{\text{Size of Raw Context Tokens}}{\text{Size of Latent Memory } M}\)

---

## 🗺️ Project Roadmap

* [x] **Phase 0:** Project scoping, initial formulation, and architecture design.
* [ ] **Phase 1:** Implement activation extraction using PyTorch / HuggingFace hooks.
* [ ] **Phase 2:** Build synthetic dataset generator and single-model benchmark.
* [ ] **Phase 3:** Test memory extraction methods ($f(H)$), starting with pooling and linear projections.
* [ ] **Phase 4:** Test transfer between different models (Source Model $\neq$ Target Model).
* [ ] **Phase 5:** Compare results with the selected baselines.
* [ ] **Phase 6:** Analyze results and document findings.

---

## 📄 Citation & Attribution

If you reference or build upon this project, please cite this repository:

```bibtex
@misc{patil2026cmlmt,
  author       = {Arjun Vinod Patil},
  title        = {Cross-Model Latent Memory Transfer: Extracting and Transferring Internal Representations Across Stateless Language Models},
  year         = {2026},
  howpublished = {\url{https://github.com/arjun05-tf/cross-model-latent-memory}},
  note         = {GitHub repository}
}
```
