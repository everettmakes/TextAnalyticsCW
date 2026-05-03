# PICO Information Extraction from Biomedical Abstracts

A comparative study of three families of approaches — **unsupervised clustering**, **classical machine learning**, and **large language models** — for automatically extracting PICO elements (Population, Intervention, Outcome) from clinical trial abstracts.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Pipelines](#pipelines)
- [Evaluation](#evaluation)
- [Results Summary](#results-summary)
- [Usage](#usage)
- [Output Format](#output-format)

---

## Project Overview

Evidence-Based Medicine (EBM) requires systematically locating key clinical elements inside published research abstracts. This project benchmarks how well different NLP paradigms can extract the three core PICO fields:

| Field | Abbreviation | Description |
|---|---|---|
| Population | Pop | The patients or study subjects |
| Intervention | Int | The treatment or exposure being studied |
| Outcome | Out | The primary measured result |

Nine pipeline variants are evaluated end-to-end on the EBM-NLP 2.0 test set (184 abstracts), using a unified prediction-table format that enables fair cross-method comparison.

---

## Dataset

**EBM-NLP 2.0** — 4,457 training abstracts and 184 test abstracts from PubMed, each with token-level P/I/O annotations. Hierarchical labels are converted to binary masks (1 = PICO token, 0 = non-PICO). Raw data should be placed at `<repo_root>/ebm_nlp_2_00/`.

---

## Repository Structure

```
.
├── PrepareData.ipynb           # Parse raw EBM-NLP → .npz (spaCy embeddings)
├── Evaluation.ipynb            # Unified multi-metric evaluation notebook
├── README.md
│
├── Task1-Cluster/
│   ├── Initial_Clustering.ipynb      # KMeans over sentence embeddings
│   ├── Prepare_Data_nltk.ipynb       # Sentence splitting & embedding (NLTK variant)
│   └── cluster_kmeans_predictions.csv
│
├── Task2-MachineLearning/
│   ├── ML-Pipeline.py                # End-to-end + decomposed LinearSVC
│   ├── ml_end_to_end_predictions.csv
│   ├── ml_decomposed_predictions.csv
│   └── token_dataset_cache.npz       # Feature cache (generated on first run)
│
├── Task2-LargeLanguageModel/
│   ├── Single-step.ipynb             # Zero-shot extraction
│   ├── Multi-step.ipynb              # Dynamic few-shot (1–5 shots, RAG)
│   ├── llm_zero_shot_predictions.csv
│   └── llm_dynamic_{1-5}_shot_predictions.csv
│
└── evaluation_outputs/         # Auto-generated CSVs + plots
```

---

## Pipelines

### Task 1 — Clustering (Unsupervised)

**Notebooks:** `Task1-Cluster/Initial_Clustering.ipynb`, `Task1-Cluster/Prepare_Data_nltk.ipynb`

Abstract sentences are embedded with `all-MiniLM-L6-v2` and partitioned into clusters using KMeans (k=3). Clusters are then heuristically mapped to P/I/O fields based on token-level annotation rates. Requires no training labels; operates at sentence level. Silhouette score: 0.025 — clusters are not well-separated, which is expected given semantic overlap between PICO fields.

---

### Task 2 — Token-level ML Classifiers

**Script:** `Task2-MachineLearning/ML-Pipeline.py`

Two LinearSVC pipelines operating at token level with a context window of ±2 tokens (features: lowercased token, case flags, digit flag, token length, context neighbours).

**End-to-End:** Three independent binary classifiers, one per PICO category, trained on all tokens directly.

**Decomposed (two-step):** Step 1 identifies any PICO token vs. non-PICO; Step 2 classifies only the predicted PICO tokens into P/I/O categories. Contiguous predicted tokens are merged into phrase spans.

> A token-dataset cache (`token_dataset_cache.npz`) is written on first run to speed up subsequent executions.

---

### Task 2 — LLM-based Extraction

**Model:** `llama3.1` via [Ollama](https://ollama.com) (local, temperature=0)

**Zero-shot** (`Single-step.ipynb`): The model is prompted to return Population, Intervention, and Outcome as exact substrings of the abstract in structured JSON, with no examples.

**Dynamic Few-shot** (`Multi-step.ipynb`): For each test abstract, the `k` most similar training abstracts are retrieved via cosine similarity over pre-computed sentence embeddings and injected as in-context demonstrations. Shot counts evaluated: 1, 2, 3, 4, 5.

---

## Evaluation

**Notebook:** `Evaluation.ipynb`

Loads all `*_predictions.csv` files and computes metrics across multiple dimensions:

| Dimension | Metrics |
|---|---|
| **Field-level quality** | Token Precision / Recall / F1; Macro-F1; Micro-F1; Exact & Relaxed match |
| **Coverage** | Per-field and complete PIO row coverage |
| **Boundary quality** | Length ratio; over- and under-extraction rates |
| **Faithfulness** | Extractiveness rate; verbatim match rate; source hallucination rate |
| **Cross-field contamination** | Rate of predictions overlapping the wrong gold field |
| **Downstream query evaluation** | Term queries (e.g. "elderly patients", "placebo", "mortality"); P/R/F1/P@20 |
| **Robustness** | F1 stratified by abstract length and gold field density |

Outputs saved to `evaluation_outputs/`.

---

## Results Summary

Results on 184 test abstracts. Token F1 is the primary metric.

| Method | Pop F1 | Int F1 | Out F1 | Macro F1 | Hallucination | Coverage |
|---|---|---|---|---|---|---|
| `ml_token_decomposed` | 0.515 | **0.471** | **0.556** | **0.514** | 0.0% | 99.8% |
| `llm_dynamic_1_shot` | **0.560** | 0.440 | 0.505 | 0.502 | 5.0% | 98.7% |
| `llm_dynamic_2_shot` | 0.520 | 0.420 | 0.504 | 0.481 | 3.3% | 98.2% |
| `ml_token_end_to_end` | 0.497 | 0.400 | 0.505 | 0.467 | 0.0% | 99.8% |
| `llm_dynamic_3_shot` | 0.554 | 0.384 | 0.477 | 0.471 | 9.4% | 99.8% |
| `llm_dynamic_4_shot` | 0.496 | 0.368 | 0.454 | 0.440 | 10.0% | 99.3% |
| `llm_dynamic_5_shot` | 0.494 | 0.373 | 0.442 | 0.436 | 9.6% | 99.8% |
| `llm_zero_shot` | 0.512 | 0.336 | 0.344 | 0.397 | 0.6% | 97.1% |
| `cluster_kmeans` | 0.282 | 0.389 | 0.387 | 0.353 | 0.0% | 63.9% |

**Key findings:**

- **ML (decomposed)** achieves the best overall Macro F1 with near-perfect coverage and zero hallucination, though it tends to over-extract (median length ratio ~1.75×).
- **LLM (1-shot dynamic)** achieves the highest Population F1 and best precision–recall trade-off for Pop, but hallucination rate grows with shot count — more examples do not consistently help.
- **Zero-shot LLM** is competitive on Pop but struggles on Intervention and Outcome.
- **KMeans clustering** suffers from low coverage (~64%) and poor field separation, but never hallucinates.

---

## Usage

Requires Python 3.10+, plus `scikit-learn`, `sentence-transformers`, `langchain-ollama`, `spaCy`/`nltk`, `pandas`, and `ollama` (for LLM tasks).

**1. Prepare the data** — run `PrepareData.ipynb`. Generates `ebm_abstracts_full.npz` and `sentence_vectors.npz` under `ebm_nlp_2_00/processed/`.

**2. Run a pipeline:**
```bash
jupyter nbconvert --to notebook --execute Task1-Cluster/Initial_Clustering.ipynb
python Task2-MachineLearning/ML-Pipeline.py
jupyter nbconvert --to notebook --execute Task2-LargeLanguageModel/Single-step.ipynb
jupyter nbconvert --to notebook --execute Task2-LargeLanguageModel/Multi-step.ipynb
```

**3. Evaluate:**
```bash
jupyter nbconvert --to notebook --execute Evaluation.ipynb
```

---

## Output Format

All pipelines write predictions to a shared CSV schema:

| Column | Description |
|---|---|
| `method` | Pipeline identifier |
| `doc_id` | PubMed document ID |
| `text` | Full abstract text |
| `Pop_gold` / `Pop_pred` | Gold and predicted Population span(s), separated by ` ; ` |
| `Int_gold` / `Int_pred` | Gold and predicted Intervention span(s) |
| `Out_gold` / `Out_pred` | Gold and predicted Outcome span(s) |

Absent predictions are stored as `null`.
