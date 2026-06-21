# Verified Reproducibility Package

This document is the canonical guide for reproducing the **paper validation pipeline**
(Cohen's κ, confusion matrix, and BERTopic assignments reported in the manuscript).
Repository: [github.com/Aslmalki/NLP-to-Modeling-pipeline](https://github.com/Aslmalki/NLP-to-Modeling-pipeline)

## Verification status

Under the pinned configuration below, three independent end-to-end runs were executed
from scratch on **2026-06-21**. All three produced **identical** results:

| Run | Cohen's κ | Paired documents (n) |
|-----|-----------|----------------------|
| 1 | 0.7078414839797639 | 63 |
| 2 | 0.7078414839797639 | 63 |
| 3 | 0.7078414839797639 | 63 |

Recorded in `outputs/validation/determinism_verification.json`.
Command: `python run_three_pipeline_kappa_check.py`

Reference run metadata: `outputs/validation/methodology_summary.json`  
Per-class metrics: `outputs/validation/classification_report.txt`

---

## What to run (paper vs. supplementary)

| Script | Purpose |
|--------|---------|
| **`run_blue_ocean_pipeline.py`** | **Paper validation** — Doc2Vec, BERTopic (min topic size 4), LLM codifier, Cohen's κ, confusion matrix |
| `run_three_pipeline_kappa_check.py` | Determinism check — three fresh runs; exits 0 only if κ matches |
| `run_pipeline.py` | **Supplementary only** — embedding ablation, co-occurrence networks, extra figures (not the κ path) |

Do **not** use `run_pipeline.py` alone if your goal is to reproduce κ = 0.7078.

---

## Prerequisites checklist

1. **Python 3.11.9** (see `.python-version`)
2. Install: `pip install -r requirements_pinned.txt`
3. **Corpus** — place 108 `.txt` files in `data/corpus/`  
   (see `data/corpus/README.md`; not redistributed due to copyright)
4. **Annotations** — `data/annotations/HumanBehaviorInExtremeEnvironments.csv`
5. **Codebook** — `data/codebook/topic_definitions.csv`
6. **Validation subset** — `data/validation_subset_ids.txt` (63 Paper IDs; do not edit)
7. **API key** — `CLAUDE_API_KEY` in `.env` (see `.env.example`)

---

## Step-by-step reproduction

```bash
git clone https://github.com/Aslmalki/NLP-to-Modeling-pipeline.git
cd NLP-to-Modeling-pipeline
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements_pinned.txt
cp .env.example .env   # add CLAUDE_API_KEY
# Place corpus + CSV as described above
python run_blue_ocean_pipeline.py
```

Optional determinism verification:

```bash
python run_three_pipeline_kappa_check.py
```

---

## Pinned methodology parameters

All constants live in `config.py` and are logged in `methodology_summary.json` after each run.

### Random seeds
- `REPRODUCIBILITY_SEED = 42` (Python, NumPy, PyTorch, UMAP via `set_all_seeds()`)
- `DOC2VEC_SEED = 42`

### Doc2Vec (`src/embeddings.py`, used by `run_blue_ocean_pipeline.py`)
| Parameter | Value |
|-----------|-------|
| vector_size | 100 |
| window | 5 |
| min_count | 1 |
| epochs | 20 |
| workers | 1 |
| seed | 42 |

### BERTopic / HDBSCAN / UMAP (`run_blue_ocean_pipeline.py`)
| Parameter | Value |
|-----------|-------|
| embedding | Doc2Vec |
| min_topic_size / min_cluster_size | 4 |
| UMAP n_neighbors | 15 |
| UMAP n_components | 5 |
| UMAP min_dist | 0.0 |
| UMAP metric | cosine |
| UMAP random_state | **42** (explicit) |
| HDBSCAN metric | euclidean |
| HDBSCAN cluster_selection_method | eom |
| reduce_outliers | strategy `c-tf-idf`, threshold **0.1** |

### LLM expert codifier (Module A)
| Parameter | Value |
|-----------|-------|
| Model | **`claude-sonnet-4-6`** (single pinned model) |
| temperature | 0 |
| Gemini fallback | **disabled** (`ENABLE_GEMINI_FALLBACK = False`) |
| Per-document model log | column `llm_model_used` in codifier / validation CSVs |
| Prompts | inline in pipeline; templates in `prompts/` |

If the Claude API fails, the run **stops with an error** (no silent model switching).

### Cohen's κ (Module B)
- Function: `sklearn.metrics.cohen_kappa_score` (default, unweighted)
- Labels: integer prefix before `_` in topic names; outliers → -1
- **Scope:** only the 63 Paper IDs in `data/validation_subset_ids.txt`
- κ compares LLM-coded expert labels vs. BERTopic topic IDs on those 63 pairs

---

## Pipeline modules (paper path)

```
Module 0  Load & preprocess 108 corpus documents
Module 1  Doc2Vec embeddings (seed=42, workers=1)
Module 2  BERTopic + reduce_outliers → topic assignments (full corpus)
Module A  LLM codifier — classify expert annotation text (Discussion/Details)
Module B  Cohen's κ, classification report, confusion matrix (n=63 subset)
Module C  Latent variable prototypes (optional; use --skip-latent to skip)
```

---

## Expected outputs (after `run_blue_ocean_pipeline.py`)

| File | Content |
|------|---------|
| `outputs/validation/methodology_summary.json` | Run metadata, seeds, model, κ |
| `outputs/validation/validation_labels.csv` | 63 paired rows + `llm_model_used` |
| `outputs/validation/module_a_codifier_output.csv` | Full codifier output |
| `outputs/validation/classification_report.txt` | Per-topic precision/recall (n=63) |
| `outputs/figures/confusion_matrix.png` | Normalized confusion matrix |
| `outputs/topic_assignments/document_topic_assignments.csv` | Corpus-wide BERTopic IDs |

**Success criteria:** `cohen_kappa` ≈ **0.7078414839797639**, `kappa_paired_rows` = **63**.

---

## Scope notes for reviewers

- **N = 108** documents in BERTopic; **n = 63** in κ / confusion matrix (fixed subset).
- **Corpus files** are not in this repository (copyright); request from corresponding author (see `data/corpus/README.md`).
- **`run_pipeline.py`** performs ablation over multiple embeddings; the **published final model** uses Doc2Vec @ min topic size 4 via `run_blue_ocean_pipeline.py`.
- **`src/validation.py`** implements thematic comparison only; LLM + κ is in `run_blue_ocean_pipeline.py`.

---

## Limitations (scientific, not documentation gaps)

- LLM classification requires a valid Anthropic API key and network access.
- Reproducibility is verified under the **pinned** model string and environment above; changing model, corpus, or subset file will change κ.
- Expert annotations are a collaborative consensus (Cohen's κ, not Fleiss' κ — see `methodology_summary.json`).
