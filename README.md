[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20542690.svg)](https://doi.org/10.5281/zenodo.20542690)
# NLP-to-Modeling Pipeline

Official reproducibility release for the ICE behavioral literature NLP pipeline
([GitHub](https://github.com/Aslmalki/NLP-to-Modeling-pipeline)): BERTopic topic modeling,
LLM expert-annotation validation (Cohen's κ), and published figures.

**This repository is the canonical source** for the manuscript submitted to the
Journal of Computational Social Science. Future updates land here first.

## Layout

```text
NLP-to-Modeling-pipeline/
├── config.py                      # paths, seeds, LLM model, validation subset
├── run_pipeline.py                # ablation + optional networks/figures
├── run_blue_ocean_pipeline.py     # paper validation: Doc2Vec, κ, confusion matrix
├── run_three_pipeline_kappa_check.py  # 3× determinism check
├── requirements_pinned.txt
├── .python-version                # 3.11.9
├── data/
│   ├── corpus/                    # 108 preprocessed .txt documents
│   ├── annotations/               # HumanBehaviorInExtremeEnvironments.csv
│   ├── codebook/                  # topic_definitions.csv
│   └── validation_subset_ids.txt  # fixed 63 Paper IDs for κ (do not edit casually)
├── prompts/                       # LLM prompt templates (verbatim)
├── outputs/
│   ├── topic_assignments/
│   ├── validation/                # κ, labels, methodology_summary.json
│   └── figures/
├── src/
└── notebooks/
```

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements_pinned.txt
cp .env.example .env
# Set CLAUDE_API_KEY in .env (required for LLM validation)
```

## What to run

| Goal | Command (from repository root) |
|------|--------------------------------|
| **Paper validation** (Doc2Vec, `min_topic_size=4`, Cohen's κ, confusion matrix) | `python run_blue_ocean_pipeline.py` |
| Ablation study + co-occurrence networks + extra figures | `python run_pipeline.py` |
| **Determinism check** (3 fresh runs; κ must match) | `python run_three_pipeline_kappa_check.py` |
| Notebook walkthrough | `notebooks/full_pipeline.ipynb` |

## Reproducibility (v1.1+)

- **Validation subset**: exactly **63 Paper IDs** in `data/validation_subset_ids.txt` (loaded at runtime; not computed dynamically).
- **LLM**: single pinned model `claude-sonnet-4-6`, `temperature=0`; no silent Gemini fallback in reported runs (`ENABLE_GEMINI_FALLBACK=False` in `config.py`).
- **Logging**: `llm_model_used` column in `outputs/validation/module_a_codifier_output.csv` and `validation_labels.csv`.
- **Environment**: Python **3.11.9**; pinned versions in `requirements_pinned.txt`.
- **Frozen paper artifacts**: `outputs/validation/frozen_paper_run/` (κ ≈ 0.708 reference run). Re-run `run_blue_ocean_pipeline.py` for fresh LLM labels under the pinned config above.

## Note on `src/validation.py`

This module implements **thematic comparison** (model topics vs. human summary text). The **LLM classification + Cohen's κ** path lives in `run_blue_ocean_pipeline.py`.

## Data and copyright

See `data/corpus/README.md` and `data/annotations/README.md`. Do not commit `.env`.

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Abdullah Almalki, Anamaria Berea.
