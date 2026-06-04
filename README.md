# NLP-to-ABM Pipeline

Public reproducibility release for the ICE behavioral literature NLP pipeline:
BERTopic topic modeling, expert-annotation validation (Cohen's κ = 0.7078), and
published figures. Derived from `ice-nlp-paper-repro/`; source folder left unchanged.

## Layout

```text
nlp-to-abm-pipeline/
├── config.py
├── run_pipeline.py
├── requirements_pinned.txt
├── data/
│   ├── corpus/              # 108 preprocessed .txt documents
│   ├── annotations/         # HumanBehaviorInExtremeEnvironments.csv
│   └── codebook/            # topic_definitions.csv + codebook_description.md
├── prompts/                 # LLM prompts used in validation (verbatim)
├── outputs/
│   ├── topic_assignments/
│   ├── validation/
│   └── figures/             # Paper figures (incl. pipeline_architecture.pdf)
├── src/
└── notebooks/
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_pinned.txt
cp .env.example .env
# Set CLAUDE_API_KEY and optionally GEMINI_API_KEY in .env
```

## How to Reproduce

From the repository root:

```bash
python run_pipeline.py
```

Or open `notebooks/full_pipeline.ipynb` and run all cells top-to-bottom.

New runs write under `outputs/`. Reference artifacts from the paper are already
included under `outputs/topic_assignments/`, `outputs/validation/`, and
`outputs/figures/`.

## File Structure

```text
data/corpus/          ← 108 plain-text ICE documents (not tracked by git)
data/annotations/     ← expert annotation spreadsheet
data/codebook/        ← topic definitions used for LLM classification
prompts/              ← exact prompt templates used in the paper
src/                  ← pipeline source code (7 modules)
outputs/              ← generated results and figures
config.py             ← all parameters in one place
run_pipeline.py       ← single entry point to reproduce results
notebooks/            ← end-to-end walkthrough notebook
```

## Note on src/validation.py

This file contains the LLM-based classification and Cohen's kappa computation. It was adapted from `thematic_comparison.py` in the original codebase. The prompt templates it uses are stored in `prompts/` — do not modify those files if your goal is to reproduce the reported kappa = 0.7078 or closer to that.

## Configuration

All paths, seeds, and model constants live in `config.py` at the repository root.
Set `ENABLE_OPTIONAL_STEPS` in `run_pipeline.py` to `False` to skip network and
extra figure generation (faster; kappa-focused runs only).

## Important Reproducibility Notes

- The LLM validation step depends on external API routing. Three independent runs in the same session produced kappa = 0.7078 identically. A later rerun produced a slightly different value due to silent model version drift between sessions. This is documented in the paper (Section 5, Limitations). To reproduce kappa without re-running the LLM step, load the archived classifications from `outputs/validation/frozen_paper_run/` directly and run the kappa computation only. This bypasses the LLM entirely and will produce kappa = 0.7078 exactly.

## Data and copyright

See `data/corpus/README.md` and `data/annotations/README.md`. Do not commit `.env`.

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Abdullah Almalki, Anamaria Berea.
