# With Argus Eyes

This repository provides the dataset construction and model fine‑tuning code, along with scripts to reproduce the figures reported in the paper.

## Repository structure

- `configs/` configuration files for dataset construction and training runs
- `scripts/` runnable scripts for dataset construction and training/analysis
- `src/with_argus_eyes/` reusable library code
- `data/` raw/interim/processed datasets (with `data/processed/8_Emb_Rank` tracked)
- `outputs/` (not committed) generated figures, logs, and artifacts
- `paper/` final figures/tables used in the manuscript

## Quickstart

1) Create an environment and install dependencies.
   - `pip install -e .`
2) Run dataset construction scripts from `scripts/dataset/`.
3) Run training/analysis scripts from `scripts/training/`.
4) Copy final figures into `paper/figures/`.

See individual scripts/configs for details.
