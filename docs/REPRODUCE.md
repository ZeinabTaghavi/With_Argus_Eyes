# Reproduce figures

This document explains how to reproduce the figures reported in the paper. The pipeline is:
1) Construct the dataset (see `docs/DATASET.md`).
2) Run training/analysis scripts (see `docs/MODELS.md`).
3) Generate figures and copy final artifacts into `paper/figures/`.

## Figure mapping
Update the figure numbers below to match the paper. The script names are stable and correspond to the analysis tasks.

| Paper figure | Script | Config | Expected output |
| --- | --- | --- | --- |
| Fig. X | `scripts/training/11_embedding_rank.py` | `configs/training/11_embedding_rank.sh` | `outputs/` (embedding rank plots) |
| Fig. X | `scripts/training/12_analysis_rank.py` | `configs/training/12_analysis_rank.sh` | `outputs/` (summary analysis) |
| Fig. X | `scripts/training/13_rp_highlow_ratio.py` | `configs/training/13_rp_highlow_ratio.sh` | `outputs/` (ratio plots) |

## Notes
- Most scripts print their output paths during execution. Capture logs for reproducibility.
- After generation, copy final figures into `paper/figures/` using the paper naming convention.
