# Model fine-tuning and analysis

This document describes how to run the training and analysis scripts. Configs live under `configs/training/`, and scripts live under `scripts/training/`.

## Recommended workflow
1) Pick a config shell script under `configs/training/`.
2) Review and adjust environment variables (GPU selection, HF cache paths).
3) Run the config script from the repo root.

Example:
```bash
bash configs/training/11_embedding_rank.sh
```

## Training and analysis scripts
Common scripts include:
- `scripts/training/11_embedding_rank.py`
- `scripts/training/12_analysis_rank.py`
- `scripts/training/13_rp_highlow_ratio.py`
- `scripts/training/13b_rp_highlow_ratio.py`
- `scripts/training/13c_avg_scores.py`
- `scripts/training/14_retrieval_bias_analysis.py`
- `scripts/training/15_retrieval_bias_polar.py`
- `scripts/training/16_score_label_context_pairs.py`

Run `--help` on any script for CLI options:
```bash
python scripts/training/11_embedding_rank.py --help
```

## Outputs
Most scripts write to `outputs/` and log progress to stdout. The exact output paths are printed by each script and should be captured when reproducing figures.
