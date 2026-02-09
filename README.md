# ![Argus Logo](Argus_Logo.png) With Argus Eyes

**With Argus Eyes** is the official repository for the paper. It provides dataset construction, model training/analysis, and figure reproduction with an emphasis on end‑to‑end reproducibility.

## At a glance
- Dataset pipeline (Wikidata/Wikipedia): `configs/dataset/`, `scripts/dataset/`, `docs/DATASET.md`
- Training and analysis: `configs/training/`, `scripts/training/`, `docs/MODELS.md`
- Figure reproduction: `docs/REPRODUCE.md`, outputs in `outputs/`
- Data layout and policy: `data/README.md`
- Source package: `src/with_argus_eyes/`
- Tests: `tests/`

## Requirements
- Python `>=3.9`

## Quickstart
```bash
pip install -e .
```

## Dataset
The dataset pipeline builds curated labels and context pairs from Wikidata/Wikipedia. The runnable pipeline lives in `configs/dataset/` (e.g., `run_all.sh`) with Python implementations in `scripts/dataset/`. See `docs/DATASET.md` for the full workflow.

## Training and analysis
Model training and analysis steps are orchestrated via `configs/training/` and implemented in `scripts/training/`. See `docs/MODELS.md` for the experimental setup and analysis details.

## Reproducing figures
Figure mappings and exact reproduction steps are documented in `docs/REPRODUCE.md`. Outputs and plots are stored under `outputs/`.

## Tests
```bash
pytest -q
```

## Notes
Argus details will be available.

## License
MIT. See `LICENSE`.
