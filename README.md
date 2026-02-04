# With Argus Eyes

This repository provides **dataset construction**, **model fine-tuning**, and **figure reproduction** for the With Argus Eyes paper. The goal is end-to-end reproducibility of the results reported in the manuscript.

## What is included
- Dataset construction pipeline (Wikidata/Wikipedia processing).
- Model fine-tuning and analysis scripts.
- Figure reproduction pipeline for all paper figures.
- Processed dataset versioned in git under `data/processed/8_Emb_Rank`.

## What is not included
- A RAG pipeline. This work focuses on the dataset and model contributions. A RAG system is intentionally out of scope.

## Quickstart
1) Create a Python environment (>= 3.9) and install dependencies:
   - `pip install -e .`
2) Review the dataset pipeline:
   - `docs/DATASET.md`
   - Runnable configs: `configs/dataset/`
3) Review model training and analysis:
   - `docs/MODELS.md`
4) Reproduce figures:
   - `docs/REPRODUCE.md`

## Reproduce the paper
All steps and figure mappings are documented here:
- `docs/REPRODUCE.md`

## Compute requirements
- Embedding generation requires GPU(s) and depends on the embedding model.
- For the paper, most runs used **2x A6000** or **A100** GPUs.
- The full Argus pipeline with Qwen-30B requires higher-memory GPUs; most experiments do not.

## Data policy (tracked processed data)
- `data/processed/8_Emb_Rank` is versioned in git and can make the repo large.
- If you only want code, use sparse checkout to avoid pulling the dataset folder:
  ```bash
  git clone --filter=blob:none --no-checkout <REPO_URL>
  cd <REPO_DIR>
  git sparse-checkout init --cone
  git sparse-checkout set configs docs scripts src README.md data/README.md
  git checkout
  ```
- If you do need the dataset, verify integrity via hashes:
  - `shasum -a 256 data/processed/8_Emb_Rank/*.jsonl | shasum -a 256 -c data/processed/8_Emb_Rank/SHA256SUMS`

## Tests
Minimal smoke tests (no heavy compute):
- `pip install pytest`
- `pytest -q`

## Citation
Please cite the paper if you use this repository:

```bibtex
@article{with-argus-eyes,
  title     = {With Argus Eyes},
  author    = {TBD},
  journal   = {TBD},
  year      = {2026},
  doi       = {TBD},
  url       = {TBD}
}
```

A `CITATION.cff` file is provided for reference managers.

## License
MIT. See `LICENSE`.
