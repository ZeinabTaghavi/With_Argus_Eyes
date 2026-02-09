# Data Directory

This directory stores dataset artifacts in two phases:
- `data/interim/`: temporary outputs from dataset-construction stages.
- `data/processed/`: finalized inputs for training and published outputs.

## Finalized Dataset Inputs (used by training)
After running `configs/dataset/run_all.sh`, these canonical files should exist:
- `data/processed/main_dataset.jsonl`
- `data/processed/wikipedia_all_relevant_results.jsonl`
- `data/processed/wiki_unrelevants_results.jsonl`

These are consumed by:
- `scripts/training/11_embedding_rank.py`
- `scripts/training/12_analysis_rank.py`

## Processed Training Outputs
`data/processed/8_Emb_Rank/` stores ranking outputs from:
- `scripts/training/11_embedding_rank.py`

Typical file pattern:
- `data/processed/8_Emb_Rank/8_main_dataset_{order}_{retriever}.jsonl`

## Tracked Published Data
`data/processed/8_Emb_Rank/` is versioned in git because it is part of the published dataset. This can make the repo large.

If you want to avoid downloading large files, use sparse checkout (see root `README.md`).

## Integrity Verification
A hash manifest is provided at:
- `data/processed/8_Emb_Rank/SHA256SUMS`

Verify files with:
```bash
shasum -a 256 data/processed/8_Emb_Rank/*.jsonl | shasum -a 256 -c data/processed/8_Emb_Rank/SHA256SUMS
```
