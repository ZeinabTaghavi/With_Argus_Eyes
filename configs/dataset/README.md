# Dataset Config Scripts

This folder contains runnable shell configs for the dataset-construction pipeline.
Each `*.sh` script maps to a stage in `scripts/dataset/`.

## Full Pipeline
Run all dataset stages in order:
```bash
bash configs/dataset/run_all.sh
```

`run_all.sh` executes:
1. `01_get_items.sh`
2. `02_get_tags.sh`
3. `03_cleaning.sh`
4. `04_wikipedia_parse.sh`
5. `05_get_tags_second_depth.sh`
6. `06_final_cleaning.sh`
7. `07_build_all_related_tags.sh`
8. `08_add_unrelevants.sh`
9. `09_unrelevant_with_wiki.sh`
10. `10_clear_directories.sh`

Stage 10 moves the final dataset artifacts from `data/interim/` to `data/processed/` using canonical names:
- `data/processed/main_dataset.jsonl`
- `data/processed/wikipedia_all_relevant_results.jsonl`
- `data/processed/wiki_unrelevants_results.jsonl`

## Optional Interim Cleanup
To clear temporary files in `data/interim/` after finalization:
```bash
RUN_CLEAR_INTERIM=true bash configs/dataset/run_all.sh
```

Keep specific files/folders while clearing:
```bash
RUN_CLEAR_INTERIM=true RUN_KEEP_INTERIM="wikidata_random.db,2_tag_only" bash configs/dataset/run_all.sh
```

## Run Single Stage
Run an individual stage from repo root:
```bash
bash configs/dataset/08_add_unrelevants.sh
```

## Shared Environment
`00_env.sh` defines common variables:
- `ARGUS_DATA_ROOT` (default: `./data`)
- `ARGUS_INTERIM_ROOT`
- `ARGUS_PROCESSED_ROOT`
- `WIKIDATA_DB_PATH`
- `WIKI_MAX_WORKERS`, `WIKI_CHUNKSIZE`, `WIKI_CHUNK_SIZE`
- `ARGUS_HF_BASE`

Example overrides:
```bash
export ARGUS_DATA_ROOT=/path/to/data
export WIKI_MAX_WORKERS=16
bash configs/dataset/run_all.sh
```

## Handoff To Training
After `run_all.sh` completes, training scripts start from:
- `scripts/training/11_embedding_rank.py`
