# Dataset configs

This folder provides runnable configuration scripts for the dataset construction pipeline. Each script calls a corresponding file in `scripts/dataset/` with explicit paths.

## Usage
Run individual steps from the repo root:
```bash
bash configs/dataset/01_get_items.sh
```

Or run the full pipeline:
```bash
bash configs/dataset/run_all.sh
```

## Shared environment
`00_env.sh` defines common variables:
- `ARGUS_DATA_ROOT` (defaults to `./data`)
- `ARGUS_INTERIM_ROOT`
- `ARGUS_PROCESSED_ROOT`
- `WIKIDATA_DB_PATH`
- `WIKI_MAX_WORKERS`, `WIKI_CHUNKSIZE`, `WIKI_CHUNK_SIZE`
- `ARGUS_HF_BASE`

Adjust these before running if needed:
```bash
export ARGUS_DATA_ROOT=/path/to/data
export WIKI_MAX_WORKERS=16
```
