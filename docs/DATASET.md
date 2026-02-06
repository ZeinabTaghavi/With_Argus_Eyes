# Dataset construction

This document describes the end-to-end dataset construction pipeline. All scripts live in `scripts/dataset/` and write intermediate outputs under `data/interim/` and final outputs under `data/processed/`.

## Configs
Runnable configs are provided in `configs/dataset/`:
- `configs/dataset/run_all.sh` runs the full pipeline.
- `configs/dataset/00_env.sh` contains shared environment variables.

## Pipeline overview (in order)
1) **Collect items from Wikidata**
   - Script: `scripts/dataset/01_get_items.py`
   - Output: `data/interim/1_*` (see script defaults)

2) **Fetch related tags**
   - Script: `scripts/dataset/02_get_tags.py`
   - Output: `data/interim/2_*`

3) **Clean and deduplicate**
   - Script: `scripts/dataset/03_cleaning.py`
   - Output: `data/interim/3_*`

4) **Fetch Wikipedia summaries**
   - Script: `scripts/dataset/04_wikipedia_parse.py`
   - Output: `data/interim/4_*`

5) **Second-depth tags (optional)**
   - Script: `scripts/dataset/05_get_tags_second_depth.py`
   - Output: `data/interim/5_*`

6) **Final cleaning + dataset export**
    - Script: `scripts/dataset/06_final_cleaning.py`
    - Outputs: `data/interim/6_*` and final processed outputs under `data/processed/` (see `data/processed/8_Emb_Rank/`)

7) **Build all relation tags**
   - Script: `scripts/dataset/07_build_all_relation_tags.py`
   - Output: `data/interim/6_all_relation_tags.json`

8) **Add irrelevant tags**
   - Script: `scripts/dataset/08_add_unrelevants.py`
   - Output: `data/interim/6_unrelevant_qids.jsonl`

9) **Filter irrelevant tags with Wikipedia**
    - Script: `scripts/dataset/09_unrelevant_with_wiki.py`
    - Output: `data/interim/6_wiki_unrelevants_results.jsonl`

## Notes
- Individual scripts have CLI arguments for overriding paths. Use `--help` to view options.
- API calls depend on Wikidata/Wikipedia availability; cached outputs are recommended.
- The active pipeline is Wikipedia-first-paragraph based; no Wikidata-description augmentation step is used.
