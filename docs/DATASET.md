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
    - Outputs: `data/interim/6_all_wikipedia_pages.jsonl`, `data/interim/6_main_dataset.jsonl`

7) **Build all related tags map**
   - Script: `scripts/dataset/07_build_all_related_tags.py`
   - Output: `data/interim/7_all_related_tags.json`

8) **Add irrelevant tags**
   - Script: `scripts/dataset/08_add_unrelevants.py`
   - Output: `data/interim/8_unrelevant_qids.jsonl`

9) **Filter irrelevant tags with Wikipedia**
    - Script: `scripts/dataset/09_unrelevant_with_wiki.py`
    - Output: `data/interim/9_wiki_unrelevants_results.jsonl`

10) **Finalize processed files + optional interim cleanup**
    - Script: `scripts/dataset/10_clear_directories.py`
    - Moves these files into `data/processed/` with canonical names:
    - `data/processed/main_dataset.jsonl`
    - `data/processed/wikipedia_all_relevant_results.jsonl`
    - `data/processed/wiki_unrelevants_results.jsonl`
    - Optional mode: `--clear_interim` removes remaining temporary files/folders in `data/interim/`.

## Notes
- Individual scripts have CLI arguments for overriding paths. Use `--help` to view options.
- API calls depend on Wikidata/Wikipedia availability; cached outputs are recommended.
- The active pipeline is Wikipedia-first-paragraph based; no Wikidata-description augmentation step is used.
