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
   - Script: `scripts/dataset/04_cleaning.py`
   - Output: `data/interim/4_*`

4) **Fetch Wikipedia summaries**
   - Script: `scripts/dataset/05_wikipedia_parse.py`
   - Output: `data/interim/5_*`

5) **Optional extra Wikipedia pass**
   - Script: `scripts/dataset/05b_wikipedia_parse_extra.py`
   - Output: `data/interim/5b_*`

6) **Enrich with Wikidata descriptions**
   - Script: `scripts/dataset/06_wikidata_desc.py`
   - Output: `data/interim/6_*`

7) **Second-depth tags (optional)**
   - Script: `scripts/dataset/07_get_tags_second_depth.py`
   - Output: `data/interim/6_*`

8) **Final cleaning + dataset export**
    - Script: `scripts/dataset/10_final_cleaning.py`
    - Outputs: `data/interim/7_*` and final processed outputs under `data/processed/` (see `data/processed/8_Emb_Rank/`)

9) **Add irrelevant tags**
   - Script: `scripts/dataset/08_add_unrelevants.py`
   - Output: `data/interim/7_*` (updated)

10) **Filter irrelevant tags with Wikipedia**
    - Script: `scripts/dataset/09_unrelevant_with_wiki.py`
    - Output: `data/interim/7_*`

11) **Resolve tag QIDs (optional)**
    - Script: `scripts/dataset/03_resolve_tag_qids.py`
    - Output: `data/interim/*_with_qids.jsonl`

## Notes
- Individual scripts have CLI arguments for overriding paths. Use `--help` to view options.
- API calls depend on Wikidata/Wikipedia availability; cached outputs are recommended.
