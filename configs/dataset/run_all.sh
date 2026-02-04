#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

bash "${THIS_DIR}/01_get_items.sh"
bash "${THIS_DIR}/02_get_tags.sh"
bash "${THIS_DIR}/04_cleaning.sh"
bash "${THIS_DIR}/05_wikipedia_parse.sh"
# Optional: extra Wikipedia pass
# bash "${THIS_DIR}/05b_wikipedia_parse_extra.sh"
bash "${THIS_DIR}/06_wikidata_desc.sh"
bash "${THIS_DIR}/07_get_tags_second_depth.sh"
bash "${THIS_DIR}/10_final_cleaning.sh"
bash "${THIS_DIR}/08_add_unrelevants.sh"
bash "${THIS_DIR}/09_unrelevant_with_wiki.sh"

# Optional: resolve tag QIDs on enriched items
# bash "${THIS_DIR}/03_resolve_tag_qids.sh"
