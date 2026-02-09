#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

bash "${THIS_DIR}/01_get_items.sh"
bash "${THIS_DIR}/02_get_tags.sh"
bash "${THIS_DIR}/03_cleaning.sh"
bash "${THIS_DIR}/04_wikipedia_parse.sh"
bash "${THIS_DIR}/05_get_tags_second_depth.sh"
bash "${THIS_DIR}/06_final_cleaning.sh"
bash "${THIS_DIR}/07_build_all_related_tags.sh"
bash "${THIS_DIR}/08_add_unrelevants.sh"
bash "${THIS_DIR}/09_unrelevant_with_wiki.sh"
bash "${THIS_DIR}/10_clear_directories.sh"
