#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_ROOT="$( cd "${THIS_DIR}/../.." && pwd )"
source "${THIS_DIR}/00_env.sh"

PYTHON=${PYTHON:-python}

"${PYTHON}" "${WORKSPACE_ROOT}/scripts/dataset/06_final_cleaning.py" \
  --items_with_tags "${ARGUS_INTERIM_ROOT}/5_items_with_tags_qids.jsonl" \
  --wikipedia_tags "${ARGUS_INTERIM_ROOT}/5_tags_wikipedia_first_paragraphs.jsonl" \
  --output_wikipedia_pages "${ARGUS_INTERIM_ROOT}/6_all_wikipedia_pages.jsonl" \
  --output_main_dataset "${ARGUS_INTERIM_ROOT}/6_main_dataset.jsonl"
