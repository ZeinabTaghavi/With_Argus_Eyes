#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_ROOT="$( cd "${THIS_DIR}/../.." && pwd )"
source "${THIS_DIR}/00_env.sh"

PYTHON=${PYTHON:-python}

"${PYTHON}" "${WORKSPACE_ROOT}/scripts/dataset/09_unrelevant_with_wiki.py" \
  --wikipedia_pages "${ARGUS_INTERIM_ROOT}/4_tags_wikipedia_first_paragraphs_cache.jsonl" \
  --unrelevant_tags "${ARGUS_INTERIM_ROOT}/8_unrelevant_qids.jsonl" \
  --output "${ARGUS_INTERIM_ROOT}/9_wiki_unrelevants_results.jsonl"
