#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_ROOT="$( cd "${THIS_DIR}/../.." && pwd )"
source "${THIS_DIR}/00_env.sh"

PYTHON=${PYTHON:-python}

"${PYTHON}" "${WORKSPACE_ROOT}/scripts/dataset/05_get_tags_second_depth.py" \
  --hf_base "${ARGUS_HF_BASE}" \
  --input_items "${ARGUS_INTERIM_ROOT}/4_items_with_wikipedia.jsonl" \
  --output_items "${ARGUS_INTERIM_ROOT}/5_items_with_tags_qids.jsonl" \
  --tags_wikipedia_out "${ARGUS_INTERIM_ROOT}/5_tags_wikipedia_first_paragraphs.jsonl" \
  --tags_wikipedia_seed_cache "${ARGUS_INTERIM_ROOT}/4_tags_wikipedia_first_paragraphs_cache.jsonl"
