#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_ROOT="$( cd "${THIS_DIR}/../.." && pwd )"
source "${THIS_DIR}/00_env.sh"

PYTHON=${PYTHON:-python}

"${PYTHON}" "${WORKSPACE_ROOT}/scripts/dataset/05_wikidata_desc.py" \
  --hf_base "${ARGUS_HF_BASE}" \
  --items_in "${ARGUS_INTERIM_ROOT}/4_items_with_wikipedia.jsonl" \
  --items_out "${ARGUS_INTERIM_ROOT}/5_items_with_wikipedia_and_desc.jsonl" \
  --tags_wd_out "${ARGUS_INTERIM_ROOT}/5_tags_wikidata_descriptions.jsonl"
