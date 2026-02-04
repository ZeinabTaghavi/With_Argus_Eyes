#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_ROOT="$( cd "${THIS_DIR}/../.." && pwd )"
source "${THIS_DIR}/00_env.sh"

PYTHON=${PYTHON:-python}

"${PYTHON}" "${WORKSPACE_ROOT}/scripts/dataset/06_wikidata_desc.py" \
  --input_path "${ARGUS_INTERIM_ROOT}/4_items_with_wikipedia.jsonl" \
  --output_path "${ARGUS_INTERIM_ROOT}/5_items_with_wikipedia_and_desc.jsonl" \
  --hf_base "${ARGUS_HF_BASE}"
