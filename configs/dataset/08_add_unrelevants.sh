#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_ROOT="$( cd "${THIS_DIR}/../.." && pwd )"
source "${THIS_DIR}/00_env.sh"

PYTHON=${PYTHON:-python}

"${PYTHON}" "${WORKSPACE_ROOT}/scripts/dataset/08_add_unrelevants.py" \
  --all_relation_tags "${ARGUS_INTERIM_ROOT}/7_all_relation_tags.json" \
  --main_dataset "${ARGUS_INTERIM_ROOT}/6_main_dataset.jsonl" \
  --out "${ARGUS_INTERIM_ROOT}/8_unrelevant_qids.jsonl"
