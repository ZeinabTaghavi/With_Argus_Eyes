#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_ROOT="$( cd "${THIS_DIR}/../.." && pwd )"
source "${THIS_DIR}/00_env.sh"

PYTHON=${PYTHON:-python}

"${PYTHON}" "${WORKSPACE_ROOT}/scripts/dataset/08_build_all_relation_tags.py" \
  --input "${ARGUS_INTERIM_ROOT}/7_main_dataset.jsonl" \
  --output "${ARGUS_INTERIM_ROOT}/7_all_relation_tags.json"
