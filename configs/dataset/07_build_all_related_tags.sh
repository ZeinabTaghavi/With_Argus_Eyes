#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_ROOT="$( cd "${THIS_DIR}/../.." && pwd )"
source "${THIS_DIR}/00_env.sh"

PYTHON=${PYTHON:-python}

"${PYTHON}" "${WORKSPACE_ROOT}/scripts/dataset/07_build_all_related_tags.py" \
  --input "${ARGUS_INTERIM_ROOT}/6_main_dataset.jsonl" \
  --output "${ARGUS_INTERIM_ROOT}/7_all_related_tags.json"
