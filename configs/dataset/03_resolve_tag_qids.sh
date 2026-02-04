#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_ROOT="$( cd "${THIS_DIR}/../.." && pwd )"
source "${THIS_DIR}/00_env.sh"

PYTHON=${PYTHON:-python}

"${PYTHON}" "${WORKSPACE_ROOT}/scripts/dataset/03_resolve_tag_qids.py" \
  --input "${ARGUS_INTERIM_ROOT}/5_items_with_wikipedia_and_desc.jsonl" \
  --output-dir "${ARGUS_INTERIM_ROOT}/with_qids"
