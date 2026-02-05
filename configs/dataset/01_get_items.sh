#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_ROOT="$( cd "${THIS_DIR}/../.." && pwd )"
source "${THIS_DIR}/00_env.sh"

PYTHON=${PYTHON:-python}

"${PYTHON}" "${WORKSPACE_ROOT}/scripts/dataset/01_get_items.py" \
  --db "${WIKIDATA_DB_PATH}" \
  --target_total 7000000 \
  --qid_workers 16 \
  --label_workers 16 \
  --lang en \
  --batch_unlabeled 50000 \
  --export_batch 500000 
  # --export_only \
  # --overwrite
