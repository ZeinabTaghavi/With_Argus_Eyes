#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_ROOT="$( cd "${THIS_DIR}/../.." && pwd )"
source "${THIS_DIR}/00_env.sh"

PYTHON=${PYTHON:-python}

"${PYTHON}" "${WORKSPACE_ROOT}/scripts/dataset/04_cleaning.py" \
  --input_dir "${ARGUS_INTERIM_ROOT}/2_tag_only" \
  --out_cleaned "${ARGUS_INTERIM_ROOT}/3_cleaned_items_tag_only.jsonl" \
  --low_in "${ARGUS_INTERIM_ROOT}/2_landmarks_low_freq.jsonl" \
  --low_out "${ARGUS_INTERIM_ROOT}/3_landmarks_low_freq.jsonl" \
  --high_in "${ARGUS_INTERIM_ROOT}/2_landmarks_high_freq.jsonl" \
  --high_out "${ARGUS_INTERIM_ROOT}/3_landmarks_high_freq.jsonl"
