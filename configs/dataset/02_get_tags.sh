#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_ROOT="$( cd "${THIS_DIR}/../.." && pwd )"
source "${THIS_DIR}/00_env.sh"

PYTHON=${PYTHON:-python}

"${PYTHON}" "${WORKSPACE_ROOT}/scripts/dataset/02_get_tags.py" \
  --hf_base "${ARGUS_HF_BASE}" \
  --base_path "${ARGUS_INTERIM_ROOT}/2_tag_only" \
  --m_million 7 \
  --parallelism 16 \
  --split_size 100000