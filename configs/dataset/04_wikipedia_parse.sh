#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_ROOT="$( cd "${THIS_DIR}/../.." && pwd )"
source "${THIS_DIR}/00_env.sh"

PYTHON=${PYTHON:-python}

# Uses WIKI_MAX_WORKERS and WIKI_CHUNKSIZE from 00_env.sh
"${PYTHON}" "${WORKSPACE_ROOT}/scripts/dataset/04_wikipedia_parse.py"
