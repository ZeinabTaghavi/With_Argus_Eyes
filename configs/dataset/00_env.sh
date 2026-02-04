#!/usr/bin/env bash
set -euo pipefail

# Shared environment for dataset construction configs
THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_ROOT="$( cd "${THIS_DIR}/../.." && pwd )"

export ARGUS_DATA_ROOT="${ARGUS_DATA_ROOT:-${WORKSPACE_ROOT}/data}"
export ARGUS_INTERIM_ROOT="${ARGUS_DATA_ROOT}/interim"
export ARGUS_PROCESSED_ROOT="${ARGUS_DATA_ROOT}/processed"

# Wikidata/Wikipedia tuning knobs
export WIKIDATA_DB_PATH="${WIKIDATA_DB_PATH:-${ARGUS_INTERIM_ROOT}/wikidata_random.db}"
export WIKI_MAX_WORKERS="${WIKI_MAX_WORKERS:-8}"
export WIKI_CHUNKSIZE="${WIKI_CHUNKSIZE:-16}"
export WIKI_CHUNK_SIZE="${WIKI_CHUNK_SIZE:-25}"

# HF cache base (used by scripts that accept --hf_base)
export ARGUS_HF_BASE="${ARGUS_HF_BASE:-./}"
export HF_HOME="${HF_HOME:-${ARGUS_HF_BASE}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
