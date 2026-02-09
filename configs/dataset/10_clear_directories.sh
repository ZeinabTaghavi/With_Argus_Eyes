#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_ROOT="$( cd "${THIS_DIR}/../.." && pwd )"
source "${THIS_DIR}/00_env.sh"

PYTHON=${PYTHON:-python}

# Set CLEAR_INTERIM=true to remove all remaining temporary files/folders
# from data/interim after moving final artifacts.
CLEAR_INTERIM=${CLEAR_INTERIM:-false}
KEEP_INTERIM=${KEEP_INTERIM:-""}

CMD=( "${PYTHON}" "${WORKSPACE_ROOT}/scripts/dataset/10_clear_directories.py"
  --interim_dir "${ARGUS_INTERIM_ROOT}"
  --processed_dir "${ARGUS_PROCESSED_ROOT}"
  --main_dataset_src "6_main_dataset.jsonl"
  --wikipedia_cache_src "4_tags_wikipedia_first_paragraphs_cache.jsonl"
  --wiki_unrelevants_src "9_wiki_unrelevants_results.jsonl"
  --main_dataset_name "main_dataset.jsonl"
  --wikipedia_cache_name "wikipedia_all_relevant_results.jsonl"
  --wiki_unrelevants_name "wiki_unrelevants_results.jsonl" )

if [[ "${CLEAR_INTERIM}" == "true" ]]; then
  CMD+=( --clear_interim )
  if [[ -n "${KEEP_INTERIM}" ]]; then
    CMD+=( --keep_interim "${KEEP_INTERIM}" )
  fi
fi

"${CMD[@]}"

