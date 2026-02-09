#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

run_step() {
  local script_name="$1"
  echo "=================================================="
  echo "[DATASET] Running ${script_name}"
  echo "=================================================="
  bash "${THIS_DIR}/${script_name}"
}

# Optional controls for stage 10 (finalization + cleanup)
# - RUN_CLEAR_INTERIM=true  -> remove remaining temporary files/folders in data/interim
# - RUN_KEEP_INTERIM="a,b"  -> comma-separated names to keep while clearing
RUN_CLEAR_INTERIM="${RUN_CLEAR_INTERIM:-false}"
RUN_KEEP_INTERIM="${RUN_KEEP_INTERIM:-}"

run_step "01_get_items.sh"
run_step "02_get_tags.sh"
run_step "03_cleaning.sh"
run_step "04_wikipedia_parse.sh"
run_step "05_get_tags_second_depth.sh"
run_step "06_final_cleaning.sh"
run_step "07_build_all_related_tags.sh"
run_step "08_add_unrelevants.sh"
run_step "09_unrelevant_with_wiki.sh"

if [[ "${RUN_CLEAR_INTERIM}" == "true" ]]; then
  echo "[DATASET] Stage 10 will clear interim files after finalization."
  echo "=================================================="
  echo "[DATASET] Running 10_clear_directories.sh (CLEAR_INTERIM=true)"
  echo "=================================================="
  CLEAR_INTERIM=true KEEP_INTERIM="${RUN_KEEP_INTERIM}" bash "${THIS_DIR}/10_clear_directories.sh"
else
  run_step "10_clear_directories.sh"
fi

echo "[DATASET] Completed. Final files are in data/processed/:"
echo " - main_dataset.jsonl"
echo " - wikipedia_all_relevant_results.jsonl"
echo " - wiki_unrelevants_results.jsonl"
echo "[DATASET] You can now run training starting from scripts/training/11_embedding_rank.py"
