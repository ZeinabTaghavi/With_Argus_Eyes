#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_ROOT="$( cd "${SCRIPT_DIR}/../.." && pwd )"
source "${WORKSPACE_ROOT}/configs/dataset/00_env.sh"

PYTHON="${PYTHON:-python}"
SCRIPT="${WORKSPACE_ROOT}/scripts/training/14_Score_Label_Context_Pairs.py"
export PYTHONPATH="${WORKSPACE_ROOT}/src:${PYTHONPATH:-}"

# Hugging Face cache root used by training scripts.
export ARGUS_HF_BASE="${ARGUS_HF_BASE:-../../../../data/proj/zeinabtaghavi}"
export HF_BASE="${ARGUS_HF_BASE}"
export HF_HOME="${HF_HOME:-${HF_BASE}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

RETRIEVER="${RETRIEVER:-contriever}"
ORDER="${ORDER:-800}"
K="${K:-50}"
TEXT_MODE="${TEXT_MODE:-context}"   # context | canonical | span
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_LENGTH="${MAX_LENGTH:-1048}"

# Input JSONL rows should contain at least: {"label":"...", "context":"..."}
INPUT_JSONL="${INPUT_JSONL:-${WORKSPACE_ROOT}/data/interim/14_selected_label_context_pairs.jsonl}"

ANALYSIS_OUT_DIR="${ANALYSIS_OUT_DIR:-${WORKSPACE_ROOT}/outputs/12_risk_outputs}"
RESULTS_TAG="${RESULTS_TAG:-}"       # optional explicit stage-12 tag
MODELS_DIR="${MODELS_DIR:-}"         # optional explicit models directory
MODEL_ARTIFACT="${MODEL_ARTIFACT:-}" # optional explicit .joblib path

OUT_DIR="${OUT_DIR:-${WORKSPACE_ROOT}/outputs/14_Score_Label_Context_Pairs}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${OUT_DIR}/14_scored_${RETRIEVER}_o_${ORDER}_k_${K}.jsonl}"

mkdir -p "${OUT_DIR}"

echo "[INFO] Workspace root: ${WORKSPACE_ROOT}"
echo "[INFO] Input JSONL: ${INPUT_JSONL}"
echo "[INFO] Output JSONL: ${OUTPUT_JSONL}"
echo "[INFO] Retriever: ${RETRIEVER}"
echo "[INFO] Order: ${ORDER}"
echo "[INFO] k: ${K}"
echo "[INFO] Text mode: ${TEXT_MODE}"
echo "[INFO] Analysis out dir: ${ANALYSIS_OUT_DIR}"

if [[ ! -f "${INPUT_JSONL}" ]]; then
  echo "[ERROR] Input file not found: ${INPUT_JSONL}"
  echo "[ERROR] Create it as JSONL rows like: {\"label\":\"...\",\"context\":\"...\"}"
  exit 1
fi

CMD=( "${PYTHON}" "${SCRIPT}"
      --retriever "${RETRIEVER}"
      --order "${ORDER}"
      --k "${K}"
      --analysis_out_dir "${ANALYSIS_OUT_DIR}"
      --input_jsonl "${INPUT_JSONL}"
      --output_jsonl "${OUTPUT_JSONL}"
      --text_mode "${TEXT_MODE}"
      --batch_size "${BATCH_SIZE}"
      --max_length "${MAX_LENGTH}"
      --hf_base "${ARGUS_HF_BASE}" )

if [[ -n "${RESULTS_TAG}" ]]; then
  CMD+=( --results_tag "${RESULTS_TAG}" )
fi
if [[ -n "${MODELS_DIR}" ]]; then
  CMD+=( --models_dir "${MODELS_DIR}" )
fi
if [[ -n "${MODEL_ARTIFACT}" ]]; then
  CMD+=( --model_artifact "${MODEL_ARTIFACT}" )
fi

echo "[CMD] ${CMD[*]}"
"${CMD[@]}"
