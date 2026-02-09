#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_ROOT="$( cd "${THIS_DIR}/../.." && pwd )"
source "${WORKSPACE_ROOT}/configs/dataset/00_env.sh"

PYTHON="${PYTHON:-python}"
SCRIPT="${WORKSPACE_ROOT}/scripts/training/13_rp_highlow_ratio.py"
export PYTHONPATH="${WORKSPACE_ROOT}/src:${PYTHONPATH:-}"

# Figure outputs
OUT_DIR="${WORKSPACE_ROOT}/outputs/10_RP_HighLow_Ratio"
SAVE_NAME="3_rp_high_low_ratio_by_order.png"

# Defaults used in the paper workflow
RETRIEVERS="${RETRIEVERS:-contriever,reasonir,qwen3,jina,bge-m3,reason-embed,nv-embed,gritlm}"
ORDERS="${ORDERS:-100,200,400,600,800}"
K="${K:-50}"
THRESHOLD="${THRESHOLD:-0.5}"

mkdir -p "${OUT_DIR}"

echo "[INFO] Workspace root: ${WORKSPACE_ROOT}"
echo "[INFO] Processed root: ${ARGUS_PROCESSED_ROOT}"
echo "[INFO] Output dir: ${OUT_DIR}"

"${PYTHON}" "${SCRIPT}" \
  --processed_root "${ARGUS_PROCESSED_ROOT}" \
  --retrievers "${RETRIEVERS}" \
  --orders "${ORDERS}" \
  --k "${K}" \
  --threshold "${THRESHOLD}" \
  --out_dir "${OUT_DIR}" \
  --save_name "${SAVE_NAME}"
