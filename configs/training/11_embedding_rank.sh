#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_ROOT="$( cd "${THIS_DIR}/../.." && pwd )"
source "${WORKSPACE_ROOT}/configs/dataset/00_env.sh"

PYTHON="${PYTHON:-python}"
SCRIPT="${WORKSPACE_ROOT}/scripts/training/11_embedding_rank.py"
export PYTHONPATH="${WORKSPACE_ROOT}/src:${PYTHONPATH:-}"

# Hugging Face cache root used by training scripts.
export ARGUS_HF_BASE="${ARGUS_HF_BASE:-../../../../data/proj/zeinabtaghavi}"
export HF_BASE="${ARGUS_HF_BASE}"
export HF_HOME="${HF_HOME:-${HF_BASE}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

# Dedicated embedding cache directory (new unified folder).
export ARGUS_EMBED_CACHE_ROOT="${ARGUS_EMBED_CACHE_ROOT:-${WORKSPACE_ROOT}/outputs/cache/embeddings}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Which embedding backends to use for the qid bank
RETRIEVERS=("contriever")

# Rank universe sizes
ORDERS=(800)

# Where to write per-worker shards (optional)
SHARD_DIR="${WORKSPACE_ROOT}/outputs/11_embedding_rank_shards"

# Optional flat YAML config (can be empty)
CONFIG_PATH=""

# Batch size for encoding texts
EMBED_BATCH_SIZE=32

# ---------------------------
# Run
# ---------------------------

echo "[INFO] Workspace root: ${WORKSPACE_ROOT}"
echo "[INFO] Data root: ${ARGUS_DATA_ROOT}"
echo "[INFO] Interim root: ${ARGUS_INTERIM_ROOT}"
echo "[INFO] Processed root: ${ARGUS_PROCESSED_ROOT}"
echo "[INFO] Embedding cache: ${ARGUS_EMBED_CACHE_ROOT}"
echo "[INFO] Shard dir: ${SHARD_DIR}"

mkdir -p "${SHARD_DIR}" "${ARGUS_EMBED_CACHE_ROOT}"

for RET in "${RETRIEVERS[@]}"; do
  for O in "${ORDERS[@]}"; do
    echo "==============================================="
    echo "[RUN] retriever=${RET} order=${O}"
    echo "==============================================="

    CMD=( "${PYTHON}" "${SCRIPT}"
          --data_root "${ARGUS_DATA_ROOT}"
          --processed_root "${ARGUS_PROCESSED_ROOT}"
          --interim_root "${ARGUS_INTERIM_ROOT}"
          --embedding_cache_dir "${ARGUS_EMBED_CACHE_ROOT}"
          --hf_base "${ARGUS_HF_BASE}"
          --retriever "${RET}"
          --o "${O}"
          --shard_output_dir "${SHARD_DIR}"
          --embed_batch_size "${EMBED_BATCH_SIZE}" )

    if [[ -n "${CONFIG_PATH}" ]]; then
      CMD+=( --config "${CONFIG_PATH}" )
    fi

    echo "[CMD] ${CMD[*]}"
    "${CMD[@]}"
  done
done
