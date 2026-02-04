#!/usr/bin/env bash
set -euo pipefail


# ---------------------------
# Config
# ---------------------------

# Root of the repo (this script lives in configs/training)
THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_ROOT="$( cd "${THIS_DIR}/../.." && pwd )"

export HF_BASE="../../../../data/proj/zeinabtaghavi"
export HF_HOME="${HF_BASE}"
export HF_HUB_CACHE="${HF_BASE}/hub"
export HF_DATASETS_CACHE="${HF_BASE}/datasets"
export CUDA_VISIBLE_DEVICES="7"

# Set HF_TOKEN in your environment if needed for private models.
export HUGGINGFACE_TOKEN="${HF_TOKEN:-}"
export HF_AUTH_TOKEN="${HF_TOKEN:-}"

export CUDA_VISIBLE_DEVICES="0"
PYTHON=python
SCRIPT="${WORKSPACE_ROOT}/scripts/training/08_embedding_rank.py"

# Which embedding backends to use for the qid bank
RETRIEVERS=("contriever")   # add "contriever" "reasonir" "qwen3" "jina" "bge-m3" "reason-embed" "nv-embed" "gritlm" if you like

# Rank universe sizes
ORDERS=(800)                # you can add e.g. 100 200

# Number of CPU workers for the ranking step
NUM_WORKERS=None


# Where to write per-worker shards (optional)
SHARD_DIR="${WORKSPACE_ROOT}/outputs/8_rank_shards"

# Optional flat YAML config (can be empty)
CONFIG_PATH=""

# Batch size for encoding texts
EMBED_BATCH_SIZE=32

# ---------------------------
# Run
# ---------------------------

echo "[INFO] Workspace root: ${WORKSPACE_ROOT}"
echo "[INFO] Shard dir: ${SHARD_DIR}"

mkdir -p "${SHARD_DIR}"

for RET in "${RETRIEVERS[@]}"; do
  cp ../../../../data/proj/zeinabtaghavi/embeddings/"${RET}"_span.npz outputs/cache/embedding_cache/embeddings/
  for O in "${ORDERS[@]}"; do
    echo "==============================================="
    echo "[RUN] retriever=${RET} order=${O}"
    echo "==============================================="

    CMD=( "${PYTHON}" "${SCRIPT}"
          --retriever "${RET}"
          --o "${O}"
        #   --num_workers "${NUM_WORKERS}"
          --shard_output_dir "${SHARD_DIR}"
          --embed_batch_size "${EMBED_BATCH_SIZE}" )

    if [[ -n "${CONFIG_PATH}" ]]; then
      CMD+=( --config "${CONFIG_PATH}" )
    fi

    echo "[CMD] ${CMD[*]}"
    "${CMD[@]}"
  done
done
