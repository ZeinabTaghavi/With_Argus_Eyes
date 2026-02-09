#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_ROOT="$( cd "${SCRIPT_DIR}/../.." && pwd )"
source "${WORKSPACE_ROOT}/configs/dataset/00_env.sh"

PYTHON="${PYTHON:-python}"
SCRIPT="${WORKSPACE_ROOT}/scripts/training/12_analysis_rank.py"

# Hugging Face cache root used by training scripts.
export ARGUS_HF_BASE="${ARGUS_HF_BASE:-../../../../data/proj/zeinabtaghavi}"
export HF_BASE="${ARGUS_HF_BASE}"
export HF_HOME="${HF_HOME:-${HF_BASE}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

# Dedicated embedding cache directory (new unified folder).
export ARGUS_EMBED_CACHE_ROOT="${ARGUS_EMBED_CACHE_ROOT:-${WORKSPACE_ROOT}/outputs/cache/embeddings}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
mkdir -p "${ARGUS_EMBED_CACHE_ROOT}"

echo "HF_HOME:             ${HF_HOME}"
echo "HF_HUB_CACHE:        ${HF_HUB_CACHE}"
echo "HF_DATASETS_CACHE:   ${HF_DATASETS_CACHE}"
echo "ARGUS_DATA_ROOT:     ${ARGUS_DATA_ROOT}"
echo "ARGUS_INTERIM_ROOT:  ${ARGUS_INTERIM_ROOT}"
echo "ARGUS_PROCESSED_ROOT:${ARGUS_PROCESSED_ROOT}"
echo "EMBED_CACHE_DIR:     ${ARGUS_EMBED_CACHE_ROOT}"

RETRIEVERS=("nv-embed")
ORDERS=(800)
K=(10 20)
OUT_DIR="outputs/9_risk_outputs"

# Histogram sampling configuration
HISTOGRAM_BINS=100
HISTOGRAM_COUNT_THRESHOLD=50
SAMPLE_BY_HISTOGRAM_BINS="true"

# Choose a sampling mode (equal, percentage, count, average)
HISTOGRAM_SAMPLING_MODE="average"

# MLP configuration
USE_MLP_CONFIGS="true"
TRAIN_MODE="true"
SAVE_MODELS="true"
# Set to "true" to plot the combined LDA+histogram; "false" to skip
PLOT="false"

for k in "${K[@]}"; do
  for retriever in "${RETRIEVERS[@]}"; do
    for order in "${ORDERS[@]}"; do
      echo "==============================================="
      echo "[RUN] retriever=${retriever} order=${order}"
      echo "==============================================="

      "${PYTHON}" "${SCRIPT}" \
        --data_root "${ARGUS_DATA_ROOT}" \
        --processed_root "${ARGUS_PROCESSED_ROOT}" \
        --interim_root "${ARGUS_INTERIM_ROOT}" \
        --embedding_cache_dir "${ARGUS_EMBED_CACHE_ROOT}" \
        --hf_base "${ARGUS_HF_BASE}" \
        --retriever "${retriever}" \
        --order "${order}" \
        --k "${k}" \
        --out_dir "${OUT_DIR}" \
        --sample_by_histogram_bins "${SAMPLE_BY_HISTOGRAM_BINS}" \
        --histogram_bins "${HISTOGRAM_BINS}" \
        --histogram_count_threshold "${HISTOGRAM_COUNT_THRESHOLD}" \
        --histogram_sampling_mode "${HISTOGRAM_SAMPLING_MODE}" \
        --use_mlp_configs "${USE_MLP_CONFIGS}" \
        --train_mode "${TRAIN_MODE}" \
        --plot "${PLOT}" \
        --save_models "${SAVE_MODELS}"
    done
  done
done
