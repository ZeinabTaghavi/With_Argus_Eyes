#  !/usr/bin/env bash
set -euo pipefail



# ls ../../../../data/proj/zeinabtaghavi/embeddings/
# bge-m3_span.npz  contriever_span.npz  gritlm_span.npz  jina_span.npz  nv-embed_span.npz  qwen3_span.npz  reason-embed_span.npz  reasonir_span.npz

# ------------------------------------------------------------------
# Basic paths (adapt if your layout differs)
# ------------------------------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "${SCRIPT_DIR}/../.." && pwd )"
cd "${ROOT_DIR}"

# ------------------------------------------------------------------
# HF caching (same convention as 11_embedding_rank.py)
# ------------------------------------------------------------------
export HF_BASE="../../../../data/proj/zeinabtaghavi"
export HF_HOME="${HF_BASE}"
export HF_HUB_CACHE="${HF_BASE}/hub"
export HF_DATASETS_CACHE="${HF_BASE}/datasets"


echo "HF_HOME:             ${HF_HOME}"
echo "HF_HUB_CACHE:        ${HF_HUB_CACHE}"
echo "HF_DATASETS_CACHE:   ${HF_DATASETS_CACHE}"

# ------------------------------------------------------------------
# Configurable knobs
# ------------------------------------------------------------------
# Choose retrievers and orders to evaluate; these must match the output
# of your 11_embedding_rank.py script.
# "contriever" "reasonir" "qwen3" "jina" "bge-m3" "reason-embed" "nv-embed" "gritlm" 
# 0: reasonir  jina
# 1: qwen3 bge-m3
# 2: reason-embed nv-embed gritlm
# "qwen3" "reason-embed" "nv-embed" "reasonir"
# configs/training/12_analysis_rank.sh
export CUDA_VISIBLE_DEVICES="2"
RETRIEVERS=("nv-embed")
ORDERS=(800)
K=(10 20)
OUT_DIR="outputs/9_risk_outputs"

cp ../../../../data/proj/zeinabtaghavi/embeddings/"${RETRIEVERS}"_span.npz outputs/cache/embedding_cache/embeddings/

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


# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------
for k in "${K[@]}"; do
  for retriever in "${RETRIEVERS[@]}"; do
    for order in "${ORDERS[@]}"; do
      echo "==============================================="
      echo "[RUN] retriever=${retriever} order=${order}"
      echo "==============================================="

      # Run the analysis script; ensure the final line does not end with a backslash.
      python scripts/training/12_analysis_rank.py \
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
