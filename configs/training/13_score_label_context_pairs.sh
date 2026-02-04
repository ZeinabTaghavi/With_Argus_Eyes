#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0,1,2,3
python scripts/training/13_score_label_context_pairs.py \
  --retriever reasonir \
  --models_dir outputs/9_risk_outputs/reasonir_ratio_unrelevant_below_k_50_o_800_k_50_sampled_average/models \
  --input_jsonl outputs/13_Score_Label_Context_Pairs/input_data.jsonl \
  --output_jsonl outputs/13_Score_Label_Context_Pairs/output_data.jsonl