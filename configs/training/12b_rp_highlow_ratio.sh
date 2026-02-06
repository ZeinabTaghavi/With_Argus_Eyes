python scripts/training/12b_rp_highlow_ratio.py \
  --retrievers contriever reasonir qwen3 jina bge-m3  reason-embed nv-embed gritlm \
  --ks 10 25 50 100 \
  --orders 800 \
  --threshold 0.5 \
  --out_dir outputs/10_RP_HighLow_Ratio \
  --save_name rp_high_low_ratio_by_order_800_all.png
