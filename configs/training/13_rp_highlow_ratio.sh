python scripts/training/13_rp_highlow_ratio.py \
  --retrievers "contriever,reasonir,qwen3,jina,bge-m3,reason-embed,nv-embed,gritlm"\
  --orders "100,200,400,600,800" \
  --k 50 \
  --threshold 0.5 \
  --out_dir "outputs/10_RP_HighLow_Ratio"
