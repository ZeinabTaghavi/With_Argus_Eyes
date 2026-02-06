python scripts/training/12c_avg_scores.py \
  --retrievers "contriever,reasonir,qwen3,jina,bge-m3,reason-embed,nv-embed,gritlm" \
  --k 50 \
  --order 800 \
  --out_dir "outputs/10_RP_HighLow_Ratio" \
  --save_name "avg_rp_score_by_retriever.png"
