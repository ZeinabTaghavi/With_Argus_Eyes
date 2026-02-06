python scripts/training/13_retrieval_bias_analysis.py \
    --retriever contriever \
    --model_path risk_outputs/models/contriever_ratio_unrelevant_below_k_50_o_400_logistic_p0500_model.pkl \
    --output_path outputs/ \
    --threshold 0.5
