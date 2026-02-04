"""Analyze ranking outputs and generate figures."""

from __future__ import annotations
from typing import List, Dict, Callable
import os, json, sys
import numpy as np
import random
import argparse
def _load_flat_yaml(path: str) -> dict:
    cfg = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" not in line:
                    continue
                k, v = line.split(":", 1)
                k = k.strip()
                v = v.strip()
                # strip quotes
                if (len(v) >= 2) and ((v[0] == v[-1]) and v[0] in ("'", '"')):
                    v = v[1:-1]
                cfg[k] = v
    except Exception as e:
        print(f"[warn] could not parse YAML config {path}: {e}")
    return cfg

def parse_args() -> argparse.Namespace:
    """Parse CLI args (mirroring 6_Emb_Rank.py style)."""
    parser = argparse.ArgumentParser(description="Analyze ranking outputs and generate figures.")
    parser.add_argument("--config", type=str, default="", help="Path to flat YAML config.")
    parser.add_argument("--retriever", type=str, default="contriever", choices=["contriever", "reasonir", "qwen3", "jina"])
    parser.add_argument("--order", type=str, default="all", help="Tag universe order per item, e.g., 'all' or '10000'.")
    parser.add_argument("--out_dir", type=str, default="risk_outputs")
    parser.add_argument("--bins_for_lda", type=int, default=2)
    parser.add_argument("--sample_histogram_bins", type=lambda x: str(x).lower() in ('1','true','yes','y'), default=True)
    parser.add_argument("--histogram_bins", type=int, default=100)
    parser.add_argument("--histogram_count_threshold", type=int, default=50)
    parser.add_argument("--histogram_sampling_mode", type=str, default="equal")
    parser.add_argument("--plot_external_samples", type=lambda x: str(x).lower() in ('1','true','yes','y'), default=True)
    parser.add_argument("--num_external_samples", type=int, default=10)
    args, _ = parser.parse_known_args()
    return args


# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from with_argus_eyes.training.helpers import load_rank_file, load_phrase_cache, phrase_key_for_item, stub_name, compute_and_print_risk_scores_for_jsonl
from with_argus_eyes.utils.risk.scores import get_risk_fn, normalized_score
from with_argus_eyes.utils.models import split_data, get_risk_prediction_fn, sample_by_histogram_bins, select_extreme_risk_samples, load_model
from with_argus_eyes.utils.plots.risk_score import (
    plot_pca2d_risk, plot_pca3d_risk, plot_lda2d_risk, plot_lda3d_risk, plot_risk_histogram
)

def build_dataset_from_items(
    items: List[dict],
    phrase_type: str,
    retriever: str,
    risk_name: str,
    *,
    risk_kwargs: dict,
    normalization_kwargs: dict,
) -> tuple[np.ndarray, np.ndarray, List[dict]]: # return (X, y_scores_raw, items_kept)
    """
    Returns (X, y_scores_raw, items_kept).
    X is embedding matrix from cache; y_scores_raw are *RAW* risk scores (no normalization here).
    """
    phrase_to_vec = load_phrase_cache(retriever, phrase_type) # {text -> embedding}
    risk_fn = get_risk_fn(risk_name)

    X_list, y_list, kept = [], [], []
    for it in items:
        key = phrase_key_for_item(it, phrase_type)
        vec = phrase_to_vec.get(key)
        assert vec is not None, f"[build_dataset_from_items] vec is None for key: {key}"
        assert key is not None, f"[build_dataset_from_items] key is None for item: {it}"

        # compute RAW score
        y_val = risk_fn(it, phrase_type, **risk_kwargs)  # float, raw
        X_list.append(vec.astype(np.float32))
        y_list.append(float(y_val))
        kept.append(it)

    y_list = normalized_score(y_list, **normalization_kwargs)

    X = np.vstack(X_list) if X_list else np.zeros((0, 1), dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y, kept


def run_risk_pipeline(
    retriever: str,
    phrase_type: str,
    *,
    using_file_path: bool = False,
    file_path: str = "",
    risk_name: str = "worst",
    risk_kwargs: dict | None = None,   
    normalization_kwargs: dict,
    out_dir: str = "risk_outputs",
    bins_for_lda: int = 2,
    sample_histogram_bins: bool = True,
    histogram_bins: int = 100,
    histogram_count_threshold: int = 50,
    histogram_sampling_mode: str = "equal", # "count" | "percentage" | "equal"
    train_mode: bool = False,
    test_size: float = 0.3,
    random_state: int = 42,
    plot_external_samples: bool = True,
    num_external_samples: int = 10,
    risk_prediction_function: str = 'mlp',
    risk_classification_threshold: float = 0.5,
    load_risk_predictions_from_file: bool = False,
    order: str = "all",
):
    os.makedirs(out_dir, exist_ok=True)

    # If no explicit file is provided, load the order-specific items file produced by 6_Emb_Rank.py
    if not using_file_path or not file_path:
        file_path = f"./outputs/6_items_with_tag_ranks_{retriever}_o_{order}.jsonl"
        using_file_path = True
    items = load_rank_file(retriever, using_file_path, file_path)

    # Prepare risk kwargs for RAW computation
    risk_kwargs = dict(risk_kwargs or {})  # do not mutate caller's dict

    # 1) Build dataset (RAW scores)
    X, y, kept = build_dataset_from_items(
        items, phrase_type, retriever, risk_name, risk_kwargs=risk_kwargs, normalization_kwargs=normalization_kwargs)
    if len(X) == 0:
        raise RuntimeError("No data assembled for training. Check caches and phrase_type.")
    print(f"[dataset] {retriever} / {phrase_type} → X={X.shape}, y∈[{y.min():.3f},{y.max():.3f}]")

    # Tag for outputs
    if risk_name == "anti_portion_below_k" or risk_name == "exists_in_top_k":
        tag = stub_name(phrase_type, retriever, f"{risk_name}_{risk_kwargs['k']}")
    elif risk_name == "combined":
        tag = stub_name(phrase_type, retriever, f"{risk_name}_{'_'.join(risk_kwargs['risk_names'])}_weights_{'_'.join(str(w) for w in risk_kwargs['risk_weights'])}")
    else:
        tag = stub_name(phrase_type, retriever, f"{risk_name}")
    tag = f"{tag}_o_{order}"

    # 3) Histograms
    os.makedirs(f"{out_dir}/plots/histograms", exist_ok=True)
    plot_risk_histogram(
        y, bins="auto", density=False, log=False,
        save_path=f"{out_dir}/plots/histograms/{tag}_hist.png",
        title=(
            f"Risk score histogram risk {risk_name} — {retriever}/{phrase_type}"
            if risk_name != "combined"
            else f"Risk score histogram risk combined_{'_'.join(risk_kwargs['risk_names'])}_weights_{'_'.join(str(w) for w in risk_kwargs['risk_weights'])} — {retriever}/{phrase_type}"
        ),
    )

    # 4) Optional histogram-based subsampling (NB: this operates on normalized y)
    if sample_histogram_bins:
        X, y = sample_by_histogram_bins(
            X, y,
            bins=histogram_bins,
            sampling_mode=histogram_sampling_mode,
            count_threshold=histogram_count_threshold,
            random_state=123
        )
        print(f"[sampled] {retriever} / {phrase_type} → X={X.shape}, y={y.shape}")

    # Additional sampled histogram (post-sampling)
    plot_risk_histogram(
        y, bins="auto", density=False, log=False,
        save_path=f"{out_dir}/plots/histograms/{tag}_hist_sampled_{histogram_bins}_{histogram_sampling_mode}_{histogram_count_threshold}.png",
        title=(
            f"Risk score histogram risk {risk_name} — {retriever}/{phrase_type} sampled"
            if risk_name != "combined"
            else f"Risk score histogram risk combined_{'_'.join(risk_kwargs['risk_names'])}_weights_{'_'.join(str(w) for w in risk_kwargs['risk_weights'])} — {retriever}/{phrase_type} sampled"
        ),
    )

    # 5) Train/test split on (X, y) — y is now normalized according to desired_norm
    X_tr, X_te, y_tr, y_te, i_tr, i_te = split_data(
        X, y, test_size=test_size, random_state=random_state, stratify=None
    )

    # # 6) Extremes (use 5% since frac=0.05 below)
    # low_lbl_tr, low_emb_tr, low_scr_tr, low_ranks_tr, high_lbl_tr, high_emb_tr, high_scr_tr, high_ranks_tr = select_extreme_risk_samples(
    #     X_tr, y_tr, kept, k=10, frac=0.05, index_map=i_tr, phrase_type=phrase_type, random_state=123
    # )
    # print("\n[extremes-train] bottom 5%:")
    # for lbl, scr, rks in zip(low_lbl_tr, low_scr_tr, low_ranks_tr):
    #     print(f"  score={scr:.6f}  label={lbl}  ranks={json.dumps(rks, ensure_ascii=False)}")
    # print("[extremes-train] top 5%:")
    # for lbl, scr, rks in zip(high_lbl_tr, high_scr_tr, high_ranks_tr):
    #     print(f"  score={scr:.6f}  label={lbl}  ranks={json.dumps(rks, ensure_ascii=False)}")

    # 7) Scan external JSONLs using the SAME normalization context
    os.makedirs(f"{out_dir}/scans", exist_ok=True)

    scan_file = f"./outputs/6_landmarks_high_freq_with_tag_ranks_{retriever}_o_{order}.jsonl"
    low_risk_sample_items = random.sample(load_rank_file('', using_file_path=True, file_path=scan_file), num_external_samples)
    x_low_risk, y_low_risk, kept_low_risk = build_dataset_from_items(
        low_risk_sample_items, phrase_type, retriever, risk_name, risk_kwargs=risk_kwargs, normalization_kwargs=normalization_kwargs)
    labels_low_risk = [(it.get("label") or it.get("item_label") or it.get("title") or "") for it in kept_low_risk]

    compute_and_print_risk_scores_for_jsonl(
        scan_file,
        phrase_type=phrase_type,
        risk_name=risk_name,
        risk_kwargs=risk_kwargs,
        normalization_kwargs=normalization_kwargs,
        out_path=f"{out_dir}/scans/landmarks_high_freq_{tag}.jsonl"
    )

    scan_file = f"./outputs/6_landmarks_low_freq_with_tag_ranks_{retriever}_o_{order}.jsonl"
    high_risk_sample_items = random.sample(load_rank_file('', using_file_path=True, file_path=scan_file), num_external_samples)
    x_high_risk, y_high_risk, kept_high_risk = build_dataset_from_items(
        high_risk_sample_items, phrase_type, retriever, risk_name, risk_kwargs=risk_kwargs, normalization_kwargs=normalization_kwargs)
    labels_high_risk  = [(it.get("label") or it.get("item_label") or it.get("title") or "") for it in kept_high_risk]
    compute_and_print_risk_scores_for_jsonl(
        scan_file,
        phrase_type=phrase_type,
        risk_name=risk_name,
        risk_kwargs=risk_kwargs,
        normalization_kwargs=normalization_kwargs,
        out_path=f"{out_dir}/scans/landmarks_low_freq_{tag}.jsonl"
    )

    
    # PCA 2D / 3D
    base = tag
    if sample_histogram_bins:    
        base = f"{tag}_sampled_{histogram_bins}_{histogram_sampling_mode}_{histogram_count_threshold}"



    os.makedirs(f"{out_dir}/plots/pca_2d", exist_ok=True)
    # os.makedirs(f"{out_dir}/plots/pca_3d", exist_ok=True)
    plot_pca2d_risk(X_tr, y_tr, X_test=X_te, scores_test=y_te, alpha=0.5, save_path=f"{out_dir}/plots/pca_2d/{base}_pca2d.png",
                    title=f"PCA (2D) risk {risk_name} — {retriever}/{phrase_type}")
    # plot_pca3d_risk(X_tr, y_tr, X_test=X_te, scores_test=y_te, alpha=0.5, save_path=f"{out_dir}/plots/pca_3d/{base}_pca3d.png",
    #                 title=f"PCA (3D) risk {risk_name} — {retriever}/{phrase_type}")

    # LDA 2D with optional overlay of high/low landmarks and their labels
    
    os.makedirs(f"{out_dir}/plots/lda_2d", exist_ok=True)
    plot_lda2d_risk(
        X_tr, y_tr,
        X_test=X_te, scores_test=y_te,
        bins_for_lda=bins_for_lda,
        alpha=0.5,
        save_path=f"{out_dir}/plots/lda_2d/{base}_overlay_lda2d.png",
        title=f"LDA (2D) risk {risk_name} — {retriever}/{phrase_type}",
        overlay_points=plot_external_samples,
        X_high=x_high_risk, scores_high=y_high_risk, labels_high=labels_high_risk,
        X_low=x_low_risk,  scores_low=y_low_risk,  labels_low=labels_low_risk,
        label_fontsize=4,
    )


    # PCS 2D / 3D (bins only used to learn the projection; colors remain continuous by risk)
    os.makedirs(f"{out_dir}/plots/lda_2d", exist_ok=True)
    # os.makedirs(f"{out_dir}/plots/lda_3d", exist_ok=True)
    plot_lda2d_risk(X_tr, y_tr, X_test=X_te, scores_test=y_te, bins_for_lda=bins_for_lda, alpha=0.5,
                    save_path=f"{out_dir}/plots/lda_2d/{base}_lda2d.png", title=f"LDA (2D) risk {risk_name} — {retriever}/{phrase_type}")
    # plot_lda3d_risk(X_tr, y_tr, X_test=X_te, scores_test=y_te, bins_for_lda=bins_for_lda, alpha=0.5,    
    #                 save_path=f"{out_dir}/plots/lda_3d/{base}_lda3d.png", title=f"LDA (3D) risk {risk_name} — {retriever}/{phrase_type}")

    plot_pca2d_risk(
        X_tr, y_tr,
        X_test=X_te, scores_test=y_te,
        title=f"PCA (2D) risk {risk_name} — {retriever}/{phrase_type}",
        overlay_points=plot_external_samples,
        X_high=x_high_risk, scores_high=y_high_risk, labels_high=labels_high_risk,  # e.g., ["Paris", "Berlin", ...]
        X_low=x_low_risk,  scores_low=y_low_risk,  labels_low=labels_low_risk,
        label_fontsize=4,
        save_path=f"{out_dir}/plots/pca_2d/{base}_overlay_pca2d.png",
    )


    if train_mode:
        # 3) Train MLP regressor to predict risk scores
        if not load_risk_predictions_from_file:
            print("[run_risk_pipeline] ...")
            risk_prediction_fn = get_risk_prediction_fn(risk_prediction_function)
            reg, y_pred, splits, metrics = risk_prediction_fn(X_tr, y_tr, X_te=X_te, y_te=y_te, i_tr=i_tr, i_te=i_te, out_dir=out_dir, tag=tag, kept=kept )
            print(f"[metrics] R2={metrics['r2']:.3f}  MAE={metrics['mae']:.3f}  RMSE={metrics['rmse']:.3f}")
        else:
            print("[run_risk_pipeline] Loading risk predictions from file...")
            model_path = f"{out_dir}/models/{tag}_model.pkl"
            model = load_model(model_path)
            y_pred = model.predict(X_te)
            print(f"[loaded] {len(y_pred)} risk predictions loaded from file")
        
        







    print(f"[done] outputs under: {out_dir}")

    # main_risk.py

# Example: worst-rank risk, normalized by max possible (total unique tags)
total_unique_tags = 5000000  # <- set from your data
# run_risk_pipeline(
#     retriever="contriever",
#     phrase_type="wikipedia_first_paragraph",
#     risk_name="mean",
#     risk_kwargs={"normalize": "max", "context": {"max_possible": total_unique_tags}},
#     out_dir="risk_outputs",
#     sample_histogram_bins=True,
#     bins_for_lda=2,   # 2 bins (median split) to learn LDA projection
#     train_mode=False,
#     histogram_bins=100,
#     histogram_count_threshold=50,
#     histogram_sampling_mode="equal",
# )


# total_unique_tags = 5000000  # <- set from your data
# run_risk_pipeline(
#     retriever="contriever",
#     phrase_type="wikipedia_first_paragraph",
#     risk_name="dev_num_tags_inverse_power",
#     risk_kwargs={"normalize": "none", "context": {"max_possible": total_unique_tags}},
#     out_dir="risk_outputs",
#     sample_histogram_bins=True,
#     bins_for_lda=2,   # 2 bins (median split) to learn LDA projection
#     train_mode=False,
#     histogram_bins=100,
#     histogram_count_threshold=50,
#     histogram_sampling_mode="equal",
# )
# -------------------------
# Example run
# -------------------------
# total_unique_tags = 5_000_000  # if you ever use "max" normalization, that's where it matters
# run_risk_pipeline(
#     retriever="contriever",
#     phrase_type="wikipedia_first_paragraph",
#     risk_name="combined",
#     risk_names=["mean", "dev_num_tags_inverse_power"],
#     risk_weights=[0.5, 0.5],
#     risk_kwargs={"normalize": "zscore"},   # or {"normalize": "minmax"}
#     out_dir="risk_outputs",
#     sample_histogram_bins=True,
#     bins_for_lda=2,
#     train_mode=False,
#     histogram_bins=100,
#     histogram_count_threshold=50,
#     histogram_sampling_mode="equal",
# )

# run_risk_pipeline(
#     retriever="contriever",
#     phrase_type="wikipedia_first_paragraph",
#     risk_name="combined",
#     risk_names=["mean", "dev_num_tags_inverse_power"],
#     risk_weights=[0.2, 0.8],
#     risk_kwargs={"normalize": "zscore"},   # or {"normalize": "minmax"}
#     out_dir="risk_outputs",
#     sample_histogram_bins=True,
#     bins_for_lda=2,
#     train_mode=False,
#     histogram_bins=100,
#     histogram_count_threshold=50,
#     histogram_sampling_mode="equal",
# )

# run_risk_pipeline(
#     retriever="contriever",
#     phrase_type="wikipedia_first_paragraph",
#     risk_name="combined",
#     risk_kwargs={"risk_names": ["mean", "dev_num_tags_inverse_power"], 
#                 "risk_weights": [0.5, 0.5]},
#     out_dir="risk_outputs",
#     normalization_kwargs={"method": "zscore"},
#     sample_histogram_bins=True,
#     bins_for_lda=2,
#     histogram_bins=100,
#     histogram_count_threshold=50,
#     histogram_sampling_mode="equal",
#     plot_external_samples=True,
#     num_external_samples=10,
#     train_mode=True,
#     load_risk_predictions_from_file=True,
#     risk_prediction_function='mlp',
#     risk_classification_threshold=0.0, # because normalized scores are in [0, 1]
# )

# run_risk_pipeline(
#     retriever="contriever",
#     phrase_type="wikipedia_first_paragraph",
#     risk_name="anti_portion_below_k",
#     risk_kwargs={"k": 8000},           # 👈 set your cutoff here
#     normalization_kwargs={"method": "none"},
#     out_dir="risk_outputs",
#     sample_histogram_bins=True,
#     bins_for_lda=2,
#     histogram_bins=100,
#     histogram_count_threshold=50,
#     histogram_sampling_mode="equal",
#     plot_external_samples=True,
#     num_external_samples=10,
#     train_mode=False
# )

# run_risk_pipeline(
#     retriever="contriever",
#     phrase_type="wikipedia_first_paragraph",
#     risk_name="anti_portion_below_k",
#     risk_kwargs={"k": 4000},           # 👈 set your cutoff here
#     normalization_kwargs={"method": "none"},
#     out_dir="risk_outputs",
#     sample_histogram_bins=True,
#     bins_for_lda=2,
#     histogram_bins=100,
#     histogram_count_threshold=50,
#     histogram_sampling_mode="equal",
#     plot_external_samples=True,
#     num_external_samples=10,
#     train_mode=False
# )

# run_risk_pipeline(
#     retriever="contriever",
#     phrase_type="wikipedia_first_paragraph",
#     risk_name="exists_in_top_k",
#     risk_kwargs={"k": 8000},           # 👈 set your cutoff here
#     normalization_kwargs={"method": "none"},
#     out_dir="risk_outputs",
#     sample_histogram_bins=True,
#     bins_for_lda=2,
#     histogram_bins=100,
#     histogram_count_threshold=50,
#     histogram_sampling_mode="equal",
#     plot_external_samples=True,
#     num_external_samples=10,
#     train_mode=False
# )

def main() -> None:
    args = parse_args()

    # Optional config override (flat YAML)
    if args.config:
        _cfg = _load_flat_yaml(args.config)
        if "retriever" in _cfg: args.retriever = _cfg["retriever"]
        if "order" in _cfg: args.order = _cfg["order"]
        if "out_dir" in _cfg: args.out_dir = _cfg["out_dir"]
        if "bins_for_lda" in _cfg: args.bins_for_lda = int(_cfg["bins_for_lda"])
        if "sample_histogram_bins" in _cfg: args.sample_histogram_bins = str(_cfg["sample_histogram_bins"]).lower() in ("1","true","yes","y")
        if "histogram_bins" in _cfg: args.histogram_bins = int(_cfg["histogram_bins"])
        if "histogram_count_threshold" in _cfg: args.histogram_count_threshold = int(_cfg["histogram_count_threshold"])
        if "histogram_sampling_mode" in _cfg: args.histogram_sampling_mode = _cfg["histogram_sampling_mode"]
        if "plot_external_samples" in _cfg: args.plot_external_samples = str(_cfg["plot_external_samples"]).lower() in ("1","true","yes","y")
        if "num_external_samples" in _cfg: args.num_external_samples = int(_cfg["num_external_samples"])
        # optional environment overrides for consistency with 6_Emb_Rank.py
        if "cuda_visible_devices" in _cfg:
            os.environ["CUDA_VISIBLE_DEVICES"] = _cfg["cuda_visible_devices"]
        if "hf_base" in _cfg:
            BASE = _cfg["hf_base"]
            os.environ["HF_BASE"] = BASE
            os.environ["HF_HOME"] = BASE
            os.environ["HF_HUB_CACHE"] = f"{BASE}/hub"
            os.environ["HF_DATASETS_CACHE"] = f"{BASE}/datasets"

    order_value = args.order
    retriever_value = args.retriever
    out_dir_value = args.out_dir

    run_risk_pipeline(
        retriever=retriever_value,
        phrase_type="wikipedia_first_paragraph",
        risk_name="exists_in_top_k",
        risk_kwargs={"k": 100},           # 👈 set your cutoff here
        normalization_kwargs={"method": "none"},
        out_dir=out_dir_value,
        sample_histogram_bins=args.sample_histogram_bins,
        bins_for_lda=args.bins_for_lda,
        histogram_bins=args.histogram_bins,
        histogram_count_threshold=args.histogram_count_threshold,
        histogram_sampling_mode=args.histogram_sampling_mode,
        plot_external_samples=args.plot_external_samples,
        num_external_samples=args.num_external_samples,
        train_mode=False,
        order=order_value,
    )

    run_risk_pipeline(
        retriever=retriever_value,
        phrase_type="wikipedia_first_paragraph",
        risk_name="exists_in_top_k",
        risk_kwargs={"k": 500},           # 👈 set your cutoff here
        normalization_kwargs={"method": "none"},
        out_dir=out_dir_value,
        sample_histogram_bins=args.sample_histogram_bins,
        bins_for_lda=args.bins_for_lda,
        histogram_bins=args.histogram_bins,
        histogram_count_threshold=args.histogram_count_threshold,
        histogram_sampling_mode=args.histogram_sampling_mode,
        plot_external_samples=args.plot_external_samples,
        num_external_samples=args.num_external_samples,
        train_mode=False,
        order=order_value,
    )

    run_risk_pipeline(
        retriever=retriever_value,
        phrase_type="wikipedia_first_paragraph",
        risk_name="exists_in_top_k",
        risk_kwargs={"k": 1000},           # 👈 set your cutoff here
        normalization_kwargs={"method": "none"},
        out_dir=out_dir_value,
        sample_histogram_bins=args.sample_histogram_bins,
        bins_for_lda=args.bins_for_lda,
        histogram_bins=args.histogram_bins,
        histogram_count_threshold=args.histogram_count_threshold,
        histogram_sampling_mode=args.histogram_sampling_mode,
        plot_external_samples=args.plot_external_samples,
        num_external_samples=args.num_external_samples,
        train_mode=False,
        order=order_value,
    )


if __name__ == "__main__":
    main()
