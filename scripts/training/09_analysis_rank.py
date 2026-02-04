from __future__ import annotations
from typing import List, Dict, Callable, Tuple, Any
import os, json, sys
import numpy as np
import random
import argparse


from with_argus_eyes.training.helpers import load_rank_file, stub_name, compute_and_print_risk_scores_for_jsonl
from with_argus_eyes.utils.risk.scores import get_risk_fn, normalized_score
from with_argus_eyes.utils.models.evaluating_models import evaluate_regression_models
from with_argus_eyes.utils.models.splitting_data import sample_by_histogram_bins_fn
from with_argus_eyes.utils.plots.risk_score import (
    plot_pca2d_risk, plot_pca3d_risk, plot_lda2d_risk, plot_lda3d_risk, plot_risk_histogram, 
    plot_combined_lda_histogram
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, roc_auc_score, classification_report
)
import matplotlib.pyplot as plt

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="", help="Path to flat YAML config.")
    parser.add_argument("--retriever", type=str, default="contriever", choices=["contriever", "reasonir", "qwen3", "jina", "bge-m3", "rader", "reason-embed", "nv-embed", "gritlm"])
    parser.add_argument("--order", type=str, default=800, help="Tag universe order per item, e.g., 'all' or '10000'.")
    parser.add_argument('--k', type=int, default=50, help="Number of top tags to consider for risk calculation.")
    parser.add_argument("--out_dir", type=str, default="risk_outputs")
    parser.add_argument("--bins_for_lda", type=int, default=2)
    parser.add_argument("--sample_by_histogram_bins", type=lambda x: str(x).lower() in ('1','true','yes','y'), default=False)
    parser.add_argument("--histogram_bins", type=int, default=100)
    parser.add_argument("--histogram_count_threshold", type=int, default=50)
    parser.add_argument("--histogram_sampling_mode", type=str, default="average")
    parser.add_argument("--plot_external_samples", type=lambda x: str(x).lower() in ('1','true','yes','y'), default=True)
    parser.add_argument("--num_external_samples", type=int, default=10)
    parser.add_argument("--save_models", type=lambda x: str(x).lower() in ('1','true','yes','y'), default=False)
    parser.add_argument("--use_mlp_configs", type=lambda x: str(x).lower() in ('1','true','yes','y'), default=False)
    parser.add_argument("--train_mode", type=lambda x: str(x).lower() in ('1','true','yes','y'), default=False)
    parser.add_argument("--plot", type=lambda x: str(x).lower() in ('1','true','yes','y'), default=False)
    args, _ = parser.parse_known_args()
    return args


# Global state populated in main()
args: argparse.Namespace | None = None
order_value: str | int | None = None
k_value: int | None = None
retriever_value: str | None = None
out_dir_value: str | None = None
workspace_root: str | None = None
WIKIPEDIA_PAGES_PATH: str | None = None
cache_dir: str | None = None

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

_WIKIPEDIA_TEXT_BY_QID: Dict[str, str] | None = None
_RETRIEVER_CACHE: Dict[str, Any] = {}
EMBED_BATCH_SIZE_DEFAULT = 64



def _ensure_wikipedia_texts() -> Dict[str, str]:
    """Load and cache canonical Wikipedia paragraphs keyed by qid."""
    global _WIKIPEDIA_TEXT_BY_QID
    if _WIKIPEDIA_TEXT_BY_QID is not None:
        return _WIKIPEDIA_TEXT_BY_QID

    mapping: Dict[str, str] = {}
    total_rows = 0
    try:
        with open(WIKIPEDIA_PAGES_PATH, "r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                total_rows += 1

                record = json.loads(raw)
                qid = record.get("qid")
                paragraph = record["wikipedia_first_paragraph"]
                if not qid or not paragraph:
                    raise ValueError(f"No qid or paragraph found for record: {record}")
                label = record["label"]
                text = f"{label} : {paragraph}" if label and label not in paragraph else paragraph
                mapping[qid] = text
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Could not load wikipedia pages file: {WIKIPEDIA_PAGES_PATH}"
        ) from exc

    _WIKIPEDIA_TEXT_BY_QID = mapping
    print(f"[wikipedia] Loaded {len(mapping)} qids (processed {total_rows} rows) -> {WIKIPEDIA_PAGES_PATH}")
    return mapping


def _canonical_text_for_qid(qid: str) -> str:
    texts = _ensure_wikipedia_texts()
    try:
        return texts[qid]
    except KeyError as exc:
        raise KeyError(f"No wikipedia paragraph found for qid={qid}") from exc


def _extract_qid_from_item(item: Dict[str, Any]) -> str:
    candidates = [
        item.get("qid"),
        item.get("item_qid"),
        item.get("itemID"),
        item.get("item_id"),
    ]
    for cand in candidates:
        if cand:
            return cand

    nested = item.get("item")
    if isinstance(nested, dict):
        for key in ("qid", "id"):
            val = nested.get(key)
            if val:
                return val

    raise KeyError(f"Item does not contain a qid: keys={list(item.keys())}")


def _resolve_phrase_text(item: Dict[str, Any]) -> str:
    qid = _extract_qid_from_item(item)
    try:
        return _canonical_text_for_qid(qid)
    except KeyError:
        fallback = _build_fallback_phrase_text(item)
        if fallback:
            return fallback
        raise ValueError(f"Unable to build fallback text for item {qid}")


def _build_fallback_phrase_text(item: Dict[str, Any]) -> str:
    """Construct a best-effort textual representation if cached wikipedia text is missing."""
    paragraph = (item.get("wikipedia_first_paragraph") or "").strip()
    label = (
        item.get("label")
        or item.get("item_label")
        or item.get("title")
        or item.get("name")
        or ""
    ).strip()
    if paragraph:
        return f"{label} : {paragraph}" if label and label not in paragraph else paragraph

    related_labels: list[str] = []
    related = item.get("related_tags")
    if isinstance(related, list):
        for tag in related:
            if isinstance(tag, dict):
                tag_label = (tag.get("label") or tag.get("name") or "").strip()
            else:
                tag_label = str(tag).strip()
            if tag_label:
                related_labels.append(tag_label)
            if len(related_labels) == 5:
                break

    if label and related_labels:
        return f"{label} : related tags -> {', '.join(related_labels)}"
    if label:
        return label
    if related_labels:
        return ", ".join(related_labels)
    return ""


def _get_retriever_instance(retriever_name: str) -> Any:
    retriever = _RETRIEVER_CACHE.get(retriever_name)
    if retriever is None:
        from with_argus_eyes.utils.embeddings import build_retriever

        retriever = build_retriever(retriever_name)
        _RETRIEVER_CACHE[retriever_name] = retriever
    return retriever


def _guess_label(item: Dict[str, Any] | None) -> str:
    if not item:
        return ""
    candidates = (
        item.get("label"),
        item.get("item_label"),
        item.get("title"),
        item.get("name"),
    )
    for cand in candidates:
        if isinstance(cand, str):
            trimmed = cand.strip()
            if trimmed:
                return trimmed
    nested = item.get("item") if isinstance(item, dict) else None
    if isinstance(nested, dict):
        for cand in (nested.get("label"), nested.get("name")):
            if isinstance(cand, str):
                trimmed = cand.strip()
                if trimmed:
                    return trimmed
    return ""


def _encode_texts_with_cache(
    retriever_name: str,
    texts: List[str],
    *,
    source_items: List[dict] | None = None,
    batch_size: int = EMBED_BATCH_SIZE_DEFAULT,
) -> List[np.ndarray]:
    """Encode texts with persistent disk caching.
    
    Uses Cache/embeddings/{retriever_name}_span.npz for storage.
    """
    if not texts:
        return []

    cache_path = os.path.join(cache_dir, f"{retriever_name}_span.npz")
    
    # Load existing cache from disk
    cached_texts: List[str] = []
    cached_qids: List[str] | None = None
    cached_embeddings: np.ndarray | None = None
    text_to_idx: Dict[str, int] = {}
    
    if os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=True)
            cached_texts = data["texts"].tolist()
            cached_embeddings = data["embeddings"].astype(np.float32)
            cached_qids = data["qids"].tolist() if "qids" in data.files else None
            text_to_idx = {text: idx for idx, text in enumerate(cached_texts)}
            print(f"[Cache] Loaded {len(cached_texts)} cached embeddings from {cache_path}")
        except Exception as e:
            raise RuntimeError(f"Could not load cache from {cache_path}: {e}")
    else:
        print(f"[Cache] Cache file not found, creating new cache at {cache_path}")
        cached_texts = []
        cached_embeddings = None
        cached_qids = []
        text_to_idx = {}
    if cached_qids is None:
        cached_qids = [""] * len(cached_texts)

    # Find texts that need encoding
    missing_info: dict[str, dict] = {}
    for idx, text in enumerate(texts):
        if text in text_to_idx or text in missing_info:
            continue
        item = source_items[idx] if source_items and idx < len(source_items) else None
        try:
            qid = _extract_qid_from_item(item) if item else ""
        except Exception:
            qid = ""
        label = _guess_label(item)
        missing_info[text] = {
            "qid": qid or "",
            "label": label,
        }
    
    # Encode missing texts (if any) and extend cache
    if missing_info:
        retriever = _get_retriever_instance(retriever_name)
        missing_texts = list(missing_info.keys())
        missing_labels: List[str] = []
        missing_qids: List[str] = []
        for text in missing_texts:
            meta = missing_info[text]
            qid = meta.get("qid", "")
            label = meta.get("label", "")
            if label:
                phrase = label
            else:
                prefix = text.split(":", 1)[0].strip()
                phrase = prefix if prefix else text[:128]
            missing_labels.append(phrase)
            missing_qids.append(qid)
        print(f"[Cache] Encoding {len(missing_texts)} missing texts for {retriever_name}")
        new_embeddings = retriever.encode_spans(
            missing_texts,
            missing_labels,
            batch_size=batch_size,
            max_length=1048,
        )
        new_embeddings = np.asarray(new_embeddings, dtype=np.float32)
        if cached_embeddings is not None and cached_embeddings.size > 0:
            if new_embeddings.shape[1] != cached_embeddings.shape[1]:
                raise RuntimeError(
                    f"Embedding dimension mismatch: cache={cached_embeddings.shape[1]}, "
                    f"new={new_embeddings.shape[1]}"
                )
            cached_embeddings = np.vstack([cached_embeddings, new_embeddings])
        else:
            cached_embeddings = new_embeddings
        start_idx = len(cached_texts)
        cached_texts.extend(missing_texts)
        cached_qids.extend(missing_qids)
        for offset, text in enumerate(missing_texts):
            text_to_idx[text] = start_idx + offset
        np.savez_compressed(
            cache_path,
            qids=np.array(cached_qids, dtype=object),
            texts=np.array(cached_texts, dtype=object),
            embeddings=cached_embeddings.astype(np.float32),
        )
        print(f"[Cache] Saved updated embeddings to {cache_path}")
    else:
        print(f"[Cache] All {len(texts)} texts are already cached")

    if cached_embeddings is None:
        raise RuntimeError("Embeddings not available after cache update")

    # Return embeddings for requested texts
    result = []
    for text in texts:
        idx = text_to_idx.get(text)
        if idx is None:
            raise KeyError(f"Text not found in cache after encoding: {text[:50]}...")
        if cached_embeddings is not None:
            result.append(cached_embeddings[idx].copy())
        else:
            raise RuntimeError("Embeddings not available")
    
    return result


def build_dataset_from_items(
    items: List[dict],
    retriever: str,
    risk_name: str,
    *,
    risk_kwargs: dict,
    normalization_kwargs: dict,
) -> tuple[np.ndarray, np.ndarray, List[dict]]:  # return (X, y_scores_raw, items_kept)
    """Build an embedding matrix and corresponding risk scores for the given items."""
    risk_fn = get_risk_fn(risk_name)
    _ensure_wikipedia_texts()

    phrase_texts: List[str] = []
    kept_items: List[dict] = []
    skipped_count = 0

    for it in items:
        try:
            phrase_text = _resolve_phrase_text(it)
        except (KeyError, ValueError):
            skipped_count += 1
            continue
        if not phrase_text:
            skipped_count += 1
            continue
        phrase_texts.append(phrase_text)
        kept_items.append(it)

    if not kept_items:
        raise RuntimeError("No items available after resolving phrase texts.")
    if skipped_count:
        print(f"[dataset] skipped {skipped_count} items with no available text.")

    embeddings = _encode_texts_with_cache(
        retriever,
        phrase_texts,
        source_items=kept_items,
        batch_size=EMBED_BATCH_SIZE_DEFAULT,
    )

    print(f"[embeddings] {retriever} → {len(embeddings)} embeddings")

    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    for vec, it in zip(embeddings, kept_items):
        X_list.append(vec.astype(np.float32))
        score = risk_fn(it, **(risk_kwargs or {}))
        y_list.append(float(score))

    y_list = normalized_score(y_list, **normalization_kwargs)

    X = np.vstack(X_list) if X_list else np.zeros((0, 1), dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y, kept_items


def run_risk_pipeline(
    retriever: str,
    *,
    using_file_path: bool = False,
    file_path: str = "",
    risk_name: str = "ratio_unrelevant_below_k",
    risk_kwargs: dict | None = None,   
    normalization_kwargs: dict,
    out_dir: str = "risk_outputs",
    bins_for_lda: int = 2,
    sample_by_histogram_bins: bool = True,
    histogram_bins: int = 100,
    histogram_count_threshold: int = 50,
    histogram_sampling_mode: str = "average", # "count" | "percentage" | "equal"
    train_mode: bool = False,
    plot_external_samples: bool = True, 
    save_models: bool = False,
    order: str = "all",
    mlp_configs: list[dict] | None = None,
    plot: bool = False,
):
    os.makedirs(out_dir, exist_ok=True)

    # If no explicit file is provided, load the order-specific items file produced by 8_Emb_Rank.py
    if not using_file_path or not file_path:
        # 8_Emb_Rank.py writes its output as:
        #   {workspace_root}/data/processed/8_Emb_Rank/8_main_dataset_{order}_{retriever}.jsonl
        from pathlib import Path
        script_dir = Path(__file__).resolve().parent
        workspace_root_local = script_dir.parents[2]  # go up from scripts/training to workspace root
        file_path = str(
            workspace_root_local
            / "data"
            / "processed"
            / "8_Emb_Rank"
            / f"8_main_dataset_{order}_{retriever}.jsonl"
        )
        using_file_path = True

    items = load_rank_file(retriever, using_file_path, file_path)

    # Prepare risk kwargs for RAW computation
    risk_kwargs = dict(risk_kwargs or {})  # do not mutate caller's dict

    # 1) Build dataset (RAW scores)
    X, y, kept = build_dataset_from_items(
        items, retriever, risk_name, risk_kwargs=risk_kwargs, normalization_kwargs=normalization_kwargs)
    if len(X) == 0:
        raise RuntimeError("No data assembled for training. Check caches.")
    print(f"[dataset] {retriever} → X={X.shape}, y∈[{y.min():.3f},{y.max():.3f}]")

    # Tag for outputs
    if risk_name == "anti_portion_below_k" or risk_name == "exists_in_top_k" or risk_name == "ratio_unrelevant_below_k":
        tag = stub_name(retriever, f"{risk_name}_{risk_kwargs['k']}")
    elif risk_name == "combined":
        tag = stub_name(retriever, f"{risk_name}_{'_'.join(risk_kwargs['risk_names'])}_weights_{'_'.join(str(w) for w in risk_kwargs['risk_weights'])}")
    else:
        tag = stub_name(retriever, f"{risk_name}")
    if risk_name == "ratio_unrelevant_below_k":
        tag = f"{tag}_o_{order}_k_{risk_kwargs['k']}"
    print(f"[tag] {tag}")

    # # 3) Histograms
    # os.makedirs(f"{out_dir}/plots/histograms", exist_ok=True)
    # plot_risk_histogram(
    #     y, bins="auto", density=False, log=False,
    #     save_path=f"{out_dir}/plots/histograms/{tag}_hist.png",
    #     title=(
    #         f"Score histogram risk {risk_name} — {retriever}"
    #         if risk_name != "combined"
    #         else f"Score histogram risk combined_{'_'.join(risk_kwargs['risk_names'])}_weights_{'_'.join(str(w) for w in risk_kwargs['risk_weights'])} — {retriever}"
    #     ),
    # )
    if plot:
        print("[run_risk_pipeline] plotting combined lda and histogram")
        plot_combined_lda_histogram(
            X, y,
            bins_for_lda=bins_for_lda,
            hist_bins=histogram_bins,
            hist_density=False,
            hist_log=False,
            figsize=(8, 8),
            title=f"LDA and histogram — {retriever}",
            save_path=f"{out_dir}/plots/lda_histogram/{tag}.png",
            show=False,
        )
        print("[run_risk_pipeline] plotting combined lda and histogram")

        # 4) Optional histogram-based subsampling (NB: this operates on normalized y)
    if sample_by_histogram_bins:
        X, y = sample_by_histogram_bins_fn(
            X, y,
            bins=histogram_bins,
            sampling_mode=histogram_sampling_mode,
            count_threshold=histogram_count_threshold,
            random_state=123
        )
        if histogram_sampling_mode == "average":
            tag = f"{tag}_sampled_{histogram_sampling_mode}"
        else:
            tag = f"{tag}_sampled_{histogram_sampling_mode}_{histogram_count_threshold}"
        print(f"[sampled] {retriever} → X={X.shape}, y={y.shape}")

        # # Additional sampled histogram (post-sampling)
        # plot_risk_histogram(
        #     y, bins=histogram_bins, density=False, log=False,
        #     save_path=f"{out_dir}/plots/histograms/{tag}_hist.png",
        #     title=(
        #         f"Score histogram risk {risk_name} — {retriever} sampled by {histogram_sampling_mode}"
        #         if risk_name != "combined"
        #         else f"Score histogram risk combined_{'_'.join(risk_kwargs['risk_names'])}_weights_{'_'.join(str(w) for w in risk_kwargs['risk_weights'])} — {retriever} sampled by {histogram_sampling_mode}"
        #     ),
        # )

    # 5) Train/test split on (X, y) — y is now normalized according to desired_norm

    # # 6) Extremes (use 5% since frac=0.05 below)
    # low_lbl_tr, low_emb_tr, low_scr_tr, low_ranks_tr, high_lbl_tr, high_emb_tr, high_scr_tr, high_ranks_tr = select_extreme_risk_samples(
    #     X_tr, y_tr, kept, k=10, frac=0.05, index_map=i_tr, random_state=123
    # )
    # print("\n[extremes-train] bottom 5%:")
    # for lbl, scr, rks in zip(low_lbl_tr, low_scr_tr, low_ranks_tr):
    #     print(f"  score={scr:.6f}  label={lbl}  ranks={json.dumps(rks, ensure_ascii=False)}")
    # print("[extremes-train] top 5%:")
    # for lbl, scr, rks in zip(high_lbl_tr, high_scr_tr, high_ranks_tr):
    #     print(f"  score={scr:.6f}  label={lbl}  ranks={json.dumps(rks, ensure_ascii=False)}")

    # # 7) Scan external JSONLs using the SAME normalization context
    # os.makedirs(f"{out_dir}/scans", exist_ok=True)

    # scan_file = f"./outputs/6_landmarks_high_freq_with_tag_ranks_{retriever}_o_{order}.jsonl"
    # low_risk_sample_items = random.sample(load_rank_file('', using_file_path=True, file_path=scan_file), num_external_samples)
    # x_low_risk, y_low_risk, kept_low_risk = build_dataset_from_items(
    #     low_risk_sample_items, retriever, risk_name, risk_kwargs=risk_kwargs, normalization_kwargs=normalization_kwargs)
    # labels_low_risk = [it["label"] for it in kept_low_risk]

    # compute_and_print_risk_scores_for_jsonl(
    #     scan_file,
    #     risk_name=risk_name,
    #     risk_kwargs=risk_kwargs,
    #     normalization_kwargs=normalization_kwargs,
    #     out_path=f"{out_dir}/scans/landmarks_high_freq_{tag}.jsonl"
    # )

    # scan_file = f"./outputs/6_landmarks_low_freq_with_tag_ranks_{retriever}_o_{order}.jsonl"
    # high_risk_sample_items = random.sample(load_rank_file('', using_file_path=True, file_path=scan_file), num_external_samples)
    # x_high_risk, y_high_risk, kept_high_risk = build_dataset_from_items(
    #     high_risk_sample_items, retriever, risk_name, risk_kwargs=risk_kwargs, normalization_kwargs=normalization_kwargs)
    # labels_high_risk  = [(it.get("label") or it.get("item_label") or it.get("title") or "") for it in kept_high_risk]
    # compute_and_print_risk_scores_for_jsonl(
    #     scan_file,
    #     risk_name=risk_name,
    #     risk_kwargs=risk_kwargs,
    #     normalization_kwargs=normalization_kwargs,
    #     out_path=f"{out_dir}/scans/landmarks_low_freq_{tag}.jsonl"
    # )

    
    # PCA 2D / 3D
    base = tag
    
    # os.makedirs(f"{out_dir}/plots/pca_2d", exist_ok=True)
    # # os.makedirs(f"{out_dir}/plots/pca_3d", exist_ok=True)
    # plot_pca2d_risk(X_tr, y_tr, X_test=X_te, scores_test=y_te, alpha=0.5, save_path=f"{out_dir}/plots/pca_2d/{base}_pca2d.png",
    #                 title=f"PCA (2D) risk {risk_name} — {retriever}")
    # plot_pca3d_risk(X_tr, y_tr, X_test=X_te, scores_test=y_te, alpha=0.5, save_path=f"{out_dir}/plots/pca_3d/{base}_pca3d.png",
    #                 title=f"PCA (3D) risk {risk_name} — {retriever}")

    # LDA 2D with optional overlay of high/low landmarks and their labels
    # os.makedirs(f"{out_dir}/plots/lda_2d", exist_ok=True)
    # plot_lda2d_risk(
    #     X, y,
    #     bins_for_lda=bins_for_lda,
    #     alpha=0.5,
    #     save_path=f"{out_dir}/plots/lda_2d/{base}_overlay_lda2d.png",
    #     title=f"LDA (2D) score {risk_name} — {retriever}",
    #     overlay_points=plot_external_samples,
    #     # X_high=x_high_risk, scores_high=y_high_risk, labels_high=labels_high_risk,
    #     # X_low=x_low_risk,  scores_low=y_low_risk,  labels_low=labels_low_risk,
    #     label_fontsize=4,
    # )

    if train_mode:
        print("[run_risk_pipeline] Training regression model...")
        evaluate_regression_models(
            X,
            y,
            results_dir=out_dir+f'/{tag}',
            mlp_configs=mlp_configs,
            tag=tag,
            save_models=save_models,
        )


def main() -> None:
    global args, order_value, k_value, retriever_value, out_dir_value, workspace_root, WIKIPEDIA_PAGES_PATH, cache_dir

    args = parse_args()

    # Optional config override (flat YAML)
    if args.config:
        _cfg = _load_flat_yaml(args.config)
        if "retriever" in _cfg: args.retriever = _cfg["retriever"]
        if "order" in _cfg: args.order = _cfg["order"]
        if "k" in _cfg: args.k = int(_cfg["k"])
        if "out_dir" in _cfg: args.out_dir = _cfg["out_dir"]
        if "bins_for_lda" in _cfg: args.bins_for_lda = int(_cfg["bins_for_lda"])
        if "sample_by_histogram_bins" in _cfg: args.sample_by_histogram_bins = str(_cfg["sample_by_histogram_bins"]).lower() in ("1","true","yes","y")
        if "histogram_bins" in _cfg: args.histogram_bins = int(_cfg["histogram_bins"])
        if "histogram_count_threshold" in _cfg: args.histogram_count_threshold = int(_cfg["histogram_count_threshold"])
        if "histogram_sampling_mode" in _cfg: args.histogram_sampling_mode = _cfg["histogram_sampling_mode"]
        if "plot_external_samples" in _cfg: args.plot_external_samples = str(_cfg["plot_external_samples"]).lower() in ("1","true","yes","y")
        if "num_external_samples" in _cfg: args.num_external_samples = int(_cfg["num_external_samples"])
        if "save_models" in _cfg: args.save_models = str(_cfg["save_models"]).lower() in ("1","true","yes","y")
        if "use_mlp_configs" in _cfg: args.use_mlp_configs = str(_cfg["use_mlp_configs"]).lower() in ("1","true","yes","y")
        if "train_mode" in _cfg: args.train_mode = str(_cfg["train_mode"]).lower() in ("1","true","yes","y")
        if "plot" in _cfg: args.plot = str(_cfg["plot"]).lower() in ("1","true","yes","y")

    # Allow override by config or env
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Optional: set HF caches
    BASE = "../../../../data/proj/zeinabtaghavi"
    os.environ["HF_HOME"]           = BASE
    os.environ["HF_HUB_CACHE"]      = f"{BASE}/hub"
    os.environ["HF_DATASETS_CACHE"] = f"{BASE}/datasets"

    print("HF_HOME:", os.environ.get("HF_HOME"))
    print("HF_HUB_CACHE:", os.environ.get("HF_HUB_CACHE"))
    print("HF_DATASETS_CACHE:", os.environ.get("HF_DATASETS_CACHE"))
    print("TRANSFORMERS_CACHE:", os.environ.get("TRANSFORMERS_CACHE"))

    order_value = args.order
    k_value = args.k
    retriever_value = args.retriever
    out_dir_value = args.out_dir

    # Workspace paths
    workspace_root = project_root
    WIKIPEDIA_PAGES_PATH = os.path.join(
        workspace_root, "data", "processed", "8_Emb_Rank", "7_all_wikipedia_pages.jsonl"
    )

    # Cache directory
    cache_dir = os.path.join(workspace_root, "outputs", "cache", "embedding_cache", "embeddings")
    os.makedirs(cache_dir, exist_ok=True)

    if args.use_mlp_configs:
        if args.retriever.lower()  in ["contriever", "cnt"]:
            retriever_value = "contriever"
        elif args.retriever.lower()  in ["reasonir", "reason"]:
            retriever_value = "reasonir"
        elif args.retriever.lower()  in ["jina", "jina"]:
            retriever_value = "jina"
        elif args.retriever.lower()  in ["qwen3", "qwen"]:
            retriever_value = "qwen3"
        elif args.retriever.lower()  in ["bge-m3", "bge_m3", "bge"]:
            retriever_value = "bge-m3"
        elif args.retriever.lower()  in ["rader", "rader"]:
            retriever_value = "rader"
        elif args.retriever.lower()  in ["reason-embed", "reason_embed"]:
            retriever_value = "reason-embed"
        elif args.retriever.lower()  in ["nv-embed", "nv_embed"]:
            retriever_value = "nv-embed"
        elif args.retriever.lower()  in ["gritlm", "gritlm"]:
            retriever_value = "gritlm"
        else:
            raise RuntimeError(f"Unknown retriever: {args.retriever}")
        model_configs_file = f"configs/training/analysis_rank_models/{retriever_value}.jsonl"
        if not os.path.exists(model_configs_file):
            raise RuntimeError(f"Model configs file not found: {model_configs_file}")
        mlp_configs = []
        with open(model_configs_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                mlp_configs.append(json.loads(line))
        print(f"[mlp_configs] have {len(mlp_configs)} model configs")
    else:
        mlp_configs = None

    run_risk_pipeline(
        retriever=retriever_value,
        risk_name="ratio_unrelevant_below_k",
        risk_kwargs={"order": order_value, "k": k_value},
        normalization_kwargs={"method": "none"},
        out_dir=out_dir_value,
        sample_by_histogram_bins=args.sample_by_histogram_bins,
        bins_for_lda=args.bins_for_lda,
        histogram_bins=args.histogram_bins,
        histogram_count_threshold=args.histogram_count_threshold,
        histogram_sampling_mode=args.histogram_sampling_mode,
        plot_external_samples=args.plot_external_samples,
        train_mode=args.train_mode,
        mlp_configs=mlp_configs,
        save_models=args.save_models,
        order=order_value,
        plot=args.plot,
    )


if __name__ == "__main__":
    main()
