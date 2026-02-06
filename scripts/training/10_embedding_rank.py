"""Run embedding-rank experiments for multiple retrievers."""

import os
import sys
import json
import random
import argparse
from typing import List, Dict, Tuple, Any
import numpy as np
from tqdm import tqdm

# ---------------------------
# Bootstrap paths & defaults
# ---------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

 

# ---------------------------
# CLI
# ---------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run embedding-rank experiments for multiple retrievers.")
    parser.add_argument("--config", type=str, default="", help="Path to flat YAML config.")
    parser.add_argument(
        "--retriever",
        type=str,
        default="contriever",
        choices=[
            "contriever", "reasonir", "qwen3",
            "jina", "bge-m3", "rader",
            "reason-embed", "nv-embed", "gritlm"  # include gritlm
        ],
    )
    parser.add_argument("--reasonir_instruction", type=str, default="")
    parser.add_argument("--qwen3_instruction", type=str, default="")
    parser.add_argument("--o", type=int, default=400,
                        help="Order of the tag universe per item. it should be less than 827")
    parser.add_argument("--order", type=int, default=None, help="Alias for --o.")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU IDs to use (overrides CUDA_VISIBLE_DEVICES).")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of parallel workers; default=len(gpus).")
    parser.add_argument("--shard_output_dir", type=str, default=None,
                        help="If set, write each worker's results as shard-XXXX.jsonl here.")
    parser.add_argument("--embed_batch_size", type=int, default=64,
                        help="Batch size used when precomputing embeddings.")
    args, _ = parser.parse_known_args()
    return args

# --- Load config (flat YAML) if provided and override args/env ---
def _load_flat_yaml(path: str) -> dict:
    cfg = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#") or ":" not in line:
                    continue
                k, v = line.split(":", 1)
                k = k.strip()
                v = v.strip()
                if (len(v) >= 2) and ((v[0] == v[-1]) and v[0] in ("'", '"')):
                    v = v[1:-1]
                cfg[k] = v
    except Exception as e:
        print(f"[warn] could not parse YAML config {path}: {e}")
    return cfg

# Global state populated in main()
args: argparse.Namespace | None = None
order_value: int | None = None
non_relation_tag_data: Dict[str, List[str]] = {}
wikipedia_pages_dict: Dict[str, Dict] = {}
main_items: List[dict] = []
cache_dir: str = ""

# ---------------------------
# Helpers
# ---------------------------
WIKI_FILTER_KEYWORDS = ("wikipedia", "wikidata")


def chunkify(lst: List[Any], n_chunks: int) -> List[Tuple[int, int]]:
    """Return (start, end) index pairs splitting lst into n_chunks contiguous chunks."""
    n = len(lst)
    if n_chunks <= 0:
        return [(0, n)]
    base = n // n_chunks
    rem = n % n_chunks
    bounds = []
    start = 0
    for i in range(n_chunks):
        extra = 1 if i < rem else 0
        end = start + base + extra
        bounds.append((start, end))
        start = end
    return bounds


def _value_contains_wiki_keyword(value: Any) -> bool:
    """Return True if value (recursively) includes banned wiki keywords."""
    if isinstance(value, str):
        lowered = value.lower()
        return any(keyword in lowered for keyword in WIKI_FILTER_KEYWORDS)
    if isinstance(value, dict):
        return any(_value_contains_wiki_keyword(v) for v in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_value_contains_wiki_keyword(v) for v in value)
    return False


def _filter_wiki_keywords_from_tags(tags: List[Any]) -> Tuple[List[Any], int]:
    """Filter tags that contain banned wiki keywords; returns (filtered_tags, removed_count)."""
    filtered: List[Any] = []
    removed = 0
    for tag in tags:
        if _value_contains_wiki_keyword(tag):
            removed += 1
            continue
        filtered.append(tag)
    return filtered, removed


def _strip_wiki_keywords_from_items(items: List[dict]) -> Tuple[int, int]:
    """Remove wiki-branded tags from each item, returning (affected_items, removed_tags)."""
    affected_items = 0
    removed_tags = 0
    for item in items:
        tags = item.get("related_tags")
        if not isinstance(tags, list) or not tags:
            continue
        filtered_tags, removed = _filter_wiki_keywords_from_tags(tags)
        if removed:
            item["related_tags"] = filtered_tags
            affected_items += 1
            removed_tags += removed
    return affected_items, removed_tags


def parse_gpu_ids() -> List[str]:
    if args.gpus:
        return [g.strip() for g in args.gpus.split(",") if g.strip() != ""]
    env = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    return [g.strip() for g in env.split(",") if g.strip() != ""]


# ---------------------------
# Global embedding bank
# ---------------------------
qid2idx: Dict[str, int] = {}
emb_matrix: np.ndarray | None = None

# Cache directory
# cache_dir is set in main()




def _make_canonical_text_for_qid(qid: str) -> str:
    """Canonical text used to embed a Wikipedia qid (paragraph only)."""
    rec = wikipedia_pages_dict.get(qid)
    if rec is None:
        raise KeyError(f"qid {qid} not found in wikipedia_pages_dict")
    para = (rec.get("wikipedia_first_paragraph") or "").strip()
    if not para:
        raise ValueError(f"wikipedia_first_paragraph is empty for qid {qid}")
    return para


def build_qid_embedding_bank(
    retriever_name: str,
    reasonir_instruction: str = "",
    qwen3_instruction: str = "",
    batch_size: int = 64,
) -> None:
    """Precompute a paragraph embedding for every Wikipedia qid and store it in memory and on disk.

    Uses persistent disk cache: Cache/embeddings/{retriever_name}_wikipedia_paragraph.npz with three aligned arrays:
      - qids:       list of qid strings
      - texts:      list of canonical texts used for encoding (Wikipedia first paragraph)
      - embeddings: paragraph embeddings, shape (N, D)
    """
    global qid2idx, emb_matrix

    cache_path = os.path.join(cache_dir, f"{retriever_name}_wikipedia_paragraph.npz")

    # Load existing cache: qids, texts, embeddings
    cached_qids: List[str] = []
    cached_texts: List[str] = []
    cached_embeddings: np.ndarray | None = None
    qid_to_idx_cache: Dict[str, int] = {}

    if os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=True)
            if not {"qids", "texts", "embeddings"}.issubset(set(data.files)):
                raise KeyError("Cache missing required entries 'qids', 'texts', 'embeddings'")
            cached_qids = data["qids"].tolist()
            cached_texts = data["texts"].tolist()
            cached_embeddings = data["embeddings"].astype(np.float32)
            qid_to_idx_cache = {qid: idx for idx, qid in enumerate(cached_qids)}
            print(f"[Cache] Loaded {len(cached_qids)} cached paragraph embeddings from {cache_path}")
        except Exception as e:
            print(f"[warn] Could not load cache from {cache_path}: {e}")
            cached_qids = []
            cached_texts = []
            cached_embeddings = None
            qid_to_idx_cache = {}

    # Prepare canonical paragraph texts for all qids
    all_qids = list(wikipedia_pages_dict.keys())
    qid_to_text: Dict[str, str] = {}
    for qid in all_qids:
        text = _make_canonical_text_for_qid(qid)
        qid_to_text[qid] = text

    # Determine which qids need encoding (based on cache)
    qids_to_encode: List[str] = []
    texts_to_encode: List[str] = []
    for qid in all_qids:
        if qid not in qid_to_idx_cache:
            qids_to_encode.append(qid)
            texts_to_encode.append(qid_to_text[qid])

    final_embeddings: np.ndarray | None = None
    final_qids: List[str] = []
    final_texts: List[str] = []

    if qids_to_encode:
        print(f"[Cache] Encoding {len(qids_to_encode)} new qids (total qids: {len(all_qids)})...")

        from with_argus_eyes.utils.embeddings import build_retriever
        retriever = build_retriever(
            retriever_name,
            reasonir_instruction=reasonir_instruction,
            qwen3_instruction=qwen3_instruction,
        )

        new_embs = []
        for i in tqdm(range(0, len(texts_to_encode), batch_size), desc="Encoding new paragraph embeddings", unit="batch"):
            batch_texts = texts_to_encode[i:i + batch_size]
            if not batch_texts:
                continue
            emb_batch = retriever.encode_texts(batch_texts, batch_size=batch_size, max_length=1048)
            new_embs.append(np.asarray(emb_batch, dtype=np.float32))

        new_embeddings = np.vstack(new_embs) if new_embs else np.zeros((0, 1), dtype=np.float32)
        del retriever

        if cached_embeddings is not None and len(cached_embeddings) > 0:
            if new_embeddings.shape[0] > 0 and cached_embeddings.shape[1] != new_embeddings.shape[1]:
                print(f"[warn] Dimension mismatch: cached={cached_embeddings.shape[1]}, new={new_embeddings.shape[1]}. Resetting cache.")
                cached_embeddings = None
                cached_qids = []
                cached_texts = []
                qid_to_idx_cache = {}

        if cached_embeddings is not None and len(cached_embeddings) > 0:
            final_embeddings = np.vstack([cached_embeddings, new_embeddings])
            final_qids = cached_qids + qids_to_encode
            final_texts = cached_texts + texts_to_encode
        else:
            final_embeddings = new_embeddings
            final_qids = qids_to_encode
            final_texts = texts_to_encode

        # Rebuild cache mapping and save
        qid_to_idx_cache = {qid: idx for idx, qid in enumerate(final_qids)}
        np.savez_compressed(
            cache_path,
            qids=np.array(final_qids, dtype=object),
            texts=np.array(final_texts, dtype=object),
            embeddings=final_embeddings.astype(np.float32),
        )
        print(f"[Cache] Saved {len(final_qids)} qids to {cache_path}")
    else:
        if cached_embeddings is None:
            raise RuntimeError("Cache file exists but embeddings are None")
        final_embeddings = cached_embeddings
        final_qids = cached_qids
        final_texts = cached_texts
        qid_to_idx_cache = {qid: idx for idx, qid in enumerate(final_qids)}
        print(f"[Cache] All required qids already cached ({len(final_qids)})")

    emb_matrix = final_embeddings
    qid2idx = qid_to_idx_cache
    print(f"Built embedding bank: {emb_matrix.shape[0]} qids, dim={emb_matrix.shape[1]}")


def get_qid_vec(qid: str) -> np.ndarray:
    """Return the embedding vector for a given qid from the global bank."""
    if emb_matrix is None:
        raise RuntimeError("Embedding matrix not initialized. Call build_qid_embedding_bank() first.")
    idx = qid2idx.get(qid)
    if idx is None:
        raise KeyError(f"qid {qid} not found in embedding bank")
    return emb_matrix[idx]


# ---------------------------
# Worker
# ---------------------------
def process_chunk(start_end_worker: Tuple[int, int, int, int, str | None]) -> Tuple[int, List[dict]]:
    """Worker function to process a contiguous slice of main_items.

    Parameters:
      (start, end, worker_idx, order_value_local, shard_dir)

    Returns:
      (start, updated_items_slice)
    """
    (start, end, worker_idx, order_value_local, shard_dir) = start_end_worker

    # Per-worker deterministic-ish rng
    random.seed(2025 + worker_idx)
    np.random.seed(2025 + worker_idx)

    local_items: List[dict] = []
    shard_fp = None
    if shard_dir:
        os.makedirs(shard_dir, exist_ok=True)
        shard_path = os.path.join(shard_dir, f"shard_{worker_idx:02d}.jsonl")
        shard_fp = open(shard_path, "w", encoding="utf-8")

    rank_key = f"rank_among_unrelevant_tags_{order_value_local}"
    sample_size_key = f"{rank_key}_sample_size"

    for item in tqdm(main_items[start:end], desc=f"worker[{worker_idx}]", unit="item", leave=False):
        try:
            qid = item['qid']
            relevant_tags = item['related_tags']
            items_ranks: List[int] = []
            rank_sample_sizes: List[int] = []

            # vector for the main item's qid
            item_vec = get_qid_vec(qid)

            for relevant_tag in relevant_tags:
                rel_qid = relevant_tag['qid']
                if rel_qid is None:
                    raise ValueError(f"rel_qid is None for relevant_tag: {relevant_tag}")

                unrelevant_tags = list(non_relation_tag_data[rel_qid])
                if not unrelevant_tags:
                    raise ValueError(f"unrelevant_tags is empty for rel_qid: {rel_qid}")

                random.shuffle(unrelevant_tags)

                # pick up to order_value_local - 1 distractors
                distractor_qids = unrelevant_tags[: max(0, order_value_local - 1)]

                # construct candidate matrix: [item] + unrelevant candidates
                candidate_vecs = [item_vec]
                for u_qid in distractor_qids:
                    try:
                        candidate_vecs.append(get_qid_vec(u_qid))
                    except KeyError:
                        # Skip qids not present in the embedding bank
                        continue

                if len(candidate_vecs) <= 1:
                    # not enough candidates; skip this relevant tag
                    continue

                all_paragraph_embeddings = np.vstack(candidate_vecs)  # (K, D)

                # vector for relevant tag qid
                rel_vec = get_qid_vec(rel_qid)

                # Similarity = dot-product between each candidate and the relevant tag
                similarities = all_paragraph_embeddings @ rel_vec  # (K,)

                # The main item is at index 0
                rank = int(np.sum(similarities > similarities[0]) + 1)
                items_ranks.append(rank)
                rank_sample_sizes.append(len(all_paragraph_embeddings))

            item[rank_key] = items_ranks
            item[sample_size_key] = rank_sample_sizes
            local_items.append(item)
            if shard_fp:
                shard_fp.write(json.dumps(item, ensure_ascii=False) + "\n")

        except Exception as ex:
            # Fail soft on this item
            item[rank_key] = []
            item[sample_size_key] = []
            local_items.append(item)
            if shard_fp:
                shard_fp.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"[worker {worker_idx}] error on qid={item.get('qid')}: {ex}")

    if shard_fp:
        shard_fp.close()

    return (start, local_items)


# ---------------------------
# Main
# ---------------------------
def main():
    global args, order_value, non_relation_tag_data, wikipedia_pages_dict, main_items, cache_dir

    args = parse_args()

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

    if args.config:
        _cfg = _load_flat_yaml(args.config)
        if "retriever" in _cfg:
            args.retriever = _cfg["retriever"]
        if "reasonir_instruction" in _cfg:
            args.reasonir_instruction = _cfg["reasonir_instruction"]
        if "qwen3_instruction" in _cfg:
            args.qwen3_instruction = _cfg["qwen3_instruction"]
        if "order" in _cfg:
            try:
                args.o = int(_cfg["order"])
                args.order = int(_cfg["order"])
            except ValueError:
                print(f"[warn] Invalid order value in config: {_cfg['order']}, using default")
        if "cuda_visible_devices" in _cfg:
            os.environ["CUDA_VISIBLE_DEVICES"] = _cfg["cuda_visible_devices"]
        if "hf_base" in _cfg:
            BASE = _cfg["hf_base"]
            os.environ["HF_BASE"] = BASE
            os.environ["HF_HOME"] = BASE
            os.environ["HF_HUB_CACHE"] = f"{BASE}/hub"
            os.environ["HF_DATASETS_CACHE"] = f"{BASE}/datasets"

    order_value = args.order if args.order is not None else args.o

    # ---------------------------
    # Input paths
    # ---------------------------
    NON_RELATION_TAGS_PATH = os.path.join(
        workspace_root, "data", "interim", "6_wiki_unrelevants_results.jsonl"
    )
    WIKIPEDIA_PAGES_PATH = os.path.join(
        workspace_root, "data", "interim", "6_all_wikipedia_pages.jsonl"
    )
    MAIN_DATASET_PATH = os.path.join(
        workspace_root, "data", "interim", "6_main_dataset.jsonl"
    )

    # ---------------------------
    # Load aux data
    # ---------------------------
    non_relation_tag_data = {}
    if os.path.exists(NON_RELATION_TAGS_PATH):
        with open(NON_RELATION_TAGS_PATH, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    non_relation_tag_data[item['qid']] = item['wiki_unrelevants']
                except json.JSONDecodeError:
                    continue
        print(f"Loaded non relation tag map -> {NON_RELATION_TAGS_PATH}")
    else:
        print(f"[warn] non relation tags file not found: {NON_RELATION_TAGS_PATH}")

    wikipedia_pages: List[Dict] = []
    if os.path.exists(WIKIPEDIA_PAGES_PATH):
        with open(WIKIPEDIA_PAGES_PATH, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    wikipedia_pages.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        print(f"Loaded {len(wikipedia_pages)} wikipedia rows -> {WIKIPEDIA_PAGES_PATH}")
    else:
        print(f"[warn] wikipedia pages file not found: {WIKIPEDIA_PAGES_PATH}")

    wikipedia_pages_dict = {page['qid']: page for page in wikipedia_pages}
    del wikipedia_pages

    main_items = []
    if os.path.exists(MAIN_DATASET_PATH):
        with open(MAIN_DATASET_PATH, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    main_items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        print(f"Loaded {len(main_items)} items -> {MAIN_DATASET_PATH}")
    else:
        raise FileNotFoundError(f"Main dataset file not found: {MAIN_DATASET_PATH}")

    _cleaned_items, _removed_tags = _strip_wiki_keywords_from_items(main_items)
    print(
        f"Removed {_removed_tags} related_tag(s) containing 'Wikipedia'/'Wikidata' across {_cleaned_items} item(s)."
    )

    cache_dir = os.path.join(workspace_root, "outputs", "cache", "embedding_cache", "embeddings")
    os.makedirs(cache_dir, exist_ok=True)

    gpus = parse_gpu_ids()
    if not gpus:
        gpus = ["0"]

    num_workers = args.num_workers if args.num_workers is not None else len(gpus)
    if num_workers <= 0:
        num_workers = 1

    print(f"Using {num_workers} worker(s); visible GPUs for embedding step: {gpus}")

    # Build the global qid embedding bank once (on main process, using retriever/GPU)
    build_qid_embedding_bank(
        retriever_name=args.retriever,
        reasonir_instruction=args.reasonir_instruction,
        qwen3_instruction=args.qwen3_instruction,
        batch_size=args.embed_batch_size,
    )

    bounds = chunkify(main_items, num_workers)
    tasks = []
    for worker_idx, (start, end) in enumerate(bounds):
        tasks.append((start, end, worker_idx, order_value, args.shard_output_dir))

    # Run workers (CPU-side) to compute ranks using the precomputed embeddings
    results = None
    try:
        from p_tqdm import p_map
        results = p_map(process_chunk, tasks)
    except Exception as e:
        print(f"[warn] p_tqdm unavailable or failed ({e}); falling back to multiprocessing.")
        import multiprocessing as mp
        ctx = mp.get_context("fork")  # safe as long as CUDA isn't inited in parent beyond emb bank
        with ctx.Pool(processes=num_workers) as pool:
            results = pool.map(process_chunk, tasks)

    # Assemble results back in order
    results = sorted(results, key=lambda x: x[0])
    updated: List[dict] = []
    for _, part in results:
        updated.extend(part)

    # Save merged results
    out_path = os.path.join(workspace_root, "data", "processed", "8_Emb_Rank", f"8_main_dataset_{order_value}_{args.retriever}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for it in updated:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"Processed {len(updated)} items across {num_workers} worker(s)")
    print(f"Saved merged results -> {out_path}")


if __name__ == "__main__":
    main()
