import os
import sys


def main():
    # Add workspace root to Python path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    if workspace_root not in sys.path:
        sys.path.insert(0, workspace_root)

    # Allow override by config or env
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")

    import json
    from typing import List, Dict, Tuple
    import numpy as np
    from tqdm import tqdm

    # 1) Optional: set HF caches (like your original)
    BASE = os.environ.get("HF_BASE", "../../../../data/proj/zeinabtaghavi")
    os.environ.setdefault("HF_HOME",           BASE)
    os.environ.setdefault("HF_HUB_CACHE",      f"{BASE}/hub")
    os.environ.setdefault("HF_DATASETS_CACHE", f"{BASE}/datasets")

    # 2) CLI
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="", help="Path to YAML config file with flat key:value pairs.")
    parser.add_argument("--retriever", type=str, default="contriever", choices=["contriever", "reasonir", "qwen3", "jina"])
    parser.add_argument("--reasonir_instruction", type=str, default="")
    parser.add_argument("--qwen3_instruction", type=str, default="")
    parser.add_argument("--o", type=str, default="all", help="Order of the tag universe per item (e.g., 'all' or '10000').")
    parser.add_argument("--order", type=str, default=None, help="Alias for --o (string).")
    args, _ = parser.parse_known_args()

    # --- Load config (flat YAML) if provided and override args/env ---
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

    if args.config:
        _cfg = _load_flat_yaml(args.config)
        # override argparse values if present
        if "retriever" in _cfg:
            args.retriever = _cfg["retriever"]
        if "reasonir_instruction" in _cfg:
            args.reasonir_instruction = _cfg["reasonir_instruction"]
        if "qwen3_instruction" in _cfg:
            args.qwen3_instruction = _cfg["qwen3_instruction"]
        if "order" in _cfg:
            args.o = _cfg["order"]
            args.order = _cfg["order"]
        if "cuda_visible_devices" in _cfg:
            os.environ["CUDA_VISIBLE_DEVICES"] = _cfg["cuda_visible_devices"]
        if "hf_base" in _cfg:
            BASE = _cfg["hf_base"]
            os.environ["HF_BASE"] = BASE
            os.environ["HF_HOME"] = BASE
            os.environ["HF_HUB_CACHE"] = f"{BASE}/hub"
            os.environ["HF_DATASETS_CACHE"] = f"{BASE}/datasets"

    order_value = args.order if args.order is not None else args.o
    rng = np.random.default_rng(2025)  # deterministic RNG for sub-sampling

    # 3) Build retriever
    from with_argus_eyes.utils.embeddings import build_retriever
    retriever = build_retriever(
        args.retriever,
        reasonir_instruction=args.reasonir_instruction,
        qwen3_instruction=args.qwen3_instruction,
    )

    # 4) Load items
    all_items: List[dict] = []
    with open('./data/interim/5_items_with_wikipedia_and_desc.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_items.append(json.loads(line))
    print(f"Loaded {len(all_items)} items")

    # 5) Collect all unique tags
    all_tags: List[str] = []
    for it in all_items:
        all_tags.extend([t for t in (it.get("related_tags") or []) if isinstance(t, str) and t.strip()])
    all_tags = sorted(set(all_tags))
    print(f"Unique tags: {len(all_tags)}")

    # 6) Cache dir
    cache_dir = "Cache/embeddings"
    os.makedirs(cache_dir, exist_ok=True)

    # 7) Load or build tag embeddings
    tags_cache_path = os.path.join(cache_dir, f"{args.retriever}_all_tags.npz")
    if os.path.exists(tags_cache_path):
        data = np.load(tags_cache_path, allow_pickle=True)
        cached_texts = data["texts"].tolist()
        cached_embeds = data["embeddings"].astype(np.float32)  # (T,H)
        all_tags = cached_texts
        all_tags_embeddings = cached_embeds
        print(f"[Cache] Loaded {len(all_tags)} tags vectors -> {tags_cache_path}")
    else:
        tag_vec_list = retriever.encode_texts(all_tags, batch_size=64, max_length=256)  # List[np.ndarray]
        all_tags_embeddings = np.vstack(tag_vec_list).astype(np.float32)               # (T,H)
        np.savez_compressed(tags_cache_path, texts=np.array(all_tags, dtype=object), embeddings=all_tags_embeddings)
        print(f"[Cache] Saved tags vectors -> {tags_cache_path}")

    tag_to_index: Dict[str, int] = {t: i for i, t in enumerate(all_tags)}

    # 8) Phrase encoders + caching
    def _phrase_key(item: dict, phrase_type: str) -> Tuple[str, str]:
        """Return (text_for_encoding, label_for_span) required by encoder."""
        label = (item.get("label") or "").strip()
        if phrase_type == "label":
            return label, label
        if phrase_type == "wikipedia_first_paragraph":
            paragraph = (item.get("wikipedia_first_paragraph") or "").strip()
            if not paragraph:
                return "", label
            txt = f"{label} : {paragraph}"
            return txt, label
        if phrase_type == "wikidata_description":
            desc_text = (item.get("wikidata_description") or "").strip()
            if not desc_text:
                return "", label
            txt = desc_text if label in desc_text else f"{label} : {desc_text}"
            return txt, label
        raise ValueError(phrase_type)

    def precompute_phrase_embeddings(items: List[dict], phrase_type: str) -> Dict[str, np.ndarray]:
        """
        Returns dict: phrase_text -> embedding (np.ndarray, L2-normalized)
        Uses span-mode for Contriever on paragraph/description to mimic your previous behavior.
        """
        cache_path = os.path.join(cache_dir, f"{args.retriever}_{phrase_type}.npz")
        loaded: Dict[str, np.ndarray] = {}
        if os.path.exists(cache_path):
            data = np.load(cache_path, allow_pickle=True)
            for txt, vec in zip(data["texts"].tolist(), data["embeddings"].astype(np.float32)):
                loaded[txt] = vec
            print(f"[Cache] Loaded {len(loaded)} {phrase_type} vectors -> {cache_path}")

        # Collect unique texts to encode
        texts: List[str] = []
        phrases: List[str] = []
        idx_map: Dict[int, str] = {}
        for idx, it in enumerate(items):
            txt, lab = _phrase_key(it, phrase_type)
            if not txt:
                continue
            if txt in loaded:
                continue
            idx_map[len(texts)] = txt
            texts.append(txt)
            phrases.append(lab)

        if texts:
            if args.retriever == "contriever" and phrase_type in ("wikipedia_first_paragraph", "wikidata_description"):
                vecs = retriever.encode_spans(texts, phrases, occurrence_index=0, max_length=512)  # List
            else:
                vecs = retriever.encode_texts(texts, batch_size=64, max_length=256)

            # Save/merge
            for i, v in enumerate(vecs):
                loaded[idx_map[i]] = v.astype(np.float32)
            all_txts = list(loaded.keys())
            all_vecs = np.vstack([loaded[t] for t in all_txts]).astype(np.float32)
            np.savez_compressed(cache_path, texts=np.array(all_txts, dtype=object), embeddings=all_vecs)
            print(f"[Cache] Saved {len(all_txts)} {phrase_type} vectors -> {cache_path}")

        return loaded

    # 9) Ranking
    def compute_related_tag_ranks_for_phrase_type(
        items: List[dict],
        phrase_type: str,
        order: str = "all",
        rng: np.random.Generator | None = None
    ) -> List[dict]:
        out = []
        phrase_map = precompute_phrase_embeddings(items, phrase_type)
        T = len(all_tags)
        use_all = (str(order).lower() == "all")
        if not use_all:
            try:
                o_val = int(str(order))
            except Exception:
                raise ValueError(f"Invalid order '{order}'. Use 'all' or an integer like 10000.")
            # clamp to [0, T]
            o_val = max(0, min(o_val, T))
        else:
            o_val = T
        if rng is None:
            rng = np.random.default_rng(2025)

        for it in tqdm(items, desc=f"Ranking tags ({phrase_type})", unit="item"):
            label = (it.get("label") or "").strip()
            rel_tags = [t.strip() for t in (it.get("related_tags") or []) if isinstance(t, str) and t.strip()]
            txt, lab = _phrase_key(it, phrase_type)
            if not txt:
                raise ValueError(f"txt is None for item: {it}")

            # Get (or encode) vector
            if txt in phrase_map:
                vec = phrase_map[txt]  # (H,)
            else:
                if args.retriever == "contriever" and phrase_type in ("wikipedia_first_paragraph", "wikidata_description"):
                    vec = retriever.encode_spans([txt], [lab], occurrence_index=0, max_length=512)[0]
                else:
                    vec = retriever.encode_texts([txt], batch_size=1, max_length=256)[0]

            # Build indices for item tags (always included)
            item_tag_indices = []
            for tag in rel_tags:
                idx = tag_to_index.get(tag)
                if idx is not None:
                    item_tag_indices.append(idx)
            item_tag_indices = np.unique(np.array(item_tag_indices, dtype=int)) if item_tag_indices else np.array([], dtype=int)

            if use_all:
                subset_indices = np.arange(len(all_tags), dtype=int)
            else:
                need = max(o_val - int(item_tag_indices.size), 0)
                if need > 0:
                    # candidate pool = all tags excluding the item's own tags
                    all_idx = np.arange(T, dtype=int)
                    if item_tag_indices.size > 0:
                        mask = np.ones(T, dtype=bool)
                        mask[item_tag_indices] = False
                        pool = all_idx[mask]
                    else:
                        pool = all_idx
                    k = min(need, pool.size)
                    sampled = rng.choice(pool, size=k, replace=False) if k > 0 else np.array([], dtype=int)
                else:
                    sampled = np.array([], dtype=int)

                subset_indices = sampled if item_tag_indices.size == 0 else np.concatenate([item_tag_indices, sampled])

            # Cosine sim = dot product (already L2) on the selected subset
            sims_sub = vec @ all_tags_embeddings[subset_indices].T  # (S,)
            order_sub = np.argsort(-sims_sub)
            ranks_sub = np.empty_like(order_sub)
            ranks_sub[order_sub] = np.arange(1, len(order_sub) + 1)

            # Map back to per-tag ranks/scores for the item's tags
            pos_in_subset = {int(gidx): int(pos) for pos, gidx in enumerate(subset_indices.tolist())}

            tag_ranks: Dict[str, int] = {}
            tag_scores: Dict[str, float] = {}
            for tag in rel_tags:
                gidx = tag_to_index.get(tag)
                if gidx is None:
                    continue
                pos = pos_in_subset.get(int(gidx))
                if pos is None:
                    continue
                tag_ranks[tag] = int(ranks_sub[pos])
                tag_scores[tag] = float(sims_sub[pos])

            out.append({**it,
                f"{phrase_type}_related_tags_ranks": tag_ranks,
                f"{phrase_type}_related_tags_scores": tag_scores,
            })
        return out

    # 10) Run all phrase types
    augmented_items = compute_related_tag_ranks_for_phrase_type(
        all_items, "wikipedia_first_paragraph", order=order_value, rng=rng
    )

    # 11) Save
    out_path = f"./outputs/6_items_with_tag_ranks_{args.retriever}_o_{order_value}.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for obj in tqdm(augmented_items, desc="Saving", unit="item"):
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Saved {len(augmented_items)} items -> {out_path}")

    # 12) Optional: process landmark splits
    def process_landmarks_file(file_path: str, output_suffix: str, order: str, rng: np.random.Generator):
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return
        items = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        items.append(json.loads(line))
                    except Exception:
                        pass
        if not items:
            print(f"No valid items in {file_path}")
            return

        print(f"Processing {len(items)} items from {file_path}")
        a = compute_related_tag_ranks_for_phrase_type(items, "label", order=order, rng=rng)
        if any(it.get("wikipedia_first_paragraph") for it in items):
            a = compute_related_tag_ranks_for_phrase_type(a, "wikipedia_first_paragraph", order=order, rng=rng)
        if any(it.get("wikidata_description") for it in items):
            a = compute_related_tag_ranks_for_phrase_type(a, "wikidata_description", order=order, rng=rng)

        out_path = f"./outputs/6_landmarks_{output_suffix}_with_tag_ranks_{args.retriever}_o_{order}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for obj in tqdm(a, desc=f"Saving {output_suffix}", unit="item"):
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        print(f"Saved {len(a)} {output_suffix} items -> {out_path}")

    process_landmarks_file("landmarks_low_freq.jsonl",  "low_freq", order=order_value, rng=rng)
    process_landmarks_file("landmarks_high_freq.jsonl", "high_freq", order=order_value, rng=rng)

if __name__ == "__main__":
    main()
