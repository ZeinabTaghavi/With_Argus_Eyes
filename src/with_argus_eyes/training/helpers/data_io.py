from __future__ import annotations
from typing import List, Dict, Tuple
import os, json
import numpy as np

def load_jsonl(path: str) -> List[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items

def load_rank_file(retriever: str, using_file_path: bool = False, file_path: str = "") -> List[dict]:
    if using_file_path:
        return load_jsonl(file_path)
    else:
        path = f"./outputs/6_items_with_tag_ranks_{retriever}.jsonl"
        return load_jsonl(path)

def load_phrase_cache(retriever: str, phrase_type: str) -> Dict[str, np.ndarray]:
    cache_path = os.path.join("./Cache/embeddings", f"{retriever}_{phrase_type}.npz")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cached phrase embeddings not found: {cache_path}")
    data = np.load(cache_path, allow_pickle=True)
    texts = data["texts"].tolist()
    embeds = data["embeddings"].astype(np.float32)

    # print("[load_phrase_cache] len(texts):", len(texts))
    # print("[load_phrase_cache] texts[0]:", texts[0])
    # print("[load_phrase_cache] len(embeds):", len(embeds))
    return {t: e for t, e in zip(texts, embeds)}

def phrase_key_for_item(item: dict, phrase_type: str) -> str:
    if phrase_type != "wikipedia_first_paragraph":
        raise ValueError(
            f"Unsupported phrase_type={phrase_type!r}. "
            "Only 'wikipedia_first_paragraph' is supported in the current pipeline."
        )
    paragraph = (item.get("wikipedia_first_paragraph") or "").strip()
    return paragraph
