from __future__ import annotations
from typing import Dict, List, Tuple
import json
import os

import numpy as np

from .data_io import load_jsonl
from with_argus_eyes.utils.risk.scores import get_risk_fn, normalized_score


def compute_and_print_risk_scores_for_jsonl(
    file_path: str,
    *,
    phrase_type: str,
    risk_name: str,
    risk_kwargs: dict | None = None,
    normalization_kwargs: dict,
    out_path: str | None = None,
) -> list[dict]:
    """
    Compute the risk score for every item in `file_path` using the same risk_fn
    (get_risk_fn) and print them. Optionally saves a JSONL with "risk_score" added.

    Returns a list of dicts: [{"label": ..., "qid": ..., "risk_score": float}, ...]
    """
    items = load_jsonl(file_path)
    if not items:
        print(f"[risk-scan] No items found in: {file_path}")
        return []


    risk_fn = get_risk_fn(risk_name)

    rows: list[dict] = []
    scores = []
    for i, it in enumerate(items):
        try:
            score = risk_fn(it, phrase_type, **risk_kwargs)
        except Exception as e:
            # Be robust to weird rows; record NaN and continue
            print(f"[risk-scan][warn] row {i} failed ({e}); setting score=nan")
            score = float("nan")
        scores.append(score)

    scores = normalized_score(scores, **normalization_kwargs)

    for i, score, it in zip(range(len(items)), scores, items):
        label = it.get("label") or it.get("itemLabel") or it.get("item_label") or ""
        qid   = it.get("qid") or it.get("Q_number") or it.get("itemQ") or ""
        ranks = it.get(f"{phrase_type}_related_tags_ranks") or {}
        rows.append({"idx": i, "label": label, "qid": qid, "risk_score": float(score), "ranks": ranks})

    # Print nicely
    print(f"\n[risk-scan] {len(rows)} items from {file_path}:")
    for r in rows:
        print(f"  [{r['idx']:05d}] risk={r['risk_score']:.6f}  qid={r['qid']}  label={r['label']}")

    # Optional save
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[risk-scan] wrote: {out_path}")

    return rows
