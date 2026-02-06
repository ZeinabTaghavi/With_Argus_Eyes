"""Build a qid -> related_tags mapping for unrelevant sampling.

Input: JSONL of items with `qid` and `related_tags` (e.g., 6_main_dataset.jsonl
or 5_items_with_tags_qids.jsonl).
Output: JSON dict mapping qid -> {qid, label, related_tags:[{qid,label},...]}.
"""

import argparse
import json
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 6_all_relation_tags.json from a JSONL dataset.")
    parser.add_argument(
        "--input",
        type=str,
        default="./data/interim/6_main_dataset.jsonl",
        help="Input JSONL (e.g., 6_main_dataset.jsonl or 5_items_with_tags_qids.jsonl).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/interim/6_all_relation_tags.json",
        help="Output JSON path for qid->related_tags mapping.",
    )
    return parser.parse_args()


def _normalize_related_tags(item_qid: str, related_tags) -> list[dict]:
    """Return list of {qid,label} dicts, deduped and without self."""
    out = []
    seen = set()
    if not isinstance(related_tags, list):
        return out

    for tag in related_tags:
        if isinstance(tag, dict):
            tqid = tag.get("qid")
            if not tqid or tqid == item_qid or tqid in seen:
                continue
            seen.add(tqid)
            out.append({"qid": tqid, "label": tag.get("label", "")})
        elif isinstance(tag, str):
            # If input tags are strings (unexpected), keep as label-only
            label = tag.strip()
            if not label:
                continue
            out.append({"qid": None, "label": label})
    return out


def main() -> None:
    args = parse_args()

    mapping = {}
    total_items = 0
    total_tags = 0

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue

            qid = item.get("qid") or item.get("Q_number") or item.get("QID")
            if not qid:
                continue

            related = _normalize_related_tags(qid, item.get("related_tags", []))
            mapping[qid] = {
                "qid": qid,
                "label": item.get("label", ""),
                "related_tags": related,
            }
            total_items += 1
            total_tags += len(related)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False)

    print(f"Wrote {len(mapping)} qids to {args.output}")
    print(f"Total items processed: {total_items}")
    print(f"Total related tags kept: {total_tags}")


if __name__ == "__main__":
    main()
