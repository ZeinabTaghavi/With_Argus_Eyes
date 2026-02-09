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
        default="./data/interim/7_all_related_tags.json",
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

    written_files: list[str] = []

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
    if os.path.exists(args.output):
        written_files.append(args.output)

    # ---------------------------------------------
    # Output summary (what was stored and where)
    # ---------------------------------------------

    def _human_size(num_bytes: int) -> str:
        if num_bytes < 1024:
            return f"{num_bytes} B"
        if num_bytes < 1024**2:
            return f"{num_bytes / 1024:.2f} KB"
        if num_bytes < 1024**3:
            return f"{num_bytes / (1024**2):.2f} MB"
        return f"{num_bytes / (1024**3):.2f} GB"

    def _jsonl_preview(path: str) -> str:
        try:
            if path.endswith(".jsonl"):
                with open(path, "r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            keys = sorted(obj.keys())
                            return f"first_record_keys={keys}"
                        return f"first_record_type={type(obj).__name__}"
                return "empty_file"
        except Exception as exc:
            return f"preview_error={type(exc).__name__}"

        # For this script the output is JSON (qid -> record). Preview the *record* keys.
        try:
            with open(path, "r", encoding="utf-8") as handle:
                obj = json.load(handle)
            if isinstance(obj, dict) and obj:
                first_val = next(iter(obj.values()))
                if isinstance(first_val, dict):
                    keys = sorted(first_val.keys())
                    return f"first_record_keys={keys}"
                return f"first_record_type={type(first_val).__name__}"
        except Exception as exc:
            return f"preview_error={type(exc).__name__}"
        return "empty_file"

    uniq_written: list[str] = []
    seen = set()
    for p in written_files:
        if p in seen:
            continue
        seen.add(p)
        uniq_written.append(p)

    print("\n[7_BUILD_ALL_RELATION_TAGS] Output summary")
    if not uniq_written:
        print("No output files recorded.")
    else:
        print(f"Recorded {len(uniq_written)} output file(s):")
        for p in uniq_written:
            abs_p = os.path.abspath(p)
            if not os.path.exists(p):
                print(f" - {abs_p} (missing)")
                continue
            size = _human_size(os.path.getsize(p))
            preview = _jsonl_preview(p)
            print(f" - {abs_p} ({size}) {preview}")


if __name__ == "__main__":
    main()
