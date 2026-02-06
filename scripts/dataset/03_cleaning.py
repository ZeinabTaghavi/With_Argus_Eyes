"""Clean and deduplicate tag-only items."""

import argparse
import glob
import json
import os
import unicodedata

# Helper: canonicalize labels for robust uniqueness
def _canon_label(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return unicodedata.normalize("NFKC", s).casefold().strip()

# Prefer QID for uniqueness; fall back to canonicalized label
def _unique_key(item: dict) -> str:
    qid = item.get("Q_number") or item.get("qid") or item.get("QID")
    if isinstance(qid, str) and qid.strip():
        return qid.strip()
    return _canon_label(item.get("label", ""))



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean and deduplicate tag-only items.")
    parser.add_argument("--input_dir", type=str, default="./data/interim/2_tag_only", help="Directory of tag-only JSONL files.")
    parser.add_argument("--out_cleaned", type=str, default="./data/interim/3_cleaned_items_tag_only.jsonl", help="Output JSONL for cleaned items.")
    parser.add_argument(
        "--prune_inputs",
        action="store_true",
        help="After writing cleaned outputs, delete processed items_with_tags_*.jsonl input splits.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    written_files = []

    file_list = []
    # Only consume the per-split tag outputs from 02_get_tags.py.
    # (Avoid mixing in cache/debug files like all_prev_items.jsonl or failed_items_*.jsonl.)
    for file in glob.glob(os.path.join(args.input_dir, "items_with_tags_*.jsonl")):
        file_list.append(file)

    print(f"Input splits found: {len(file_list)} (from {args.input_dir})")

    # Initialize empty list to store all items
    all_items = []

    # Load and combine items from all files
    for file_path in file_list:
        with open(file_path, "r") as f:
            for line in f:
                item = json.loads(line)
                all_items.append(item)

    print(f"Total items loaded: {len(all_items)}")
    print("Sample item:", all_items[0] if all_items else None)

    cleaned_items = []
    seen_keys = set()
    for item in all_items:
        all_tags = [tag for tag in item.get("related_tags", []) if not (tag.startswith('Q') and tag[1:].isdigit())]
        all_tags = [tag for tag in all_tags if tag not in ["Wikimedia category", "Wikimedia template", "Wikimedia disambiguation page"]]
        all_tags = [tag for tag in all_tags if "_" not in tag and '/' not in tag and 'wiki' not in tag.lower()]

        if len(all_tags) == 0:
            continue

        key = _unique_key(item)
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)

        item["related_tags"] = all_tags
        cleaned_items.append(item)

    print(f"Total items loaded: {len(cleaned_items)}")
    print("Sample item:", cleaned_items[0] if cleaned_items else None)
    print(f"Unique items kept (main) by key (QID preferred): {len(seen_keys)}")

    with open(args.out_cleaned, "w") as f:
        for item in cleaned_items:
            f.write(json.dumps(item) + "\n")
    written_files.append(args.out_cleaned)

    if args.prune_inputs:
        deleted = 0
        for path in file_list:
            try:
                os.remove(path)
                deleted += 1
            except OSError:
                continue
        print(f"Pruned {deleted}/{len(file_list)} processed split file(s) from {args.input_dir}")

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
        except Exception as exc:
            return f"preview_error={type(exc).__name__}"
        return "empty_file"

    # de-dupe while preserving order
    uniq_written = []
    seen = set()
    for p in written_files:
        if p in seen:
            continue
        seen.add(p)
        uniq_written.append(p)

    print("\n[3_CLEANING] Output summary")
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
