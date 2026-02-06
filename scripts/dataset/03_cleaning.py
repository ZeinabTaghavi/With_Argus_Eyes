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
    parser.add_argument("--low_in", type=str, default="./data/interim/2_landmarks_low_freq.jsonl")
    parser.add_argument("--low_out", type=str, default="./data/interim/3_landmarks_low_freq.jsonl")
    parser.add_argument("--high_in", type=str, default="./data/interim/2_landmarks_high_freq.jsonl")
    parser.add_argument("--high_out", type=str, default="./data/interim/3_landmarks_high_freq.jsonl")
    parser.add_argument(
        "--prune_inputs",
        action="store_true",
        help="After writing cleaned outputs, delete processed items_with_tags_*.jsonl input splits.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    file_list = []
    # Only consume the per-split tag outputs from 02_get_tags.py.
    # (Avoid mixing in cache/debug files like all_prev_items.jsonl or failed_items_*.jsonl.)
    for file in glob.glob(os.path.join(args.input_dir, "items_with_tags_*.jsonl")):
        file_list.append(file)

    print(len(file_list))
    print(sorted(file_list))

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

    if args.prune_inputs:
        deleted = 0
        for path in file_list:
            try:
                os.remove(path)
                deleted += 1
            except OSError:
                continue
        print(f"Pruned {deleted}/{len(file_list)} processed split file(s) from {args.input_dir}")

    # -------------------------------
    # Clean landmarks_low_freq
    # -------------------------------
    low_freq_in = args.low_in
    low_freq_out = args.low_out
    if os.path.exists(low_freq_in):
        low_items = []
        with open(low_freq_in, "r") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                low_items.append(obj)

        cleaned_low = []
        seen_low_keys = set()
        for item in low_items:
            all_tags = [tag for tag in item.get("related_tags", []) if not (tag.startswith('Q') and tag[1:].isdigit())]
            all_tags = [tag for tag in all_tags if tag not in ["Wikimedia category", "Wikimedia template", "Wikimedia disambiguation page"]]
            all_tags = [tag for tag in all_tags if ("_" not in tag) and ("wiki" not in tag.lower()) and ('/' not in tag)]
            if len(all_tags) == 0:
                continue

            key = _unique_key(item)
            if not key or key in seen_low_keys:
                continue
            seen_low_keys.add(key)

            item["related_tags"] = all_tags
            cleaned_low.append(item)

        print(f"Unique items kept (low_freq) by key (QID preferred): {len(seen_low_keys)}")

        with open(low_freq_out, "w") as f:
            for item in cleaned_low:
                f.write(json.dumps(item) + "\n")

    # -------------------------------
    # Clean landmarks_high_freq
    # -------------------------------
    high_freq_in = args.high_in
    high_freq_out = args.high_out
    if os.path.exists(high_freq_in):
        high_items = []
        with open(high_freq_in, "r") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                high_items.append(obj)

        cleaned_high = []
        seen_high_keys = set()
        for item in high_items:
            all_tags = [tag for tag in item.get("related_tags", []) if not (tag.startswith('Q') and tag[1:].isdigit())]
            all_tags = [tag for tag in all_tags if tag not in ["Wikimedia category", "Wikimedia template", "Wikimedia disambiguation page"]]
            all_tags = [tag for tag in all_tags if  ("_" not in tag) and ("wiki" not in tag.lower()) and ('/' not in tag)]
            if len(all_tags) == 0:
                continue

            key = _unique_key(item)
            if not key or key in seen_high_keys:
                continue
            seen_high_keys.add(key)

            item["related_tags"] = all_tags
            cleaned_high.append(item)

        print(f"Unique items kept (high_freq) by key (QID preferred): {len(seen_high_keys)}")

        with open(high_freq_out, "w") as f:
            for item in cleaned_high:
                f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    main()
