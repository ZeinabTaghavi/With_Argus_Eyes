"""Intersect irrelevant tags with Wikipedia QIDs."""

import argparse
import json
import os
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Intersect unrelevant tags with Wikipedia QIDs.")
    parser.add_argument(
        "--main_dataset",
        type=str,
        default=os.path.join("data", "processed", "8_Emb_Rank", "8_main_dataset.jsonl"),
        help="Main dataset path (kept for parity; not directly used).",
    )
    parser.add_argument(
        "--wikipedia_pages",
        type=str,
        default=os.path.join("data", "interim", "6_all_wikipedia_pages.jsonl"),
        help="Wikipedia pages JSONL path.",
    )
    parser.add_argument(
        "--unrelevant_tags",
        type=str,
        default="data/interim/6_unrelevant_qids.jsonl",
        help="Unrelevant tags JSONL path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/interim/6_wiki_unrelevants_results.jsonl",
        help="Output JSONL path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    written_files: list[str] = []

    unrelevant_tags = []
    with open(args.unrelevant_tags, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading unrelevant tags"):
            line = line.strip()
            if not line:
                continue
            unrelevant_tags.append(json.loads(line))

    wikipedia_pages = []
    with open(args.wikipedia_pages, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                wikipedia_pages.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    print(f"Loaded {len(wikipedia_pages)} wikipedia pages from {args.wikipedia_pages}")

    all_wiki_qids = {page['qid'] for page in wikipedia_pages}

    all_unrelevants = []
    for unrelevant_tag in tqdm(unrelevant_tags):
        all_unrelevants.extend(unrelevant_tag['unrelevant'])

    all_unrelevants_set = set(all_unrelevants)

    all_unrelevants_in_wiki = all_unrelevants_set.intersection(all_wiki_qids)
    print(f"Number of unrelevant tags that are also wikipedia qids: {len(all_unrelevants_in_wiki)}")

    results = []
    for item in tqdm(unrelevant_tags):
        qid = item['qid']
        set_unrelevants = set(item['unrelevant'])
        set_unrelevants_in_wiki = set_unrelevants.intersection(all_unrelevants_in_wiki)
        results.append({
            'qid': qid,
            'wiki_unrelevants': list(set_unrelevants_in_wiki)
        })

    with open(args.output, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Saved results as JSONL to {args.output}")
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
    uniq_written: list[str] = []
    seen = set()
    for p in written_files:
        if p in seen:
            continue
        seen.add(p)
        uniq_written.append(p)

    print("\n[9_UNRELEVANT_WITH_WIKI] Output summary")
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
