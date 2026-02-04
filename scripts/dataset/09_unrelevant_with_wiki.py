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
        default=os.path.join("data", "interim", "7_all_wikipedia_pages.jsonl"),
        help="Wikipedia pages JSONL path.",
    )
    parser.add_argument(
        "--unrelevant_tags",
        type=str,
        default="data/interim/7_unrelevant_qids_sampled.jsonl",
        help="Unrelevant tags JSONL path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/interim/7_wiki_unrelevants_results.jsonl",
        help="Output JSONL path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

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


if __name__ == "__main__":
    main()
