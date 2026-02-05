"""Final cleaning and export of the main dataset."""

import argparse
import json
from tqdm import tqdm

def wikipedia_page_cleaning(item):
    """Return True if the wikipedia_first_paragraph is meaningful, otherwise False."""
    paragraph = item.get('wikipedia_first_paragraph', '').strip()
    if ((len(paragraph) < 60 and paragraph.endswith('may refer to:')) or
        (len(paragraph) < 60 and paragraph.endswith('is a given name and a surname.')) or
        (len(paragraph) < 60 and paragraph.endswith('is a given name.')) or
        (len(paragraph) < 60 and paragraph.endswith('is a surname.'))):
        return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final cleaning to build main dataset and Wikipedia pages file.")
    parser.add_argument("--items_with_tags", type=str, default="./data/interim/6_items_with_tags_qids.jsonl")
    parser.add_argument("--wikipedia_tags", type=str, default="./data/interim/6_tags_wikipedia_first_paragraphs.jsonl")
    parser.add_argument("--output_wikipedia_pages", type=str, default="./data/interim/7_all_wikipedia_pages.jsonl")
    parser.add_argument("--output_main_dataset", type=str, default="./data/interim/7_main_dataset.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading items with tags from:", args.items_with_tags)
    with open(args.items_with_tags, "r") as f:
        data = [json.loads(line) for line in tqdm(f, desc="Reading items with tags")]

    print(f"Loaded {len(data)} items with tags.")

    wikipedia_path = args.wikipedia_tags
    print("Loading Wikipedia first paragraphs from:", wikipedia_path)
    with open(wikipedia_path, 'r') as f:
        wikipedia_data = [json.loads(line) for line in tqdm(f, desc="Reading Wikipedia paragraphs")]

    print(f"Loaded {len(wikipedia_data)} wikipedia entries.")

    all_wikipedia_qids = [item['qid'] for item in wikipedia_data]
    assert len(all_wikipedia_qids) == len(wikipedia_data)
    print("All Wikipedia QIDs loaded: ", len(all_wikipedia_qids))

    print("Merging missing items from data into wikipedia_data if they don't already exist...")
    newly_added = 0
    for item in tqdm(data, desc="Merging items"):
        if item['qid'] not in all_wikipedia_qids:
            wiki_item = {}
            wiki_item['qid'] = item['qid']
            wiki_item['label'] = item['label']
            wiki_item['wikipedia_first_paragraph'] = item.get('wikipedia_first_paragraph', '')
            all_wikipedia_qids.append(item['qid'])
            wikipedia_data.append(wiki_item)
            newly_added += 1
    if newly_added > 0:
        print(f"Added {newly_added} new wikipedia entries from data.")
    else:
        print("No new wikipedia entries were added from data.")

    print(f"Merged Wikipedia data size: {len(wikipedia_data)}")

    clean_related_tags = []
    output_path = args.output_wikipedia_pages
    print(f"Saving cleaned wikipedia pages to: {output_path}")
    with open(output_path, 'w') as f:
        num_cleaned = 0
        num_skipped = 0
        for item in tqdm(wikipedia_data, desc="Cleaning Wikipedia entries and writing to file"):
            if wikipedia_page_cleaning(item):
                clean_related_tags.append(item['qid'])
                f.write(json.dumps(item) + '\n')
                num_cleaned += 1
            else:
                num_skipped += 1
    print(f"Saved {num_cleaned} valid wikipedia pages. Skipped {num_skipped} invalid entries.")

    main_dataset = []
    removing_Data = []

    print("Filtering main data to retain only items and their related_tags with cleaned wikipedia paragraphs...")
    for item in tqdm(data, desc="Filtering main dataset"):
        if item['qid'] in clean_related_tags:
            new_item = {}
            new_item['qid'] = item['qid']
            new_item['label'] = item['label']
            new_item['related_tags'] = [
                {'qid': sub_item['qid'], 'label': sub_item['label']}
                for sub_item in item.get('related_tags', [])
                if sub_item['qid'] in clean_related_tags
            ]
            if len(new_item['related_tags']) == 0:
                removing_Data.append(item)
                continue
            else:
                main_dataset.append(new_item)
        else:
            removing_Data.append(item)

    print(f"Final: {len(main_dataset)} items remaining in main dataset after cleaning.")
    print(f"Removed: {len(removing_Data)} items that didn't have valid wikipedia entries.")

    output_main_dataset_path = args.output_main_dataset
    print(f"Saving main dataset to: {output_main_dataset_path}")
    with open(output_main_dataset_path, 'w') as f:
        for item in main_dataset:
            f.write(json.dumps(item) + '\n')
    print(f"Saved {len(main_dataset)} items to {output_main_dataset_path}")


if __name__ == "__main__":
    main()
