"""Add irrelevant tags to the dataset."""

import os
import json
import argparse
import random
from tqdm import tqdm
import multiprocessing as mp

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add irrelevant tags to the dataset.")
    parser.add_argument("--order", type=int, default=5000, help="Number of unrelevant tags to sample per qid.")
    parser.add_argument("--out", type=str, default='./data/interim/7_unrelevant_qids.jsonl', help="Output JSONL path.")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of worker processes.")
    return parser.parse_args()

# -------------------------------------------------------------
# Parallel unrelevant sampling
# -------------------------------------------------------------
_GLOBAL_RELATED = None
_GLOBAL_ALL_QIDS = None
_GLOBAL_OUT_PATH = None
_GLOBAL_LOCK = None
_GLOBAL_ORDER = None

def _init_pool(related_map, all_qids_seq, out_path, lock, order):
    """Initializer for worker processes; set read-only globals."""
    global _GLOBAL_RELATED, _GLOBAL_ALL_QIDS, _GLOBAL_OUT_PATH, _GLOBAL_LOCK, _GLOBAL_ORDER
    _GLOBAL_RELATED = related_map
    _GLOBAL_ALL_QIDS = list(all_qids_seq)
    _GLOBAL_OUT_PATH = out_path
    _GLOBAL_LOCK = lock
    _GLOBAL_ORDER = int(order) if order is not None else 0


def _compute_and_write_unrelevant(target_qid: str):
    """Randomly sample up to `_GLOBAL_ORDER` unrelevant qids for one target and append one JSONL line."""
    related_qids = _GLOBAL_RELATED.get(target_qid, set())

    # Determine how many to sample; cannot exceed the available unrelevant pool
    max_possible = max(0, len(_GLOBAL_ALL_QIDS) - len(related_qids) - 1)  # minus self
    k = _GLOBAL_ORDER if _GLOBAL_ORDER is not None else 0
    if k < 0:
        k = 0
    if k > max_possible:
        k = max_possible

    picked = set()
    if k > 0 and len(_GLOBAL_ALL_QIDS) > 0:
        rng = random.SystemRandom()
        # Rejection-sample unique unrelevant qids without materializing the full complement
        while len(picked) < k:
            cand = _GLOBAL_ALL_QIDS[rng.randrange(len(_GLOBAL_ALL_QIDS))]
            if cand == target_qid:
                continue
            if cand in related_qids:
                continue
            if cand in picked:
                continue
            picked.add(cand)

    line = json.dumps({"qid": target_qid, "unrelevant": list(picked)}) + "\n"
    with _GLOBAL_LOCK:
        with open(_GLOBAL_OUT_PATH, "a", encoding="utf-8") as f:
            f.write(line)
    return True


def return_unrelevant_tags_to_jsonl(main_dataset_items, all_qid_related_tags, all_qids, out_jsonl, num_workers=None, order=100):
    """
    Parallel computation that writes each (qid, sampled_unrelevant_list) as one JSONL line to `out_jsonl`.
    Randomly samples `order` unrelevant qids per target qid without constructing massive complement lists.
    """
    # Collect unique 'relevant' qids appearing in main dataset
    relevant_qids = []
    seen = set()
    for item in tqdm(main_dataset_items, total=len(main_dataset_items), desc="Collecting relevant qids"):
        for tag in item.get('related_tags', []):
            q = tag.get('qid')
            if q is not None and q in all_qid_related_tags and q not in seen:
                seen.add(q)
                relevant_qids.append(q)

    if not relevant_qids:
        # touch empty file for consistency
        with open(out_jsonl, "w", encoding="utf-8") as _:
            pass
        return

    # Fresh output file
    if os.path.exists(out_jsonl):
        os.remove(out_jsonl)

    if num_workers is None:
        num_workers = min(len(relevant_qids), mp.cpu_count() or 1)

    mgr = mp.Manager()
    lock = mgr.RLock()

    # Use chunksize to reduce IPC overhead for large inputs
    chunksize = max(1, len(relevant_qids) // (num_workers * 8))

    with mp.Pool(
        processes=num_workers,
        initializer=_init_pool,
        initargs=(all_qid_related_tags, all_qids, out_jsonl, lock, order),
    ) as pool:
        for _ in tqdm(
            pool.imap_unordered(_compute_and_write_unrelevant, relevant_qids, chunksize=chunksize),
            total=len(relevant_qids),
            desc="Writing unrelevant qids",
        ):
            pass


def main() -> None:
    print("Number of CPUs available:", os.cpu_count())

    args = parse_args()

    # -------------------------------------------------------------
    # Load inputs
    # -------------------------------------------------------------
    with open('./data/interim/7_all_relation_tags.json', 'r') as f:
        all_relation_tags = json.load(f)
    print(f"Loaded {len(all_relation_tags)} entries from 7_all_relation_tags.json")

    with open('./data/interim/7_main_dataset.jsonl', 'r') as f:
        main_dataset = [json.loads(line) for line in f]
    print(f"Loaded {len(main_dataset)} entries from 7_main_dataset.jsonl")

    # Build mapping: qid -> set(related_qids) and the global universe of qids
    all_qids_set = set()
    all_qid_related_tags = {}
    for qid, tags in tqdm(all_relation_tags.items(), total=len(all_relation_tags), desc="Gathering all qids"):
        tags_qids = [t['qid'] for t in tags['related_tags']]
        all_qid_related_tags[qid] = set(tags_qids)
        all_qids_set.update(tags_qids)

    all_qids = list(all_qids_set)

    out_path = args.out
    return_unrelevant_tags_to_jsonl(
        main_dataset,
        all_qid_related_tags,
        all_qids,
        out_path,
        num_workers=args.workers,
        order=args.order,
    )
    print(f"Wrote per-qid unrelevant lists to: {out_path}")


if __name__ == "__main__":
    main()


'''
python scripts/dataset/09_add_unrelevants.py \
  --order 5000 \
  --out ./data/interim/7_unrelevant_qids.jsonl \

'''
