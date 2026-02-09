"""Add irrelevant tags to the dataset."""

import os
import json
import argparse
import random
from tqdm import tqdm
import multiprocessing as mp
from typing import Any

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add irrelevant tags to the dataset.")
    parser.add_argument("--order", type=int, default=800, help="Number of unrelevant tags to sample per qid.")
    parser.add_argument("--out", type=str, default='./data/interim/8_unrelevant_qids.jsonl', help="Output JSONL path.")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of worker processes.")
    parser.add_argument(
        "--all_relation_tags",
        type=str,
        default="./data/interim/6_all_relation_tags.json",
        help="Input JSON mapping qid -> {qid,label,related_tags:[{qid,label},...]} (output of stage 07).",
    )
    parser.add_argument(
        "--main_dataset",
        type=str,
        default="./data/interim/6_main_dataset.jsonl",
        help="Input JSONL main dataset (output of stage 06).",
    )
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


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def return_unrelevant_tags_to_jsonl(
    main_dataset_items,
    all_qid_related_tags,
    all_qids,
    out_jsonl,
    num_workers=None,
    order=100,
):
    """
    Parallel computation that writes each (qid, sampled_unrelevant_list) as one JSONL line to `out_jsonl`.
    Randomly samples `order` unrelevant qids per target qid without constructing massive complement lists.
    """
    _ensure_parent_dir(out_jsonl)

    # Collect unique target qids from main dataset items.
    # This is the expected behavior for generating one unrelevant list per item qid.
    relevant_qids = []
    seen = set()
    for item in tqdm(main_dataset_items, total=len(main_dataset_items), desc="Collecting relevant qids"):
        q = item.get("qid") or item.get("Q_number") or item.get("QID")
        if q is not None and q in all_qid_related_tags and q not in seen:
            seen.add(q)
            relevant_qids.append(q)

    # Backward-compatible fallback for legacy schemas where item qid is missing.
    if not relevant_qids:
        for item in tqdm(main_dataset_items, total=len(main_dataset_items), desc="Fallback: collecting qids from tags"):
            for tag in item.get('related_tags', []):
                if not isinstance(tag, dict):
                    continue
                q = tag.get('qid')
                if q is not None and q in all_qid_related_tags and q not in seen:
                    seen.add(q)
                    relevant_qids.append(q)

    if not relevant_qids:
        # touch empty file for consistency
        _ensure_parent_dir(out_jsonl)
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


def _human_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024**2:
        return f"{num_bytes / 1024:.2f} KB"
    if num_bytes < 1024**3:
        return f"{num_bytes / (1024**2):.2f} MB"
    return f"{num_bytes / (1024**3):.2f} GB"


def _jsonl_preview(path: str) -> str:
    """
    For JSONL: preview first record keys.
    For JSON: preview first value record keys if mapping-like.
    """
    try:
        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        return f"first_record_keys={sorted(obj.keys())}"
                    return f"first_record_type={type(obj).__name__}"
            return "empty_file"
    except Exception as exc:
        return f"preview_error={type(exc).__name__}"

    # JSON: for this pipeline the expected structure is dict[qid] -> record
    try:
        with open(path, "r", encoding="utf-8") as handle:
            obj = json.load(handle)
        if isinstance(obj, dict) and obj:
            first_val = next(iter(obj.values()))
            if isinstance(first_val, dict):
                return f"first_record_keys={sorted(first_val.keys())}"
            return f"first_record_type={type(first_val).__name__}"
        return "empty_file"
    except Exception as exc:
        return f"preview_error={type(exc).__name__}"


def main() -> None:
    print("Number of CPUs available:", os.cpu_count())

    args = parse_args()

    # -------------------------------------------------------------
    # Load inputs
    # -------------------------------------------------------------
    print("Loading all relation tags from:", args.all_relation_tags)
    with open(args.all_relation_tags, "r", encoding="utf-8") as f:
        all_relation_tags: dict[str, Any] = json.load(f)
    print(f"Loaded {len(all_relation_tags)} entries from {args.all_relation_tags}")

    print("Loading main dataset from:", args.main_dataset)
    with open(args.main_dataset, "r", encoding="utf-8") as f:
        main_dataset = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(main_dataset)} entries from {args.main_dataset}")

    # Build mapping: qid -> set(related_qids) and the global universe of qids
    all_qids_set = set()
    all_qid_related_tags = {}
    for qid, rec in tqdm(all_relation_tags.items(), total=len(all_relation_tags), desc="Gathering all qids"):
        all_qids_set.add(qid)
        tags_qids = [t.get("qid") for t in (rec.get("related_tags") or []) if isinstance(t, dict) and t.get("qid")]
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

    # ---------------------------------------------
    # Output summary (what was stored and where)
    # ---------------------------------------------
    written_files: list[str] = []
    if os.path.exists(out_path):
        written_files.append(out_path)

    uniq_written: list[str] = []
    seen = set()
    for p in written_files:
        if p in seen:
            continue
        seen.add(p)
        uniq_written.append(p)

    print("\n[8_ADD_UNRELEVANTS] Output summary")
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


'''
python scripts/dataset/08_add_unrelevants.py \
  --order 5000 \
  --all_relation_tags ./data/interim/7_all_relation_tags.json \
  --main_dataset ./data/interim/6_main_dataset.jsonl \
  --out ./data/interim/6_unrelevant_qids.jsonl \

'''
