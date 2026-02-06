"""Fetch related tags for items and build tag-only datasets."""

import argparse
import os
import fnmatch

import gzip
import json
import requests
import unicodedata
import time
import random
import numpy as np

import math
from p_tqdm import p_map
from tqdm import trange
import glob
from tqdm import tqdm




WDQS_ENDPOINT = "https://query.wikidata.org/sparql"

# -------------------------------
# HTTP helpers (robust + polite)
# -------------------------------

def make_session(user_agent="YourAppName/1.0 (you@example.com)"):
    s = requests.Session()
    s.headers.update({
        "User-Agent": user_agent,
        "Accept": "application/sparql-results+json",
        "Accept-Encoding": "gzip, deflate",
    })
    return s

def request_json_with_backoff(session, url, *, params=None, max_retries=6, base_sleep=0.5):
    """
    Get JSON with WDQS-friendly backoff:
      - Honor Retry-After for 429
      - Back off on 502/503/504
      - Check content-type before json()
    """
    last_resp = None
    for attempt in range(1, max_retries + 1):
        resp = session.get(url, params=params, timeout=60)
        last_resp = resp
        status = resp.status_code

        # Handle throttling / transient errors
        if status in (429, 502, 503, 504):
            ra = resp.headers.get("Retry-After")
            if ra:
                try:
                    wait = float(ra)
                except ValueError:
                    wait = base_sleep * (2 ** (attempt - 1))
            else:
                wait = base_sleep * (2 ** (attempt - 1))
            wait += random.uniform(0.0, 0.5)  # jitter
            time.sleep(wait)
            continue

        # Non-2xx
        resp.raise_for_status()

        # Sanity check on content-type
        ctype = resp.headers.get("Content-Type", "")
        if "json" not in ctype.lower():
            snippet = resp.text[:200]
            raise RuntimeError(f"Expected JSON but got Content-Type={ctype}. Snippet: {snippet!r}")

        return resp.json()

    # Exhausted retries
    snippet = ""
    try:
        snippet = last_resp.text[:200] if last_resp is not None else ""
    except Exception:
        pass
    raise RuntimeError(f"Failed after {max_retries} attempts. Last status={getattr(last_resp,'status_code',None)}. Snippet: {snippet!r}")


# -------------------------------
# Query builders + callers
# -------------------------------

# -------------------------------
# Query builders + callers
# -------------------------------

def _build_query(qid: str, include_subclasses: bool, limit: int, offset: int, language: str) -> str:
    # P31 = instance of; P279 = subclass of
    path = "wdt:P31/wdt:P279*" if include_subclasses else "wdt:P31"
    return f"""
    SELECT ?item ?itemLabel ?qid WHERE {{
      ?item {path} wd:{qid} .
      BIND(STRAFTER(STR(?item), "entity/") AS ?qid)
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{language},[AUTO_LANGUAGE]". }}
    }}
    LIMIT {int(limit)}
    OFFSET {int(offset)}
    """

def get_category_items(
    qid: str,
    *,
    step: int = 300,
    include_subclasses: bool = False,
    language: str = "en",
    user_agent: str = "YourAppName/1.0 (contact@example.com)",
    max_retries: int = 5,
    min_delay: float = 0.25,
    max_delay: float = 1.0,
):
    """
    Iterate over all items that are `instance of` wd:<qid>, requesting pages of size `step`.
    Stops when a page returns zero rows. Includes simple exponential backoff on transient errors.
    """
    offset = 0
    session = make_session(user_agent=user_agent)

    all_rows = []
    while True:
        query = _build_query(qid, include_subclasses, step, offset, language)

        # retry loop (wrap inside our backoff helper)
        for attempt in range(1, max_retries + 1):
            try:
                data = request_json_with_backoff(
                    session,
                    WDQS_ENDPOINT,
                    params={"query": query},
                    max_retries=6,
                    base_sleep=0.5,
                )
                rows = [
                    {
                        "Q_number": b["qid"]["value"],
                        "label": b["itemLabel"]["value"],
                        "uri": b["item"]["value"],
                    }
                    for b in data["results"]["bindings"]
                ]
                # If this page is empty, we are done
                if not rows:
                    return all_rows
                # Accumulate current page
                all_rows.extend(rows)

                # brief polite delay between pages
                time.sleep(random.uniform(min_delay, max_delay))
                break  # success; exit retry loop

            except (requests.exceptions.RequestException, ValueError, RuntimeError):
                if attempt == max_retries:
                    raise
                # exponential backoff with jitter
                time.sleep((2 ** (attempt - 1)) + random.uniform(0.0, 0.5))

        offset += step

    return all_rows

def get_item_properties(
    qid: str,
    *,
    language: str = "en",
    only_entity_values: bool = False,
    user_agent: str = "YourAppName/1.0 (you@example.com)"
):
    """
    Return all property values for a Wikidata item using the statement graph (p:/ps:),
    but keep the SAME output format as the original function.
    """
    session = make_session(user_agent=user_agent)

    query = f"""
    SELECT ?item ?itemLabel
           ?wdProp ?wdPropLabel
           (STRAFTER(STR(?wdProp), "/entity/") AS ?pid)
           ?value ?valueLabel
           (IF(isIRI(?value), "wikibase-item", DATATYPE(?value)) AS ?valueType)
           (IF(isIRI(?value), STRAFTER(STR(?value), "/entity/"), "") AS ?valueQid)
           ( ?wdPropLabel AS ?propertyLabel )
    WHERE {{
      VALUES ?item {{ wd:{qid} }}
      ?item ?p ?statement .
      ?wdProp wikibase:claim ?p .
      ?wdProp wikibase:statementProperty ?ps .
      ?statement ?ps ?value .
      SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "{language},[AUTO_LANGUAGE]" .
      }}
    }}
    """

    data = request_json_with_backoff(
        session,
        WDQS_ENDPOINT,
        params={"query": query},
        max_retries=6,
        base_sleep=0.5,
    )

    # Initialize container with the item label if present
    item_label = None
    claims = {}

    for b in data["results"]["bindings"]:
        # Item label (same for all rows)
        if item_label is None and "itemLabel" in b:
            item_label = b["itemLabel"]["value"]

        pid = b["pid"]["value"]                           # e.g., "P571"
        prop_label = b.get("propertyLabel", {}).get("value", pid)

        # Build the value object with the SAME shape as before
        if b["valueType"]["value"] == "wikibase-item":
            value_obj = {
                "type": "wikibase-item",
                "Q_number": b.get("valueQid", {}).get("value", ""),
                "label": b.get("valueLabel", {}).get("value", ""),
                "uri": b["value"]["value"],
            }
        else:
            # Literal (strings, numbers, times, coords, etc.)
            value_obj = {
                "type": "literal",
                "datatype": b["valueType"]["value"],  # e.g., xsd:dateTime, geo:wktLiteral, xsd:decimal
                "value": b["value"]["value"],
            }

        if pid not in claims:
            claims[pid] = {"property_label": prop_label, "values": [value_obj]}
        else:
            claims[pid]["values"].append(value_obj)

    if not claims:
        # Fallback to direct property query if no statement-graph results
        fallback_query = f'''
        SELECT ?prop ?propLabel ?value ?valueLabel
            (STRAFTER(STR(?prop), "/prop/direct/") AS ?pid)
            (IF(isIRI(?value), "wikibase-item", DATATYPE(?value)) AS ?valueType)
            (IF(isIRI(?value), STRAFTER(STR(?value), "/entity/"), "") AS ?valueQid)
        WHERE {{
        VALUES ?item {{ wd:{qid} }}
        ?item ?prop ?value .
        FILTER(STRSTARTS(STR(?prop), "http://www.wikidata.org/prop/direct/"))
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{language},[AUTO_LANGUAGE]" . }}
        }}
        '''
        fallback_data = request_json_with_backoff(
            session, WDQS_ENDPOINT,
            params={"query": fallback_query},
            max_retries=6, base_sleep=0.5,
        )
        for b in fallback_data["results"]["bindings"]:
            pid = b["pid"]["value"]
            prop_label = b.get("propLabel", {}).get("value", pid)
            if b["valueType"]["value"] == "wikibase-item":
                value_obj = {
                    "type": "wikibase-item",
                    "Q_number": b.get("valueQid", {}).get("value", ""),
                    "label": b.get("valueLabel", {}).get("value", ""),
                    "uri": b["value"]["value"],
                }
            else:
                value_obj = {
                    "type": "literal",
                    "datatype": b["valueType"]["value"],
                    "value": b["value"]["value"],
                }
            if pid not in claims:
                claims[pid] = {"property_label": prop_label, "values": [value_obj]}
            else:
                claims[pid]["values"].append(value_obj)

    return {"QID": qid, "label": item_label or qid, "claims": claims}


# pip install p_tqdm if you haven't already
from p_tqdm import p_map
import random, time

# --- Tunables (be polite to WDQS!) ---
PARALLELISM = 8        # 2–4 is a good citizen; raise carefully
PER_CALL_SLEEP = (0.15, 0.45)  # small jitter per process between calls
# NOTE: The new wbgetentities-batched path is preferred and ignores these.

M_DEFAULT = 20
SPLIT_SIZE_DEFAULT = 100000
HF_BASE_DEFAULT = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch related tags for Wikidata items.")
    parser.add_argument(
        "--hf_base",
        type=str,
        default=HF_BASE_DEFAULT,
        help="Base path for HF caches (falls back to $ARGUS_HF_BASE, then $HF_HOME, then ./).",
    )
    parser.add_argument("--m_million", type=int, default=M_DEFAULT, help="Read input labels from 1_wikidata_random_labels_{M}M.jsonl.")
    parser.add_argument("--parallelism", type=int, default=PARALLELISM, help="Parallel workers for WDQS calls.")
    parser.add_argument("--split_size", type=int, default=SPLIT_SIZE_DEFAULT, help="Chunk size per output split.")
    parser.add_argument("--base_path", type=str, default="./data/interim/2_tag_only", help="Output folder for tag-only splits.")
    parser.add_argument(
        "--prune_merged_inputs",
        action="store_true",
        help=(
            "After merging existing tag-only JSONLs into all_prev_items.jsonl, delete the merged "
            "split files (items_with_tags_*.jsonl and failed_items_*.jsonl) to avoid redundancy."
        ),
    )
    return parser.parse_args()
def _compute_related_tags_for_item(item,
                                   language="en",
                                   user_agent="YourAppName/1.0 (you@example.com)"):
    """
    Worker function run in parallel. Returns a *new* item dict with 'related_tags' added.
    """
    # Your all_items sometimes use 'Q_number' (from get_category_items), but the later loop used 'qid'.
    # Make this robust to either.
    qid = item.get("Q_number") or item.get("qid") or item.get("QID")
    if not qid:
        return {**item, "related_tags": [], "_error": "missing_qid"}

    # Gentle pause per request (helps reduce synchronized bursts across workers)
    time.sleep(random.uniform(*PER_CALL_SLEEP))

    try:
        props = get_item_properties(
            qid,
            language=language,
            only_entity_values=False,
            user_agent=user_agent
        )

        item_name = item.get("label", "")

        values_list = []
        for pid, prop_data in props.get("claims", {}).items():
            values = prop_data.get("values", [])
            # Skip properties with no values
            if not values:
                continue
            # Check every value in the list
            for v in values:
                # Only keep values that are Wikidata items
                if v.get("type") != "wikibase-item":
                    continue
                q_number = v.get("Q_number", "").strip()
                label_v = v.get("label", "")
                # Skip if no Q-number or if the value’s label is just the item name
                if not q_number:
                    continue
                if item_name and item_name in label_v:
                    continue
                values_list.append(label_v)

        # return a fresh dict (don’t mutate the input)
        return {**item, "related_tags": values_list}

    except Exception as e:
        # keep going even if a single item fails
        return {**item, "related_tags": [], "_error": f"{type(e).__name__}: {e}"}




def _canon_label(s: str) -> str:
    """Canonicalize labels for stable, fast membership checks."""
    if not isinstance(s, str):
        return ""
    # Normalize + casefold + strip to unify variants
    return unicodedata.normalize("NFKC", s).casefold().strip()


def merge_items_with_tags(all_items, base_path, *, prune_inputs: bool = False):
    """
    Build a set of previously-seen labels (canonicalized) from JSONL files under
    base_path, then filter all_items by that set. Uses a set (O(1) membership)
    and streams the combined file to avoid large in-memory lists.
    """
    os.makedirs(base_path, exist_ok=True)

    # Write via a temp file and swap to avoid truncating our own input
    save_path = os.path.join(base_path, "all_prev_items.jsonl")
    tmp_path = save_path + ".tmp"

    pattern = os.path.join(base_path, "*.jsonl")
    all_files = glob.glob(pattern)

    # Include the existing all_prev_items.jsonl as input (if present),
    # but never write to it directly (we write to tmp_path instead).
    input_files = []
    save_abs = os.path.abspath(save_path)
    tmp_abs = os.path.abspath(tmp_path)
    for file in all_files:
        abs_file = os.path.abspath(file)
        if abs_file == tmp_abs:
            continue
        input_files.append(file)

    seen_labels = set()
    deleted = 0
    delete_patterns = ("items_with_tags_*.jsonl", "failed_items_*.jsonl")

    with open(tmp_path, "w") as out_f:
        for file in tqdm(sorted(input_files), desc="Merging previous items"):
            try:
                with open(file, "r") as f:
                    for line in f:
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        lbl = _canon_label(obj.get("label", ""))
                        if not lbl or lbl in seen_labels:
                            continue
                        seen_labels.add(lbl)
                        out_f.write(json.dumps(obj) + "\n")
            except FileNotFoundError:
                # If a file got moved/deleted during the scan, skip it
                continue

            # Optionally prune merged split files to avoid redundancy.
            if prune_inputs:
                try:
                    abs_file = os.path.abspath(file)
                    if abs_file != save_abs:
                        base = os.path.basename(file)
                        if any(fnmatch.fnmatch(base, pat) for pat in delete_patterns):
                            os.remove(file)
                            deleted += 1
                except Exception:
                    # Best-effort cleanup only
                    pass

    os.replace(tmp_path, save_path)

    print(f"Saved {len(seen_labels)} unique previous items -> {save_path}")
    print(f"Total previous unique labels: {len(seen_labels)}")
    if prune_inputs:
        print(f"Pruned {deleted} merged split file(s) from {base_path}")

    # Fast O(1) membership using the canonicalized label
    filtered = [it for it in tqdm(all_items, desc="Filtering new items")
                if _canon_label(it.get("label", "")) not in seen_labels]
    return filtered


def main() -> None:
    args = parse_args()

    # HF cache setup
    base = args.hf_base or os.environ.get("ARGUS_HF_BASE") or os.environ.get("HF_HOME") or "./"
    if args.hf_base:
        os.environ["HF_HOME"] = base
        os.environ["HF_HUB_CACHE"] = f"{base}/hub"
        os.environ["HF_DATASETS_CACHE"] = f"{base}/datasets"
    else:
        os.environ.setdefault("HF_HOME", base)
        os.environ.setdefault("HF_HUB_CACHE", f"{base}/hub")
        os.environ.setdefault("HF_DATASETS_CACHE", f"{base}/datasets")

    global PARALLELISM
    PARALLELISM = int(args.parallelism)

    written_files = []

    all_qs = []
    total_all_items = []
    all_num = 0
    M = args.m_million

    input_path = f"./data/interim/1_wikidata_random_labels_{M}M.jsonl"
    if not os.path.exists(input_path) and os.path.exists(input_path + ".gz"):
        input_path = input_path + ".gz"

    open_fn = gzip.open if input_path.endswith(".gz") else open
    with open_fn(input_path, "rt", encoding="utf-8") as f:
        for line in f:
            all_num += 1
            item = json.loads(line)
            label = item['label'].strip()
            if (
                len(label) > 0 and
                not (label.startswith('Q') and label[1:].isdigit()) and
                not label.isdigit() and
                "wikidata" not in label.lower() and
                "wikipedia" not in label.lower() and
                "wikimedia" not in label.lower()
            ):
                total_all_items.append(item)

    # Compute splits *after* filtering/merging and use the filtered list
    base_path = args.base_path
    print(f"Merging items with tags from {base_path}")
    print(f"Total items: {len(total_all_items)}")
    all_items = merge_items_with_tags(
        total_all_items,
        base_path,
        prune_inputs=bool(getattr(args, "prune_merged_inputs", False)),
    )
    # merge_items_with_tags always writes this combined file
    written_files.append(f"{base_path}/all_prev_items.jsonl")
    print(f"Total items after merging: {len(all_items)}")
    print(f"Total items to process: {len(all_items)}")

    split_size = args.split_size
    num_splits = (len(all_items) // split_size)

    for i in trange(num_splits, desc="Processing splits"):
        start_idx = i * split_size
        end_idx = start_idx + split_size
        out_put_path = f"{base_path}/items_with_tags_{start_idx}_{end_idx}.jsonl"

        print(f"[2_GET_TAGS] Processing items from index {start_idx} to {end_idx}")
        chunk = all_items[start_idx:end_idx]
        print(len(chunk))

        all_items_with_tags = p_map(
            _compute_related_tags_for_item,
            chunk,
            num_cpus=PARALLELISM,
            desc="Fetching item properties (WDQS)"
        )

        failures = [it for it in all_items_with_tags if it.get("_error")]
        if failures:
            print(f"{len(failures)} items failed; example:", failures[0].get("_error"))

        os.makedirs(base_path, exist_ok=True)
        with open(out_put_path, "w") as f:
            for item in all_items_with_tags:
                f.write(json.dumps(item) + "\n")
        written_files.append(out_put_path)

        if failures:
            failed_path = f"{base_path}/failed_items_{start_idx}_{end_idx}.jsonl"
            with open(failed_path, "w") as f:
                for item in failures:
                    f.write(json.dumps(item) + "\n")
            written_files.append(failed_path)

    # ---------------------------------------------
    # Output summary (what was stored and where)
    # ---------------------------------------------

    def _human_size(num_bytes: int) -> str:
        if num_bytes < 1024:
            return f"{num_bytes} B"
        if num_bytes < 1024 ** 2:
            return f"{num_bytes / 1024:.2f} KB"
        if num_bytes < 1024 ** 3:
            return f"{num_bytes / (1024 ** 2):.2f} MB"
        return f"{num_bytes / (1024 ** 3):.2f} GB"

    def _jsonl_preview(path: str) -> str:
        """
        Return a short preview string describing what the JSONL contains,
        without scanning the entire file.
        """
        try:
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        keys = sorted(obj.keys())
                        return f"first_record_keys={keys}"
                    return f"first_record_type={type(obj).__name__}"
        except Exception as e:
            return f"preview_error={type(e).__name__}"
        return "empty_file"

    # de-dupe while preserving order
    uniq_written = []
    seen = set()
    for p in written_files:
        if p in seen:
            continue
        seen.add(p)
        uniq_written.append(p)

    print("\n[2_GET_TAGS] Output summary")
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
