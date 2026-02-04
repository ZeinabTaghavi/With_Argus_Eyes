"""Fetch Wikipedia summaries for tag QIDs."""

import os
# 1️⃣ Pick an absolute path that has enough space (ARGUS_HF_BASE/HF_HOME or default)
BASE = os.environ.get("ARGUS_HF_BASE") or os.environ.get("HF_HOME") or "./"

# 2️⃣ Point both caches there ─ before any HF import
os.environ.setdefault("HF_HOME", BASE)          # makes <BASE>/hub and <BASE>/datasets
os.environ.setdefault("HF_HUB_CACHE", f"{BASE}/hub") # optional, explicit
os.environ.setdefault("HF_DATASETS_CACHE", f"{BASE}/datasets")

import json
import glob
from tqdm import tqdm


import requests
from urllib.parse import quote
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from multiprocessing import cpu_count
# Prefer p_tqdm if available; else use tqdm.contrib.concurrent; else plain tqdm
try:
    from p_tqdm import p_map  # pip install p_tqdm
    _HAVE_P_TQDM = True
except Exception:
    _HAVE_P_TQDM = False
    try:
        from tqdm.contrib.concurrent import process_map as _process_map  # part of tqdm
        _HAVE_PROCESS_MAP = True
    except Exception:
        _HAVE_PROCESS_MAP = False
        from tqdm import tqdm  # fallback progress bar

# Concurrency knobs (can be overridden via env vars)
MAX_WORKERS = int(os.getenv("WIKI_MAX_WORKERS", os.cpu_count() or 8))
print(f"[4_wikipedia_par] Using {MAX_WORKERS} workers.")
CHUNKSIZE = int(os.getenv("WIKI_CHUNKSIZE", "16"))
print(f"[4_wikipedia_par] Using chunk size {CHUNKSIZE}.")
# Path for tag→Wikipedia first paragraph cache
TAGS_OUT_PATH = "./data/interim/4_tags_wikipedia_first_paragraphs_cache.jsonl"

WIKIDATA_API = "https://www.wikidata.org/w/api.php"

def _make_session():
    s = requests.Session()
    s.headers.update({
        # put your contact here per Wikimedia best practice
        "User-Agent": "qid-first-paragraph/1.0 (contact: you@example.com)",
        "Accept": "application/json",
    })
    retry = Retry(
        total=3,
        backoff_factor=0.4,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s

def _get_wikipedia_title_from_qid(qid: str, lang: str = "en", session: requests.Session | None = None):
    """
    Resolve a Wikidata QID to the Wikipedia page title (and lang) using wbgetentities.
    Requires the given language's wiki (e.g., 'enwiki'); if absent, we skip the item (no fallback to other languages).
    """
    if not qid or not qid.startswith("Q"):
        raise ValueError("QID must look like 'Q…' (e.g., 'Q92561').")
    session = session or _make_session()

    # First try: ask only for the requested language wiki via sitefilter (less payload).
    params_pref = {
        "action": "wbgetentities",
        "format": "json",
        "formatversion": "2",
        "ids": qid,
        "props": "sitelinks/urls",
        "sitefilter": f"{lang}wiki",
    }
    r = session.get(WIKIDATA_API, params=params_pref, timeout=15)
    r.raise_for_status()
    data = r.json()
    ent = (data.get("entities") or {}).get(qid) or {}
    sitelinks = ent.get("sitelinks") or {}

    if f"{lang}wiki" in sitelinks:
        return sitelinks[f"{lang}wiki"]["title"], lang

    raise LookupError(f"No {lang}.wikipedia.org sitelink found for {qid}.")

def _fetch_first_paragraph(title: str, wiki_lang: str, session: requests.Session | None = None) -> str:
    """
    Fetch the first paragraph of a Wikipedia article.
    Tries REST /page/summary first; falls back to Action API TextExtracts exintro.
    """
    session = session or _make_session()

    # 1) REST summary (plain-text 'extract'); handles redirects via ?redirect=true
    rest_api = f"https://{wiki_lang}.wikipedia.org/api/rest_v1/page/summary/{quote(title.replace(' ', '_'), safe='')}"
    r = session.get(rest_api, params={"redirect": "true"}, timeout=15)
    if r.ok:
        j = r.json()
        extract = (j or {}).get("extract") or ""
        if extract.strip():
            # REST summary is already a concise lead; return as "first paragraph"
            return extract.strip()

    # 2) Fallback: Action API + TextExtracts: exintro&explaintext returns the lead section as plain text.
    action_api = f"https://{wiki_lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "exintro": 1,
        "explaintext": 1,
        "redirects": 1,
        "formatversion": "2",
        "titles": title,
    }
    r = session.get(action_api, params=params, timeout=15)
    r.raise_for_status()
    j = r.json()
    pages = (j.get("query") or {}).get("pages") or []
    if not pages:
        raise LookupError(f"No page returned for '{title}' on {wiki_lang}.wikipedia.org.")
    extract = (pages[0] or {}).get("extract") or ""
    if not extract.strip():
        raise LookupError(f"No extract text found for '{title}' on {wiki_lang}.wikipedia.org.")
    # First paragraph = text up to the first blank line
    return extract.strip().split("\n\n", 1)[0].strip()

def first_paragraph_from_qid(qid: str, lang: str = "en") -> str:
    """
    Main entry: QID → (title, wiki_lang) → first paragraph.
    """
    session = _make_session()
    title, wiki_lang = _get_wikipedia_title_from_qid(qid, lang=lang, session=session)
    return _fetch_first_paragraph(title, wiki_lang, session=session)

def _process_one(item: dict):
    """Return enriched item with English Wikipedia first paragraph, or None if not available."""
    qid = item.get('qid')
    if not qid:
        raise LookupError(f"No qid found for {item}")
        return None
    try:
        title, wiki_lang = _get_wikipedia_title_from_qid(qid, lang='en')
        if wiki_lang != 'en':
            raise LookupError(f"Wikipedia language is not English for {qid}")
            return None
        first_para = _fetch_first_paragraph(title, wiki_lang)
        enriched = dict(item)
        enriched['qid'] = qid
        enriched['wikipedia_lang'] = wiki_lang
        enriched['wikipedia_title'] = title
        enriched['wikipedia_first_paragraph'] = first_para
        return enriched
    except (LookupError, ValueError, requests.RequestException):
        return None


def _normalize_item_with_qid(original: dict) -> dict | None:
    """
    Return a shallow copy of the original item with a 'qid' field populated
    from any of: 'qid', 'Q_number', or 'itemQ'. Returns None if none present.
    """
    qid = original.get('qid') or original.get('Q_number') or original.get('itemQ')
    if not qid:
        return None
    out = dict(original)
    out['qid'] = qid
    return out


def _augment_items_with_wikipedia_keep(items: list[dict]) -> list[dict]:
    """
    Try to add wikipedia_* fields to each item; keep original item if lookup fails.
    """
    # Normalize to include 'qid' for the worker
    normalized: list[dict | None] = [_normalize_item_with_qid(it) for it in items]
    work_items = [it for it in normalized if it is not None]

    # Map using available parallelism choices
    if _HAVE_P_TQDM:
        results = p_map(_process_one, work_items, num_cpus=MAX_WORKERS)
    elif _HAVE_PROCESS_MAP:
        results = _process_map(_process_one, work_items, max_workers=MAX_WORKERS, chunksize=CHUNKSIZE)
    else:
        results = []
        for it in work_items:
            results.append(_process_one(it))

    # Stitch results back to original order
    enriched: list[dict] = []
    idx = 0
    for orig, norm in zip(items, normalized):
        if norm is None:
            enriched.append(orig)
        else:
            res = results[idx]
            idx += 1
            enriched.append(res if res is not None else orig)
    return enriched


def _augment_file_inplace_with_wikipedia(path: str, path_out: str):
    """
    Load a JSONL file, add wikipedia fields where available, write back to the output file.

    This helper considers an existing output file and avoids reprocessing items
    that have already been enriched with a non-empty wikipedia_first_paragraph.
    Only items absent from the existing output or lacking a wikipedia_first_paragraph
    are sent through the enrichment pipeline. Newly processed items are merged into
    the existing output and written back to path_out.
    """
    if not os.path.exists(path):
        return

    # Load existing enriched items from path_out
    existing_by_qid: dict[str, dict] = {}
    if os.path.exists(path_out):
        try:
            with open(path_out, 'r', encoding='utf-8') as f_out:
                for line in f_out:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except Exception:
                        continue
                    qid = item.get('qid') or item.get('Q_number') or item.get('itemQ')
                    if qid:
                        item['qid'] = qid
                        existing_by_qid[qid] = item
        except Exception:
            existing_by_qid = {}

    # Load items from the input file
    loaded: list[dict] = []
    with open(path, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                loaded.append(json.loads(line))
            except Exception:
                continue

    if not loaded:
        return

    # Determine which items need enrichment
    items_to_process: list[dict] = []
    for orig in loaded:
        qid = orig.get('qid') or orig.get('Q_number') or orig.get('itemQ')
        if not qid:
            continue
        existing_item = existing_by_qid.get(qid)
        if (existing_item is None) or (not existing_item.get('wikipedia_first_paragraph')):
            items_to_process.append(orig)

    # Enrich the items
    if items_to_process:
        if _HAVE_P_TQDM:
            results = p_map(_process_one, items_to_process, num_cpus=MAX_WORKERS)
        elif _HAVE_PROCESS_MAP:
            results = _process_map(_process_one, items_to_process, max_workers=MAX_WORKERS, chunksize=CHUNKSIZE)
        else:
            results = [ _process_one(it) for it in items_to_process ]

        # Merge back into existing_by_qid
        idx = 0
        for orig in items_to_process:
            qid = orig.get('qid') or orig.get('Q_number') or orig.get('itemQ')
            if not qid:
                continue
            processed_item = results[idx] if idx < len(results) else None
            idx += 1
            if processed_item is not None:
                processed_item = dict(processed_item)
                processed_item['qid'] = qid
                existing_by_qid[qid] = processed_item

    # Write out all items with a wikipedia_first_paragraph
    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    with open(path_out, 'w', encoding='utf-8') as f_out:
        for it in existing_by_qid.values():
            if it.get('wikipedia_first_paragraph'):
                f_out.write(json.dumps(it, ensure_ascii=False) + '\n')
    print(f"[4_wikipedia_par update] items saved so far: {len(existing_by_qid)}")


# === Tag Wikipedia first paragraph cache utilities ===
def _process_one_tag(tag_name: str) -> dict | None:
    """
    Resolve a tag name (assumed to be an English Wikipedia title) to the first paragraph.
    Returns a record suitable for JSONL or None if not resolvable.
    """
    if not isinstance(tag_name, str):
        return None
    name = tag_name.strip()
    if not name:
        raise ValueError(f"Tag name is empty for {tag_name}")
    try:
        first_para = _fetch_first_paragraph(name, "en")
        return {
            "tag": name,
            "wikipedia_lang": "en",
            "wikipedia_title": name,
            "wikipedia_first_paragraph": first_para,
        }
    except (LookupError, ValueError, requests.RequestException):
        return None

def _load_tag_cache(path: str) -> dict[str, dict]:
    """
    Load an existing tags cache JSONL into a dict keyed by 'tag'.
    If the file does not exist, create an empty file.
    """
    cache: dict[str, dict] = {}
    if not os.path.exists(path):
        # Make parent directories as needed and create an empty file
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8"):
            pass  # Create the empty file
        return cache
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    raise Exception(f"Error loading tag cache from {path}: {line}")
                tag = obj.get("tag")
                if isinstance(tag, str) and tag:
                    cache[tag] = obj
    except Exception:
        raise Exception(f"Error loading tag cache from {path}")
    return cache

def _save_tag_cache(cache: dict[str, dict], path: str) -> None:
    """
    Persist the tag cache as JSONL (one record per line).
    Only writes entries that actually have a first paragraph.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in cache.values():
            if rec.get("wikipedia_first_paragraph"):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def _collect_unique_tags_from_items(items: list[dict]) -> set[str]:
    """
    Collect a set of unique tag names from item dictionaries.
    Looks for list-valued fields commonly used for tags.
    """
    tags: set[str] = set()
    candidate_keys = ("tags", "tag_list", "labels", "categories", 'related_tags')
    for it in items:
        for k in candidate_keys:
            v = it.get(k)
            if isinstance(v, list):
                for t in v:
                    if isinstance(t, str):
                        s = t.strip()
                        if s:
                            tags.add(s)
    return tags

def _collect_unique_tags_from_jsonl(path: str) -> set[str]:
    """
    Read a JSONL file of items and collect unique tag names using the same logic
    as _collect_unique_tags_from_items.
    """
    if not os.path.exists(path):
        return set()
    items: list[dict] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return set()
    return _collect_unique_tags_from_items(items)

def _augment_tags_first_paragraphs(tag_names: set[str], path_out: str, batch_size: int = 10000) -> None:
    """
    Ensure that for every tag in `tag_names`, we have an entry with a first paragraph
    in the JSONL file at `path_out`. Uses the same parallelism strategy as items.
    """
    if not tag_names:
        return

    cache = _load_tag_cache(path_out) # the jsonl file with the format of {'tag', 'wikipedia_lang', 'wikipedia_title', 'wikipedia_first_paragraph'}
    cached_tags = set([item['tag'] for item in cache])
    missing = [t for t in tag_names if t not in cached_tags]

    if not missing:
        print(f"No new tags to process. Cache already has {len(cache)} entries.")
        return

    print(f"Processing {len(missing)} missing tags in batches of {batch_size}...")
    for start in tqdm(range(0, len(missing), batch_size), desc="Processing tag batches"):
        batch = missing[start:start + batch_size]
        # Parallel map over tags
        if _HAVE_P_TQDM:
            results = p_map(_process_one_tag, batch, num_cpus=MAX_WORKERS)
        elif _HAVE_PROCESS_MAP:
            results = _process_map(_process_one_tag, batch, max_workers=MAX_WORKERS, chunksize=CHUNKSIZE)
        else:
            results = [_process_one_tag(t) for t in batch]

        # Merge successful lookups into cache
        for tag, rec in zip(batch, results):
            if rec is not None:
                cache[tag] = rec

        # Save progress after each batch for fault tolerance
        print(f"Saving progress to {path_out}.")
        _save_tag_cache(cache, path_out)
        print(f"Processed {min(start + batch_size, len(missing))} / {len(missing)} tags. Saved progress to {path_out}.")

    print(f"Tags cache now contains {len(cache)} entries. Saved to: {path_out}")

def main():
    """
    Load Wikidata items, enrich them with English Wikipedia first paragraphs and
    write them to an output JSONL file. If the output file already exists, previously
    processed items with complete Wikipedia fields are reused to avoid redundant
    network calls. Items that are missing the wikipedia_first_paragraph or other
    wikipedia_* keys are reprocessed. Newly processed items are merged into the
    existing output. The script writes progress to disk after each 10,000 items.

    Additionally, this function now ensures that for every tag found across the input
    items (and landmark files), we also store the English Wikipedia first paragraph
    for that tag in `tags_wikipedia_first_paragraphs.jsonl`. That cache is updated
    incrementally and saved after each batch as well.
    """
    out_path = './data/interim/4_items_with_wikipedia.jsonl'

    # Load existing processed items from out_path, keyed by QID
    existing_by_qid: dict[str, dict] = {}
    if os.path.exists(out_path):
        try:
            with open(out_path, 'r', encoding='utf-8') as f_out:
                for line in f_out:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except Exception:
                        continue
                    qid = item.get('qid') or item.get('Q_number') or item.get('itemQ')
                    if qid:
                        item['qid'] = qid
                        existing_by_qid[qid] = item
        except Exception:
            existing_by_qid = {}

    # Load input items and determine which need processing
    wikidata_items: list[dict] = []
    items_to_process: list[dict] = []
    added_qids: set[str] = set()
    try:
        with open('./data/interim/3_cleaned_items_tag_only.jsonl', 'r', encoding='utf-8') as f_in:
            all_cleaned_items_labels = set()
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                label = item.get('label')
                if label:
                    all_cleaned_items_labels.add(label)
                wikidata_items.append(item)
                qid = item.get('qid') 
                if not qid:
                    continue
                existing_item = existing_by_qid.get(qid)
                if (existing_item is None):
                    items_to_process.append(item)
                elif (not existing_item.get('wikipedia_first_paragraph')):
                    raise Exception(f"Item {qid} has no wikipedia first paragraph")

            for item in items_to_process:
                assert item.get('qid') not in existing_by_qid.keys()

    except FileNotFoundError:
        pass

    # Process items in batches of 10,000, saving after each batch
    initial_saved = len(existing_by_qid)
    total_added = 0
    BATCH_SIZE = 1000
    for start in tqdm(range(0, len(items_to_process), BATCH_SIZE), desc="[4_wikipedia_par] Processing batches"):
        batch = items_to_process[start:start + BATCH_SIZE]

        # Enrich the current batch
        if _HAVE_P_TQDM:
            results = p_map(_process_one, batch, num_cpus=MAX_WORKERS)
        elif _HAVE_PROCESS_MAP:
            results = _process_map(_process_one, batch, max_workers=MAX_WORKERS, chunksize=CHUNKSIZE)
        else:
            results = [ _process_one(it) for it in batch ]

        # Merge processed items into existing_by_qid
        batch_added = 0
        for orig, processed in zip(batch, results):
            qid = orig.get('qid')
            if not qid:
                raise Exception(f"Item {orig} has no qid")
            if processed is not None:
                processed = dict(processed)
                processed['qid'] = qid
                if qid not in existing_by_qid:
                    batch_added += 1
                existing_by_qid[qid] = processed
            else: # there is no wikipedia first paragraph
                processed = dict(orig)
                processed['qid'] = qid
                processed['wikipedia_first_paragraph'] = ''
                processed['wikipedia_exists'] = False
                existing_by_qid[qid] = processed

        total_added += batch_added

        # Immediately write out current progress for fault tolerance
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f_out:
            for it in existing_by_qid.values():
                if it.get('wikipedia_first_paragraph'):
                    f_out.write(json.dumps(it, ensure_ascii=False) + '\n')
        remaining_est = max(0, len(items_to_process) - total_added)
        print(f"[4_wikipedia_par update] items saved so far: {len(existing_by_qid)} (+{batch_added} this batch, +{total_added} this run), estimated remaining (qids): {remaining_est}")

        print(f"Processed {min(start + BATCH_SIZE, len(items_to_process))} out of {len(items_to_process)} items and saved progress.")

    # Summaries
    print(f"Total input items: {len(wikidata_items)}")
    print(f"Items needing (re)processing: {len(items_to_process)}")
    print(f"Items with an English Wikipedia page (final): {len(existing_by_qid)}")
    print(f"Saved enriched items to: {out_path}")

    # Update landmark files similarly (items with QIDs)
    _augment_file_inplace_with_wikipedia('./data/interim/3_landmarks_low_freq.jsonl', './data/interim/4_landmarks_low_freq.jsonl')
    _augment_file_inplace_with_wikipedia('./data/interim/3_landmarks_high_freq.jsonl', './data/interim/4_landmarks_high_freq.jsonl')
    print("Augmented Wikipedia fields in: data/interim/landmarks_low_freq.jsonl, data/interim/landmarks_high_freq.jsonl")

    # === NEW: Ensure Wikipedia first paragraphs for all encountered tags ===
    # Collect tags from the main items and landmark files.
    print(f"Collecting tags from items and landmark files...")
    all_tags: set[str] = set()
    print(len(wikidata_items))
    all_tags |= _collect_unique_tags_from_items(wikidata_items)
    all_tags |= _collect_unique_tags_from_jsonl('./data/interim/3_landmarks_low_freq.jsonl')
    all_tags |= _collect_unique_tags_from_jsonl('./data/interim/3_landmarks_high_freq.jsonl')

    print(f"Collected {len(all_tags)} unique tags from items and landmark files.")

    # Build/extend the tags cache JSONL with first paragraphs.
    _augment_tags_first_paragraphs(all_tags, TAGS_OUT_PATH, batch_size=10000)

if __name__ == "__main__":
    main()
