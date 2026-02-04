"""Enrich items with Wikidata descriptions."""

import argparse
import json
import os
import requests
from typing import Dict, Iterable, List, Optional, Tuple, Set
from p_tqdm import p_map
from functools import partial
import time
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Concurrency, chunking, and retry/backoff configuration (override via env)
MAX_WORKERS = int(os.environ.get("WIKI_MAX_WORKERS", "4"))          # cap parallelism to be polite
CHUNK_SIZE  = int(os.environ.get("WIKI_CHUNK_SIZE",  "25"))         # <= 50 per API constraints
MAX_RETRIES = int(os.environ.get("WIKI_MAX_RETRIES", "6"))          # retries per chunk on 429/5xx
BACKOFF_BASE = float(os.environ.get("WIKI_BACKOFF_BASE", "1.5"))    # exponential backoff base
BACKOFF_MAX  = float(os.environ.get("WIKI_BACKOFF_MAX",  "60"))     # max sleep between retries (s)

API_URL = "https://www.wikidata.org/w/api.php"
MAX_IDS_PER_CALL = 50  # MediaWiki Action API typical limit per request

# Cache file for tag → Wikidata description
TAGS_WD_OUT_PATH = "./data/interim/tags_wikidata_descriptions.jsonl"


def make_session() -> requests.Session:
    """Create a requests session with robust retries and connection pooling."""
    sess = requests.Session()
    retries = Retry(
        total=MAX_RETRIES,
        backoff_factor=1.0,  # base factor; we'll also do our own exponential backoff
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=64, pool_maxsize=64)
    sess.mount("https://", adapter)
    sess.headers.update({
        "User-Agent": "KBAugmented-RAG/1.0 wikidata-desc-fetcher (wikidata-desc-fetcher)",
    })
    return sess


def _fetch_desc_chunk(chunk: List[str], lang: str = "en") -> Dict[str, Optional[str]]:
    """Fetch descriptions for a chunk of QIDs (<=50) and return {qid: desc}."""
    session = make_session()
    try:
        attempts = 0
        while True:
            try:
                params = {
                    "action": "wbgetentities",
                    "format": "json",
                    "ids": "|".join(chunk),
                    "props": "descriptions",
                    "languages": lang,
                    "languagefallback": "1",
                    "maxlag": "5",  # be nice to WMF clusters
                }
                resp = session.get(API_URL, params=params, timeout=30)
                # If server asks us to slow down, sleep according to Retry-After
                if resp.status_code in (429, 503):
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after:
                        try:
                            delay = float(retry_after)
                        except ValueError:
                            delay = min(BACKOFF_MAX, BACKOFF_BASE * (2 ** attempts) + random.uniform(0, 0.5))
                    else:
                        delay = min(BACKOFF_MAX, BACKOFF_BASE * (2 ** attempts) + random.uniform(0, 0.5))
                    attempts += 1
                    if attempts > MAX_RETRIES:
                        resp.raise_for_status()
                    time.sleep(delay)
                    continue

                resp.raise_for_status()
                data = resp.json()
                entities = data.get("entities", {}) or {}
                out: Dict[str, Optional[str]] = {}
                for q in chunk:
                    desc_val: Optional[str] = None
                    ent = entities.get(q)
                    if isinstance(ent, dict):
                        descs = ent.get("descriptions", {}) or {}
                        if lang in descs and isinstance(descs[lang], dict):
                            desc_val = descs[lang].get("value")
                        else:
                            for d in descs.values():
                                if isinstance(d, dict) and "value" in d:
                                    desc_val = d["value"]
                                    break
                    out[q] = desc_val
                # small jitter to avoid lockstep hammering
                time.sleep(random.uniform(0.01, 0.05))
                return out

            except requests.exceptions.HTTPError as e:
                status = getattr(e.response, "status_code", None)
                if status in (429, 500, 502, 503, 504):
                    attempts += 1
                    if attempts > MAX_RETRIES:
                        raise
                    # honor Retry-After if present
                    retry_after = None if e.response is None else e.response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            delay = float(retry_after)
                        except ValueError:
                            delay = min(BACKOFF_MAX, BACKOFF_BASE * (2 ** (attempts - 1)) + random.uniform(0, 0.5))
                    else:
                        delay = min(BACKOFF_MAX, BACKOFF_BASE * (2 ** (attempts - 1)) + random.uniform(0, 0.5))
                    time.sleep(delay)
                    continue
                else:
                    raise
    finally:
        session.close()


def fetch_wikidata_descriptions(
    qids: Iterable[str],
    lang: str = "en",
) -> Dict[str, Optional[str]]:
    """
    Returns a mapping {qid -> description or None} using Wikidata's action API
    with language fallback. This parallel version splits QIDs into chunks of
    up to 50 IDs (API limit per call) and fetches them concurrently using
    p_tqdm.p_map across all CPUs.
    """
    # Normalize and deduplicate while preserving order
    seen = set()
    qid_list: List[str] = []
    for q in qids:
        if q and q not in seen:
            seen.add(q)
            qid_list.append(q)

    # Split into <=CHUNK_SIZE-ID chunks
    chunks: List[List[str]] = [qid_list[i:i + CHUNK_SIZE] for i in range(0, len(qid_list), CHUNK_SIZE)]
    if not chunks:
        return {}

    fetch_partial = partial(_fetch_desc_chunk, lang=lang)
    workers = max(1, min(MAX_WORKERS, os.cpu_count() or 1))
    chunk_results: List[Dict[str, Optional[str]]] = p_map(fetch_partial, chunks, num_cpus=workers)

    # Merge
    out: Dict[str, Optional[str]] = {}
    for d in chunk_results:
        out.update(d)
    return out


# ---------- Tag helpers: fetch Wikidata descriptions for tag titles ----------

def _norm_title(s: str) -> str:
    """Normalize a Wikipedia title for matching (spaces, case)."""
    return (s or "").replace("_", " ").strip().lower()

def _fetch_tag_desc_chunk_by_titles(titles: List[str], lang: str = "en", site: str = "enwiki") -> Dict[str, Optional[str]]:
    """
    Given up to 50 Wikipedia page titles, resolve to Wikidata entities via wbgetentities
    using the sites/titles path, and return {original_title: description or None}.
    """
    session = make_session()
    try:
        # Prepare default mapping with None for each requested title
        result: Dict[str, Optional[str]] = {t: None for t in titles}
        attempts = 0
        while True:
            try:
                params = {
                    "action": "wbgetentities",
                    "format": "json",
                    "sites": site,                       # e.g., enwiki
                    "titles": "|".join(titles),          # <= 50 per call
                    "props": "descriptions|sitelinks",   # need sitelinks to map back to titles
                    "sitefilter": site,                  # limit sitelinks to the site we care about
                    "languages": lang,
                    "languagefallback": "1",
                    "maxlag": "5",
                }
                resp = session.get(API_URL, params=params, timeout=30)
                if resp.status_code in (429, 503):
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after:
                        try:
                            delay = float(retry_after)
                        except ValueError:
                            delay = min(BACKOFF_MAX, BACKOFF_BASE * (2 ** attempts) + random.uniform(0, 0.5))
                    else:
                        delay = min(BACKOFF_MAX, BACKOFF_BASE * (2 ** attempts) + random.uniform(0, 0.5))
                    attempts += 1
                    if attempts > MAX_RETRIES:
                        resp.raise_for_status()
                    time.sleep(delay)
                    continue

                resp.raise_for_status()
                data = resp.json()
                entities = data.get("entities", {}) or {}

                # Build a mapping from normalized sitelink title to description
                by_norm_title: Dict[str, Optional[str]] = {}
                for ent in entities.values():
                    if not isinstance(ent, dict):
                        continue
                    # Get description with fallback to any available language if needed
                    desc_val: Optional[str] = None
                    descs = ent.get("descriptions", {}) or {}
                    if lang in descs and isinstance(descs[lang], dict):
                        desc_val = descs[lang].get("value")
                    else:
                        for d in descs.values():
                            if isinstance(d, dict) and "value" in d:
                                desc_val = d["value"]
                                break
                    # Map back via sitelink title on the requested site
                    sitelinks = ent.get("sitelinks", {}) or {}
                    sl = sitelinks.get(site)
                    if isinstance(sl, dict):
                        page_title = sl.get("title")
                        if isinstance(page_title, str) and page_title:
                            by_norm_title[_norm_title(page_title)] = desc_val

                # Now assign results for all requested titles using normalization
                for t in titles:
                    result[t] = by_norm_title.get(_norm_title(t))
                # jitter
                time.sleep(random.uniform(0.01, 0.05))
                return result

            except requests.exceptions.HTTPError as e:
                status = getattr(e.response, "status_code", None)
                if status in (429, 500, 502, 503, 504):
                    attempts += 1
                    if attempts > MAX_RETRIES:
                        raise
                    retry_after = None if e.response is None else e.response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            delay = float(retry_after)
                        except ValueError:
                            delay = min(BACKOFF_MAX, BACKOFF_BASE * (2 ** (attempts - 1)) + random.uniform(0, 0.5))
                    else:
                        delay = min(BACKOFF_MAX, BACKOFF_BASE * (2 ** (attempts - 1)) + random.uniform(0, 0.5))
                    time.sleep(delay)
                    continue
                else:
                    raise
    finally:
        session.close()

def fetch_tag_descriptions_from_titles(tag_titles: Iterable[str], lang: str = "en", site: str = "enwiki") -> Dict[str, Optional[str]]:
    """
    Parallel batched fetch of Wikidata descriptions for tag titles using wbgetentities
    (sites/titles path). Returns {title: description or None}.
    """
    # Normalize and de-dup while preserving order
    seen = set()
    title_list: List[str] = []
    for t in tag_titles:
        if t and t not in seen:
            seen.add(t)
            title_list.append(t)

    chunks: List[List[str]] = [title_list[i:i + CHUNK_SIZE] for i in range(0, len(title_list), CHUNK_SIZE)]
    if not chunks:
        return {}

    fetch_partial = partial(_fetch_tag_desc_chunk_by_titles, lang=lang, site=site)
    workers = max(1, min(MAX_WORKERS, os.cpu_count() or 1))
    chunk_results: List[Dict[str, Optional[str]]] = p_map(fetch_partial, chunks, num_cpus=workers)

    out: Dict[str, Optional[str]] = {}
    for d in chunk_results:
        out.update(d)
    return out

# ---------- Tag description JSONL cache ----------

def _load_tag_wd_cache(path: str) -> Dict[str, dict]:
    """
    Load a tags→wikidata_description JSONL into a dict keyed by 'tag'.
    """
    cache: Dict[str, dict] = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    tag = obj.get("tag")
                    if isinstance(tag, str) and tag:
                        cache[tag] = obj
        except Exception:
            pass
    return cache

def _save_tag_wd_cache(cache: Dict[str, dict], path: str) -> None:
    """
    Persist the tag wikidata cache as JSONL, writing only entries with a non-empty description.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in cache.values():
            desc = rec.get("wikidata_description")
            if isinstance(desc, str) and desc.strip():
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ---------- Tag collection from items / jsonl ----------

def _collect_unique_tags_from_items(items: List[dict]) -> Set[str]:
    """
    Collect a set of unique tag names from item dictionaries.
    Looks for list-valued fields commonly used for tags.
    """
    tags: Set[str] = set()
    candidate_keys = ("tags", "tag_list", "labels", "categories")
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

def _collect_unique_tags_from_jsonl(path: str) -> Set[str]:
    """
    Read a JSONL file of items and collect unique tag names using the same logic
    as _collect_unique_tags_from_items.
    """
    if not os.path.exists(path):
        return set()
    items: List[dict] = []
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

# ---------- Orchestrator: ensure tag→wikidata_description coverage ----------

def _augment_tags_wikidata_descriptions(tag_names: Set[str], path_out: str, batch_size: int = 10000, lang: str = "en", site: str = "enwiki") -> None:
    """
    Ensure that for every tag in `tag_names`, we have a Wikidata description stored
    in the JSONL file at `path_out`. Saves progress after each batch.
    """
    if not tag_names:
        print("No tags found to augment for Wikidata descriptions.")
        return

    cache = _load_tag_wd_cache(path_out)
    missing = [t for t in tag_names if t not in cache]

    if not missing:
        print(f"No new tags to process for Wikidata. Cache already has {len(cache)} entries.")
        return

    print(f"Processing {len(missing)} missing tags for Wikidata descriptions in batches of {batch_size} ...")
    total = len(missing)
    for start in range(0, total, batch_size):
        batch = missing[start:start + batch_size]
        # Fetch descriptions (this function parallelizes internally over CHUNK_SIZE)
        desc_map = fetch_tag_descriptions_from_titles(batch, lang=lang, site=site)

        # Merge successful lookups into cache
        for t in batch:
            desc = desc_map.get(t)
            if isinstance(desc, str) and desc.strip():
                cache[t] = {
                    "tag": t,
                    "wikidata_description": desc,
                    "lang": lang,
                    "site": site
                }

        # Save progress after each batch for robustness
        _save_tag_wd_cache(cache, path_out)
        print(f"[5_wikidata_desc] Processed and saved {min(start + batch_size, total)} / {total} tags to {path_out}")

    print(f"Tag Wikidata cache now contains {len(cache)} entries. Saved to: {path_out}")


# ---------- Incremental I/O helpers ----------

def load_existing_output(path: str) -> Dict[str, dict]:
    """Load a JSONL of enriched items into a dict keyed by QID."""
    existing: Dict[str, dict] = {}
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                qid = item.get('qid')
                if qid:
                    existing[qid] = item
    return existing


def write_output(items_by_qid: Dict[str, dict], path: str) -> None:
    """Write the items dictionary to JSONL, creating parent folder if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as out_f:
        for it in items_by_qid.values():
            out_f.write(json.dumps(it, ensure_ascii=False) + '\n')


def process_and_clean_batch(batch_items: List[dict], lang: str = 'en') -> Tuple[List[dict], List[dict], int]:
    """
    Enrich a batch with Wikidata descriptions, then apply the same filtering rules
    used previously. Returns (kept_items, examples_not_in_wikipedia, count_not_in_wikipedia).
    """
    qids = [it.get('qid') for it in batch_items]
    desc_map = fetch_wikidata_descriptions(qids, lang=lang)

    kept: List[dict] = []
    not_in_wikipedia_examples: List[dict] = []
    not_in_wikipedia_count = 0

    for it in batch_items:
        q = it.get('qid')
        desc = desc_map.get(q)
        enriched = dict(it)
        enriched['wikidata_description'] = desc

        # Apply the original cleaning rules
        first_par = it.get('wikipedia_first_paragraph', '') or ''
        if desc is None:
            continue
        if '<?xml' in first_par:
            continue
        if 'href' in first_par:
            continue
        if '{{' in desc:
            continue
        if desc.startswith('Wiki') or desc.startswith('wiki'):
            continue
        if it.get('label') not in first_par:
            # Track a few examples and a full count
            if len(not_in_wikipedia_examples) < 5:
                not_in_wikipedia_examples.append(enriched)
            not_in_wikipedia_count += 1
            continue

        kept.append(enriched)

    return kept, not_in_wikipedia_examples, not_in_wikipedia_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich items with Wikidata descriptions.")
    parser.add_argument("--input_path", type=str, default="./data/interim/4_items_with_wikipedia.jsonl")
    parser.add_argument("--output_path", type=str, default="./data/interim/5_items_with_wikipedia_and_desc.jsonl")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument(
        "--hf_base",
        type=str,
        default=None,
        help="Base path for HF caches (falls back to $ARGUS_HF_BASE, then $HF_HOME, then ./).",
    )
    return parser.parse_args()


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

    input_path = args.input_path
    output_path = args.output_path

    # Ensure the output folder exists ("see if the folder exists")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load any previously enriched items to support resuming
    existing_by_qid = load_existing_output(output_path)

    # Load all input items once
    wikidata_items: List[dict] = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                wikidata_items.append(json.loads(line))
    print(f"Length of wikidata_items: {len(wikidata_items)}")

    # Determine which items still need a non-empty wikidata_description
    items_to_process: List[dict] = []
    for it in wikidata_items:
        qid = it.get('qid')
        if not qid:
            continue
        prev = existing_by_qid.get(qid)
        if (prev is None) or (not prev.get('wikidata_description')):
            items_to_process.append(it)

    print(f"Length of items_to_process: {len(items_to_process)}")

    total_to_process = len(items_to_process)

    global_not_in_wikipedia_count = 0
    global_not_in_wikipedia_examples: List[dict] = []

    for start in range(0, total_to_process, args.batch_size):
        batch = items_to_process[start:start + args.batch_size]
        kept_items, not_in_wiki_examples, not_in_wiki_count = process_and_clean_batch(batch, lang=args.lang)

        # Merge into existing_by_qid
        for it in kept_items:
            existing_by_qid[it['qid']] = it

        # Aggregate diagnostics
        global_not_in_wikipedia_count += not_in_wiki_count
        for ex in not_in_wiki_examples:
            if len(global_not_in_wikipedia_examples) < 5:
                global_not_in_wikipedia_examples.append(ex)

        # Save after each batch
        write_output(existing_by_qid, output_path)
        print(f"Processed and saved {min(start + args.batch_size, total_to_process)} of {total_to_process} items")

    print(f"Items not in wikipedia: {global_not_in_wikipedia_count}")
    print(f"Enriched {len(existing_by_qid)} items with 'wikidata_description'. Saved to {output_path}")

    # === NEW: Ensure Wikidata descriptions for all encountered tags ===
    try:
        all_tags: Set[str] = set()
        # Include tags from the main items we just processed
        all_tags |= _collect_unique_tags_from_items(wikidata_items)
        # Also include tags from landmark files if present
        all_tags |= _collect_unique_tags_from_jsonl('./data/interim/3_landmarks_low_freq.jsonl')
        all_tags |= _collect_unique_tags_from_jsonl('./data/interim/3_landmarks_high_freq.jsonl')
        print(f"Collected {len(all_tags)} unique tags from items and landmark files for Wikidata descriptions.")
        _augment_tags_wikidata_descriptions(all_tags, TAGS_WD_OUT_PATH, batch_size=args.batch_size, lang=args.lang, site='enwiki')
    except Exception as e:
        print(f"[WARN] Tag Wikidata augmentation skipped due to error: {e}")


if __name__ == '__main__':
    main()
