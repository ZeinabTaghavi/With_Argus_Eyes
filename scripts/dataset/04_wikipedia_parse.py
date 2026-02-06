"""Fetch Wikipedia summaries for tag QIDs."""

import argparse
import os
import json
import glob
from tqdm import tqdm


import requests
from urllib.parse import quote
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from multiprocessing import cpu_count
import re
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
CHUNKSIZE = int(os.getenv("WIKI_CHUNKSIZE", "16"))
# Path for tag→Wikipedia first paragraph cache
TAGS_OUT_PATH = "./data/interim/4_tags_wikipedia_first_paragraphs_cache.jsonl"
WIKI_LANGUAGE = "en"
USER_AGENT = "qid-first-paragraph/1.0 (contact: you@example.com)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Wikipedia summaries for Wikidata items and tags.")
    parser.add_argument(
        "--hf_base",
        type=str,
        default=None,
        help="Base path for HF caches (falls back to $ARGUS_HF_BASE, then $HF_HOME, then ./).",
    )
    parser.add_argument(
        "--hf_hub_cache",
        type=str,
        default=None,
        help="Path for Hugging Face hub cache ($HF_HUB_CACHE). Defaults to <hf_base>/hub if not set.",
    )
    parser.add_argument(
        "--hf_datasets_cache",
        type=str,
        default=None,
        help="Path for Hugging Face datasets cache ($HF_DATASETS_CACHE). Defaults to <hf_base>/datasets if not set.",
    )

    parser.add_argument(
        "--items_in",
        type=str,
        default="./data/interim/3_cleaned_items_tag_only.jsonl",
        help="Input JSONL from 03_cleaning.py (cleaned tag-only items).",
    )
    parser.add_argument(
        "--items_out",
        type=str,
        default="./data/interim/4_items_with_wikipedia.jsonl",
        help="Output JSONL for items enriched with Wikipedia fields.",
    )

    parser.add_argument(
        "--tags_cache_out",
        type=str,
        default="./data/interim/4_tags_wikipedia_first_paragraphs_cache.jsonl",
        help="Output JSONL cache: tag -> Wikipedia first paragraph.",
    )

    parser.add_argument("--language", type=str, default="en", help="Wikipedia language (default: en).")
    parser.add_argument(
        "--user-agent",
        type=str,
        default="qid-first-paragraph/1.0 (contact: you@example.com)",
        help="User-Agent for Wikimedia requests (please include contact info).",
    )

    parser.add_argument(
        "--max_workers",
        type=int,
        default=MAX_WORKERS,
        help="Parallel workers (defaults to $WIKI_MAX_WORKERS or CPU count).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=CHUNKSIZE,
        help="Chunk size for tqdm process_map fallback (defaults to $WIKI_CHUNKSIZE).",
    )
    parser.add_argument(
        "--items_batch_size",
        type=int,
        default=1000,
        help="Batch size when enriching items; progress is checkpointed each batch.",
    )
    parser.add_argument(
        "--tags_batch_size",
        type=int,
        default=10000,
        help="Batch size when enriching tags cache; progress is checkpointed each batch.",
    )
    args = parser.parse_args()

    # HF cache setup (mirrors other scripts)
    base = args.hf_base or os.environ.get("ARGUS_HF_BASE") or os.environ.get("HF_HOME") or "./"
    if args.hf_base:
        os.environ["HF_HOME"] = base
    else:
        os.environ.setdefault("HF_HOME", base)

    if args.hf_hub_cache:
        os.environ["HF_HUB_CACHE"] = args.hf_hub_cache
    else:
        os.environ.setdefault("HF_HUB_CACHE", f"{base}/hub")

    if args.hf_datasets_cache:
        os.environ["HF_DATASETS_CACHE"] = args.hf_datasets_cache
    else:
        os.environ.setdefault("HF_DATASETS_CACHE", f"{base}/datasets")

    return args

WIKIDATA_API = "https://www.wikidata.org/w/api.php"

def _make_session(user_agent: str = "qid-first-paragraph/1.0 (contact: you@example.com)"):
    s = requests.Session()
    s.headers.update({
        # put your contact here per Wikimedia best practice
        "User-Agent": user_agent,
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

def _get_wikipedia_title_from_qid(
    qid: str,
    lang: str = "en",
    session: requests.Session | None = None,
    *,
    user_agent: str = "qid-first-paragraph/1.0 (contact: you@example.com)",
):
    """
    Resolve a Wikidata QID to the Wikipedia page title (and lang) using wbgetentities.
    Requires the given language's wiki (e.g., 'enwiki'); if absent, we skip the item (no fallback to other languages).
    """
    if not qid or not qid.startswith("Q"):
        raise ValueError("QID must look like 'Q…' (e.g., 'Q92561').")
    session = session or _make_session(user_agent=user_agent)

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

def _fetch_first_paragraph(
    title: str,
    wiki_lang: str,
    session: requests.Session | None = None,
    *,
    user_agent: str = "qid-first-paragraph/1.0 (contact: you@example.com)",
) -> str:
    """
    Fetch the first paragraph of a Wikipedia article.
    Tries REST /page/summary first; falls back to Action API TextExtracts exintro.
    """
    session = session or _make_session(user_agent=user_agent)

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


# ---------------------------------------------------------------------------
# Wikipedia paragraph cleaning / validation
# ---------------------------------------------------------------------------

_DISALLOWED_LEADS = (
    "may refer to",
    "may also refer to",
    "can refer to",
    "can also refer to",
    "may stand for",
    "may mean",
    "this category should only contain articles on species",
)

_DISALLOWED_SUBSTRINGS = (
    "disambiguation",
    "may refer to one of the following",
    "refer to one of the following",
    "this page lists articles",
    "this disambiguation page",
    "wikipedia does not have an article with this exact name",
    "this category should only contain articles on species",
)


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def clean_wikipedia_first_paragraph(text: str, *, min_words: int = 10) -> str | None:
    """
    Return a cleaned first paragraph if it looks like a real lead section.
    Otherwise return None (illegal / not useful / disambiguation / too short).
    """
    if not isinstance(text, str):
        return None
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return None

    lead = cleaned.casefold()
    for prefix in _DISALLOWED_LEADS:
        if lead.startswith(prefix):
            return None

    head = lead[:200]
    if any(s in head for s in _DISALLOWED_SUBSTRINGS):
        return None

    if _word_count(cleaned) < int(min_words):
        return None

    return cleaned

def first_paragraph_from_qid(qid: str, lang: str = "en") -> str:
    """
    Main entry: QID → (title, wiki_lang) → first paragraph.
    """
    session = _make_session()
    title, wiki_lang = _get_wikipedia_title_from_qid(qid, lang=lang, session=session)
    return _fetch_first_paragraph(title, wiki_lang, session=session)

def _process_one(item: dict) -> dict:
    """
    Return enriched item with Wikipedia fields.

    Always returns a dict with:
      - wikipedia_checked: bool
      - wikipedia_exists: bool
      - wikipedia_first_paragraph: str ("" when not found)
    """
    qid = item.get("qid") or item.get("Q_number") or item.get("itemQ")
    if not qid:
        raise LookupError(f"No qid found for {item}")

    try:
        title, wiki_lang = _get_wikipedia_title_from_qid(qid, lang=WIKI_LANGUAGE, user_agent=USER_AGENT)
        first_para = _fetch_first_paragraph(title, wiki_lang, user_agent=USER_AGENT)
        first_para_clean = clean_wikipedia_first_paragraph(first_para)
        enriched = dict(item)
        enriched["qid"] = qid
        enriched["wikipedia_checked"] = True
        enriched["wikipedia_exists"] = bool(first_para_clean)
        enriched["wikipedia_lang"] = wiki_lang
        enriched["wikipedia_title"] = title
        enriched["wikipedia_first_paragraph"] = first_para_clean or ""
        enriched.pop("_user_agent", None)
        return enriched
    except (LookupError, ValueError, requests.RequestException):
        enriched = dict(item)
        enriched["qid"] = qid
        enriched["wikipedia_checked"] = True
        enriched["wikipedia_exists"] = False
        enriched.setdefault("wikipedia_lang", WIKI_LANGUAGE)
        enriched.setdefault("wikipedia_title", "")
        enriched["wikipedia_first_paragraph"] = ""
        enriched.pop("_user_agent", None)
        return enriched


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
        checked = False
        if existing_item is not None:
            checked = bool(existing_item.get("wikipedia_checked")) or ("wikipedia_exists" in existing_item)
        if (existing_item is None) or (not checked):
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
            processed_item = dict(processed_item) if processed_item is not None else dict(orig)
            processed_item["qid"] = qid
            existing_by_qid[qid] = processed_item

    # Write out all items (including those without a paragraph) so we don't reprocess forever.
    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    with open(path_out, 'w', encoding='utf-8') as f_out:
        for it in existing_by_qid.values():
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
        first_para = _fetch_first_paragraph(name, WIKI_LANGUAGE, user_agent=USER_AGENT)
        first_para_clean = clean_wikipedia_first_paragraph(first_para)
        if not first_para_clean:
            return None
        return {
            "tag": name,
            "wikipedia_lang": WIKI_LANGUAGE,
            "wikipedia_title": name,
            "wikipedia_first_paragraph": first_para_clean,
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
                    # Drop any cached paragraphs that are no longer considered "legal"
                    para = obj.get("wikipedia_first_paragraph", "")
                    para_clean = clean_wikipedia_first_paragraph(para)
                    if not para_clean:
                        continue
                    obj["wikipedia_first_paragraph"] = para_clean
                    cache[tag.strip()] = obj
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
    cached_tags = set(cache.keys())
    wanted = {t.strip() for t in tag_names if isinstance(t, str) and t.strip()}
    missing = sorted([t for t in wanted if t not in cached_tags])

    if not missing:
        print(f"No new tags to process. Cache already has {len(cache)} entries.")
        # Still rewrite the cache so any previously-stored "soft"/illegal paragraphs get removed.
        _save_tag_cache(cache, path_out)
        print(f"Re-saved cleaned tag cache to: {path_out}")
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


def _filter_related_tags_in_item(item: dict, legal_tags: set[str]) -> dict:
    """
    Remove related_tags that do not have a legal cached paragraph.
    Returns a (possibly) new dict; also normalizes kept tags to stripped strings.
    """
    tags = item.get("related_tags")
    if not isinstance(tags, list) or not tags:
        return item
    filtered: list[str] = []
    for t in tags:
        if not isinstance(t, str):
            continue
        s = t.strip()
        if s and (s in legal_tags):
            filtered.append(s)
    if filtered == tags:
        return item
    out = dict(item)
    out["related_tags"] = filtered
    return out

def main() -> None:
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
    args = parse_args()

    global MAX_WORKERS, CHUNKSIZE, TAGS_OUT_PATH, WIKI_LANGUAGE, USER_AGENT
    MAX_WORKERS = max(1, int(args.max_workers))
    CHUNKSIZE = max(1, int(args.chunksize))
    TAGS_OUT_PATH = args.tags_cache_out
    WIKI_LANGUAGE = args.language
    USER_AGENT = args.user_agent

    print(f"[4_wikipedia_par] Using {MAX_WORKERS} workers.")
    print(f"[4_wikipedia_par] Using chunk size {CHUNKSIZE}.")

    written_files: list[str] = []

    out_path = args.items_out

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
                        # Only keep items with a legal cached paragraph
                        para = item.get("wikipedia_first_paragraph", "")
                        para_clean = clean_wikipedia_first_paragraph(para)
                        if not para_clean:
                            continue
                        item["wikipedia_first_paragraph"] = para_clean
                        item.setdefault("wikipedia_checked", True)
                        item.setdefault("wikipedia_exists", True)
                        existing_by_qid[qid] = item
        except Exception:
            existing_by_qid = {}

    # Load input items and determine which need processing
    wikidata_items: list[dict] = []
    items_to_process: list[dict] = []
    try:
        with open(args.items_in, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                qid = item.get('qid') or item.get('Q_number') or item.get('itemQ')
                if not qid:
                    continue
                item['qid'] = qid
                wikidata_items.append(item)
                existing_item = existing_by_qid.get(qid)
                checked = False
                if existing_item is not None:
                    checked = bool(existing_item.get("wikipedia_checked")) or ("wikipedia_exists" in existing_item)
                if existing_item is None:
                    items_to_process.append(item)
                elif not checked:
                    items_to_process.append(item)

    except FileNotFoundError:
        print(f"[4_wikipedia_par] Missing input file: {args.items_in}")

    # Process items in batches of 10,000, saving after each batch
    total_added = 0
    batch_size = max(1, int(args.items_batch_size))
    for start in tqdm(range(0, len(items_to_process), batch_size), desc="[4_wikipedia_par] Processing item batches"):
        batch = items_to_process[start:start + batch_size]

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
            processed = dict(processed) if processed is not None else dict(orig)
            processed['qid'] = qid
            if qid not in existing_by_qid:
                batch_added += 1
            existing_by_qid[qid] = processed

        total_added += batch_added

        # Immediately write out current progress for fault tolerance
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f_out:
            for it in existing_by_qid.values():
                # Only store items with a legal, non-empty Wikipedia first paragraph
                if it.get("wikipedia_first_paragraph"):
                    f_out.write(json.dumps(it, ensure_ascii=False) + '\n')
        if out_path not in written_files:
            written_files.append(out_path)
        remaining_est = max(0, len(items_to_process) - total_added)
        print(f"[4_wikipedia_par update] items saved so far: {len(existing_by_qid)} (+{batch_added} this batch, +{total_added} this run), estimated remaining (qids): {remaining_est}")

        print(f"Processed {min(start + batch_size, len(items_to_process))} out of {len(items_to_process)} items and saved progress.")

    # Final rewrite: ensure output only contains items with legal paragraphs (and cleaned text)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_final = out_path + ".tmp"
    with open(tmp_final, "w", encoding="utf-8") as f_out:
        for it in existing_by_qid.values():
            para = it.get("wikipedia_first_paragraph", "")
            para_clean = clean_wikipedia_first_paragraph(para)
            if not para_clean:
                continue
            obj = dict(it)
            obj["wikipedia_first_paragraph"] = para_clean
            obj.setdefault("wikipedia_checked", True)
            obj.setdefault("wikipedia_exists", True)
            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
    os.replace(tmp_final, out_path)
    if out_path not in written_files:
        written_files.append(out_path)

    # Summaries
    print(f"Total input items: {len(wikidata_items)}")
    print(f"Items needing (re)processing: {len(items_to_process)}")
    stored_items = sum(1 for it in existing_by_qid.values() if it.get("wikipedia_first_paragraph"))
    print(f"Items stored with a legal Wikipedia paragraph (final): {stored_items}")
    print(f"Saved enriched items to: {out_path}")

    # === Tag Wikipedia first paragraph cache ===
    # IMPORTANT: only process tags for items we are actually going to keep,
    # i.e., items that have a legal, non-empty Wikipedia first paragraph.
    kept_items = [it for it in existing_by_qid.values() if it.get("wikipedia_first_paragraph")]
    print("Collecting tags from kept items...")
    all_tags: set[str] = set()
    all_tags |= _collect_unique_tags_from_items(kept_items)
    print(f"Collected {len(all_tags)} unique tags from kept items.")

    # Build/extend the tags cache JSONL with first paragraphs.
    # This function loads the existing cache (if any) and only processes missing tags.
    if all_tags:
        _augment_tags_first_paragraphs(all_tags, TAGS_OUT_PATH, batch_size=max(1, int(args.tags_batch_size)))
        if os.path.exists(TAGS_OUT_PATH):
            written_files.append(TAGS_OUT_PATH)
    else:
        print("[4_wikipedia_par] No kept items with tags; skipping tag paragraph cache update.")

    # Filter related_tags based on legal tag paragraphs (present in the tag cache).
    # If an item ends up with no related tags, we drop it (it won't help next-step tag expansion).
    if os.path.exists(out_path) and os.path.exists(TAGS_OUT_PATH):
        try:
            tag_cache = _load_tag_cache(TAGS_OUT_PATH)
            legal_tags = set(tag_cache.keys())
        except Exception:
            legal_tags = set()

        if legal_tags:
            tmp_out = out_path + ".tmp"
            kept = 0
            with open(out_path, "r", encoding="utf-8") as fin, open(tmp_out, "w", encoding="utf-8") as fout:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    obj = _filter_related_tags_in_item(obj, legal_tags)
                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    kept += 1
            os.replace(tmp_out, out_path)
            print(f"[4_wikipedia_par] Filtered related_tags using tag cache; kept {kept} items.")

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

    print("\n[4_WIKIPEDIA_PARSE] Output summary")
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
