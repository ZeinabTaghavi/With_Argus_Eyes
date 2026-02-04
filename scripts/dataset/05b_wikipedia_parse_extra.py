"""Fetch additional Wikipedia summaries for missed tags."""

import glob
import json
import os
from collections.abc import Iterable
from typing import Any

from tqdm import tqdm

import requests
from requests.adapters import HTTPAdapter
from urllib.parse import quote
from urllib3.util.retry import Retry


# 1️⃣ Pick an absolute path that has enough space (ARGUS_HF_BASE/HF_HOME or default)
BASE = os.environ.get("ARGUS_HF_BASE") or os.environ.get("HF_HOME") or "./"

# 2️⃣ Point both caches there ─ before any HF import
os.environ.setdefault("HF_HOME", BASE)
os.environ.setdefault("HF_HUB_CACHE", f"{BASE}/hub")
os.environ.setdefault("HF_DATASETS_CACHE", f"{BASE}/datasets")


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
print(f"[4_2_wikipedia_par] Using {MAX_WORKERS} workers.")
CHUNKSIZE = int(os.getenv("WIKI_CHUNKSIZE", "16"))
print(f"[4_2_wikipedia_par] Using chunk size {CHUNKSIZE}.")

INPUT_DIR = os.getenv(
    "WIKI_EMB_RANK_DIR", "./data/processed/8_Emb_Rank"
)
OUTPUT_PATH = os.getenv(
    "WIKI_EMB_RANK_OUT",
    "./data/processed/8_Emb_Rank/7_all_wikipedia_pages.jsonl",
)
BATCH_SIZE = int(os.getenv("WIKI_EMB_RANK_BATCH_SIZE", "1000"))
LANG = os.getenv("WIKI_EMB_LANG", "en")

WIKIDATA_API = "https://www.wikidata.org/w/api.php"


def _make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "qid-first-paragraph/1.0 (contact: you@example.com)",
            "Accept": "application/json",
        }
    )
    retry = Retry(
        total=3,
        backoff_factor=0.4,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _get_wikipedia_title_from_qid(
    qid: str, lang: str = "en", session: requests.Session | None = None
):
    if not qid or not qid.startswith("Q"):
        raise ValueError("QID must look like 'Q…' (e.g., 'Q92561').")
    session = session or _make_session()

    params_pref = {
        "action": "wbgetentities",
        "format": "json",
        "formatversion": "2",
        "ids": qid,
        "props": "sitelinks/urls",
        "sitefilter": f"{lang}wiki",
    }
    response = session.get(WIKIDATA_API, params=params_pref, timeout=15)
    response.raise_for_status()
    data = response.json()
    entity = (data.get("entities") or {}).get(qid) or {}
    sitelinks = entity.get("sitelinks") or {}

    if f"{lang}wiki" in sitelinks:
        return sitelinks[f"{lang}wiki"]["title"], lang

    raise LookupError(f"No {lang}.wikipedia.org sitelink found for {qid}.")


def _fetch_first_paragraph(
    title: str, wiki_lang: str, session: requests.Session | None = None
) -> str:
    session = session or _make_session()

    rest_api = f"https://{wiki_lang}.wikipedia.org/api/rest_v1/page/summary/{quote(title.replace(' ', '_'), safe='')}"
    response = session.get(rest_api, params={"redirect": "true"}, timeout=15)
    if response.ok:
        payload = response.json()
        extract = (payload or {}).get("extract") or ""
        if extract.strip():
            return extract.strip()

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
    response = session.get(action_api, params=params, timeout=15)
    response.raise_for_status()
    payload = response.json()
    pages = (payload.get("query") or {}).get("pages") or []
    if not pages:
        raise LookupError(f"No page returned for '{title}' on {wiki_lang}.wikipedia.org.")
    extract = (pages[0] or {}).get("extract") or ""
    if not extract.strip():
        raise LookupError(f"No extract text found for '{title}' on {wiki_lang}.wikipedia.org.")
    return extract.strip().split("\n\n", 1)[0].strip()


def _normalize_qid_value(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate:
        return None
    if candidate[0].upper() != "Q":
        return None
    number = candidate[1:]
    if not number:
        return None
    if not number.isdigit():
        return None
    return f"Q{int(number)}"


def _clean_label(value: Any) -> str:
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned:
            return cleaned
    return ""


def _extract_label_from_item(item: dict[str, Any]) -> str:
    for key in ("label", "item_label", "title", "name"):
        label = _clean_label(item.get(key))
        if label:
            return label
    nested = item.get("item")
    if isinstance(nested, dict):
        for key in ("label", "name", "title"):
            label = _clean_label(nested.get(key))
            if label:
                return label
    return ""


def _iter_related_tag_entries(related_tags: Any):
    if related_tags is None:
        return
    if isinstance(related_tags, dict):
        qid = _normalize_qid_value(
            related_tags.get("qid")
            or related_tags.get("Q_number")
            or related_tags.get("itemQ")
        )
        label = _clean_label(
            related_tags.get("label")
            or related_tags.get("name")
            or related_tags.get("title")
        )
        if qid:
            yield qid, label
        return
    if isinstance(related_tags, str):
        qid = _normalize_qid_value(related_tags)
        label = _clean_label(related_tags)
        if qid:
            yield qid, (label if label != qid else "")
        return
    if not isinstance(related_tags, Iterable):
        return
    for tag in related_tags:
        if isinstance(tag, dict) or isinstance(tag, str):
            yield from _iter_related_tag_entries(tag)
        elif isinstance(tag, Iterable):
            yield from _iter_related_tag_entries(tag)


def _maybe_update_label(label_map: dict[str, str], qid: str | None, label: str) -> None:
    if not qid:
        return
    existing = label_map.get(qid)
    cleaned = _clean_label(label)
    if existing is None:
        label_map[qid] = cleaned
    elif (not existing) and cleaned:
        label_map[qid] = cleaned


def _collect_qid_label_map(directory: str) -> dict[str, str]:
    pattern = os.path.join(directory, "*.jsonl")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[4_2_wikipedia_par] No JSONL files found in {directory}")
        return {}

    label_map: dict[str, str] = {}
    for path in tqdm(files, desc="[4_2_wikipedia_par] Collecting qids"):
        try:
            with open(path, "r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except Exception:
                        continue
                    main_qid = _normalize_qid_value(
                        item.get("qid") or item.get("Q_number") or item.get("itemQ")
                    )
                    if main_qid:
                        _maybe_update_label(label_map, main_qid, _extract_label_from_item(item))
                    for rel_qid, rel_label in _iter_related_tag_entries(item.get("related_tags")):
                        _maybe_update_label(label_map, rel_qid, rel_label)
        except FileNotFoundError:
            continue
    print(f"[4_2_wikipedia_par] Collected {len(label_map)} unique QIDs (items + related tags).")
    return label_map


def _load_existing_records(path: str) -> dict[str, dict]:
    if not os.path.exists(path):
        return {}
    records: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            qid = _normalize_qid_value(obj.get("qid") or obj.get("Q_number") or obj.get("itemQ"))
            if not qid:
                continue
            obj["qid"] = qid
            obj["label"] = _clean_label(obj.get("label"))
            records[qid] = obj
    print(f"[4_2_wikipedia_par] Loaded {len(records)} existing QID entries from cache.")
    return records


def _save_records(records: dict[str, dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for qid in sorted(records.keys()):
            entry = dict(records[qid])
            entry["label"] = _clean_label(entry.get("label"))
            if not entry.get("wikipedia_first_paragraph"):
                continue
            entry["qid"] = qid
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _process_qid(qid: str) -> dict | None:
    try:
        session = _make_session()
        title, wiki_lang = _get_wikipedia_title_from_qid(qid, lang=LANG, session=session)
        first_para = _fetch_first_paragraph(title, wiki_lang, session=session)
        return {
            "qid": qid,
            "wikipedia_lang": wiki_lang,
            "wikipedia_title": title,
            "wikipedia_first_paragraph": first_para,
            "wikipedia_exists": True,
        }
    except (LookupError, ValueError, requests.RequestException):
        return None


def _needs_processing(qid: str, cache: dict[str, dict]) -> bool:
    record = cache.get(qid)
    if record is None:
        return True
    if record.get("wikipedia_first_paragraph"):
        return False
    if record.get("wikipedia_exists") is False:
        return False
    return True


def _process_batch(batch: list[str]) -> list[dict | None]:
    if _HAVE_P_TQDM:
        return p_map(_process_qid, batch, num_cpus=MAX_WORKERS)
    if _HAVE_PROCESS_MAP:
        return _process_map(
            _process_qid, batch, max_workers=MAX_WORKERS, chunksize=CHUNKSIZE
        )
    return [_process_qid(qid) for qid in batch]


def main():
    qid_label_map = _collect_qid_label_map(INPUT_DIR)
    if not qid_label_map:
        print("[4_2_wikipedia_par] Nothing to process.")
        return

    existing_records = _load_existing_records(OUTPUT_PATH)
    for qid, label in qid_label_map.items():
        if not label:
            continue
        record = existing_records.get(qid)
        if record is not None and not record.get("label"):
            record["label"] = label

    all_qids = sorted(qid_label_map.keys())
    qids_to_process = [qid for qid in all_qids if _needs_processing(qid, existing_records)]
    if not qids_to_process:
        print("[4_2_wikipedia_par] Cache already contains all requested QIDs.")
        return

    total = len(qids_to_process)
    print(f"[4_2_wikipedia_par] Processing {total} QIDs (items + related tags).")
    processed_count = 0

    for start in tqdm(
        range(0, total, BATCH_SIZE), desc="[4_2_wikipedia_par] Enriching batches"
    ):
        batch = qids_to_process[start : start + BATCH_SIZE]
        results = _process_batch(batch)
        for qid, processed in zip(batch, results):
            if processed is not None:
                record = dict(processed)
            else:
                record = {
                    "qid": qid,
                    "wikipedia_lang": "",
                    "wikipedia_title": "",
                    "wikipedia_first_paragraph": "",
                    "wikipedia_exists": False,
                }
            label = qid_label_map.get(qid, "") or record.get("label") or ""
            record["label"] = label
            existing_records[qid] = record
        processed_count += len(batch)
        _save_records(existing_records, OUTPUT_PATH)
        print(
            f"[4_2_wikipedia_par] Saved progress for {processed_count} / {total} QIDs "
            f"(cached total: {len(existing_records)})."
        )

    print(
        f"[4_2_wikipedia_par] Finished. Cached paragraphs for {len(existing_records)} QIDs. "
        f"Output written to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()

