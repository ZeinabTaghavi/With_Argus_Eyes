"""Attach unrelevant qids to a single Wikipedia paragraph cache.

This stage keeps a single paragraph source of truth at:
  data/interim/4_tags_wikipedia_first_paragraphs_cache.jsonl

Flow:
1) Read unrelevant qids per item from stage 08.
2) Ensure each unrelevant qid has a Wikipedia paragraph in the cache (fetch missing qids).
3) Write stage-09 index file: qid -> wiki_unrelevants (only qids with cached paragraphs).
"""

import argparse
import json
import os
import re
from typing import Any
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

WIKIDATA_API = "https://www.wikidata.org/w/api.php"

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update Wikipedia cache for unrelevant qids and write qid->wiki_unrelevants."
    )
    parser.add_argument(
        "--wikipedia_pages",
        type=str,
        default=os.path.join("data", "interim", "4_tags_wikipedia_first_paragraphs_cache.jsonl"),
        help="Wikipedia paragraph cache JSONL path (read/write). Supports tag-cache and qid-cache rows.",
    )
    parser.add_argument(
        "--unrelevant_tags",
        type=str,
        default=os.path.join("data", "interim", "6_unrelevant_qids.jsonl"),
        help="Unrelevant tags JSONL path from stage 08.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("data", "interim", "6_wiki_unrelevants_results.jsonl"),
        help="Output JSONL path: one line {'qid','wiki_unrelevants'} per main item.",
    )
    parser.add_argument("--language", type=str, default="en", help="Wikipedia language code.")
    parser.add_argument(
        "--user_agent",
        type=str,
        default="ArgusEyes/1.0 (research; contact@example.com)",
        help="HTTP User-Agent for Wikipedia/Wikidata requests.",
    )
    parser.add_argument(
        "--skip_cache_update",
        action="store_true",
        help="Do not fetch missing unrelevant qids; only intersect against existing cache.",
    )
    return parser.parse_args()


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def _clean_wikipedia_first_paragraph(text: str, *, min_words: int = 10) -> str | None:
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


def _make_session(user_agent: str) -> requests.Session:
    sess = requests.Session()
    sess.headers.update({"User-Agent": user_agent})
    retries = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=32, pool_maxsize=64)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess


def _get_wikipedia_title_from_qid(
    qid: str, *, language: str, session: requests.Session
) -> tuple[str, str]:
    if not qid or not qid.startswith("Q"):
        raise ValueError(f"Invalid qid: {qid}")

    params = {
        "action": "wbgetentities",
        "format": "json",
        "formatversion": "2",
        "ids": qid,
        "props": "sitelinks/urls",
        "sitefilter": f"{language}wiki",
    }
    r = session.get(WIKIDATA_API, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    ent = (data.get("entities") or {}).get(qid) or {}
    sitelinks = ent.get("sitelinks") or {}
    key = f"{language}wiki"
    if key not in sitelinks:
        raise LookupError(f"No {language}.wikipedia.org sitelink found for {qid}")
    return sitelinks[key]["title"], language


def _fetch_first_paragraph(title: str, wiki_lang: str, session: requests.Session) -> str:
    rest_api = (
        f"https://{wiki_lang}.wikipedia.org/api/rest_v1/page/summary/"
        f"{quote(title.replace(' ', '_'), safe='')}"
    )
    r = session.get(rest_api, params={"redirect": "true"}, timeout=20)
    if r.ok:
        j = r.json()
        extract = (j or {}).get("extract") or ""
        if extract.strip():
            return extract.strip().split("\n\n", 1)[0].strip()

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
    r = session.get(action_api, params=params, timeout=20)
    r.raise_for_status()
    j = r.json()
    pages = (j.get("query") or {}).get("pages") or []
    if not pages:
        raise LookupError(f"No page returned for '{title}' on {wiki_lang}.wikipedia.org")
    extract = (pages[0] or {}).get("extract") or ""
    if not extract.strip():
        raise LookupError(f"No extract text found for '{title}' on {wiki_lang}.wikipedia.org")
    return extract.strip().split("\n\n", 1)[0].strip()


def _fetch_wikipedia_record_for_qid(
    qid: str, *, language: str, session: requests.Session
) -> dict[str, Any] | None:
    try:
        title, wiki_lang = _get_wikipedia_title_from_qid(qid, language=language, session=session)
        first_para = _fetch_first_paragraph(title, wiki_lang, session=session)
        cleaned = _clean_wikipedia_first_paragraph(first_para)
        if not cleaned:
            return None
        return {
            "tag": title,
            "qid": qid,
            "label": title,
            "wikipedia_lang": wiki_lang,
            "wikipedia_title": title,
            "wikipedia_first_paragraph": cleaned,
        }
    except (LookupError, ValueError, requests.RequestException):
        return None


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


_QID_RE = re.compile(r"^Q[1-9]\d*$")


def _extract_qid_from_row(row: dict[str, Any]) -> str | None:
    qid = row.get("qid")
    if isinstance(qid, str) and _QID_RE.match(qid):
        return qid
    tag = row.get("tag")
    if isinstance(tag, str) and _QID_RE.match(tag):
        return tag
    return None


def _normalize_cache_record(row: dict[str, Any], qid: str, para_clean: str) -> dict[str, Any]:
    title = row.get("wikipedia_title") or row.get("label") or row.get("tag") or qid
    return {
        "tag": row.get("tag") or title,
        "qid": qid,
        "label": row.get("label") or title,
        "wikipedia_lang": row.get("wikipedia_lang") or "en",
        "wikipedia_title": title,
        "wikipedia_first_paragraph": para_clean,
    }


def _to_cache_row(rec: dict[str, Any]) -> dict[str, Any]:
    qid = rec["qid"]
    title = rec.get("wikipedia_title") or rec.get("label") or rec.get("tag") or qid
    return {
        "tag": rec.get("tag") or title,
        "qid": qid,
        "label": rec.get("label") or title,
        "wikipedia_lang": rec.get("wikipedia_lang") or "en",
        "wikipedia_title": title,
        "wikipedia_first_paragraph": rec["wikipedia_first_paragraph"],
    }


def _load_wikipedia_cache(path: str) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    cache: dict[str, dict[str, Any]] = {}
    rows = _load_jsonl(path)
    for row in rows:
        qid = _extract_qid_from_row(row)
        para = row.get("wikipedia_first_paragraph")
        if not qid or not isinstance(qid, str):
            continue
        if not isinstance(para, str) or not para.strip():
            continue
        cleaned = _clean_wikipedia_first_paragraph(para)
        if not cleaned:
            continue
        cache[qid] = _normalize_cache_record(row, qid, cleaned)
    return cache, rows


def _save_wikipedia_cache(
    cache: dict[str, dict[str, Any]], existing_rows: list[dict[str, Any]], path: str
) -> None:
    _ensure_parent_dir(path)
    merged_rows: list[dict[str, Any]] = []
    written_qids: set[str] = set()

    # Preserve non-qid tag-cache rows and update any qid rows already present.
    for row in existing_rows:
        if not isinstance(row, dict):
            continue
        qid = _extract_qid_from_row(row)
        if not qid:
            merged_rows.append(row)
            continue
        if qid in written_qids:
            continue
        if qid in cache:
            merged_rows.append(_to_cache_row(cache[qid]))
            written_qids.add(qid)
        else:
            merged_rows.append(row)
            written_qids.add(qid)

    # Append newly fetched qids that were not present in existing rows.
    for qid in sorted(cache.keys()):
        if qid in written_qids:
            continue
        merged_rows.append(_to_cache_row(cache[qid]))

    with open(path, "w", encoding="utf-8") as f:
        for row in merged_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _save_jsonl(rows: list[dict[str, Any]], path: str) -> None:
    _ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    written_files: list[str] = []

    unrelevant_tags = _load_jsonl(args.unrelevant_tags)
    print(f"Loaded {len(unrelevant_tags)} unrelevant rows from {args.unrelevant_tags}")

    wiki_cache, wiki_cache_rows = _load_wikipedia_cache(args.wikipedia_pages)
    print(f"Loaded {len(wiki_cache)} qid-linked Wikipedia rows from {args.wikipedia_pages}")

    all_unrelevant_qids: set[str] = set()
    for row in unrelevant_tags:
        for qid in row.get("unrelevant", []):
            if isinstance(qid, str) and qid:
                all_unrelevant_qids.add(qid)
    print(f"Unique unrelevant qids: {len(all_unrelevant_qids)}")

    if args.skip_cache_update:
        print("Skipping Wikipedia cache update (requested).")
    else:
        missing_qids = [q for q in sorted(all_unrelevant_qids) if q not in wiki_cache]
        print(f"Missing unrelevant qids in cache: {len(missing_qids)}")
        if missing_qids:
            session = _make_session(args.user_agent)
            added = 0
            for qid in tqdm(missing_qids, desc="Fetching missing unrelevant wiki paragraphs"):
                rec = _fetch_wikipedia_record_for_qid(qid, language=args.language, session=session)
                if rec is None:
                    continue
                wiki_cache[qid] = rec
                added += 1
            print(f"Added {added} new qids into unified Wikipedia cache.")
            if added > 0:
                _save_wikipedia_cache(wiki_cache, wiki_cache_rows, args.wikipedia_pages)
                print(f"Updated Wikipedia cache at: {args.wikipedia_pages}")
                if os.path.exists(args.wikipedia_pages):
                    written_files.append(args.wikipedia_pages)

    available_wiki_qids = set(wiki_cache.keys())
    results: list[dict[str, Any]] = []
    for item in tqdm(unrelevant_tags, desc="Building wiki_unrelevants index"):
        qid = item.get("qid")
        if not isinstance(qid, str) or not qid:
            continue
        unrelevants = item.get("unrelevant", [])
        if not isinstance(unrelevants, list):
            unrelevants = []
        wiki_unrelevants = [u for u in unrelevants if isinstance(u, str) and u in available_wiki_qids]
        results.append({"qid": qid, "wiki_unrelevants": wiki_unrelevants})

    _save_jsonl(results, args.output)
    print(f"Saved {len(results)} rows to {args.output}")
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
