"""Collect random Wikidata items and labels for dataset construction."""

import os

import math
import time
import json
import gzip
import sqlite3
import requests
import multiprocessing as mp
from typing import List, Tuple, Optional, Iterable
from p_tqdm import p_imap, p_umap
from tqdm import tqdm
import argparse

API = "https://www.wikidata.org/w/api.php"
HEADERS = {"User-Agent": "WikidataSampler/0.3 (research; your-email@example.com)"}

DB_PATH = "./data/interim/1_wikidata_random.db"          # disk-backed dedup + resume
OUT_PATH = "./data/interim/1_wikidata_random_labels_0M.jsonl.gz"  # compressed output


# -------------------------------
# SQLite helpers (disk dedup)
# -------------------------------
def init_db(path: Optional[str] = None):
    # Resolve DB path at call-time to allow overrides
    if path is None:
        path = os.environ.get("WIKIDATA_DB_PATH", DB_PATH)
    new = not os.path.exists(path)
    con = sqlite3.connect(path, isolation_level=None, timeout=60)
    cur = con.cursor()
    # speed knobs (safe enough for this use)
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("PRAGMA temp_store=MEMORY;")
    cur.execute("PRAGMA cache_size=-200000;")  # ~200MB cache; adjust as needed

    cur.execute("""
        CREATE TABLE IF NOT EXISTS seen (
            qid TEXT PRIMARY KEY
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS labels (
            qid TEXT PRIMARY KEY,
            label TEXT
        )
    """)
    return con

def seen_count(con) -> int:
    return con.execute("SELECT COUNT(*) FROM seen").fetchone()[0]

def labeled_count(con) -> int:
    return con.execute("SELECT COUNT(*) FROM labels").fetchone()[0]

def insert_seen(con, qids: Iterable[str]):
    con.executemany("INSERT OR IGNORE INTO seen(qid) VALUES (?)", ((q,) for q in qids))

def insert_labels(con, pairs: Iterable[Tuple[str, str]]):
    # Use OR REPLACE so a refresh can update existing labels
    con.executemany("INSERT OR REPLACE INTO labels(qid,label) VALUES (?,?)", pairs)

def next_unlabeled(con, limit: int) -> List[str]:
    # Grab QIDs that have no label yet
    return [r[0] for r in con.execute(
        "SELECT qid FROM seen WHERE qid NOT IN (SELECT qid FROM labels) LIMIT ?", (limit,)
    ).fetchall()]


# Export helpers
# --------------
def dump_all_labels_to_gzip(out_path: str, con=None, overwrite: bool = False) -> int:
    """
    Stream all rows from the labels table into a gz JSONL at out_path.
    This lets us reuse an existing DB (no new API calls) to rebuild the
    labels file. Returns the number of lines written.
    """
    close_con = False
    if con is None:
        con = init_db()
        close_con = True
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mode = "wt" if overwrite else "at"
    written = 0
    with gzip.open(out_path, mode, encoding="utf-8") as f:
        cur = con.cursor()
        cur.execute("SELECT qid, label FROM labels")
        for qid, label in cur:
            f.write(json.dumps({"qid": qid, "label": label}, ensure_ascii=False) + "\n")
            written += 1
    if close_con:
        con.close()
    print(f"[labels] exported {written} rows from DB to {out_path}")
    return written

def dump_all_labels_batched(out_path: str, batch: int = 500000, overwrite: bool = False, con=None) -> int:
    """
    Export the entire labels table using rowid pagination (memory-safe for very large DBs).
    Returns total rows written.
    """ 
    close_con = False
    if con is None:
        con = init_db()
        close_con = True
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mode = "wt" if overwrite else "at"
    total = 0
    last_rowid = 0
    with gzip.open(out_path, mode, encoding="utf-8") as f:
        cur = con.cursor()
        while True:
            rows = cur.execute(
                "SELECT rowid, qid, label FROM labels WHERE rowid > ? ORDER BY rowid LIMIT ?",
                (last_rowid, batch)
            ).fetchall()
            if not rows:
                break
            for rowid, qid, label in rows:
                f.write(json.dumps({"qid": qid, "label": label}, ensure_ascii=False) + "\n")
                last_rowid = rowid
            total += len(rows)
            print(f"[labels] exported {total} rows so far...")
    if close_con:
        con.close()
    print(f"[labels] exported {total} rows from DB to {out_path}")
    return total

def print_db_counts(con=None):
    close_con = False
    if con is None:
        con = init_db()
        close_con = True
    s = seen_count(con)
    l = labeled_count(con)
    print(f"[db] seen={s} | labeled={l} | unlabeled={s - l}")
    if close_con:
        con.close()


# -------------------------------
# HTTP helpers (retry/backoff)
# -------------------------------
def _request_with_retries(session: requests.Session, method: str, url: str,
                          *, params=None, data=None, timeout=30, retries=6):
    backoff = 1.0
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=timeout) if method == "GET" \
                else session.post(url, data=data, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.exceptions.HTTPError as e:
            if r.status_code == 429:  # Too Many Requests
                # Check for Retry-After header
                retry_after = r.headers.get('Retry-After')
                if retry_after:
                    wait_time = int(retry_after)
                    print(f"Rate limited. Waiting {wait_time} seconds as requested by server...")
                    time.sleep(wait_time)
                else:
                    # Exponential backoff for 429 without Retry-After header
                    wait_time = min(120.0, backoff * 3)
                    print(f"Rate limited. Waiting {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    backoff = wait_time
                continue
            elif attempt == retries - 1:
                raise
            else:
                time.sleep(backoff)
                backoff = min(32.0, backoff * 2)
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(backoff)
            backoff = min(32.0, backoff * 2)


# -------------------------------
# Worker functions
# -------------------------------
def fetch_random_qids_worker(target_count: int, rnlimit: int = 20, sleep: float = 2.0) -> List[str]:
    """
    Fetch ~target_count random QIDs (namespace 0) in a single process.
    May include duplicates; caller dedups via SQLite.
    """
    s = requests.Session(); s.headers.update(HEADERS)
    out: List[str] = []
    while len(out) < target_count:
        params = {
            "action": "query", "list": "random", "rnnamespace": 0,
            "rnlimit": min(rnlimit, target_count - len(out)),
            "format": "json"
        }
        r = _request_with_retries(s, "GET", API, params=params)
        for it in r.json().get("query", {}).get("random", []):
            t = it.get("title", "")
            if t.startswith("Q"):
                out.append(t)
        time.sleep(sleep)
    return out

def labels_for_chunk_worker(chunk: List[str], lang_order: List[str]) -> List[Tuple[str, str]]:
    if not chunk:
        return []
    s = requests.Session(); s.headers.update(HEADERS)
    params = {
        "action": "wbgetentities",
        "ids": "|".join(chunk),
        "props": "labels",
        "languages": "|".join(lang_order),
        "format": "json",
    }
    r = _request_with_retries(s, "POST", API, data=params)
    entities = (r.json().get("entities", {}) or {})
    out: List[Tuple[str, str]] = []
    for qid, ent in entities.items():
        label = ""
        labels = ent.get("labels") or {}
        for lang in lang_order:
            v = labels.get(lang, {})
            if "value" in v:
                label = v["value"]
                break
        out.append((qid, label))
    # Add sleep after each request to respect rate limits
    time.sleep(1.0)  # 1 second sleep between requests
    return out


# -------------------------------
# High-level pipeline
# -------------------------------
def collect_qids_streaming(target_total: int,
                           per_task: int = 2000,
                           tasks_per_round: int = None,
                           workers: Optional[int] = None):
    """
    Fill the 'seen' table up to target_total unique QIDs by streaming results
    from parallel tasks. Writes directly into SQLite; minimal RAM.
    """
    con = init_db()
    con.execute("PRAGMA busy_timeout = 60000")  # 60 seconds
    cpu = os.cpu_count() or 1
    workers = workers or min(cpu, 2)  # Very conservative: max 2 workers
    if tasks_per_round is None:
        tasks_per_round = workers  # No pressure multiplication

    round_idx = 1
    while True:
        have = seen_count(con)
        if have >= target_total:
            print(f"[qids] target reached: {have}/{target_total}")
            break

        remaining = target_total - have
        # Plan this round (don’t overshoot too much)
        planned = min(remaining * 2, per_task * tasks_per_round)  # ~2x oversample to fight duplicates
        n_tasks = math.ceil(planned / per_task)
        payloads = [per_task] * n_tasks

        print(f"[qids] round {round_idx} | have={have} | planning {n_tasks} tasks x {per_task} ≈ {planned}")
        if workers == 1:
            for payload in tqdm(payloads, desc="[qids] tasks", total=len(payloads)):
                lst = fetch_random_qids_worker(payload)
                insert_seen(con, lst)
        else:
            # p_imap streams results task-by-task (low RAM)
            for lst in p_imap(fetch_random_qids_worker, payloads, num_cpus=workers):
                insert_seen(con, lst)

        new_have = seen_count(con)
        print(f"[qids] round {round_idx} | unique total now {new_have} (+{new_have - have})")
        round_idx += 1
    con.close()

def label_qids_streaming(batch_unlabeled: int = 50000,
                         lang_order: List[str] = ["en"],
                         workers: Optional[int] = None,
                         out_path: str = OUT_PATH,
                         export_existing_if_empty: bool = False,
                         overwrite_output: bool = False):
    """
    Resolve labels for unlabeled QIDs in streaming fashion and append to gz JSONL.
    """
    con = init_db()
    con.execute("PRAGMA busy_timeout = 60000")  # 60 seconds
    cpu = os.cpu_count() or 1
    workers = workers or min(cpu, 2)  # Very conservative: max 2 workers

    # open gzip file in append mode
    # ensure file exists; if not, create it
    if not os.path.exists(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        print(f"Creating file: {out_path}")
        with gzip.open(out_path, "wt", encoding="utf-8") as _:
            pass

    print(f"Labeling {batch_unlabeled} qids in {workers} workers")

    while True:
        unlabeled = next_unlabeled(con, batch_unlabeled)
        if not unlabeled:
            if export_existing_if_empty:
                print("[labels] no unlabeled rows; exporting existing labels from DB...")
                dump_all_labels_to_gzip(out_path=out_path, con=con, overwrite=overwrite_output)
            else:
                print("[labels] nothing to label; all done.")
            break

        # chunk into 50s (API limit)
        chunks = [unlabeled[i:i+50] for i in range(0, len(unlabeled), 50)]
        print(f"[labels] labeling {len(unlabeled)} qids in {len(chunks)} chunks using {workers} workers")

        if workers == 1:
            results = []
            for chunk in tqdm(chunks, desc="[labels] chunks", total=len(chunks)):
                results.append(labels_for_chunk_worker(chunk, lang_order))
        else:
            # Parallel call; returns list of lists
            results = p_umap(labels_for_chunk_worker, chunks, [lang_order] * len(chunks), num_cpus=workers)

        # Stream insert + write to gzip
        total_written = 0
        with gzip.open(out_path, "at", encoding="utf-8") as f:
            for pairs in results:
                if not pairs:
                    continue
                insert_labels(con, pairs)
                for qid, label in pairs:
                    f.write(json.dumps({"qid": qid, "label": label}, ensure_ascii=False) + "\n")
                    total_written += 1
        have_labels = labeled_count(con)
        print(f"[labels] wrote {total_written} lines this pass | total labeled={have_labels}")
    con.close()

def run_pipeline(target_total: int = 7000000,
                 qid_workers: Optional[int] = None,
                 label_workers: Optional[int] = None,
                 out_path: str = OUT_PATH,
                 lang_order: List[str] = ["en"],
                 export_existing_if_empty: bool = False,
                 overwrite_output: bool = False):
    """
    End-to-end: collect unique QIDs up to target_total, then resolve labels,
    all streaming to disk. Safe to re-run; it resumes from the DB state.
    """
    print(f"==> Phase 1: collecting up to {target_total} unique QIDs")
    collect_qids_streaming(target_total=target_total, workers=qid_workers)

    print(f"==> Phase 2: labeling (streamed)")
    label_qids_streaming(batch_unlabeled=50000,
                         lang_order=lang_order,
                         workers=label_workers,
                         out_path=out_path,
                         export_existing_if_empty=export_existing_if_empty,
                         overwrite_output=overwrite_output)


# -------------------------------
# CLI
# -------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wikidata QID collector/labeler + exporter")
    parser.add_argument("--db", type=str, default=None,
                        help="Path to SQLite DB (defaults to DB_PATH or $WIKIDATA_DB_PATH)")
    parser.add_argument("--out", type=str, default=None,
                        help="Path to output gzip JSONL (defaults to OUT_PATH)")
    parser.add_argument("--hf_base", type=str, default=None,
                        help="Base directory for HF caches (sets $HF_HOME; falls back to $ARGUS_HF_BASE, then $HF_HOME, then ./)")
    parser.add_argument("--hf_hub_cache", type=str, default=None,
                        help="Path for Hugging Face hub cache ($HF_HUB_CACHE). Defaults to <hf_base>/hub if not set.")
    parser.add_argument("--hf_datasets_cache", type=str, default=None,
                        help="Path for Hugging Face datasets cache ($HF_DATASETS_CACHE). Defaults to <hf_base>/datasets if not set.")
    parser.add_argument("--target_total", type=int, default=7000000,
                        help="Target unique QIDs to collect into 'seen'")
    parser.add_argument("--qid_workers", type=int, default=None,
                        help="Workers for random QID collection")
    parser.add_argument("--label_workers", type=int, default=None,
                        help="Workers for labeling API calls")
    parser.add_argument("--lang", nargs="+", default=["en"],
                        help="Language fallback order for labels")
    parser.add_argument("--batch_unlabeled", type=int, default=50000,
                        help="Batch size for fetching unlabeled QIDs")
    parser.add_argument("--export_only", action="store_true",
                        help="Skip API; export all labels from existing DB to gzip")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite output gzip instead of appending")
    parser.add_argument("--export_batch", type=int, default=500_000,
                        help="Batch size for export-only rowid pagination")
    parser.add_argument("--print_counts", action="store_true",
                        help="Print DB counts before doing anything")

    args = parser.parse_args()

    # Cache/env resolution:
    # - CLI flags win
    # - otherwise keep existing env if present
    # - otherwise derive from base
    base = args.hf_base or os.environ.get("ARGUS_HF_BASE") or os.environ.get("HF_HOME") or "./"
    if args.hf_base:
        os.environ["HF_HOME"] = base
    else:
        os.environ.setdefault("HF_HOME", base)  # makes <BASE>/hub and <BASE>/datasets

    if args.hf_hub_cache:
        os.environ["HF_HUB_CACHE"] = args.hf_hub_cache
    else:
        os.environ.setdefault("HF_HUB_CACHE", f"{base}/hub")

    if args.hf_datasets_cache:
        os.environ["HF_DATASETS_CACHE"] = args.hf_datasets_cache
    else:
        os.environ.setdefault("HF_DATASETS_CACHE", f"{base}/datasets")

    return args


def main() -> None:
    args = parse_args()

    # Allow runtime DB override via env or --db
    if args.db:
        os.environ["WIKIDATA_DB_PATH"] = args.db

    out_path = args.out or OUT_PATH
    if args.target_total>1000000:
        out_path = args.out or OUT_PATH.replace("_0M.jsonl.gz", f"_{args.target_total//1000000}M.jsonl.gz")

    if args.print_counts:
        print_db_counts()


    if args.export_only:
        # Pure export path (no API calls)
        dump_all_labels_batched(out_path=out_path, batch=args.export_batch, overwrite=args.overwrite)
    else:
        # Full pipeline; if nothing to label, export what exists
        print(f"==> Phase 1: collecting up to {args.target_total} unique QIDs")
        collect_qids_streaming(target_total=args.target_total, workers=args.qid_workers)

        print(f"==> Phase 2: labeling (streamed)")
        label_qids_streaming(batch_unlabeled=args.batch_unlabeled,
                             lang_order=args.lang,
                             workers=args.label_workers,
                             out_path=out_path,
                             export_existing_if_empty=True,
                             overwrite_output=args.overwrite)


if __name__ == "__main__":
    main()
