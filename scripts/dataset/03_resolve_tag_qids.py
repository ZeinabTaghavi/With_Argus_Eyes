"""Resolve tag labels to Wikidata QIDs with optional SPARQL fallback."""

#!/usr/bin/env python3
"""Resolve Wikidata QIDs for previously harvested related tag labels.

Given one or more JSONL files produced by `2_Get_tags.py`, this script
re-queries Wikidata for each source item and augments every `related_tag`
label with its corresponding QID. The enriched data are written back to
JSONL files (one output file per input file).

Example usage:

    python 3_resolve_tag_qids.py \
        --input ./data/interim/2_tag_only/items_with_tags_0_100000.jsonl \
        --output-dir ./data/interim/2_tag_only/with_qids

Use `--input` with a directory or glob pattern to batch-process multiple
files. By default the script politely parallelises requests using p_tqdm's
`p_map`; tune `--parallelism` to adjust.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import time
import unicodedata
from collections import defaultdict, deque
from typing import Deque, Dict, Iterable, List, Optional

import requests
from p_tqdm import p_map
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Shared configuration (mirrors 2_Get_tags.py defaults)
# ---------------------------------------------------------------------------

BASE = os.environ.get("ARGUS_HF_BASE") or os.environ.get("HF_HOME") or "./"

os.environ.setdefault("HF_HOME", BASE)
os.environ.setdefault("HF_HUB_CACHE", f"{BASE}/hub")
os.environ.setdefault("HF_DATASETS_CACHE", f"{BASE}/datasets")


WDQS_ENDPOINT = "https://query.wikidata.org/sparql"


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def make_session(user_agent: str = "YourAppName/1.0 (you@example.com)") -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": "application/sparql-results+json",
            "Accept-Encoding": "gzip, deflate",
        }
    )
    return session


def request_json_with_backoff(
    session: requests.Session,
    url: str,
    *,
    params: Optional[Dict[str, str]] = None,
    max_retries: int = 6,
    base_sleep: float = 0.5,
) -> Dict:
    """Robust JSON fetch with retry, rate-limit honouring, and backoff."""

    last_resp: Optional[requests.Response] = None
    for attempt in range(1, max_retries + 1):
        resp = session.get(url, params=params, timeout=60)
        last_resp = resp
        status = resp.status_code

        if status in (429, 502, 503, 504):
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                try:
                    wait = float(retry_after)
                except ValueError:
                    wait = base_sleep * (2 ** (attempt - 1))
            else:
                wait = base_sleep * (2 ** (attempt - 1))
            wait += random.uniform(0.0, 0.5)
            time.sleep(wait)
            continue

        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")
        if "json" not in content_type.lower():
            snippet = resp.text[:200]
            raise RuntimeError(
                f"Expected JSON but got Content-Type={content_type}. Snippet: {snippet!r}"
            )

        return resp.json()

    snippet = ""
    try:
        snippet = last_resp.text[:200] if last_resp is not None else ""
    except Exception:  # pragma: no cover - best effort logging only
        pass
    raise RuntimeError(
        "Failed after {max_retries} attempts. Last status="
        f"{getattr(last_resp, 'status_code', None)}. Snippet: {snippet!r}"
    )


# ---------------------------------------------------------------------------
# Wikidata query helpers (adapted from 2_Get_tags.py)
# ---------------------------------------------------------------------------


def _build_statement_query(qid: str, language: str) -> str:
    return f"""
    SELECT ?item ?itemLabel
           ?wdProp ?wdPropLabel
           (STRAFTER(STR(?wdProp), "/entity/") AS ?pid)
           ?value ?valueLabel
           (IF(isIRI(?value), "wikibase-item", DATATYPE(?value)) AS ?valueType)
           (IF(isIRI(?value), STRAFTER(STR(?value), "/entity/"), "") AS ?valueQid)
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


def _build_direct_query(qid: str, language: str) -> str:
    return f"""
    SELECT ?prop ?propLabel ?value ?valueLabel
           (STRAFTER(STR(?prop), "/prop/direct/") AS ?pid)
           (IF(isIRI(?value), "wikibase-item", DATATYPE(?value)) AS ?valueType)
           (IF(isIRI(?value), STRAFTER(STR(?value), "/entity/"), "") AS ?valueQid)
    WHERE {{
      VALUES ?item {{ wd:{qid} }}
      ?item ?prop ?value .
      FILTER(STRSTARTS(STR(?prop), "http://www.wikidata.org/prop/direct/"))
      SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "{language},[AUTO_LANGUAGE]" .
      }}
    }}
    """


def get_item_properties(
    qid: str,
    *,
    language: str = "en",
    user_agent: str = "YourAppName/1.0 (you@example.com)",
    session: Optional[requests.Session] = None,
) -> Dict:
    """Return property claims for a Wikidata item (statement graph first)."""

    owns_session = False
    if session is None:
        session = make_session(user_agent=user_agent)
        owns_session = True

    try:
        data = request_json_with_backoff(
            session,
            WDQS_ENDPOINT,
            params={"query": _build_statement_query(qid, language)},
            max_retries=6,
            base_sleep=0.5,
        )

        claims: Dict[str, Dict[str, List[Dict[str, str]]]] = {}
        item_label: Optional[str] = None

        for binding in data["results"]["bindings"]:
            if item_label is None and "itemLabel" in binding:
                item_label = binding["itemLabel"]["value"]

            if binding["valueType"]["value"] != "wikibase-item":
                continue

            q_number = binding.get("valueQid", {}).get("value", "").strip()
            label = binding.get("valueLabel", {}).get("value", "").strip()
            uri = binding["value"]["value"]
            if not q_number or not label:
                continue

            pid = binding["pid"]["value"]
            prop_label = binding.get("wdPropLabel", {}).get("value", pid)

            claims.setdefault(pid, {"property_label": prop_label, "values": []})
            claims[pid]["values"].append(
                {
                    "type": "wikibase-item",
                    "Q_number": q_number,
                    "label": label,
                    "uri": uri,
                }
            )

        if claims:
            return {"QID": qid, "label": item_label or qid, "claims": claims}

        # Fallback to direct properties if no statement claims were returned.
        fallback = request_json_with_backoff(
            session,
            WDQS_ENDPOINT,
            params={"query": _build_direct_query(qid, language)},
            max_retries=6,
            base_sleep=0.5,
        )

        for binding in fallback["results"]["bindings"]:
            if binding["valueType"]["value"] != "wikibase-item":
                continue

            q_number = binding.get("valueQid", {}).get("value", "").strip()
            label = binding.get("valueLabel", {}).get("value", "").strip()
            if not q_number or not label:
                continue

            pid = binding["pid"]["value"]
            prop_label = binding.get("propLabel", {}).get("value", pid)

            claims.setdefault(pid, {"property_label": prop_label, "values": []})
            claims[pid]["values"].append(
                {
                    "type": "wikibase-item",
                    "Q_number": q_number,
                    "label": label,
                    "uri": binding["value"]["value"],
                }
            )

        return {"QID": qid, "label": qid, "claims": claims}

    finally:
        if owns_session:
            session.close()


def _escape_sparql_literal(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def resolve_qid_by_exact_label(
    label: str,
    *,
    language: str = "en",
    user_agent: str = "YourAppName/1.0 (you@example.com)",
    session: Optional[requests.Session] = None,
) -> Optional[str]:
    """Resolve a Wikidata QID via an exact label match."""

    owns_session = False
    if session is None:
        session = make_session(user_agent=user_agent)
        owns_session = True

    try:
        safe_label = _escape_sparql_literal(label)
        query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT (STRAFTER(STR(?item), "/entity/") AS ?qid)
        WHERE {{
          ?item rdfs:label "{safe_label}"@{language} .
        }}
        LIMIT 1
        """

        data = request_json_with_backoff(
            session,
            WDQS_ENDPOINT,
            params={"query": query},
            max_retries=6,
            base_sleep=1.0,
        )
        rows = data.get("results", {}).get("bindings", [])
        if not rows:
            return None
        return rows[0].get("qid", {}).get("value")
    finally:
        if owns_session:
            session.close()


# ---------------------------------------------------------------------------
# Tag augmentation
# ---------------------------------------------------------------------------


def _canon_label(value: str) -> str:
    if not isinstance(value, str):
        return ""
    return unicodedata.normalize("NFKC", value).casefold().strip()


def _build_value_map(
    claims: Dict[str, Dict[str, List[Dict[str, str]]]],
    *,
    item_label: str,
) -> Dict[str, Deque[Dict[str, str]]]:
    """Index candidate tag values by canonical label for quick lookup."""

    value_map: Dict[str, Deque[Dict[str, str]]] = defaultdict(deque)
    for prop in claims.values():
        for value in prop.get("values", []):
            if value.get("type") != "wikibase-item":
                continue
            q_number = value.get("Q_number", "").strip()
            label = value.get("label", "").strip()
            if not q_number or not label:
                continue
            if item_label and item_label in label:
                continue

            canon = _canon_label(label)
            value_map[canon].append(
                {
                    "qid": q_number,
                    "wikidata_label": label,
                }
            )

    return value_map


def _augment_item(
    item: Dict,
    *,
    language: str,
    user_agent: str,
    per_call_sleep: tuple[float, float],
    fallback_exact_label: bool,
) -> Dict:
    """Worker function run in parallel to enrich a single item."""

    orig_tags: List[str] = list(item.get("related_tags", []))
    if not orig_tags:
        return {**item, "related_tags": []}

    qid = item.get("Q_number") or item.get("qid") or item.get("QID")
    if not qid:
        enriched = [{"label": label, "qid": None} for label in orig_tags]
        return {**item, "related_tags": enriched, "_error": "missing_qid"}

    time.sleep(random.uniform(*per_call_sleep))
    session = make_session(user_agent=user_agent)

    try:
        props = get_item_properties(
            qid,
            language=language,
            user_agent=user_agent,
            session=session,
        )
    except Exception as exc:
        session.close()
        enriched = [{"label": label, "qid": None} for label in orig_tags]
        return {
            **item,
            "related_tags": enriched,
            "_error": f"{type(exc).__name__}: {exc}",
        }

    value_map = _build_value_map(props.get("claims", {}), item_label=item.get("label", ""))

    enriched: List[Dict[str, Optional[str]]] = []
    unresolved_indices: List[int] = []

    for idx, label in enumerate(orig_tags):
        canon = _canon_label(label)
        candidates = value_map.get(canon)
        if candidates:
            match = candidates.popleft()
            if match["wikidata_label"] != label:
                enriched.append(
                    {
                        "label": label,
                        "qid": match["qid"],
                        "resolved_label": match["wikidata_label"],
                    }
                )
            else:
                enriched.append({"label": label, "qid": match["qid"]})
        else:
            enriched.append({"label": label, "qid": None})
            unresolved_indices.append(idx)

    if fallback_exact_label and unresolved_indices:
        for idx in unresolved_indices:
            label = enriched[idx]["label"]
            try:
                time.sleep(random.uniform(*per_call_sleep))
                resolved_qid = resolve_qid_by_exact_label(
                    label,
                    language=language,
                    user_agent=user_agent,
                    session=session,
                )
            except Exception:
                resolved_qid = None

            if resolved_qid:
                enriched[idx]["qid"] = resolved_qid

    session.close()
    return {**item, "related_tags": enriched}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _expand_input_paths(path_like: str) -> List[str]:
    if os.path.isdir(path_like):
        return sorted(
            glob.glob(os.path.join(path_like, "*.jsonl"))
        )

    matches = glob.glob(path_like)
    if matches:
        return sorted(matches)

    if os.path.exists(path_like):
        return [path_like]

    raise FileNotFoundError(f"No files matched input path '{path_like}'")


def _load_jsonl(path: str) -> List[Dict]:
    items: List[Dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def _write_jsonl(path: str, items: Iterable[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def process_file(
    input_path: str,
    *,
    output_path: str,
    language: str,
    user_agent: str,
    parallelism: int,
    per_call_sleep: tuple[float, float],
    fallback_exact_label: bool,
) -> Dict[str, int]:
    items = _load_jsonl(input_path)
    if not items:
        _write_jsonl(output_path, [])
        return {"total": 0, "resolved": 0, "unresolved": 0}

    enriched_items = p_map(
        _augment_item,
        items,
        num_cpus=parallelism,
        language=language,
        user_agent=user_agent,
        per_call_sleep=per_call_sleep,
        fallback_exact_label=fallback_exact_label,
        desc=f"Resolving tags ({os.path.basename(input_path)})",
    )

    resolved = 0
    unresolved = 0
    for item in enriched_items:
        for tag in item.get("related_tags", []):
            if tag.get("qid"):
                resolved += 1
            else:
                unresolved += 1

    _write_jsonl(output_path, enriched_items)

    return {"total": resolved + unresolved, "resolved": resolved, "unresolved": unresolved}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment related tag labels with Wikidata QIDs")
    parser.add_argument(
        "--input",
        default="./data/interim/5_items_with_wikipedia_and_desc.jsonl",
        help="Path/glob to JSONL file(s) or directory containing JSONL outputs from 2_Get_tags.py",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory where enriched JSONL files will be written",
    )
    parser.add_argument(
        "--output",
        help="Explicit output JSONL path (only valid when a single input file is processed)",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Wikidata language code for labels (default: en)",
    )
    parser.add_argument(
        "--user-agent",
        default="YourAppName/1.0 (you@example.com)",
        help="Custom user-agent string for Wikidata requests",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=6,
        help="Number of worker processes for Wikidata requests",
    )
    parser.add_argument(
        "--min-sleep",
        type=float,
        default=0.15,
        help="Minimum per-call sleep (seconds) between requests",
    )
    parser.add_argument(
        "--max-sleep",
        type=float,
        default=0.45,
        help="Maximum per-call sleep (seconds) between requests",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Disable exact-label SPARQL fallback for unresolved tags",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_paths = _expand_input_paths(args.input)
    if not input_paths:
        raise SystemExit("No input files found.")

    if args.output and len(input_paths) != 1:
        raise SystemExit("--output can only be used when processing a single input file.")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    per_call_sleep = (min(args.min_sleep, args.max_sleep), max(args.min_sleep, args.max_sleep))

    overall_total = 0
    overall_resolved = 0
    overall_unresolved = 0

    for input_path in tqdm(input_paths, desc="Files"):
        if args.output:
            output_path = args.output
        else:
            file_name = os.path.splitext(os.path.basename(input_path))[0]
            if args.output_dir:
                output_path = os.path.join(args.output_dir, f"{file_name}_with_qids.jsonl")
            else:
                output_dir = os.path.dirname(input_path) or "."
                output_path = os.path.join(output_dir, f"{file_name}_with_qids.jsonl")

        stats = process_file(
            input_path,
            output_path=output_path,
            language=args.language,
            user_agent=args.user_agent,
            parallelism=max(1, args.parallelism),
            per_call_sleep=per_call_sleep,
            fallback_exact_label=not args.no_fallback,
        )

        overall_total += stats["total"]
        overall_resolved += stats["resolved"]
        overall_unresolved += stats["unresolved"]

    if overall_total:
        resolved_pct = (overall_resolved / overall_total) * 100
        unresolved_pct = (overall_unresolved / overall_total) * 100
    else:
        resolved_pct = unresolved_pct = 0.0

    print(
        "Completed tag augmentation: "
        f"resolved {overall_resolved}/{overall_total} tags ({resolved_pct:.1f}%), "
        f"unresolved {overall_unresolved} ({unresolved_pct:.1f}%)."
    )


if __name__ == "__main__":
    main()

