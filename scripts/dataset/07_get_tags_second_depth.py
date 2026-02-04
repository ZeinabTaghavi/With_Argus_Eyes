import os
# 1️⃣ Pick an absolute path that has enough space
BASE = "./"

# 2️⃣ Point both caches there ─ before any HF import
os.environ["HF_HOME"]          = BASE          # makes <BASE>/hub and <BASE>/datasets
os.environ["HF_HUB_CACHE"]     = f"{BASE}/hub" # optional, explicit
os.environ["HF_DATASETS_CACHE"]= f"{BASE}/datasets" 

import json
import requests
import time
import random
import multiprocessing
from p_tqdm import p_map
from tqdm import tqdm
import glob
from urllib.parse import quote
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

WDQS_ENDPOINT = "https://query.wikidata.org/sparql"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"

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

def make_wikipedia_session(user_agent="YourAppName/1.0 (you@example.com)"):
    """Create a session for Wikipedia API requests with retry logic."""
    s = requests.Session()
    s.headers.update({
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


def _escape_sparql_literal(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def resolve_qid_by_exact_label(label: str,
                               language: str = "en",
                               user_agent: str = "YourAppName/1.0 (you@example.com)"):
    """
    Resolve a Wikidata QID by exact label match in the given language.
    Returns the first QID found or None if not found.
    """
    session = make_session(user_agent=user_agent)
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


# --- Tunables (be polite to WDQS!) ---
# Use all available CPUs, but allow override via environment variable
PARALLELISM = int(os.getenv("MAX_WORKERS", multiprocessing.cpu_count() or 8))
print(f"Using {PARALLELISM} workers for parallel processing (available CPUs: {multiprocessing.cpu_count()})")
PER_CALL_SLEEP = (0.15, 0.45)  # small jitter per process between calls


# -------------------------------
# Get item properties (for finding related tags)
# -------------------------------

def get_item_properties(
    qid: str,
    *,
    language: str = "en",
    user_agent: str = "YourAppName/1.0 (you@example.com)",
    session: requests.Session | None = None
):
    """
    Return all property values for a Wikidata item including:
    - Statement properties (main values)
    - Direct properties
    - Qualifiers on statements
    - References on statements
    
    Returns a dict with "QID", "label", and "claims".
    """
    if session is None:
        session = make_session(user_agent=user_agent)

    # Comprehensive query that gets ALL Wikidata items linked to the page
    # This includes statement properties, qualifiers, and references
    query = f"""
    SELECT DISTINCT ?item ?itemLabel
           ?wdProp ?wdPropLabel
           (STRAFTER(STR(?wdProp), "/entity/") AS ?pid)
           ?value ?valueLabel
           (IF(isIRI(?value), "wikibase-item", DATATYPE(?value)) AS ?valueType)
           (IF(isIRI(?value), STRAFTER(STR(?value), "/entity/"), "") AS ?valueQid)
           ( ?wdPropLabel AS ?propertyLabel )
    WHERE {{
      VALUES ?item {{ wd:{qid} }}
      
      {{
        # Statement properties (main values)
        ?item ?p ?statement .
        ?wdProp wikibase:claim ?p .
        ?wdProp wikibase:statementProperty ?ps .
        ?statement ?ps ?value .
      }}
      UNION
      {{
        # Qualifiers on statements
        ?item ?p ?statement .
        ?statement ?pq ?value .
        ?wdProp wikibase:qualifier ?pq .
      }}
      UNION
      {{
        # References on statements
        ?item ?p ?statement .
        ?statement prov:wasDerivedFrom ?ref .
        ?ref ?pr ?value .
        ?wdProp wikibase:reference ?pr .
      }}
      UNION
      {{
        # Direct properties (non-statement) - convert prop/direct/ to entity/
        ?item ?prop ?value .
        FILTER(STRSTARTS(STR(?prop), "http://www.wikidata.org/prop/direct/"))
        BIND(IRI(REPLACE(STR(?prop), "/prop/direct/", "/entity/")) AS ?wdProp)
      }}
      
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

    # Initialize container
    item_label = None
    claims = {}
    seen_values = set()  # Track seen (pid, value) pairs to avoid duplicates

    for b in data["results"]["bindings"]:
        # Item label (same for all rows)
        if item_label is None and "itemLabel" in b:
            item_label = b["itemLabel"]["value"]

        pid = b["pid"]["value"]
        prop_label = b.get("propertyLabel", {}).get("value", pid)
        
        # Build the value object
        if b["valueType"]["value"] == "wikibase-item":
            q_number = b.get("valueQid", {}).get("value", "")
            if not q_number:
                continue
            
            # Create unique key to avoid duplicates
            value_key = (pid, q_number)
            if value_key in seen_values:
                continue
            seen_values.add(value_key)
            
            value_obj = {
                "type": "wikibase-item",
                "Q_number": q_number,
                "label": b.get("valueLabel", {}).get("value", ""),
                "uri": b["value"]["value"],
            }
        else:
            # For literals, use value as key
            value_val = b["value"]["value"]
            value_key = (pid, value_val)
            if value_key in seen_values:
                continue
            seen_values.add(value_key)
            
            value_obj = {
                "type": "literal",
                "datatype": b["valueType"]["value"],
                "value": value_val,
            }

        if pid not in claims:
            claims[pid] = {"property_label": prop_label, "values": [value_obj]}
        else:
            claims[pid]["values"].append(value_obj)

    return {"QID": qid, "label": item_label or qid, "claims": claims}


def _fetch_and_resolve_related_tags(item, language="en", user_agent="YourAppName/1.0 (you@example.com)"):
    """
    Worker function that:
    1. Gets the item's QID
    2. Fetches related_tags from Wikidata for that QID
    3. Extracts entity values as related_tags (with QIDs)
    4. Gets one-hop-related-tags for each tag
    
    Returns a new item dict with related_tags containing qid, label, and one_hop_related_tags.
    """
    # Get the item's QID
    qid = item.get("qid") or item.get("Q_number") or item.get("QID")
    item_label = item.get("label", "")
    
    if not qid or not isinstance(qid, str) or not qid.startswith("Q"):
        # If no QID, return item with empty related_tags
        return {**item, "related_tags": []}
    
    # Check if already processed (has related_tags with dict format and one_hop_related_tags)
    related_tags_existing = item.get("related_tags", [])
    if related_tags_existing and isinstance(related_tags_existing[0], dict):
        # Check if all tags have one_hop_related_tags
        all_have_one_hop = all("one_hop_related_tags" in tag for tag in related_tags_existing)
        if all_have_one_hop:
            return item
    
    try:
        # Gentle pause per request
        time.sleep(random.uniform(*PER_CALL_SLEEP))
        
        # Get properties for this item's QID
        props = get_item_properties(
            qid,
            language=language,
            user_agent=user_agent
        )
        
        # Extract entity values as related tags
        related_tags = []
        seen_qids = set()  # Track seen QIDs to avoid duplicates
        
        for pid, prop_data in props.get("claims", {}).items():
            values = prop_data.get("values", [])
            if not values:
                continue
            
            for v in values:
                # Only keep values that are Wikidata items
                if v.get("type") != "wikibase-item":
                    continue
                
                q_number = v.get("Q_number", "").strip()
                label_v = v.get("label", "")
                
                # Skip if no Q-number
                if not q_number:
                    continue
                
                # Skip if we've already seen this QID
                if q_number in seen_qids:
                    continue
                seen_qids.add(q_number)
                
                # Skip if the value's label contains the item label (to avoid self-references)
                if item_label and item_label.lower() in label_v.lower():
                    continue
                
                # Create tag dict with qid and label (one-hop will be added later)
                tag_dict = {"qid": q_number, "label": label_v}
                related_tags.append(tag_dict)
        
        return {**item, "related_tags": related_tags}
    
    except Exception as e:
        # On error, return item with empty related_tags
        return {**item, "related_tags": [], "_error": f"{type(e).__name__}: {e}"}


def _resolve_tags_with_qids(item, language="en", user_agent="YourAppName/1.0 (you@example.com)"):
    """
    Worker function run in parallel. Resolves QIDs for all related_tags in an item.
    Returns a new item dict with related_tags containing both qid and label.
    """
    # Get the related_tags list (should be a list of strings)
    related_tags_raw = item.get("related_tags", [])
    
    # If already processed (has dict format), verify structure and skip
    if related_tags_raw and isinstance(related_tags_raw[0], dict):
        # Assert that all tags have 'qid' field (even if None)
        for tag in related_tags_raw:
            assert isinstance(tag, dict), f"Expected dict, got {type(tag)}"
            assert "qid" in tag, f"Missing 'qid' field in tag: {tag}"
            assert "label" in tag, f"Missing 'label' field in tag: {tag}"
        return item
    
    resolved_tags = []
    for tag_label in related_tags_raw:
        if not isinstance(tag_label, str) or not tag_label.strip():
            continue
        
        # Gentle pause per request
        time.sleep(random.uniform(*PER_CALL_SLEEP))
        
        try:
            qid = resolve_qid_by_exact_label(tag_label.strip(), language=language, user_agent=user_agent)
            if qid:
                resolved_tags.append({"qid": qid, "label": tag_label.strip()})
            else:
                # Keep the tag even if QID not found, but mark it
                resolved_tags.append({"qid": None, "label": tag_label.strip()})
        except Exception as e:
            # On error, keep the tag but mark it
            resolved_tags.append({"qid": None, "label": tag_label.strip(), "_error": f"{type(e).__name__}: {e}"})
    
    # Assert that every tag in resolved_tags has a 'qid' field
    for tag in resolved_tags:
        assert isinstance(tag, dict), f"Expected dict, got {type(tag)}"
        assert "qid" in tag, f"Missing 'qid' field in tag: {tag}"
        assert "label" in tag, f"Missing 'label' field in tag: {tag}"
    
    # Return a fresh dict with resolved tags
    return {**item, "related_tags": resolved_tags}


def _add_one_hop_related_tags(item, language="en", user_agent="YourAppName/1.0 (you@example.com)"):
    """
    Worker function to add one_hop_related_tags to each tag in related_tags.
    Takes an item with resolved tags (with qid and label) and adds one_hop_related_tags to each tag.
    """
    related_tags = item.get("related_tags", [])
    if not related_tags:
        return item
    
    # Process each tag to add one_hop_related_tags
    enriched_tags = []
    for tag in related_tags:
        # Copy the tag
        enriched_tag = dict(tag)
        
        # Skip if already has one_hop_related_tags
        if "one_hop_related_tags" in enriched_tag:
            enriched_tags.append(enriched_tag)
            continue
        
        # Get one-hop related tags for this tag
        one_hop_tags = _get_one_hop_related_tags(tag, language=language, user_agent=user_agent)
        enriched_tag["one_hop_related_tags"] = one_hop_tags
        enriched_tags.append(enriched_tag)
    
    return {**item, "related_tags": enriched_tags}


def _get_one_hop_related_tags(tag_dict: dict, language: str = "en", user_agent: str = "YourAppName/1.0 (you@example.com)"):
    """
    Get one-hop related tags for a tag (find related tags of the tag itself).
    Input: {"qid": "Q349", "label": "sport"}
    Output: [{"qid": "Q123", "label": "football"}, ...] or empty list if failed
    """
    qid = tag_dict.get("qid")
    tag_label = tag_dict.get("label", "")
    
    if not qid or not isinstance(qid, str) or not qid.startswith("Q"):
        return []
    
    try:
        # Gentle pause per request
        time.sleep(random.uniform(*PER_CALL_SLEEP))
        
        # Get properties for this tag's QID
        props = get_item_properties(
            qid,
            language=language,
            user_agent=user_agent
        )
        
        # Extract entity values as related tags
        one_hop_tags = []
        seen_qids = set()  # Track seen QIDs to avoid duplicates
        
        for pid, prop_data in props.get("claims", {}).items():
            values = prop_data.get("values", [])
            if not values:
                continue
            
            for v in values:
                # Only keep values that are Wikidata items
                if v.get("type") != "wikibase-item":
                    continue
                
                q_number = v.get("Q_number", "").strip()
                label_v = v.get("label", "")
                
                # Skip if no Q-number
                if not q_number:
                    continue
                
                # Skip if we've already seen this QID
                if q_number in seen_qids:
                    continue
                seen_qids.add(q_number)
                
                # Skip if the value's label contains the tag label (to avoid self-references)
                if tag_label and tag_label.lower() in label_v.lower():
                    continue
                
                # Add the one-hop tag
                one_hop_tags.append({"qid": q_number, "label": label_v})
        
        return one_hop_tags
    
    except Exception as e:
        # On error, return empty list
        return []


# -------------------------------
# Wikipedia first paragraph helpers
# -------------------------------

def _get_wikipedia_title_from_qid(qid: str, lang: str = "en", session: requests.Session | None = None):
    """
    Resolve a Wikidata QID to the Wikipedia page title (and lang) using wbgetentities.
    Requires the given language's wiki (e.g., 'enwiki'); if absent, raises LookupError.
    """
    if not qid or not qid.startswith("Q"):
        raise ValueError("QID must look like 'Q…' (e.g., 'Q92561').")
    session = session or make_wikipedia_session()

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
    session = session or make_wikipedia_session()

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
    session = make_wikipedia_session()
    title, wiki_lang = _get_wikipedia_title_from_qid(qid, lang=lang, session=session)
    return _fetch_first_paragraph(title, wiki_lang, session=session)

def _process_tag_wikipedia(tag_dict: dict, language: str = "en") -> dict | None:
    """
    Worker function to fetch Wikipedia first paragraph for a tag.
    Input: {"qid": "Q123", "label": "sport"}
    Output: {"qid": "Q123", "label": "sport", "wikipedia_first_paragraph": "..."} or None if failed
    """
    qid = tag_dict.get("qid")
    label = tag_dict.get("label")
    
    if not qid or not isinstance(qid, str):
        return None
    
    try:
        # Gentle pause per request
        time.sleep(random.uniform(*PER_CALL_SLEEP))
        first_para = first_paragraph_from_qid(qid, lang=language)
        return {
            "qid": qid,
            "label": label,
            "wikipedia_first_paragraph": first_para
        }
    except (LookupError, ValueError, requests.RequestException) as e:
        # On error, return None (we'll skip this tag)
        return None


def process_file(input_path, output_path, tags_wikipedia_output_path=None):
    """
    Process a single JSONL file:
    1. For each item with a QID, fetch related_tags from Wikidata
    2. Extract entity values as related_tags (with QIDs)
    3. Get one-hop-related-tags for each tag
    4. Optionally fetch Wikipedia first paragraphs for tags and save to separate file.
    
    Returns statistics about the processing.
    """
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return None
    
    print(f"\nProcessing: {input_path}")
    
    # Read all items
    all_items = []
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                all_items.append(item)
            except Exception as e:
                print(f"Error parsing line: {e}")
                continue
    
    # all_items = all_items[:10]
    print(f"Loaded {len(all_items)} items")
    
    # Filter out items that already have fully processed tags
    items_to_process = []
    already_processed = []
    for item in all_items:
        related_tags = item.get("related_tags", [])
        # Check if already fully processed (has dict format with qid, label, and one_hop_related_tags)
        if related_tags and isinstance(related_tags[0], dict):
            # Check if all tags have one_hop_related_tags
            all_have_one_hop = all("one_hop_related_tags" in tag for tag in related_tags)
            if all_have_one_hop:
                # Verify structure
                for tag in related_tags:
                    assert isinstance(tag, dict), f"Expected dict in already_processed item, got {type(tag)}"
                    assert "qid" in tag, f"Missing 'qid' field in already_processed tag: {tag}"
                    assert "label" in tag, f"Missing 'label' field in already_processed tag: {tag}"
                    assert "one_hop_related_tags" in tag, f"Missing 'one_hop_related_tags' field in already_processed tag: {tag}"
                already_processed.append(item)
            else:
                items_to_process.append(item)
        else:
            items_to_process.append(item)
    
    print(f"Items already processed: {len(already_processed)}")
    print(f"Items to process: {len(items_to_process)}")
    
    # Step 1: Fetch related_tags from Wikidata (without one-hop tags yet)
    if items_to_process:
        print("Step 1: Fetching related_tags from Wikidata...")
        items_with_tags = p_map(
            _fetch_and_resolve_related_tags,
            items_to_process,
            num_cpus=PARALLELISM,
            desc="Fetching related tags"
        )
        
        # Combine with already processed items
        all_items_with_tags = already_processed + items_with_tags
    else:
        all_items_with_tags = already_processed
    
    # Step 2: Collect unique tags across all items (deduplicate by QID for efficiency)
    print("Step 2: Collecting unique tags for one-hop processing...")
    unique_tags_by_qid = {}
    tag_to_items_map = {}  # Map QID to list of (item_idx, tag_idx) tuples
    
    for item_idx, item in enumerate(all_items_with_tags):
        related_tags = item.get("related_tags", [])
        for tag_idx, tag in enumerate(related_tags):
            tag_qid = tag.get("qid")
            if tag_qid and isinstance(tag_qid, str) and tag_qid.startswith("Q"):
                # Skip if already has one_hop_related_tags
                if "one_hop_related_tags" not in tag:
                    if tag_qid not in unique_tags_by_qid:
                        unique_tags_by_qid[tag_qid] = tag
                        tag_to_items_map[tag_qid] = []
                    tag_to_items_map[tag_qid].append((item_idx, tag_idx))
    
    print(f"Found {len(unique_tags_by_qid)} unique tags needing one-hop tags")
    
    # Step 3: Fetch one-hop tags for unique tags in parallel (much faster!)
    if unique_tags_by_qid:
        print("Step 3: Fetching one-hop tags for unique tags (parallel)...")
        
        def _add_one_hop_to_single_tag(tag_dict):
            """Helper to add one-hop tags to a single tag."""
            tag_qid = tag_dict.get("qid")
            if tag_qid and isinstance(tag_qid, str) and tag_qid.startswith("Q"):
                if "one_hop_related_tags" not in tag_dict:
                    one_hop_tags = _get_one_hop_related_tags(tag_dict, language="en", user_agent="YourAppName/1.0 (you@example.com)")
                    return {**tag_dict, "one_hop_related_tags": one_hop_tags}
            return {**tag_dict, "one_hop_related_tags": tag_dict.get("one_hop_related_tags", [])}
        
        unique_tags_list = list(unique_tags_by_qid.values())
        tags_with_one_hop = p_map(
            _add_one_hop_to_single_tag,
            unique_tags_list,
            num_cpus=PARALLELISM,
            desc="Fetching one-hop tags"
        )
        
        # Build lookup dict: QID -> one_hop_related_tags
        one_hop_lookup = {tag["qid"]: tag["one_hop_related_tags"] for tag in tags_with_one_hop if tag.get("qid")}
        
        # Step 4: Merge one-hop tags back into all items
        print("Step 4: Merging one-hop tags back into items...")
        all_resolved = []
        for item_idx, item in enumerate(all_items_with_tags):
            related_tags = item.get("related_tags", [])
            enriched_tags = []
            for tag_idx, tag in enumerate(related_tags):
                enriched_tag = dict(tag)
                tag_qid = tag.get("qid")
                
                # Add one-hop tags if available
                if tag_qid and tag_qid in one_hop_lookup:
                    enriched_tag["one_hop_related_tags"] = one_hop_lookup[tag_qid]
                elif "one_hop_related_tags" not in enriched_tag:
                    enriched_tag["one_hop_related_tags"] = []
                
                enriched_tags.append(enriched_tag)
            
            all_resolved.append({**item, "related_tags": enriched_tags})
    else:
        all_resolved = all_items_with_tags
    
    # Assert that every item has proper structure before saving
    print("Verifying structure of all items...")
    for item_idx, item in enumerate(all_resolved):
        related_tags = item.get("related_tags", [])
        assert isinstance(related_tags, list), f"Item {item_idx}: related_tags must be a list, got {type(related_tags)}"
        
        for tag_idx, tag in enumerate(related_tags):
            assert isinstance(tag, dict), f"Item {item_idx}, tag {tag_idx}: Expected dict, got {type(tag)}"
            assert "qid" in tag, f"Item {item_idx}, tag {tag_idx}: Missing 'qid' field in tag: {tag}"
            assert "label" in tag, f"Item {item_idx}, tag {tag_idx}: Missing 'label' field in tag: {tag}"
            # qid can be None or a string, but must exist
            assert tag["qid"] is None or isinstance(tag["qid"], str), \
                f"Item {item_idx}, tag {tag_idx}: 'qid' must be None or str, got {type(tag['qid'])}"
            # Check one_hop_related_tags (should be present for all tags)
            assert "one_hop_related_tags" in tag, \
                f"Item {item_idx}, tag {tag_idx}: Missing 'one_hop_related_tags' field"
            assert isinstance(tag["one_hop_related_tags"], list), \
                f"Item {item_idx}, tag {tag_idx}: 'one_hop_related_tags' must be a list"
            for oh_tag_idx, oh_tag in enumerate(tag["one_hop_related_tags"]):
                assert isinstance(oh_tag, dict), \
                    f"Item {item_idx}, tag {tag_idx}, one_hop_tag {oh_tag_idx}: Expected dict"
                assert "qid" in oh_tag and "label" in oh_tag, \
                    f"Item {item_idx}, tag {tag_idx}, one_hop_tag {oh_tag_idx}: Missing qid or label"
    
    print(f"All {len(all_resolved)} items verified successfully!")
    
    # Save results
    print(f"Saving {len(all_resolved)} items to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        for item in tqdm(all_resolved, desc="Writing output"):
            # Double-check before writing
            for tag in item.get("related_tags", []):
                assert "qid" in tag and "label" in tag, f"Invalid tag structure before writing: {tag}"
            f.write(json.dumps(item) + "\n")
    
    # Calculate statistics
    total_tags = 0
    tags_with_qid = 0
    tags_without_qid = 0
    total_one_hop_tags = 0
    for item in all_resolved:
        for tag in item.get("related_tags", []):
            total_tags += 1
            if tag.get("qid"):
                tags_with_qid += 1
            else:
                tags_without_qid += 1
            # Count one-hop tags
            one_hop_tags = tag.get("one_hop_related_tags", [])
            total_one_hop_tags += len(one_hop_tags)
    
    stats = {
        "total_items": len(all_resolved),
        "total_tags": total_tags,
        "tags_with_qid": tags_with_qid,
        "tags_without_qid": tags_without_qid,
        "total_one_hop_tags": total_one_hop_tags,
    }
    
    # Fetch Wikipedia first paragraphs for tags if output path is provided
    if tags_wikipedia_output_path:
        print(f"\nFetching Wikipedia first paragraphs for tags...")
        _process_tags_wikipedia(all_resolved, tags_wikipedia_output_path)
    
    return stats


def _process_tags_wikipedia(all_items: list, output_path: str):
    """
    Collect unique tags with QIDs from all items, fetch Wikipedia first paragraphs,
    and save them to a separate JSONL file.
    """
    # Collect unique tags (by QID) - we want to process each QID only once
    unique_tags_by_qid = {}
    for item in all_items:
        for tag in item.get("related_tags", []):
            qid = tag.get("qid")
            label = tag.get("label")
            if qid and isinstance(qid, str) and qid.startswith("Q"):
                # Use QID as key to avoid duplicates
                if qid not in unique_tags_by_qid:
                    unique_tags_by_qid[qid] = {"qid": qid, "label": label}
    
    unique_tags_list = list(unique_tags_by_qid.values())
    print(f"Found {len(unique_tags_list)} unique tags with QIDs to process")
    
    if not unique_tags_list:
        print("No tags with QIDs to process for Wikipedia paragraphs")
        return
    
    # Load existing cache to avoid reprocessing
    existing_cache = {}
    if os.path.exists(output_path):
        print(f"Loading existing cache from {output_path}...")
        with open(output_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    tag_data = json.loads(line)
                    qid = tag_data.get("qid")
                    if qid:
                        existing_cache[qid] = tag_data
                except Exception:
                    continue
        print(f"Found {len(existing_cache)} tags already in cache")
    
    # Filter out tags that are already processed
    tags_to_process = [
        tag for tag in unique_tags_list 
        if tag["qid"] not in existing_cache
    ]
    
    print(f"Processing {len(tags_to_process)} new tags...")
    
    # Process tags in parallel
    if tags_to_process:
        tags_with_wikipedia = p_map(
            _process_tag_wikipedia,
            tags_to_process,
            num_cpus=PARALLELISM,
            desc="Fetching Wikipedia paragraphs for tags"
        )
        
        # Merge successful results into cache
        for tag_data in tags_with_wikipedia:
            if tag_data is not None:
                existing_cache[tag_data["qid"]] = tag_data
    
    # Write all tags (both existing and newly processed) to output file
    print(f"Saving {len(existing_cache)} tags to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        for tag_data in tqdm(existing_cache.values(), desc="Writing tag Wikipedia paragraphs"):
            # Ensure structure is correct
            assert "qid" in tag_data, f"Missing 'qid' in tag_data: {tag_data}"
            assert "label" in tag_data, f"Missing 'label' in tag_data: {tag_data}"
            assert "wikipedia_first_paragraph" in tag_data, f"Missing 'wikipedia_first_paragraph' in tag_data: {tag_data}"
            f.write(json.dumps(tag_data, ensure_ascii=False) + "\n")
    
    print(f"Saved {len(existing_cache)} tag Wikipedia paragraphs to {output_path}")


def main():
    """
    Process items from input file:
    1. For each item with a QID, fetch related_tags from Wikidata
    2. Extract entity values as related_tags (with QIDs)
    3. Get one-hop-related-tags for each tag
    4. Optionally fetch Wikipedia first paragraphs for tags and save to separate file.
    """
    input_file = "./data/interim/5_items_with_wikipedia_and_desc.jsonl"
    
    # Create output filenames
    basename = os.path.basename(input_file)
    output_file = os.path.join("./data/interim/6_items_with_tags_qids.jsonl")
    tags_wikipedia_output_file = os.path.join("./data/interim/6_tags_wikipedia_first_paragraphs.jsonl")
    
    stats = process_file(input_file, output_file, tags_wikipedia_output_path=tags_wikipedia_output_file)
    if stats:
        print(f"\n  Statistics for {basename}:")
        print(f"    Total items: {stats['total_items']}")
        print(f"    Total tags: {stats['total_tags']}")
        print(f"    Tags with QID: {stats['tags_with_qid']}")
        print(f"    Tags without QID: {stats['tags_without_qid']}")
        print(f"    Total one-hop tags: {stats['total_one_hop_tags']}")
        if stats['total_tags'] > 0:
            print(f"    Success rate: {stats['tags_with_qid']/stats['total_tags']*100:.1f}%")
            print(f"    Average one-hop tags per tag: {stats['total_one_hop_tags']/stats['total_tags']:.1f}")
        print(f"\n  Tag Wikipedia paragraphs saved to: {tags_wikipedia_output_file}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

