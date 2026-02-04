import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np

MODEL_NAME_DEFAULT = "facebook/contriever"

torch = None
AutoTokenizer = None
AutoModel = None
tokenizer = None
model = None
device = None
tqdm = None

# -------------------------------
# Embedding utilities (mean pooling)
# -------------------------------

def _mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = model_output.last_hidden_state  # (B, T, H)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask

@torch.no_grad()
def encode_texts(texts, batch_size: int = 32, max_length: int = 256, normalize: bool = True) -> torch.Tensor:
    if not texts:
        raise ValueError("No texts to encode")
    
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        if device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(**enc)
        else:
            outputs = model(**enc)
        vecs = _mean_pooling(outputs, enc["attention_mask"])  # (B, H)
        if normalize:
            vecs = torch.nn.functional.normalize(vecs, p=2, dim=1)
        all_vecs.append(vecs)
    return torch.cat(all_vecs, dim=0)

@torch.inference_mode()
def get_phrase_embedding(
    text: str,
    phrase: str,
    *,
    occurrence_index: int = 0,
    use_last4: bool = True,     # average last 4 layers; set False to use last layer: 	
                                # BERT paper (Devlin et al., 2019): in the feature-based setup for CoNLL-2003 NER, they report that concatenating the last 4 layers gives the best results among layer choices. 
                                # https://arxiv.org/pdf/1902.04267.pdf
    case_insensitive: bool = True,
    max_length: int = 512,
) -> torch.Tensor:
    """
    Returns a single tensor [hidden_size] — the contextual embedding of `phrase` within `text`.
    Uses fast-tokenizer offset mapping and averages the last 4 hidden layers by default.
    The returned vector is L2-normalized and resides on `device`.
    Raises ValueError if the phrase occurrence isn't found or gets truncated.
    """
    # 1) Find exact occurrences of `phrase` in raw text
    flags = re.IGNORECASE if case_insensitive else 0
    assert phrase in text, f"Phrase not found in text: {phrase!r} in {text!r} ...."
    matches = list(re.finditer(re.escape(phrase), text, flags=flags))
    if not matches:
        raise ValueError(f"Phrase not found in text: {phrase!r}")
    if occurrence_index >= len(matches):
        raise ValueError(f"occurrence_index {occurrence_index} out of range (found {len(matches)}).")

    m = matches[occurrence_index]
    char_start, char_end = m.start(), m.end()

    # 2) Tokenize with offsets
    enc = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    offsets = enc["offset_mapping"][0].tolist()  # List[(start_char, end_char)]

    # 3) Forward pass to get token-level representations
    if device.type == "cuda":
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            out = model(input_ids=input_ids, attention_mask=attention_mask)
    else:
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    token_reps = (
        torch.stack(out.hidden_states[-4:], dim=0).mean(0) if use_last4 else out.last_hidden_state
    )[0]  # [seq_len, hidden_size]

    # 4) Collect token indices whose char offsets overlap the phrase span
    token_idxs = [
        i for i, (a, b) in enumerate(offsets)
        if not (a == b == 0) and max(a, char_start) < min(b, char_end)
    ]
    if not token_idxs:
        raise ValueError(
            "Phrase span wasn't covered by the tokenized window (likely truncated). "
            "Shorten the text or ensure the phrase is within the first 512 tokens."
        )

    # 5) Pool subword vectors for the phrase and L2-normalize
    span_vec = token_reps[token_idxs].mean(0)
    span_vec = torch.nn.functional.normalize(span_vec, p=2, dim=0)
    return span_vec  # on `device`


def get_label_embedding(label: str) -> torch.Tensor:
    return encode_texts([label])

# -------------------------------
# Compute related tag similarity scores
# -------------------------------

def compute_related_tag_scores(items, phrase_type, *, batch_encode_tags: bool = True):
    """
    For each item in items (expects keys 'label' and 'related_tags'), compute
    cosine similarity between the item's Contriever embedding and each related tag.
    Adds 'related_tag_scores': list[(related_tag, score)] to each item.
    Returns a new list (does not mutate input).
    """
    out = []
    for item in tqdm(items, desc=f"Scoring ({phrase_type})", unit="item"):
        label = (item.get("label") or "").strip()
        tags = [t.strip() for t in (item.get("related_tags") or []) if isinstance(t, str) and t.strip()]

        if not label or not tags:
            raise ValueError(f"Label or tags are empty for item: {item}")
        
        if phrase_type == "wikipedia_first_paragraph":
            phrase = item.get("wikipedia_first_paragraph").strip()
        elif phrase_type == "wikidata_description":
            phrase = item.get("wikidata_description").strip()
            if item.get("label") not in phrase:
                phrase = item.get("label") + " : " + phrase
        elif phrase_type == "label":
            phrase = item.get("label").strip()
            item_vec = get_label_embedding(label)
            tag_vecs = encode_texts(tags) 
            # Cosine similarity via dot-product of normalized vectors
            sims = (item_vec @ tag_vecs.T).squeeze(0).tolist()
            scores = list(zip(tags, [float(s) for s in sims]))
            # Sort descending by score
            scores.sort(key=lambda x: x[1], reverse=True)

            out.append({**item, f"{phrase_type}_scores": scores})
            continue
        else:   
            raise ValueError(f"Invalid phrase type: {phrase_type}")

        # Encode item label *in context* of the first Wikipedia paragraph when available
        span_vec = get_phrase_embedding(phrase, label)           # (H) on device
        item_vec = span_vec.unsqueeze(0)                              # (1, H)

        # Encode tags (batch)
        tag_vecs = encode_texts(tags)  # (N, H)

        # Cosine similarity via dot-product of normalized vectors
        sims = (item_vec @ tag_vecs.T).squeeze(0).tolist()
        scores = list(zip(tags, [float(s) for s in sims]))
        # Sort descending by score
        scores.sort(key=lambda x: x[1], reverse=True)

        out.append({**item, f"{phrase_type}_scores": scores})
    return out

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute related-tag similarity scores with Contriever.")
    parser.add_argument(
        "--input_jsonl",
        type=str,
        default="./data/interim/5_items_with_wikipedia_and_desc.jsonl",
        help="Input JSONL file with items, labels, and related tags.",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="./outputs/6_items_with_tag_scores.jsonl",
        help="Output JSONL path for items with tag scores.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME_DEFAULT,
        help="HuggingFace model name (default: facebook/contriever).",
    )
    parser.add_argument(
        "--hf_base",
        type=str,
        default="../../../../data/proj/zeinabtaghavi",
        help="Base path for HF caches (HF_HOME/HF_HUB_CACHE/HF_DATASETS_CACHE).",
    )
    return parser.parse_args()


def main() -> None:
    global torch, AutoTokenizer, AutoModel, tokenizer, model, device, tqdm

    args = parse_args()

    # Set HF caches before importing transformers
    BASE = args.hf_base
    os.environ["HF_HOME"] = BASE
    os.environ["HF_HUB_CACHE"] = f"{BASE}/hub"
    os.environ["HF_DATASETS_CACHE"] = f"{BASE}/datasets"

    # Ensure required packages are available
    try:
        import torch as _torch
        from transformers import AutoTokenizer as _AutoTokenizer, AutoModel as _AutoModel
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers", "torch"])
        import torch as _torch
        from transformers import AutoTokenizer as _AutoTokenizer, AutoModel as _AutoModel

    try:
        from tqdm import tqdm as _tqdm
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tqdm"])
        from tqdm import tqdm as _tqdm

    torch = _torch
    AutoTokenizer = _AutoTokenizer
    AutoModel = _AutoModel
    tqdm = _tqdm

    # -------------------------------
    # Load Contriever model
    # -------------------------------
    model_name = args.model

    # --- Choose device: pick the CUDA device with most free memory; fallback to CPU if none is usable ---
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        best_idx, best_free = None, -1
        for i in range(torch.cuda.device_count()):
            try:
                free, total = torch.cuda.mem_get_info(i)
            except Exception:
                free, total = 0, 0
            if free > best_free:
                best_free, best_idx = free, i
        # Require at least ~1.0 GiB free for comfort; adjust if needed
        min_free_bytes = int(1.0 * 1024 ** 3)
        if best_idx is not None and best_free >= min_free_bytes:
            torch.cuda.set_device(best_idx)
            device = torch.device(f"cuda:{best_idx}")
            print(f"Using single GPU cuda:{best_idx} — free {best_free/1e9:.2f} GB / mem_get_info")
            torch.backends.cudnn.benchmark = True
        else:
            device = torch.device("cpu")
            print(f"No GPU with ≥1.0 GB free (best free {best_free/1e9:.2f} GB). Falling back to CPU.")
    else:
        device = torch.device("cpu")
        print("CUDA not available — using CPU")

    # Tokenizer/model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    base_model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

    # Move model to device with a safe fallback if GPU is exhausted
    try:
        if device.type == "cuda":
            # Use FP16 weights on GPU to reduce memory pressure
            base_model = base_model.to(device, dtype=torch.float16)
        else:
            base_model = base_model.to(device)
    except torch.cuda.OutOfMemoryError:
        print("[WARN] CUDA OOM while moving model — falling back to CPU.")
        device = torch.device("cpu")
        base_model = base_model.to(device)

    model = base_model  # force single‑GPU/CPU mode
    model.eval()

    # Run scoring on the previously computed all_items_with_tags
    with open(args.input_jsonl, "r") as f:
        all_items = [json.loads(line) for line in f]

    print("Computing related tag scores...")
    all_items_scores = compute_related_tag_scores(all_items, "wikipedia_first_paragraph")
    all_items_scores = compute_related_tag_scores(all_items_scores, "wikidata_description")
    all_items_scores = compute_related_tag_scores(all_items_scores, "label")

    # Save items with tag scores to file
    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
    with open(args.output_jsonl, "w") as f:
        for item in tqdm(all_items_scores, desc="Saving", unit="item"):
            f.write(json.dumps(item) + "\n")

    # Print some statistics about the saved data
    print(f"Saved {len(all_items_scores)} items with tag scores")
    print("First item example:")
    print(all_items_scores[0])


if __name__ == "__main__":
    main()
