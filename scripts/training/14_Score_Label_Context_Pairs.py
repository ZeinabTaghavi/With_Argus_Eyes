import os
import sys
import json
import argparse
from typing import List, Dict, Any

import numpy as np
import joblib

# ----------------------------------------
# Bootstrap repo paths (same style as 8_Emb_Rank.py)
# ----------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
src_root = os.path.join(workspace_root, "src")
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)
if src_root not in sys.path:
    sys.path.insert(0, src_root)

# Optional HF cache setup (reuse your defaults)
BASE = "../../../../data/proj/zeinabtaghavi"
os.environ.setdefault("HF_HOME", BASE)
os.environ.setdefault("HF_HUB_CACHE", f"{BASE}/hub")
os.environ.setdefault("HF_DATASETS_CACHE", f"{BASE}/datasets")

from with_argus_eyes.utils.embeddings.factory import build_retriever  # type: ignore


def canonicalize(label: str, context: str) -> str:
    """
    Mimic the Wikipedia setup: if label is not already in the context,
    prepend it as "label : context"; otherwise just use the context.
    """
    label = (label or "").strip()
    context = (context or "").strip()
    if label and label not in context:
        return f"{label} : {context}"
    return context


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1D vectors."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute similarity scores between labels and contexts using a given embedding model.\n"
            "Input must be a JSONL file with lines of the form: "
            '{"label": "...", "context": "..."}'
        )
    )
    parser.add_argument(
        "--retriever",
        type=str,
        required=True,
        help="Name of the retriever / embedding model (passed to utils.embeddings.build_retriever).",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        required=True,
        help="Directory containing saved model artefacts (joblib) from 9_Analysis_Rank.",
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        required=True,
        help='Path to JSONL file containing {"label": ..., "context": ...} per line.',
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default=None,
        help="Where to save scores as JSONL. If not set, defaults to <input>.scored.jsonl",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for encoding.",
    )
    args = parser.parse_args()

    input_path = args.input_jsonl
    output_path = (
        args.output_jsonl
        if args.output_jsonl is not None
        else f"{os.path.splitext(input_path)[0]}.scored.jsonl"
    )
    # Ensure output directory exists (if output_path is nested).
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # -------------------------------
    # Load label/context pairs
    # -------------------------------
    examples: List[Dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            if "label" not in ex or "context" not in ex:
                continue
            examples.append(ex)

    if not examples:
        print(f"No valid examples found in {input_path}")
        return

    # labels = [ex["label"] for ex in examples]
    contexts = [ex["context"] for ex in examples]
    labels = [ex["label"] for ex in examples]

    # -------------------------------
    # Build retriever / embedding model
    # -------------------------------
    retriever = build_retriever(args.retriever)

    
    # -------------------------------
    # Encode canonicalised texts (label + context)
    # -------------------------------
    text_embs: List[np.ndarray] = []
    for i in range(0, len(examples), args.batch_size):
        batch_contexts = contexts[i : i + args.batch_size]
        batch_labels = labels[i : i + args.batch_size]
        batch_embs = np.asarray(retriever.encode_spans(batch_contexts, batch_labels), dtype=np.float32)
        text_embs.append(batch_embs)

    X = np.vstack(text_embs)

    # -------------------------------
    # Load best MLP model from models_dir
    # -------------------------------
    models_dir = args.models_dir
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"models_dir '{models_dir}' does not exist or is not a directory")

    # Prefer mlp_best_*.joblib; fall back to any mlp_*.joblib; finally any .joblib
    candidate_paths: List[str] = []
    for fname in os.listdir(models_dir):
        if not fname.endswith(".joblib"):
            continue
        candidate_paths.append(os.path.join(models_dir, fname))

    if not candidate_paths:
        raise FileNotFoundError(f"No .joblib models found in '{models_dir}'")

    # Try to find the best MLP artefact first
    mlp_best = [p for p in candidate_paths if os.path.basename(p).startswith("mlp_best_")]
    mlp_any = [p for p in candidate_paths if os.path.basename(p).startswith("mlp_")]
    baseline_best = [p for p in candidate_paths if os.path.basename(p).startswith("baseline_best_")]

    if mlp_best:
        model_path = sorted(mlp_best)[0]
        print(f"Using best MLP model: {model_path}")
    elif mlp_any:
        model_path = sorted(mlp_any)[0]
        print(f"Using any MLP model: {model_path}")
    elif baseline_best:
        model_path = sorted(baseline_best)[0]
        print(f"Using best baseline model: {model_path}")
        model_path = sorted(baseline_best)[0]
    else:
        model_path = sorted(candidate_paths)[0]

    artefact = joblib.load(model_path)
    if isinstance(artefact, dict) and "model" in artefact:
        model = artefact["model"]
        scaler = artefact.get("scaler", None)
    else:
        model = artefact
        scaler = None

    # Apply scaler if available
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X

    # Predict risk scores
    y_pred = model.predict(X_scaled)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    # -------------------------------
    # Save scores
    # -------------------------------
    with open(output_path, "w", encoding="utf-8") as out_f:
        for idx, ex in enumerate(examples):
            score = float(y_pred[idx])
            out_ex = dict(ex)
            out_ex["risk_score"] = score
            out_f.write(json.dumps(out_ex, ensure_ascii=False) + "\n")

    print(f"Processed {len(examples)} pairs.")
    print(f"Saved scored examples -> {output_path}")


if __name__ == "__main__":
    main()