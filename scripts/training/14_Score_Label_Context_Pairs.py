#!/usr/bin/env python
"""Score selected (label, context) pairs with a stage-12 trained model."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any

import joblib
import numpy as np

# Bootstrap project paths before local imports.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
SRC_ROOT = os.path.join(WORKSPACE_ROOT, "src")
for path in (WORKSPACE_ROOT, SRC_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from with_argus_eyes.training.helpers import stub_name
from with_argus_eyes.utils.embeddings import build_retriever


def _resolve_workspace_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(WORKSPACE_ROOT, path)


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


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
                    return f"first_record_keys={sorted(obj.keys())}"
                return f"first_record_type={type(obj).__name__}"
        return "empty_file"
    except Exception as exc:
        return f"preview_error={type(exc).__name__}"


def _canonicalize(label: str, context: str) -> str:
    label = (label or "").strip()
    context = (context or "").strip()
    if label and label.lower() not in context.lower():
        return f"{label} : {context}"
    return context


def _build_default_results_tag(retriever: str, order: str, k: int) -> str:
    base = stub_name(retriever, f"ratio_unrelevant_below_k_{k}")
    return f"{base}_o_{order}_k_{k}"


def _pick_best_artifact(model_dir: str) -> str | None:
    if not os.path.isdir(model_dir):
        return None

    all_joblib = sorted(glob.glob(os.path.join(model_dir, "*.joblib")))
    if not all_joblib:
        return None

    mlp_best = [p for p in all_joblib if os.path.basename(p).startswith("mlp_best_")]
    baseline_best = [p for p in all_joblib if os.path.basename(p).startswith("baseline_best_")]
    ranked = mlp_best or baseline_best or all_joblib
    # Prefer newest if multiple exist.
    ranked.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return ranked[0]


def _resolve_model_artifact(
    *,
    model_artifact: str,
    models_dir: str,
    analysis_out_dir: str,
    retriever: str,
    order: str,
    k: int,
    results_tag: str,
) -> tuple[str, str]:
    if model_artifact:
        resolved = _resolve_workspace_path(model_artifact)
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"Model artifact not found: {resolved}")
        return resolved, os.path.dirname(resolved)

    candidate_model_dirs: list[str] = []
    if models_dir:
        candidate_model_dirs.append(_resolve_workspace_path(models_dir))
    else:
        out_dir = _resolve_workspace_path(analysis_out_dir)
        final_tag = results_tag or _build_default_results_tag(retriever, order, k)
        candidate_model_dirs.append(os.path.join(out_dir, final_tag, "models"))

        # Fallback: any directory for the same retriever/order/k pattern.
        pattern = os.path.join(
            out_dir,
            f"{retriever}_ratio_unrelevant_below_k_*_o_{order}_k_{k}*",
            "models",
        )
        for match in sorted(glob.glob(pattern)):
            if match not in candidate_model_dirs:
                candidate_model_dirs.append(match)

    for model_dir in candidate_model_dirs:
        artefact = _pick_best_artifact(model_dir)
        if artefact:
            return artefact, model_dir

    tried = ", ".join(candidate_model_dirs) if candidate_model_dirs else "(none)"
    raise FileNotFoundError(
        "No .joblib model artifact found. "
        f"Tried model directories: {tried}"
    )


def _load_examples(input_jsonl: str, label_key: str, context_key: str) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    skipped = 0
    with open(input_jsonl, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if not isinstance(record, dict):
                skipped += 1
                continue

            label_val = record.get(label_key, "")
            context_val = record.get(context_key, "")
            if not isinstance(label_val, str):
                label_val = str(label_val)
            if not isinstance(context_val, str):
                context_val = str(context_val)
            if not context_val.strip():
                skipped += 1
                continue

            row = dict(record)
            row["_score_label"] = label_val
            row["_score_context"] = context_val
            examples.append(row)

    print(
        f"[input] Loaded {len(examples)} valid rows from {input_jsonl}; "
        f"skipped={skipped}"
    )
    if not examples:
        raise RuntimeError(f"No valid rows found in {input_jsonl}")
    return examples


def _encode_examples(
    retriever_name: str,
    examples: list[dict[str, Any]],
    *,
    batch_size: int,
    max_length: int,
    text_mode: str,
) -> np.ndarray:
    retriever = build_retriever(retriever_name)

    labels = [ex["_score_label"] for ex in examples]
    contexts = [ex["_score_context"] for ex in examples]

    vectors: list[np.ndarray] = []
    for start in range(0, len(examples), batch_size):
        batch_labels = labels[start : start + batch_size]
        batch_contexts = contexts[start : start + batch_size]

        if text_mode == "context":
            batch_vec = retriever.encode_texts(
                batch_contexts, batch_size=batch_size, max_length=max_length
            )
        elif text_mode == "canonical":
            canonical = [_canonicalize(lbl, ctx) for lbl, ctx in zip(batch_labels, batch_contexts)]
            batch_vec = retriever.encode_texts(
                canonical, batch_size=batch_size, max_length=max_length
            )
        else:
            batch_vec = retriever.encode_spans(
                batch_contexts,
                batch_labels,
                batch_size=batch_size,
                max_length=max_length,
            )
        vectors.append(np.asarray(batch_vec, dtype=np.float32))

    if not vectors:
        raise RuntimeError("No vectors were generated from the input examples.")
    return np.vstack(vectors)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score selected label/context pairs using the best model artifact "
            "from stage 12 (or an explicitly provided artifact)."
        )
    )
    parser.add_argument(
        "--retriever",
        type=str,
        default="contriever",
        help="Retriever name used for embeddings and model selection.",
    )
    parser.add_argument(
        "--order",
        type=str,
        default="800",
        help="Order value used in stage-12 training tag resolution.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=50,
        help="k value used in stage-12 training tag resolution.",
    )
    parser.add_argument(
        "--analysis_out_dir",
        type=str,
        default=os.path.join("outputs", "12_risk_outputs"),
        help="Stage-12 output directory containing trained-model folders.",
    )
    parser.add_argument(
        "--results_tag",
        type=str,
        default="",
        help=(
            "Optional explicit stage-12 results tag. "
            "If empty, auto-builds from retriever/order/k."
        ),
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="",
        help="Optional explicit model directory. Overrides analysis_out_dir/tag lookup.",
    )
    parser.add_argument(
        "--model_artifact",
        type=str,
        default="",
        help="Optional direct path to a .joblib artifact. Highest priority.",
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        required=True,
        help='Input JSONL with rows containing at least {"label","context"} (configurable keys).',
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="",
        help="Output JSONL path. Defaults to outputs/14_Score_Label_Context_Pairs/...",
    )
    parser.add_argument(
        "--label_key",
        type=str,
        default="label",
        help="JSON key to read labels from input rows.",
    )
    parser.add_argument(
        "--context_key",
        type=str,
        default="context",
        help="JSON key to read contexts from input rows.",
    )
    parser.add_argument(
        "--text_mode",
        type=str,
        choices=("context", "canonical", "span"),
        default="context",
        help=(
            "Embedding input mode: "
            "context=encode context only, "
            "canonical=encode 'label : context', "
            "span=encode_spans(context, label)."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Embedding batch size.")
    parser.add_argument("--max_length", type=int, default=1048, help="Embedding max token length.")
    parser.add_argument(
        "--hf_base",
        type=str,
        default=os.environ.get("ARGUS_HF_BASE", "../../../../data/proj/zeinabtaghavi"),
        help="Base directory for Hugging Face caches.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["HF_BASE"] = args.hf_base
    os.environ["HF_HOME"] = args.hf_base
    os.environ["HF_HUB_CACHE"] = f"{args.hf_base}/hub"
    os.environ["HF_DATASETS_CACHE"] = f"{args.hf_base}/datasets"

    input_jsonl = _resolve_workspace_path(args.input_jsonl)
    if not os.path.exists(input_jsonl):
        raise FileNotFoundError(f"Input JSONL not found: {input_jsonl}")

    output_jsonl = (
        _resolve_workspace_path(args.output_jsonl)
        if args.output_jsonl
        else os.path.join(
            WORKSPACE_ROOT,
            "outputs",
            "14_Score_Label_Context_Pairs",
            f"14_scored_{args.retriever}_o_{args.order}_k_{args.k}.jsonl",
        )
    )
    _ensure_parent_dir(output_jsonl)

    model_artifact, model_dir = _resolve_model_artifact(
        model_artifact=args.model_artifact,
        models_dir=args.models_dir,
        analysis_out_dir=args.analysis_out_dir,
        retriever=args.retriever,
        order=args.order,
        k=args.k,
        results_tag=args.results_tag,
    )

    print(f"[paths] input_jsonl: {input_jsonl}")
    print(f"[paths] output_jsonl: {output_jsonl}")
    print(f"[paths] model_dir: {model_dir}")
    print(f"[paths] model_artifact: {model_artifact}")
    print(f"[args] text_mode={args.text_mode} retriever={args.retriever} order={args.order} k={args.k}")

    examples = _load_examples(input_jsonl, args.label_key, args.context_key)
    X = _encode_examples(
        args.retriever,
        examples,
        batch_size=args.batch_size,
        max_length=args.max_length,
        text_mode=args.text_mode,
    )

    artefact = joblib.load(model_artifact)
    if isinstance(artefact, dict) and "model" in artefact:
        model = artefact["model"]
        scaler = artefact.get("scaler")
    else:
        model = artefact
        scaler = None

    X_input = scaler.transform(X) if scaler is not None else X
    y_pred = np.asarray(model.predict(X_input), dtype=float).reshape(-1)
    if y_pred.shape[0] != len(examples):
        raise RuntimeError(
            f"Prediction length mismatch: predicted={y_pred.shape[0]} expected={len(examples)}"
        )

    with open(output_jsonl, "w", encoding="utf-8") as out_f:
        for ex, score in zip(examples, y_pred):
            row = dict(ex)
            row.pop("_score_label", None)
            row.pop("_score_context", None)
            row["risk_score"] = float(score)
            row["score_model_artifact"] = model_artifact
            row["score_retriever"] = args.retriever
            row["score_text_mode"] = args.text_mode
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[done] Scored {len(examples)} pairs.")
    print("[14_SCORE_LABEL_CONTEXT_PAIRS] Output summary")
    if os.path.exists(output_jsonl):
        size = os.path.getsize(output_jsonl)
        print(f"  - output: {os.path.abspath(output_jsonl)}")
        print(f"    size: {_human_size(size)}")
        print(f"    {_jsonl_preview(output_jsonl)}")
    else:
        print(f"  - output missing: {os.path.abspath(output_jsonl)}")


if __name__ == "__main__":
    main()
