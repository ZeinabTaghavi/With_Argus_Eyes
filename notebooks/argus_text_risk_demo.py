#!/usr/bin/env python
"""Command-line companion for ``argus_text_risk_demo.ipynb``.

This script follows the same raw-text -> NER -> embedding -> ARGUS scoring
procedure as the notebook, but prints each step for easier debugging.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


DEFAULT_TEXT = """
St. Martin's Church in Zillis, Switzerland, is a Romanesque church best known
for its painted wooden ceiling panels dating from the 12th century. Neanderthals
inhabited Europe and Western and Central Asia during the Middle to Late Pleistocene.
""".strip()


def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for path in (start, *start.parents):
        if (path / "src" / "with_argus_eyes").exists():
            return path
    raise RuntimeError("Could not find the With_Argus_Eyes repository root.")


def bootstrap_imports(repo_root: Path) -> None:
    src_path = repo_root / "src"
    for path in (repo_root, src_path):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the ARGUS raw-text NER and risk-scoring demo with debug prints."
    )
    parser.add_argument("--text", type=str, default="", help="Raw text to analyze.")
    parser.add_argument("--text-file", type=str, default="", help="Path to a UTF-8 text file to analyze.")
    parser.add_argument("--retriever", type=str, default="contriever", help="Retriever backend.")
    parser.add_argument("--language", type=str, default="en", help="NER/input language label.")
    parser.add_argument("--ner-model", type=str, default="dslim/bert-base-NER", help="Hugging Face NER model.")
    parser.add_argument("--risk-threshold", type=float, default=0.3, help="ARGUS risk threshold.")
    parser.add_argument("--ner-threshold", type=float, default=0.5, help="NER confidence threshold.")
    parser.add_argument("--order", type=str, default="800", help="ARGUS trained-model order value.")
    parser.add_argument("--k", type=int, default=50, help="ARGUS trained-model k value.")
    parser.add_argument(
        "--text-mode",
        choices=("context", "canonical", "span"),
        default="span",
        help="How to convert entity/context pairs to embedding inputs.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size.")
    parser.add_argument("--max-length", type=int, default=1048, help="Embedding max token length.")
    parser.add_argument(
        "--cuda-visible-devices",
        type=str,
        default="",
        help='GPU IDs, for example "0" or "0,1". Empty string uses CPU/default behavior.',
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=str,
        default="",
        help="Hugging Face cache directory. Defaults to outputs/cache/huggingface.",
    )
    parser.add_argument("--model-artifact", type=str, default="", help="Optional explicit .joblib model path.")
    parser.add_argument("--models-dir", type=str, default="", help="Optional explicit model directory.")
    parser.add_argument("--results-tag", type=str, default="", help="Optional explicit stage-12 result tag.")
    parser.add_argument("--json-output", type=str, default="", help="Optional path to write JSON results.")
    parser.add_argument("--workspace-root", type=str, default="", help="Optional repository root override.")
    return parser.parse_args()


def configure_environment(args: argparse.Namespace, repo_root: Path) -> None:
    print("[step 1] Configuring environment")

    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    else:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    hf_cache_dir = args.hf_cache_dir or str(repo_root / "outputs" / "cache" / "huggingface")
    if hf_cache_dir:
        hf_cache = Path(hf_cache_dir).expanduser()
        if not hf_cache.is_absolute():
            hf_cache = repo_root / hf_cache
        os.environ["HF_HOME"] = str(hf_cache)
        os.environ["HF_HUB_CACHE"] = str(hf_cache / "hub")
        os.environ["HF_DATASETS_CACHE"] = str(hf_cache / "datasets")
        os.environ["TRANSFORMERS_CACHE"] = str(hf_cache / "transformers")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print(f"  repo_root: {repo_root}")
    print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<default/CPU>')}")
    print(f"  HF_HOME: {os.environ.get('HF_HOME', '<system default>')}")


def load_text(args: argparse.Namespace, repo_root: Path) -> str:
    print("[step 2] Loading input text")

    if args.text_file:
        path = Path(args.text_file).expanduser()
        if not path.is_absolute():
            path = repo_root / path
        text = path.read_text(encoding="utf-8").strip()
        print(f"  source: {path}")
    elif args.text:
        text = args.text.strip()
        print("  source: --text")
    else:
        text = DEFAULT_TEXT
        print("  source: built-in demo text")

    print(f"  characters: {len(text)}")
    print("  preview:")
    print(indent_block(text[:500]))
    return text


def indent_block(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def make_config(args: argparse.Namespace, repo_root: Path) -> Any:
    from with_argus_eyes.inference import ArgusTextConfig

    print("[step 3] Building ARGUS configuration")
    config = ArgusTextConfig(
        retriever=args.retriever,
        language=args.language,
        ner_model=args.ner_model,
        risk_threshold=args.risk_threshold,
        ner_threshold=args.ner_threshold,
        order=args.order,
        k=args.k,
        text_mode=args.text_mode,
        batch_size=args.batch_size,
        max_length=args.max_length,
        workspace_root=repo_root,
        results_tag=args.results_tag,
        models_dir=args.models_dir or None,
        model_artifact=args.model_artifact or None,
    )
    for key, value in config.__dict__.items():
        print(f"  {key}: {value}")
    return config


def print_entities(entities: list[Any]) -> None:
    print(f"  extracted entities: {len(entities)}")
    if not entities:
        return
    for idx, entity in enumerate(entities, start=1):
        print(
            f"  {idx:02d}. {entity.text!r} "
            f"type={entity.entity_type} span=({entity.start},{entity.end}) ner_score={entity.score:.4f}"
        )


def print_results(results: list[Any]) -> None:
    print(f"  scored entities: {len(results)}")
    if not results:
        return

    rows = [result.as_dict() if hasattr(result, "as_dict") else dict(result) for result in results]
    columns = ("entity", "entity_type", "ner_score", "risk_score", "above_threshold", "retriever")
    widths = {
        column: max(len(column), *(len(format_value(row[column])) for row in rows))
        for column in columns
    }
    header = " | ".join(column.ljust(widths[column]) for column in columns)
    print("  " + header)
    print("  " + "-+-".join("-" * widths[column] for column in columns))
    for row in sorted(rows, key=lambda item: item["risk_score"], reverse=True):
        print("  " + " | ".join(format_value(row[column]).ljust(widths[column]) for column in columns))


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_json_output(path: str, repo_root: Path, results: list[Any]) -> None:
    if not path:
        return

    output_path = Path(path).expanduser()
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [result.as_dict() if hasattr(result, "as_dict") else dict(result) for result in results]
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[step 8] Wrote JSON output: {output_path}")


def main() -> None:
    args = parse_args()
    repo_root = Path(args.workspace_root).expanduser().resolve() if args.workspace_root else find_repo_root()
    bootstrap_imports(repo_root)

    from with_argus_eyes.inference import available_retrievers, extract_entities, resolve_model_artifact, score_entities

    configure_environment(args, repo_root)
    print("[step 1b] Available retrievers")
    print("  " + ", ".join(available_retrievers()))

    text = load_text(args, repo_root)
    config = make_config(args, repo_root)

    print("[step 4] Resolving ARGUS model artifact")
    artifact = resolve_model_artifact(config)
    print(f"  selected artifact: {artifact}")

    print("[step 5] Extracting named entities")
    entities = extract_entities(text, config)
    print_entities(entities)
    if not entities:
        print("[done] No named entities were found with the current NER threshold.")
        write_json_output(args.json_output, repo_root, [])
        return

    print("[step 6] Embedding entities and predicting ARGUS risk scores")
    results = score_entities(text, entities, config, model_artifact=artifact)

    print("[step 7] Results")
    print_results(results)
    write_json_output(args.json_output, repo_root, results)
    print("[done] ARGUS text risk demo finished successfully.")


if __name__ == "__main__":
    main()
