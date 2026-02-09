"""Finalize key dataset files into data/processed and optionally clear interim."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Move the final dataset artifacts from data/interim to data/processed "
            "with canonical (non-numbered) names."
        )
    )
    parser.add_argument("--interim_dir", type=str, default="data/interim")
    parser.add_argument("--processed_dir", type=str, default="data/processed")

    parser.add_argument("--main_dataset_src", type=str, default="6_main_dataset.jsonl")
    parser.add_argument(
        "--wikipedia_cache_src",
        type=str,
        default="4_tags_wikipedia_first_paragraphs_cache.jsonl",
    )
    parser.add_argument(
        "--wiki_unrelevants_src",
        type=str,
        default="9_wiki_unrelevants_results.jsonl",
    )

    parser.add_argument("--main_dataset_name", type=str, default="main_dataset.jsonl")
    parser.add_argument(
        "--wikipedia_cache_name",
        type=str,
        default="wikipedia_all_relevant_results.jsonl",
    )
    parser.add_argument(
        "--wiki_unrelevants_name",
        type=str,
        default="wiki_unrelevants_results.jsonl",
    )

    parser.add_argument(
        "--clear_interim",
        action="store_true",
        help="If set, remove remaining files/folders in interim_dir after moving outputs.",
    )
    parser.add_argument(
        "--keep_interim",
        type=str,
        default="",
        help="Comma-separated names to keep in interim_dir when --clear_interim is used.",
    )
    return parser.parse_args()


def _resolve_src(path_or_name: str, interim_dir: Path) -> Path:
    candidate = Path(path_or_name)
    if candidate.is_absolute():
        return candidate
    return interim_dir / path_or_name


def _move_with_overwrite(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Required input file not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if dst.is_file() or dst.is_symlink():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    shutil.move(str(src), str(dst))


def _clear_interim_dir(interim_dir: Path, keep_names: set[str]) -> tuple[int, int]:
    removed_files = 0
    removed_dirs = 0
    if not interim_dir.exists():
        return removed_files, removed_dirs

    for entry in interim_dir.iterdir():
        if entry.name in keep_names:
            continue
        if entry.is_file() or entry.is_symlink():
            entry.unlink()
            removed_files += 1
            continue
        shutil.rmtree(entry)
        removed_dirs += 1
    return removed_files, removed_dirs


def main() -> None:
    args = _parse_args()

    interim_dir = Path(args.interim_dir)
    processed_dir = Path(args.processed_dir)

    src_main = _resolve_src(args.main_dataset_src, interim_dir)
    src_wiki = _resolve_src(args.wikipedia_cache_src, interim_dir)
    src_unrel = _resolve_src(args.wiki_unrelevants_src, interim_dir)

    dst_main = processed_dir / args.main_dataset_name
    dst_wiki = processed_dir / args.wikipedia_cache_name
    dst_unrel = processed_dir / args.wiki_unrelevants_name

    print("[10_CLEAR_DIRECTORIES] Moving finalized files to processed directory...")
    _move_with_overwrite(src_main, dst_main)
    _move_with_overwrite(src_wiki, dst_wiki)
    _move_with_overwrite(src_unrel, dst_unrel)

    print(f" - main_dataset: {dst_main.resolve()}")
    print(f" - wikipedia_cache: {dst_wiki.resolve()}")
    print(f" - wiki_unrelevants: {dst_unrel.resolve()}")

    if args.clear_interim:
        keep_names = {name.strip() for name in args.keep_interim.split(",") if name.strip()}
        removed_files, removed_dirs = _clear_interim_dir(interim_dir, keep_names)
        print(
            "[10_CLEAR_DIRECTORIES] Cleared interim directory: "
            f"removed_files={removed_files}, removed_dirs={removed_dirs}, "
            f"kept={sorted(keep_names)}"
        )
    else:
        print("[10_CLEAR_DIRECTORIES] Interim cleanup disabled (use --clear_interim to enable).")


if __name__ == "__main__":
    main()
