"""Variant analysis of high/low retrieval probability ratios."""

#!/usr/bin/env python
from __future__ import annotations

import os
import sys
import argparse
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt

# Local imports (same style as 9_Analysis_Rank.py)
from with_argus_eyes.training.helpers import load_rank_file
from with_argus_eyes.utils.risk.scores import get_risk_fn, normalized_score


def compute_average_score_for_combo(
    workspace_root: str,
    retriever: str,
    order: str,
    k: int,
) -> float | None:
    """
    Compute the mean RP-score for all items belonging to a (retriever, order) pair.
    Returns None if data is missing or no finite scores are available.
    """
    temp_dir = os.path.join(workspace_root, "data", "processed", "8_Emb_Rank")
    file_path = os.path.join(
        temp_dir,
        f"8_main_dataset_{order}_{retriever}.jsonl",
    )

    if not os.path.exists(file_path):
        print(
            f"[warn] (avg) file not found for retriever={retriever}, order={order}: {file_path}"
        )
        return None

    items = load_rank_file(
        retriever=retriever,
        using_file_path=True,
        file_path=file_path,
    )
    if not items:
        print(f"[warn] (avg) no items loaded for retriever={retriever}, order={order}")
        return None

    risk_name = "ratio_unrelevant_below_k"
    risk_fn = get_risk_fn(risk_name)

    scores: list[float] = []
    for it in items:
        score = risk_fn(it, order=order, k=k)
        scores.append(float(score))

    y = normalized_score(scores, method="none")
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]

    if y.size == 0:
        print(
            f"[warn] (avg) no finite scores for retriever={retriever}, order={order}"
        )
        return None

    avg_score = float(np.mean(y))
    print(
        f"[avg] retriever={retriever:12s} order={order:>4s} → mean_score={avg_score:.4f}"
    )
    return avg_score


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute and plot average retrieval probability (RP) scores per retriever "
            "as a bar chart."
        )
    )
    parser.add_argument(
        "--retrievers",
        type=str,
        default="contriever,reasonir,qwen3,jina,bge-m3,rader,reason-embed,nv-embed,gritlm",
        help=(
            "Comma-separated list of retrievers. "
            "Must match names used in 8_Emb_Rank outputs, e.g. 'contriever,reasonir,gritlm'."
        ),
    )
    parser.add_argument(
        "--k",
        type=int,
        default=50,
        help="k parameter for ratio_unrelevant_below_k (must match what you used in 8_Emb_Rank).",
    )
    parser.add_argument(
        "--order",
        type=str,
        default="800",
        help="Order (N) to use when computing the per-retriever average RP-score.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs/9_risk_outputs",
        help="Directory where the plot will be saved.",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default="avg_rp_score_by_retriever.png",
        help="Filename for the average-score bar chart inside out_dir.",
    )
    args = parser.parse_args()

    # Resolve workspace root (same logic as 9_Analysis_Rank.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    if workspace_root not in sys.path:
        sys.path.insert(0, workspace_root)

    retrievers = [r.strip() for r in args.retrievers.split(",") if r.strip()]

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, args.save_name)

    # Dictionary mapping retrievers to display labels
    # You can modify this dictionary to customize the labels shown in the figure
    retriever_labels: Dict[str, str] = {
        "contriever": "Contriever",
        "reasonir": "ReasonIR",
        "qwen3": "Qwen3",
        "jina": "Jina",
        "bge-m3": "BGE-M3",
        "rader": "Rader",
        "reason-embed": "Reason-Embed",
        "nv-embed": "NV-Embed",
        "gritlm": "GritLM",
    }
    # Default to capitalized version if not in dictionary
    for retriever in retrievers:
        if retriever not in retriever_labels:
            retriever_labels[retriever] = retriever.capitalize()

    # Dictionary mapping retrievers to colors
    retriever_colors: Dict[str, str] = {
        "contriever": "#1f77b4",      # blue
        "reasonir": "#ff7f0e",         # orange
        "qwen3": "#2ca02c",           # green
        "jina": "#d62728",            # red
        "bge-m3": "#9467bd",          # purple
        "reason-embed": "#8c564b",     # brown
        "nv-embed": "#e377c2",        # pink
        "gritlm": "#7f7f7f",          # grey
    }

    # Compute average RP-score per retriever
    avg_scores: Dict[str, float] = {}
    print(
        f"=== Computing average RP-score per retriever (order={args.order}, k={args.k}) ==="
    )
    for retriever in retrievers:
        avg_score = compute_average_score_for_combo(
            workspace_root=workspace_root,
            retriever=retriever,
            order=args.order,
            k=args.k,
        )
        if avg_score is not None:
            avg_scores[retriever] = avg_score

    # Plot bar chart of average RP-score per retriever
    if avg_scores:
        fig_bar, ax_bar = plt.subplots(figsize=(9, 4))
        sorted_avg = sorted(avg_scores.items(), key=lambda kv: kv[1])
        bar_labels = [retriever_labels.get(name, name) for name, _ in sorted_avg]
        bar_values = [score for _, score in sorted_avg]
        bar_colors = [retriever_colors.get(name, "#4C72B0") for name, _ in sorted_avg]

        bars = ax_bar.bar(bar_labels, bar_values, color=bar_colors, alpha=0.85)
        ax_bar.set_ylabel("Average RP-score", fontweight="bold")
        ax_bar.set_xlabel("Retrievers", fontweight="bold")
        ax_bar.set_title(
            f"Average Retrieval Probability (RP) Score per Retriever (Having Top-k={args.k})",
            fontweight="bold",
        )
        # Clamp Y-axis to 0..0.6 (as requested)
        y_max = 0.6
        ax_bar.set_ylim(0, y_max)
        ax_bar.set_yticks(np.arange(0.0, y_max + 1e-9, 0.1))
        ax_bar.grid(axis="y", alpha=0.2, linestyle="--", linewidth=0.7)
        for bar, value in zip(bars, bar_values):
            # Keep labels visible within the clamped Y-range
            text_y = min(value + 0.01, y_max - 0.02)
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                text_y,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
        plt.setp(ax_bar.get_xticklabels(), rotation=25, ha="right")
        fig_bar.tight_layout()
        fig_bar.savefig(out_path, dpi=200)
        plt.close(fig_bar)
        print(f"[plot] saved average-score bar chart to: {out_path}")
    else:
        print(
            "[plot] skipped average-score bar chart (no valid scores for the requested order)."
        )


if __name__ == "__main__":
    main()
