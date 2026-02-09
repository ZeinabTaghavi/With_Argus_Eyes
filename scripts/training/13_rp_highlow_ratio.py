"""Analyze high/low ratio for retrieval probability."""

#!/usr/bin/env python
from __future__ import annotations

import os
import sys
import argparse
from typing import Dict, List

# Add project and src roots before local package imports.
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
src_root = os.path.join(workspace_root, "src")
for path in (workspace_root, src_root):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Local imports (same style as 9_Analysis_Rank.py)
from with_argus_eyes.training.helpers import load_rank_file
from with_argus_eyes.utils.risk.scores import get_risk_fn, normalized_score


def _resolve_rank_file_path(processed_root: str, retriever: str, order: str) -> str:
    """Prefer stage-11 output path and fallback to legacy stage-8 path."""
    candidates = [
        os.path.join(
            processed_root,
            "11_Emb_Rank",
            f"11_main_dataset_{order}_{retriever}.jsonl",
        ),
        os.path.join(
            processed_root,
            "8_Emb_Rank",
            f"8_main_dataset_{order}_{retriever}.jsonl",
        ),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def compute_high_low_ratio_for_combo(
    processed_root: str,
    retriever: str,
    order: str,
    k: int,
    threshold: float = 0.5,
) -> float | None:
    """
    For a given (retriever, order), load the JSONL produced by stage 11:
        data/processed/11_Emb_Rank/11_main_dataset_{order}_{retriever}.jsonl
    Falls back to legacy stage-8 naming when needed.

    Compute risk scores using ratio_unrelevant_below_k, then:
        ratio = (#items with score > threshold) / (#all items with a finite score)

    Returns None if the file is missing or no items are found.
    """
    file_path = _resolve_rank_file_path(processed_root, retriever, order)

    if not os.path.exists(file_path):
        print(f"[warn] file not found for retriever={retriever}, order={order}: {file_path}")
        return None

    # Load the rank file items
    items = load_rank_file(
        retriever=retriever,
        using_file_path=True,
        file_path=file_path,
    )
    if not items:
        print(f"[warn] no items loaded for retriever={retriever}, order={order}")
        return None

    # Risk function: same as in 9_Analysis_Rank.py
    risk_name = "ratio_unrelevant_below_k"
    risk_fn = get_risk_fn(risk_name)

    scores: List[float] = []
    for it in items:
        # risk_kwargs mirror 9_Analysis_Rank usage
        score = risk_fn(it, order=order, k=k)
        scores.append(float(score))

    # Normalization: method="none" keeps scores as-is (for consistency)
    y = normalized_score(scores, method="none")
    y = np.asarray(y, dtype=float)

    if y.size == 0:
        print(f"[warn] no scores computed for retriever={retriever}, order={order}")
        return None

    # Compute high / low ratio at the given threshold
    high_mask = y > threshold
    low_mask = ~high_mask  # scores <= threshold

    num_high = int(high_mask.sum())
    num_low = int(low_mask.sum())

    if num_low == 0:
        print(
            f"[warn] num_low == 0 for retriever={retriever}, order={order}; "
            "returning None for ratio."
        )
        return None

    ratio = num_high / (num_low + num_high)
    print(
        f"[ratio] retriever={retriever:12s} order={order:>4s} "
        f"→ high={num_high}, low={num_low}, ratio={ratio:.4f}"
    )
    return ratio


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute high/low retrieval probability score ratios for multiple "
            "retrievers and orders, and plot them on a single figure."
        )
    )
    parser.add_argument(
        "--retrievers",
        type=str,
        default="contriever,reasonir,qwen3,jina,bge-m3,rader,reason-embed,nv-embed,gritlm",
        help=(
            "Comma-separated list of retrievers. "
            "Must match names used in 11_Emb_Rank outputs, e.g. 'contriever,reasonir,gritlm'."
        ),
    )
    parser.add_argument(
        "--orders",
        type=str,
        default="100,200,400,600,800",
        help="Comma-separated list of orders, e.g. '100,200,400,600,800'.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=50,
        help="k parameter for ratio_unrelevant_below_k (must match what you used in 11_Emb_Rank).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold on RP score to define 'high' vs 'low'.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs/13_RP_HighLow_Ratio",
        help="Directory where the plot will be saved.",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default="rp_high_low_ratio_by_order.png",
        help="Filename for the saved plot inside out_dir.",
    )
    parser.add_argument(
        "--processed_root",
        type=str,
        default=os.environ.get("ARGUS_PROCESSED_ROOT", os.path.join(workspace_root, "data", "processed")),
        help="Processed data directory (contains 11_Emb_Rank/).",
    )
    args = parser.parse_args()

    processed_root = args.processed_root if os.path.isabs(args.processed_root) else os.path.join(workspace_root, args.processed_root)
    print(f"[paths] processed_root: {processed_root}")

    # Make all plot fonts 3pt bigger (ticks, labels, titles, etc.)
    # (This is +1pt on top of the previous +2pt setting.)
    plt.rcParams.update({"font.size": float(plt.rcParams.get("font.size", 10)) + 3.0})

    retrievers = [r.strip() for r in args.retrievers.split(",") if r.strip()]
    orders = [o.strip() for o in args.orders.split(",") if o.strip()]
    # numeric version for plotting on x-axis
    orders_numeric = [int(o) for o in orders]

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, args.save_name)
    # In case save_name contains subdirectories, ensure parent exists.
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Dictionary mapping retrievers to display labels
    # You can modify this dictionary to customize the labels shown in the figure
    retriever_labels: Dict[str, str] = {
        "contriever": "Contriever",
        "reasonir": "ReasonIR",
        "qwen3": "Qwen3",
        "jina": "Jina",
        "bge-m3": "BGE-M3",
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
        "rader": "#8c564b",           # brown (fallback for rader)
        "reason-embed": "#8c564b",     # brown
        "nv-embed": "#e377c2",        # pink
        "gritlm": "#7f7f7f",          # grey
    }

    # Compute ratios for all (retriever, order)
    ratios: Dict[str, Dict[str, float]] = {r: {} for r in retrievers}
    plot_entries: List[tuple[str, str, object]] = []
    legend_order = orders[0] if orders else None

    print("=== Computing high/low RP-score ratios ===")
    print(f"Retrievers: {retrievers}")
    print(f"Orders: {orders}")
    print(f"k = {args.k}, threshold = {args.threshold}")
    print("==========================================")

    for retriever in retrievers:
        for order in orders:
            ratio = compute_high_low_ratio_for_combo(
                processed_root=processed_root,
                retriever=retriever,
                order=order,
                k=args.k,
                threshold=args.threshold,
            )
            if ratio is not None:
                ratios[retriever][order] = ratio

    # Plot
    # Make the plot shorter (75% of previous height)
    fig, ax = plt.subplots(figsize=(8, 6))

    for retriever in retrievers:
        # X: numeric orders; Y: ratios (NaN where missing)
        y_values = []
        has_data = False
        for order in orders:
            if order in ratios[retriever]:
                val = ratios[retriever][order]
                y_values.append(val)
                # Track whether we have at least one finite ratio
                if not np.isnan(val):
                    has_data = True
            else:
                y_values.append(np.nan)

        # Skip retrievers with no valid points (e.g., only missing files / warnings)
        if not has_data:
            print(f"[info] skipping retriever={retriever} (no valid ratios for any order)")
            continue

        # Use the label from dictionary, or fallback to retriever name
        display_label = retriever_labels.get(retriever, retriever)
        retriever_color = retriever_colors.get(retriever, "#1f77b4")  # default to blue
        line = ax.plot(
            orders_numeric,
            y_values,
            marker="o",
            linewidth=1.5,
            label=display_label,
            color=retriever_color,
        )
        plot_entries.append((retriever, display_label, line[0]))

    ax.set_xlabel("N (Number of Neutral Items per Related Tag for each Selected Items)", fontweight="bold")
    ax.set_ylabel(
        f"Fraction of Entities Having RPS > {args.threshold:.2f}",
        fontweight="bold",
    )

    ax.set_xticks(orders_numeric)
    # Add headroom so legend doesn't clash with top lines,
    # but keep the visible tick labels capped at 1.0
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)

    # Horizontal reference lines at 1, 1/2, 1/4, 1/8 (light gray)
    reference_levels = [0.5, 0.25, 0.125]
    for level in reference_levels:
        ax.axhline(
            level,
            color="0.9",
            linestyle="--",
            linewidth=0.8,
            alpha=0.7,
            zorder=0,
        )

    # Add y-axis labels for reference lines (smaller, gray)
    current_yticks = ax.get_yticks()

    # Combine current ticks with reference levels and sort,
    # then drop any tick above 1.0 (so we don't show "1.1")
    all_yticks = sorted(set(list(current_yticks) + reference_levels))
    all_yticks = [t for t in all_yticks if t <= 1.0 + 1e-9]
    ax.set_yticks(all_yticks)

    def format_ytick(value: float, _pos: int) -> str:
        """Formatter for y-axis ticks with minimal trailing zeros."""
        if abs(value - 0.125) < 1e-6:
            return "0.125"
        formatted = f"{value:.2f}".rstrip("0").rstrip(".")
        return formatted

    ax.yaxis.set_major_formatter(FuncFormatter(format_ytick))

    # Ensure tick labels updated before styling
    fig.canvas.draw()
    ytick_positions = ax.get_yticks()
    ytick_labels = ax.get_yticklabels()
    for pos, label in zip(ytick_positions, ytick_labels):
        for ref_level in reference_levels:
            if abs(pos - ref_level) < 1e-6:
                label.set_color("0.5")  # Gray color
                label.set_fontsize(11)   # Smaller font size
                break

    # Build legend sorted by value at the first order (e.g., N=100)
    if plot_entries and legend_order is not None:
        def legend_sort_key(entry: tuple[str, str, object]) -> float:
            retriever_name = entry[0]
            return ratios.get(retriever_name, {}).get(legend_order, float("-inf"))

        sorted_entries = sorted(plot_entries, key=legend_sort_key, reverse=True)
        handles = [entry[2] for entry in sorted_entries]
        labels = [entry[1] for entry in sorted_entries]
        ax.legend(handles, labels, title="Retrievers", fontsize=10, title_fontsize=11)
    else:
        ax.legend(title="Retrievers", fontsize=10, title_fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"[plot] saved to: {out_path}")


if __name__ == "__main__":
    main()
