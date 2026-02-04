"""Analyze high/low ratio for retrieval probability."""

#!/usr/bin/env python
from __future__ import annotations

import os
import sys
import argparse
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Local imports (same style as 9_Analysis_Rank.py)
from with_argus_eyes.training.helpers import load_rank_file
from with_argus_eyes.utils.risk.scores import get_risk_fn, normalized_score


def compute_high_low_ratio_for_combo(
    workspace_root: str,
    retriever: str,
    order: str,
    k: int,
    threshold: float = 0.5,
) -> float | None:
    """
    For a given (retriever, order), load the JSONL produced by 8_Emb_Rank:
        data/processed/8_Emb_Rank/8_main_dataset_{order}_{retriever}.jsonl

    Compute risk scores using ratio_unrelevant_below_k, then:
        ratio = (#items with score > threshold) / (#all items with a finite score)

    Returns None if the file is missing or no items are found.
    """
    temp_dir = os.path.join(workspace_root, "data", "processed", "8_Emb_Rank")
    file_path = os.path.join(
        temp_dir,
        f"8_main_dataset_{order}_{retriever}.jsonl",
    )

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


# --- New helper: collect per-item (related_tag_count, RP score) for correlation ---
def collect_related_counts_and_scores_for_combo(
    workspace_root: str,
    retriever: str,
    order: str,
    k: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    For a given (retriever, order), load the JSONL produced by 8_Emb_Rank and
    return two arrays:

        related_counts[i] = number of related tags for item i
        scores[i]         = RP risk score for item i

    This is used to study the correlation between the number of related
    tags per item and the retrieval-quality score.

    NOTE: This function assumes that each item dictionary contains one of
    the keys 'related_tags', 'related', or 'positives' that stores the
    list of related tags. Adapt the key selection if your schema differs.
    """
    temp_dir = os.path.join(workspace_root, "data", "processed", "8_Emb_Rank")
    file_path = os.path.join(
        temp_dir,
        f"8_main_dataset_{order}_{retriever}.jsonl",
    )

    if not os.path.exists(file_path):
        print(f"[warn] (corr) file not found for retriever={retriever}, order={order}: {file_path}")
        return None, None

    items = load_rank_file(
        retriever=retriever,
        using_file_path=True,
        file_path=file_path,
    )
    if not items:
        print(f"[warn] (corr) no items loaded for retriever={retriever}, order={order}")
        return None, None

    risk_name = "ratio_unrelevant_below_k"
    risk_fn = get_risk_fn(risk_name)

    related_counts: List[int] = []
    scores: List[float] = []

    for it in items:
        # Heuristic: infer number of related tags from common keys.
        if "related_tags" in it:
            n_rel = len(it["related_tags"])
        elif "related" in it:
            n_rel = len(it["related"])
        elif "positives" in it:
            n_rel = len(it["positives"])
        else:
            # If no known key is present, skip this item.
            continue

        score = risk_fn(it, order=order, k=k)
        related_counts.append(int(n_rel))
        scores.append(float(score))

    if not related_counts:
        print(f"[warn] (corr) no usable items for retriever={retriever}, order={order}")
        return None, None

    rc_arr = np.asarray(related_counts, dtype=int)
    sc_arr = np.asarray(scores, dtype=float)
    return rc_arr, sc_arr


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
            "Must match names used in 8_Emb_Rank outputs, e.g. 'contriever,reasonir,gritlm'."
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
        help="k parameter for ratio_unrelevant_below_k (must match what you used in 8_Emb_Rank).",
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
        default="outputs/10_RP_HighLow_Ratio",
        help="Directory where the plot will be saved.",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default="rp_high_low_ratio_by_order.png",
        help="Filename for the saved plot inside out_dir.",
    )
    args = parser.parse_args()

    # Resolve workspace root (same logic as 9_Analysis_Rank.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    if workspace_root not in sys.path:
        sys.path.insert(0, workspace_root)

    # Make all plot fonts 3pt bigger (ticks, labels, titles, etc.)
    # (This is +1pt on top of the previous +2pt setting.)
    plt.rcParams.update({"font.size": float(plt.rcParams.get("font.size", 10)) + 3.0})

    retrievers = [r.strip() for r in args.retrievers.split(",") if r.strip()]
    orders = [o.strip() for o in args.orders.split(",") if o.strip()]
    # numeric version for plotting on x-axis
    orders_numeric = [int(o) for o in orders]

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
                workspace_root=workspace_root,
                retriever=retriever,
                order=order,
                k=args.k,
                threshold=args.threshold,
            )
            if ratio is not None:
                ratios[retriever][order] = ratio

    # --- Collect per-item (related_count, score) for correlation analysis ---
    related_counts_by_retriever: Dict[str, List[int]] = {r: [] for r in retrievers}
    scores_by_retriever: Dict[str, List[float]] = {r: [] for r in retrievers}

    for retriever in retrievers:
        for order in orders:
            rc_arr, sc_arr = collect_related_counts_and_scores_for_combo(
                workspace_root=workspace_root,
                retriever=retriever,
                order=order,
                k=args.k,
            )
            if rc_arr is None or sc_arr is None:
                continue
            related_counts_by_retriever[retriever].extend(rc_arr.tolist())
            scores_by_retriever[retriever].extend(sc_arr.tolist())

    # Compute Pearson correlation between number of related tags and score per retriever
    correlations: Dict[str, float] = {}
    for retriever in retrievers:
        rc_list = related_counts_by_retriever[retriever]
        sc_list = scores_by_retriever[retriever]
        if len(rc_list) < 2:
            correlations[retriever] = float("nan")
            continue
        rc_arr = np.asarray(rc_list, dtype=float)
        sc_arr = np.asarray(sc_list, dtype=float)
        if np.std(rc_arr) == 0.0 or np.std(sc_arr) == 0.0:
            correlations[retriever] = float("nan")
            continue
        corr = float(np.corrcoef(rc_arr, sc_arr)[0, 1])
        correlations[retriever] = corr
        print(
            f"[corr] retriever={retriever:12s} → "
            f"pearson_corr(#related_tags, RP-score) = {corr:.4f}"
        )

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

    # --- New plot: per-retriever scatter of RP-score vs #related tags ---
    # We create a 2x4 grid of subplots (up to 8 retrievers).
    n_panels = min(len(retrievers), 8)
    if n_panels > 0:
        # Compute a global max for y-axis to keep scales comparable
        global_max_rc = 0
        for r in retrievers:
            rc_list = related_counts_by_retriever[r]
            if rc_list:
                local_max = max(rc_list)
                if local_max > global_max_rc:
                    global_max_rc = local_max
        if global_max_rc <= 0:
            global_max_rc = 1

        fig_scatter, axes = plt.subplots(2, 4, figsize=(16, 9), sharex=True, sharey=True)
        axes = axes.flatten()

        for idx in range(8):
            ax_sc = axes[idx]
            if idx >= n_panels:
                # Hide unused subplot
                ax_sc.axis("off")
                continue

            retriever = retrievers[idx]
            rc_list = related_counts_by_retriever[retriever]
            sc_list = scores_by_retriever[retriever]

            # If no data for this retriever, mark as such
            if len(rc_list) == 0 or len(sc_list) == 0:
                ax_sc.set_title(retriever_labels.get(retriever, retriever), fontsize=12)
                ax_sc.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=11)
                ax_sc.set_xlim(0, 1)
                # For log scale, can't set ylim to 0; use 0.8 as small positive value if global_max_rc > 0
                ymin = 0.8 if global_max_rc > 0 else 1
                ax_sc.set_ylim(ymin, global_max_rc * 1.05)
                ax_sc.set_yscale('log')
                continue

            rc_arr = np.asarray(rc_list, dtype=float)
            sc_arr = np.asarray(sc_list, dtype=float)

            # Ensure no zero in rc_arr for log-scale plotting
            rc_arr_safe = np.where(rc_arr <= 0, 0.8, rc_arr)

            ax_sc.scatter(sc_arr, rc_arr_safe, s=5, alpha=0.5)
            ax_sc.set_title(retriever_labels.get(retriever, retriever), fontsize=12)

            # X-axis: RP-score, Y-axis: number of related tags (logarithmic)
            ax_sc.set_xlim(0, 1)
            ymin = min(rc_arr_safe.min(), 0.8)
            ymin = 0.8 if ymin <= 0 else ymin
            ax_sc.set_ylim(ymin, global_max_rc * 1.05)
            ax_sc.set_yscale('log')

            # Only show x-labels on the bottom row and y-labels on the leftmost column
            row = idx // 4
            col = idx % 4
            if row == 1:
                ax_sc.set_xlabel("RP-score")
            if col == 0:
                ax_sc.set_ylabel("Number of related tags (log)")

        fig_scatter.tight_layout()
        scatter_out_path = os.path.join(
            args.out_dir,
            "rp_score_vs_related_tags_per_retriever.png",
        )
        fig_scatter.savefig(scatter_out_path, dpi=200)
        print(f"[plot] saved scatter distribution plot to: {scatter_out_path}")
        plt.close(fig_scatter)
    else:
        print("[plot] no retrievers to plot scatter distributions.")


if __name__ == "__main__":
    main()
