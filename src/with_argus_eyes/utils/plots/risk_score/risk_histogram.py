from __future__ import annotations
import os
from typing import Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from .colors import make_risk_cmap, normalize_scores

def plot_risk_histogram(
    scores: Union[np.ndarray, list],
    *,
    scores_test: Optional[Union[np.ndarray, list]] = None,
    bins: Union[int, str] = "auto",          # e.g. 50, "auto", "fd"
    density: bool = False,
    log: bool = False,
    title: str = "Score histogram",
    save_path: Optional[str] = None,
    show: bool = False,
    label_train: str = "train",
    label_test: str = "test",
) -> None:
    """
    Draws a histogram of scores.
      - Train bars are colored by bin center (blue→red as score increases).
      - If scores_test is provided, overlays a black step curve for test.
    """
    s_train = np.asarray(scores, dtype=float).ravel()
    if s_train.size == 0:
        raise ValueError("scores_train is empty.")

    smin, smax = float(np.min(s_train)), float(np.max(s_train))
    # include test range if provided
    if scores_test is not None:
        s_test = np.asarray(scores_test, dtype=float).ravel()
        if s_test.size == 0:
            s_test = None
        else:
            smin = min(smin, float(np.min(s_test)))
            smax = max(smax, float(np.max(s_test)))
    else:
        s_test = None

    # Edge case: constant scores -> synthesize a small span so we can make bins
    if not np.isfinite(smin) or not np.isfinite(smax):
        raise ValueError("scores contain non-finite values.")
    if smax <= smin + 1e-12:
        smax = smin + 1.0

    # Build common bin edges for train/test
    if isinstance(bins, int):
        edges = np.linspace(smin, smax, bins + 1)
    else:
        # "auto" or rule name via numpy
        data_for_bins = s_train if s_test is None else np.concatenate([s_train, s_test])
        edges = np.histogram_bin_edges(data_for_bins, bins=bins)

    # Histogram counts
    h_train, edges = np.histogram(s_train, bins=edges, density=density)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = edges[1:] - edges[:-1]

    cmap = make_risk_cmap()
    # Use the **actual** score range for the color scale, not just bin centers range
    _, norm = normalize_scores(np.array([smin, smax], dtype=float))
    bin_colors = cmap(norm(centers))

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot train as colored bars
    ax.bar(
        edges[:-1], h_train, width=widths, align="edge",
        color=bin_colors, edgecolor="none", label=label_train, alpha=0.9
    )

    # Overlay test as a black step line (shares the same bins)
    if s_test is not None:
        h_test, _ = np.histogram(s_test, bins=edges, density=density)
        # step wants x as bin edges; y stepped per-bin
        ax.step(edges, np.r_[h_test, h_test[-1] if h_test.size else 0.0],
                where="post", color="k", linewidth=1.5, label=label_test, alpha=0.9)

    if log:
        ax.set_yscale("log")

    ax.set_title(title)
    ax.set_xlabel("Score")
    ax.set_ylabel("density" if density else "count")

    # Colorbar to show score→color mapping
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Score")

    ax.legend()
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)