from __future__ import annotations
from typing import Tuple
import numpy as np
import matplotlib as mpl

def make_risk_cmap() -> mpl.colors.Colormap:
    """
    Custom colormap for scores.

    Mapping on normalized scores in [0, 1]:
        0.0 -> light red (alpha 0.5)
        0.1 -> light yellow (alpha 0.5)
        0.2 -> light green (alpha 0.5)
        0.7 -> light blue (alpha 0.5)
        1.0 -> light dark blue (alpha 0.5)
    """
    from matplotlib.colors import LinearSegmentedColormap, to_rgba

    # Positions in [0, 1] and their corresponding colors (now RGBA for alpha 0.5, lighter versions)
    positions = [0.0, 0.1, 0.2, 0.7, 1.0]

    # Define lighter color values (using more pastel/lighter tones)
    # Using HTML CSS colors for softer hues, with alpha=0.5
    color_rgba = [
        to_rgba("#ff6666", alpha=0.5),    # light red
        to_rgba("#ffff99", alpha=0.5),    # light yellow
        to_rgba("#99ff99", alpha=0.5),    # light green
        to_rgba("#99ccff", alpha=0.5),    # light blue
        to_rgba("#3366cc", alpha=0.5),    # lighter dark blue (not too dark)
    ]

    # Build a smooth colormap that interpolates through these anchors
    return LinearSegmentedColormap.from_list("score_cmap", list(zip(positions, color_rgba)))


def normalize_scores(scores: np.ndarray) -> Tuple[np.ndarray, mpl.colors.Normalize]:
    s = np.asarray(scores, dtype=float)
    vmin, vmax = float(np.min(s)), float(np.max(s))
    if vmax <= vmin + 1e-12:
        vmax = vmin + 1.0
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    return s, norm