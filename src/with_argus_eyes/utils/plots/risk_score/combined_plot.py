# utils/plots/risk_score/combined_plot.py
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from .colors import make_risk_cmap

def _bin_labels_from_scores(scores: np.ndarray, bins: int = 2) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    if bins <= 1 or s.size == 0:
        return np.zeros_like(s, dtype=int)
    qs = np.quantile(s, np.linspace(0, 1, bins + 1))
    qs[0] -= 1e-9; qs[-1] += 1e-9
    return np.digitize(s, qs[1:-1], right=False)

def _pca2d_fallback(X: np.ndarray):
    d = X.shape[1]
    ncomp = max(1, min(2, d))
    pca = PCA(n_components=ncomp, random_state=42)
    Zt = pca.fit_transform(X)
    if ncomp == 1:
        Zt = np.column_stack([Zt[:, 0], np.zeros_like(Zt[:, 0])])
        xlab, ylab = "PCA 1", "zero"
    else:
        xlab, ylab = "PCA 1", "PCA 2"
    return Zt, xlab, ylab, pca

def plot_combined_lda_histogram(
    X_train: np.ndarray, scores_train: np.ndarray,
    *,
    X_test: np.ndarray | None = None,
    scores_test: np.ndarray | None = None,
    bins_for_lda: int = 2,
    hist_bins: int | str = 100,
    hist_density: bool = False,
    hist_log: bool = False,
    figsize: tuple[int, int] = (8, 8),
    title: str | None = None,
    overlay_points: bool = False,
    X_high: np.ndarray | None = None,
    scores_high: np.ndarray | None = None,
    labels_high: list[str] | None = None,
    X_low: np.ndarray | None = None,
    scores_low: np.ndarray | None = None,
    labels_low: list[str] | None = None,
    save_path: str | None = None,
    show: bool = False,
    label_fontsize: int = 5,
    # === NEW ARGUMENTS (Optional, safe for other scripts) ===
    explicit_labels: np.ndarray | None = None,
    class_names: list[str] | None = None
) -> None:
    """
    Produce a combined figure with an LDA scatter plot (top) and a histogram (bottom).
    Supports explicit class labels for LDA projection and coloring.
    """
    # Concatenate train and test for fitting if desired
    combine = X_test is not None and scores_test is not None
    X_fit = np.vstack([X_train, X_test]) if combine else X_train
    scores_fit = np.hstack([scores_train, scores_test]) if combine else scores_train

    # --- 1. Determine Labels for LDA ---
    if explicit_labels is not None:
        # Use provided disjoint categories (e.g. 0=Missed, 1=GoldRet, 2=Distractor)
        yb = explicit_labels
    else:
        # Fallback to binning by score (Old behavior)
        yb = _bin_labels_from_scores(scores_fit, bins=bins_for_lda)

    classes = np.unique(yb)
    d = X_fit.shape[1]
    max_allowed = min(2, d, max(0, len(classes) - 1))

    # --- 2. Fit LDA or Fallback to PCA ---
    use_pca = max_allowed < 1
    if not use_pca:
        lda = LDA(n_components=max_allowed)
        try:
            lda.fit(X_fit, yb)
        except Exception:
            use_pca = True
            
    if use_pca:
        Z_fit, xlab, ylab, pca = _pca2d_fallback(X_fit)
        def transform(X):
            if X is None: return None
            Z = pca.transform(X)
            if Z.shape[1] == 1: Z = np.column_stack([Z[:, 0], np.zeros_like(Z[:, 0])])
            else: Z = Z[:, :2]
            return Z
    else:
        if max_allowed == 1:
            w = lda.coef_[0]
            w = w / (np.linalg.norm(w) + 1e-12)
            X_res = X_fit - (X_fit @ w)[:, None] * w[None, :]
            u, s, vt = np.linalg.svd(X_res, full_matrices=False)
            axis2 = vt[0]
            def transform(X):
                X = np.asarray(X)
                Z1 = lda.transform(X)[:, 0]
                X_res_local = X - (X @ w)[:, None] * w[None, :]
                Z2 = (X_res_local @ axis2[:, None]).ravel()
                return np.column_stack([Z1, Z2])
            xlab, ylab = "LDA 1", "LDA residual"
        else:
            def transform(X):
                return lda.transform(X)[:, :2]
            xlab, ylab = "LDA 1", "LDA 2"

    Z_train = transform(X_train)
    Z_test = transform(X_test) if combine else None
    
    # --- 3. Setup Colors ---
    # Normalize scores for the histogram and default coloring
    all_scores = np.hstack([scores_train, scores_test]) if combine else scores_train
    smin, smax = float(np.min(all_scores)), float(np.max(all_scores))
    if smax <= smin + 1e-12: smax = smin + 1.0
    norm = Normalize(vmin=0.0, vmax=1.0)
    scores_train_norm = (scores_train - smin) / (smax - smin)

    # Randomize draw order
    rng = np.random.default_rng()
    train_order = rng.permutation(Z_train.shape[0])
    Z_train_sorted = Z_train[train_order]
    
    if explicit_labels is not None:
        # COLOR BY CLASS LABEL
        c_train_sorted = explicit_labels[train_order]
        # Colors: 0=Red, 1=Green, 2=Blue/Purple
        label_colors = np.array(['tab:red', 'tab:green', 'tab:blue', 'tab:orange', 'tab:purple'])
        # Ensure we don't go out of bounds if labels > 5
        scatter_colors = label_colors[c_train_sorted % len(label_colors)]
        cmap = None 
        scatter_norm = None
    else:
        # COLOR BY SCORE (Old behavior)
        c_train_sorted = scores_train_norm[train_order]
        cmap = make_risk_cmap()
        scatter_norm = norm
        scatter_colors = None

    # --- 4. Plotting ---
    fig, (ax_scatter, ax_hist) = plt.subplots(
        2, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1]}, sharex=False
    )

    # Scatter Plot
    if explicit_labels is not None:
        ax_scatter.scatter(
            Z_train_sorted[:, 0], Z_train_sorted[:, 1],
            c=scatter_colors, 
            s=10, alpha=0.6,
        )
        if class_names:
            from matplotlib.lines import Line2D
            handles = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor=label_colors[i], label=name)
                for i, name in enumerate(class_names)
            ]
            ax_scatter.legend(handles=handles, loc='best')
    else:
        # Old plotting logic
        sc = ax_scatter.scatter(
            Z_train_sorted[:, 0], Z_train_sorted[:, 1],
            c=c_train_sorted, cmap=cmap, norm=scatter_norm,
            s=10, alpha=0.7,
        )

    ax_scatter.set_xlabel(xlab, fontweight="bold")
    ax_scatter.set_ylabel(ylab, fontweight="bold")
    if title: ax_scatter.set_title(title)

    # Histogram Plot
    if isinstance(hist_bins, int):
        bins_edges = np.linspace(smin, smax, hist_bins + 1)
    else:
        bins_edges = np.histogram_bin_edges(all_scores, bins=hist_bins)
        
    counts_train, edges = np.histogram(scores_train, bins=bins_edges, density=hist_density)
    widths = edges[1:] - edges[:-1]
    
    # Color histogram bars by their score (gradient)
    centers = 0.5 * (edges[:-1] + edges[1:])
    centers_norm = (centers - smin) / (smax - smin)
    hist_cmap = make_risk_cmap()
    bin_colors = hist_cmap(norm(centers_norm))

    ax_hist.bar(
        edges[:-1], counts_train, width=widths, align="edge",
        color=bin_colors, edgecolor="none", alpha=0.9
    )
    
    if combine and Z_test is not None:
         counts_test, _ = np.histogram(scores_test, bins=edges, density=hist_density)
         ax_hist.step(edges, np.r_[counts_test, counts_test[-1] if counts_test.size else 0],
                      where="post", color="k", linewidth=1.5, alpha=0.8, label="test")
         ax_hist.legend()

    ax_hist.set_xlabel("Retrieval Probability Score (RPS)", fontweight="bold")
    ax_hist.set_ylabel("Count", fontweight="bold")

    # Add Colorbar (only if coloring by score)
    fig.subplots_adjust(right=0.80)
    if explicit_labels is None:
        pos_scatter = ax_scatter.get_position()
        pos_hist = ax_hist.get_position()
        cbar_ax = fig.add_axes([0.84, pos_hist.y0, 0.03, pos_scatter.y1 - pos_hist.y0])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("Score (0-1)", rotation=270, labelpad=16, fontweight="bold")

    if save_path:
        fig.savefig(save_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)