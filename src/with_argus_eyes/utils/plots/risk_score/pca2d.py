from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from .colors import make_risk_cmap, normalize_scores


def _transform2d_pca(pca: PCA, X: np.ndarray, ncomp: int) -> np.ndarray:
    """
    Transform X with a fitted PCA and ensure a (N,2) output.
    If the PCA was 1D, pad a zero second axis.
    """
    Z = pca.transform(X)
    Z = np.asarray(Z)
    if Z.ndim == 1:
        Z = Z[:, None]
    if ncomp == 1:
        # pad a zero column for the second axis
        Z = np.column_stack([Z[:, 0], np.zeros(Z.shape[0], dtype=Z.dtype)])
    else:
        Z = Z[:, :2]
    return Z


def plot_pca2d_risk(
    X_train: np.ndarray, scores_train: np.ndarray,
    *,
    X_test: np.ndarray | None = None,
    scores_test: np.ndarray | None = None,
    title: str = "PCA (2D) colored by risk score",
    alpha: float = 0.5,
    save_path: str | None = None,
    show: bool = False,
    # --- optional overlay of high/low landmarks with labels ---
    overlay_points: bool = False,
    X_high_risk: np.ndarray | None = None,
    scores_high_risk: np.ndarray | None = None,
    labels_high_risk: list[str] | None = None,
    X_low_risk: np.ndarray | None = None,
    scores_low_risk: np.ndarray | None = None,
    labels_low_risk: list[str] | None = None,
    label_fontsize: int = 6,
) -> None:
    """
    2D PCA plot colored by continuous risk scores.
    - Fits PCA on training embeddings (n_components=min(2, d)).
    - Always returns/plots 2 axes; if d==1, the 2nd axis is zero.
    - Optionally overlays "high" and "low" landmark points with labels.
    """
    X_train = np.asarray(X_train)
    if X_test is not None:
        X_test = np.asarray(X_test)

    d = X_train.shape[1]
    ncomp = max(1, min(2, d))

    # Fit PCA on training set
    pca = PCA(n_components=ncomp, random_state=42)
    Z_train_core = pca.fit_transform(X_train)
    if ncomp == 1:
        Z_train = np.column_stack([Z_train_core[:, 0], np.zeros_like(Z_train_core[:, 0])])
        xlab, ylab = "PCA 1", "zero"
    else:
        Z_train = Z_train_core[:, :2]
        xlab, ylab = "PCA 1", "PCA 2"

    Z_test = None
    if X_test is not None and len(X_test) > 0:
        Z_test = _transform2d_pca(pca, X_test, ncomp)

    # Colors
    cmap = make_risk_cmap()
    s_train, norm = normalize_scores(scores_train)

    # Optional overlays (high/low landmarks)
    def _plot_overlay(X0, s0, labels0, marker, label):
        if X0 is None or s0 is None:
            return
        if isinstance(X0, list):
            X0 = np.asarray(X0)
        if X0.size == 0:
            return
        Z0 = _transform2d_pca(pca, X0, ncomp)
        s0 = np.asarray(s0, dtype=float)
        plt.scatter(
            Z0[:, 0], Z0[:, 1],
            c=s0, cmap=cmap, norm=norm,
            s=28, marker=marker, edgecolors='k', linewidths=0.3,
            label=label, alpha=1.0
        )
        if labels0:
            for (xv, yv, txt) in zip(Z0[:, 0], Z0[:, 1], labels0):
                if txt:
                    plt.text(xv, yv, str(txt), fontsize=label_fontsize, ha='left', va='bottom', alpha=0.9)

    # Plot
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(Z_train[:, 0], Z_train[:, 1], c=s_train, cmap=cmap, norm=norm, s=10, alpha=alpha, label="train")

    if Z_test is not None and scores_test is not None and len(X_test) > 0:
        s_test = np.asarray(scores_test, dtype=float)
        plt.scatter(Z_test[:, 0], Z_test[:, 1], c=s_test, cmap=cmap, norm=norm, s=16, marker='x', alpha=alpha, label="test")

    if overlay_points:
        _plot_overlay(X_high_risk, scores_high_risk, labels_high_risk, marker='*', label='high')
        _plot_overlay(X_low_risk,  scores_low_risk,  labels_low_risk,  marker='^', label='low')

    plt.xlabel(xlab); plt.ylabel(ylab); plt.title(title)
    cb = plt.colorbar(sc); cb.set_label("risk score")
    plt.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
    if show:
        plt.show()
    plt.close()