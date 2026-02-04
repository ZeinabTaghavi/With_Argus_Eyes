from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from .colors import make_risk_cmap, normalize_scores

def _bin_labels_from_scores(scores: np.ndarray, bins: int = 2) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    if bins <= 1 or s.size == 0:
        return np.zeros_like(s, dtype=int)
    qs = np.quantile(s, np.linspace(0, 1, bins + 1))
    # guard identical quantiles (constant scores)
    qs[0] -= 1e-9
    qs[-1] += 1e-9
    yb = np.digitize(s, qs[1:-1], right=False)
    return yb

def _pca2d_fallback(X_train: np.ndarray, X_test: np.ndarray | None):
    """
    Returns: Z_train (N,2), Z_test (M,2) or None, axis labels, and a transform function `transform_test(X)->(n,2)`
    Uses PCA with n_components=min(2, d). If d==1, pads a zero 2nd axis.
    """
    d = X_train.shape[1]
    ncomp = max(1, min(2, d))
    pca = PCA(n_components=ncomp, random_state=42)
    Zt = pca.fit_transform(X_train)
    if ncomp == 1:
        Zt = np.column_stack([Zt[:, 0], np.zeros_like(Zt[:, 0])])
        def transform_test(X):
            Zx = pca.transform(X)[:, 0:1]
            return np.column_stack([Zx[:, 0], np.zeros(X.shape[0], dtype=Zx.dtype)])
        xlab, ylab = "PCA 1", "zero"
    else:
        def transform_test(X):
            return pca.transform(X)[:, :2]
        xlab, ylab = "PCA 1", "PCA 2"
    Z_test = None if X_test is None else transform_test(X_test)
    return Zt, Z_test, xlab, ylab, transform_test

def plot_lda2d_risk(
    X_train: np.ndarray, scores_train: np.ndarray,
    *,
    X_test: np.ndarray | None = None,
    scores_test: np.ndarray | None = None,
    bins_for_lda: int = 2,
    title: str = "LDA (2D) score",
    combine_train_test: bool = True,
    alpha: float = 0.5,
    save_path: str | None = None,
    show: bool = False,
    # --- new optional overlay of high/low landmarks ---
    overlay_points: bool = False,
    X_high_risk: np.ndarray | None = None,
    scores_high_risk: np.ndarray | None = None,
    labels_high_risk: list[str] | None = None,
    X_low_risk: np.ndarray | None = None,
    scores_low_risk: np.ndarray | None = None,
    labels_low_risk: list[str] | None = None,
    label_fontsize: int = 4,
) -> None:
    """
    Robust 2D LDA plot:
    - Fits LDA on binned scores (bins_for_lda classes)
    - Caps n_components <= min(2, n_features, n_classes-1)
    - If LDA is not possible (e.g., only one class or numerical issues), falls back to PCA(2).
    Colors always represent the continuous score.
    """
    X_train = np.asarray(X_train)
    scores_train = np.asarray(scores_train, dtype=float).reshape(-1)
    if scores_train.shape[0] != X_train.shape[0]:
        raise ValueError("X_train and scores_train must have the same number of samples.")

    if X_test is not None:
        X_test = np.asarray(X_test)
    if scores_test is not None:
        scores_test = np.asarray(scores_test, dtype=float).reshape(-1)

    has_test = (
        X_test is not None
        and scores_test is not None
        and X_test.shape[0] > 0
        and scores_test.shape[0] > 0
    )
    if has_test and X_test.shape[0] != scores_test.shape[0]:
        raise ValueError("X_test and scores_test must have the same number of samples.")

    if combine_train_test and has_test:
        X_fit = np.concatenate([X_train, X_test], axis=0)
        scores_fit = np.concatenate([scores_train, scores_test], axis=0)
    else:
        X_fit = X_train
        scores_fit = scores_train

    yb = _bin_labels_from_scores(scores_fit, bins=bins_for_lda)
    classes = np.unique(yb)
    d = X_fit.shape[1]
    max_allowed = min(2, d, max(0, len(classes) - 1))

    # Decide projection
    use_pca_fallback = (max_allowed < 1)
    transform_2d = None  # will become a callable

    if not use_pca_fallback:
        try:
            lda = LDA(n_components=max_allowed)
            lda.fit(X_fit, yb)
            # If only 1 axis from LDA, create a second axis from residual variance
            if max_allowed == 1:
                w = lda.coef_[0]
                w = w / (np.linalg.norm(w) + 1e-12)
                X_res = X_fit - (X_fit @ w)[:, None] * w[None, :]
                # dominant variance direction in the residual
                u, s, vt = np.linalg.svd(X_res, full_matrices=False)
                axis2 = vt[0]

                def transform_2d(X):
                    X = np.asarray(X)
                    Zt1 = lda.transform(X)[:, 0]
                    Xt_res = X - (X @ w)[:, None] * w[None, :]
                    second_t = (Xt_res @ axis2[:, None]).ravel()
                    return np.column_stack([Zt1, second_t])
                xlab, ylab = "Orthonormal axis 1", "Orthonormal axis 2"
            else:
                def transform_2d(X):
                    return lda.transform(X)[:, :2]
                xlab, ylab = "LDA orthonormal axis 1", "LDA orthonormal axis 2"
        except Exception:
            # any numerical issue -> fallback
            use_pca_fallback = True

    if use_pca_fallback:
        _, _, xlab, ylab, transform_2d = _pca2d_fallback(X_fit, X_test if has_test else None)

    if transform_2d is None:
        raise RuntimeError("Failed to build a 2D transform for LDA plot.")

    Z_train = transform_2d(X_train)
    Z_test = transform_2d(X_test) if has_test else None

    # Colors
    cmap = make_risk_cmap()
    if has_test:
        _, norm = normalize_scores(np.concatenate([scores_train, scores_test]))
    else:
        _, norm = normalize_scores(scores_train)

    # Optional overlays (high/low landmarks)
    def _plot_overlay(X0, s0, labels0, marker, label):
        if X0 is None or s0 is None:
            return
        if isinstance(X0, list):
            X0 = np.asarray(X0)
        if X0.size == 0:
            return
        Z0 = transform_2d(X0)
        s0 = np.asarray(s0, dtype=float)
        plt.scatter(
            Z0[:, 0], Z0[:, 1],
            c=s0, cmap=cmap, norm=norm,
            s=26, marker=marker, edgecolors='k', linewidths=0.3,
            label=label, alpha=1.0
        )
        if labels0:
            for (xv, yv, txt) in zip(Z0[:, 0], Z0[:, 1], labels0):
                if txt:
                    plt.text(xv, yv, str(txt), fontsize=label_fontsize, ha='left', va='bottom', alpha=0.9)

    # Plot
    plt.figure(figsize=(7, 6))
    train_label = "train" if (has_test or not combine_train_test) else None
    sc = plt.scatter(
        Z_train[:, 0],
        Z_train[:, 1],
        c=scores_train,
        cmap=cmap,
        norm=norm,
        s=10,
        alpha=alpha,
        label=train_label,
    )
    if has_test and Z_test is not None:
        plt.scatter(
            Z_test[:, 0],
            Z_test[:, 1],
            c=scores_test,
            cmap=cmap,
            norm=norm,
            s=16,
            marker='x',
            alpha=alpha,
            label="test",
        )
    
    if overlay_points:
        _plot_overlay(X_high_risk, scores_high_risk, labels_high_risk, marker='*', label='high')
        _plot_overlay(X_low_risk,  scores_low_risk,  labels_low_risk,  marker='^', label='low')

    plt.xlabel(xlab); plt.ylabel(ylab); plt.title(title)
    cb = plt.colorbar(sc); cb.set_label("score")
    # if (not combine_train_test) or has_test or overlay_points:
    #     plt.legend()
    plt.tight_layout()
    if save_path:
        dirname = os.path.dirname(save_path) or "."
        os.makedirs(dirname, exist_ok=True)
        plt.savefig(save_path, dpi=200)
    if show:
        plt.show()
    plt.close()