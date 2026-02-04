from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from .colors import make_risk_cmap, normalize_scores
from .lda2d import _bin_labels_from_scores  # reuse helper

def _pca3d_fallback(X_train: np.ndarray, X_test: np.ndarray | None):
    d = X_train.shape[1]
    ncomp = max(1, min(3, d))
    pca = PCA(n_components=ncomp, random_state=42)
    Zt = pca.fit_transform(X_train)
    if ncomp < 3:
        # pad with zeros to get 3D
        Zt = np.pad(Zt, ((0, 0), (0, 3 - ncomp)), mode="constant", constant_values=0.0)
        def transform(X):
            Zx = pca.transform(X)[:, :ncomp]
            return np.pad(Zx, ((0, 0), (0, 3 - ncomp)), mode="constant", constant_values=0.0)
    else:
        def transform(X):
            return pca.transform(X)[:, :3]
    Z_test = None if X_test is None else transform(X_test)
    return Zt, Z_test, ("PCA 1", "PCA 2", "PCA 3")

def plot_lda3d_risk(
    X_train: np.ndarray, scores_train: np.ndarray,
    *,
    X_test: np.ndarray | None = None,
    scores_test: np.ndarray | None = None,
    bins_for_lda: int = 2,
    title: str = "LDA (3D) colored by risk score",
    alpha: float = 0.5,
    save_path: str | None = None,
    show: bool = False,
) -> None:
    # derive discrete labels
    yb = _bin_labels_from_scores(scores_train, bins=bins_for_lda)
    classes = np.unique(yb)
    d = X_train.shape[1]
    max_allowed = min(3, d, max(0, len(classes) - 1))

    use_pca_fallback = (max_allowed < 1)
    if not use_pca_fallback:
        try:
            lda = LDA(n_components=max_allowed, solver="svd")
            Zt = lda.fit_transform(X_train, yb)
            Z_test = lda.transform(X_test) if X_test is not None else None

            # If <3 axes, augment with residual variance PCs
            if Zt.shape[1] < 3:
                W = lda.scalings_[:, :Zt.shape[1]]  # [d, k] may be empty if k==0
                Q, _ = np.linalg.qr(W) if W.size else (np.zeros((d, 0)), None)
                Xc = X_train - X_train.mean(0, keepdims=True)
                X_res = Xc - (Xc @ Q) @ Q.T if Q.size else Xc
                u, s, vt = np.linalg.svd(X_res, full_matrices=False)
                need = 3 - Zt.shape[1]
                extra = (X_res @ vt[:need].T) if need > 0 else np.zeros((X_train.shape[0], 0))
                Zt = np.concatenate([Zt, extra], axis=1)

                if X_test is not None:
                    Xtc = X_test - X_train.mean(0, keepdims=True)
                    Xt_res = Xtc - (Xtc @ Q) @ Q.T if Q.size else Xtc
                    extra_t = (Xt_res @ vt[:need].T) if need > 0 else np.zeros((X_test.shape[0], 0))
                    Z_test = np.concatenate([Z_test, extra_t], axis=1)

            axis_labels = ("LDA 1", "LDA 2", "LDA 3") if Zt.shape[1] == 3 else ("LDA 1", "LDA/Var 2", "LDA/Var 3")
        except Exception:
            use_pca_fallback = True

    if use_pca_fallback:
        Zt, Z_test, axis_labels = _pca3d_fallback(X_train, X_test)

    cmap = make_risk_cmap()
    s_train, norm = normalize_scores(scores_train)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(Zt[:, 0], Zt[:, 1], Zt[:, 2], c=s_train, cmap=cmap, norm=norm, s=10, alpha=alpha, label="train")

    if X_test is not None and scores_test is not None and len(X_test) > 0:
        s_test = np.asarray(scores_test, dtype=float)
        ax.scatter(Z_test[:, 0], Z_test[:, 1], Z_test[:, 2], c=s_test, cmap=cmap, norm=norm, s=18, marker='x', alpha=alpha, label="test")

    ax.set_xlabel(axis_labels[0]); ax.set_ylabel(axis_labels[1]); ax.set_zlabel(axis_labels[2])
    ax.set_title(title); ax.view_init(elev=20, azim=35)
    cb = fig.colorbar(sc, ax=ax, pad=0.1); cb.set_label("risk score")
    ax.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()