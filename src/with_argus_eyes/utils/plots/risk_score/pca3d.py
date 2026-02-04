from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from sklearn.decomposition import PCA
from .colors import make_risk_cmap, normalize_scores

def plot_pca3d_risk(
    X_train: np.ndarray, scores_train: np.ndarray,
    *,
    X_test: np.ndarray | None = None,
    scores_test: np.ndarray | None = None,
    title: str = "PCA (3D) colored by risk score",
    alpha: float = 0.5,
    save_path: str | None = None,
    show: bool = False,
) -> None:
    pca = PCA(n_components=3, random_state=42)
    Z_train = pca.fit_transform(X_train)
    cmap = make_risk_cmap()
    s_train, norm = normalize_scores(scores_train)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(Z_train[:,0], Z_train[:,1], Z_train[:,2], c=s_train, cmap=cmap, norm=norm, s=10, alpha=alpha, label="train")

    if X_test is not None and scores_test is not None and len(X_test) > 0:
        Z_test = pca.transform(X_test)
        s_test = np.asarray(scores_test, dtype=float)
        ax.scatter(Z_test[:,0], Z_test[:,1], Z_test[:,2], c=s_test, cmap=cmap, norm=norm, s=20, marker='x', alpha=alpha, label="test")

    ax.set_xlabel("PCA 1"); ax.set_ylabel("PCA 2"); ax.set_zlabel("PCA 3")
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