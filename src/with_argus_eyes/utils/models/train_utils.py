"""
train_utils.py
---------------

Utility functions to orchestrate dataset splitting, model training, evaluation,
and visualisation of learning curves.  These helpers are imported by
``main.py`` to keep the top-level script concise and easy to follow.

The functions here rely on NumPy arrays and scikit-learn for splitting data.
Matplotlib is used for plotting learning curves.  We avoid seaborn or
explicitly setting colour palettes to comply with the project's plotting
guidelines.

This version has been updated to **only persist the single best baseline model and the single best MLP model** per run (based on test RMSE), to reduce disk and memory usage.

IMPORTANT CHANGE (memory friendly):
-----------------------------------
- Baselines: we train all baseline models but, when `save_models=True`, we
  only save the *single best* baseline (lowest test RMSE).
- MLPs: we train all configurations and seeds, plot the aggregated
  learning curves using *all* runs, but, when `save_models=True`, we only
  save the *single best* MLP model across all configs and seeds
  (lowest test RMSE).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
import joblib
from scipy.stats import gaussian_kde

from .baselines import get_baseline_model
from .mlp import MLPRegressor
from .metrics import regression_metrics


logger = logging.getLogger(__name__)


def _label_eligible_for_best_save(label: str) -> bool:
    """Return True if the config label should compete for best-model saving.

    Only configurations whose label contains 'S3_WD_LOW' (case-insensitive)
    are eligible for persistence, as per the memory-usage policy.
    """
    return "S3_WD_LOW".lower() in label.lower()


def save_model_artifact(model: Any, scaler: Any | None, output_path: str | Path) -> None:
    """Save a trained model and optional feature transformer to disk.

    The artefact is stored as a joblib file containing a dictionary with
    two keys:

    - 'model':   the trained estimator (e.g., Ridge, XGBoost, MLPRegressor)
    - 'scaler':  the feature transformer such as StandardScaler (or None)

    This makes it easy to reload the full pipeline later:

        artefact = joblib.load(path)
        model = artefact['model']
        scaler = artefact['scaler']
        X_scaled = scaler.transform(X_new)
        y_pred = model.predict(X_scaled)
    """
    if isinstance(output_path, str):
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model, "scaler": scaler}
    joblib.dump(payload, output_path)
    logger.info("Saved model artefact to %s", output_path)


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Split the dataset into train/validation/test sets."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    # First split off the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=test_ratio,
        random_state=seed,
        shuffle=True,
    )
    # Compute relative validation ratio with respect to the remaining data
    val_relative = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_relative,
        random_state=seed,
        shuffle=True,
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def run_baseline_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    baseline_configs: Dict[str, Dict],
    scaler: Any | None = None,
    save_models: bool = False,
    results_dir: str | Path | None = None,
) -> Dict[str, Dict[str, float]]:
    """Train and evaluate multiple baseline models.

    When `save_models=True`, ONLY the best baseline (lowest test RMSE) is
    saved to disk, to avoid storing many large models.
    """
    results: Dict[str, Dict[str, float]] = {}

    best_name: str | None = None
    best_rmse: float = float("inf")
    best_model: Any | None = None

    for name, params in baseline_configs.items():
        logger.info("Training baseline model '%s' with params=%s", name, params)
        model = get_baseline_model(name, **params)
        model.fit(X_train, y_train)

        # Evaluate on validation and test sets
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        val_metrics = regression_metrics(y_val, val_pred)
        test_metrics = regression_metrics(y_test, test_pred)

        # Prefix metric keys with 'val_' and 'test_'
        combined = {f"val_{k}": v for k, v in val_metrics.items()}
        combined.update({f"test_{k}": v for k, v in test_metrics.items()})
        results[name] = combined

        # NOTE: we do not save every baseline anymore; we only keep the globally best one.
        # Track best baseline by test RMSE
        rmse = test_metrics["rmse"]
        if rmse < best_rmse:
            best_rmse = rmse
            best_name = name
            best_model = model

    # Save ONLY the best baseline, if requested
    if save_models and best_model is not None:
        if results_dir is None:
            results_dir = "./"
        model_dir = Path(results_dir) / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        artefact_path = model_dir / f"baseline_best_{best_name}.joblib"
        save_model_artifact(model=best_model, scaler=scaler, output_path=artefact_path)
        print(
            f"[run_baseline_models] Saved BEST baseline model '{best_name}' "
            f"(test_rmse={best_rmse:.6f}) to {artefact_path}"
        )

    return results


def run_mlp_experiments(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    mlp_configs: Iterable[Dict],
    seeds: Iterable[int] = (42,),
    results_dir: str | Path | None = None,
    scaler: Any | None = None,
    save_models: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Train multiple MLP configurations and aggregate metrics across seeds.

    - Aggregated metrics (mean/std over seeds) are returned for EVERY config.
    - Aggregated learning curves (mean + min/max band) are plotted for EVERY
      config using ALL seeds.
    - When `save_models=True`, ONLY the single best eligible MLP instance
      (labels containing 'S3_WD_LOW') across all seeds is saved.
    - Config dictionaries may include a `scheduler` entry that will be passed
      through to `MLPRegressor` to configure a PyTorch LR scheduler.
    """
    if results_dir is not None:
        Path(results_dir).mkdir(parents=True, exist_ok=True)

    aggregated: Dict[str, Dict[str, float]] = {}

    # NOTE: we track a single global best MLP (config + seed) and only save that one.
    # Global best MLP across all configs + seeds
    best_mlp_label: str | None = None
    best_mlp_seed: int | None = None
    best_mlp_rmse: float = float("inf")
    best_mlp_model: MLPRegressor | None = None

    for cfg in mlp_configs:
        # Extract label and remove it from kwargs for MLPRegressor
        label = cfg.get("label", "MLP")
        label_is_eligible = _label_eligible_for_best_save(label)
        cfg = dict(cfg)  # copy so we can pop
        cfg.pop("label", None)
        # Ensure input_dim is set
        cfg.setdefault("input_dim", X_train.shape[1])

        # Accumulators for metrics across seeds
        metrics_list: List[Dict[str, float]] = []
        histories: List[Dict[str, List[float]]] = []

        for seed in seeds:
            logger.info("Training MLP '%s' with seed=%d and params=%s", label, seed, cfg)
            mlp = MLPRegressor(seed=seed, **cfg)
            mlp.fit(X_train, y_train, X_val, y_val)

            # Predict on validation and test sets
            val_pred = mlp.predict(X_val)
            test_pred = mlp.predict(X_test)
            val_metrics = regression_metrics(y_val, val_pred)
            test_metrics = regression_metrics(y_test, test_pred)

            # Prefix metric keys
            combined = {f"val_{k}": v for k, v in val_metrics.items()}
            combined.update({f"test_{k}": v for k, v in test_metrics.items()})
            metrics_list.append(combined)

            histories.append(
                {
                    "train_loss": mlp.history["train_loss"].copy(),
                    "val_loss": mlp.history["val_loss"].copy(),
                }
            )

            # Track GLOBAL best MLP (across eligible configs + seeds) by test RMSE
            rmse = test_metrics["rmse"]
            if label_is_eligible and rmse < best_mlp_rmse:
                best_mlp_rmse = rmse
                best_mlp_label = label
                best_mlp_seed = seed
                best_mlp_model = mlp

        # Aggregate metrics across seeds: compute mean and std
        agg: Dict[str, float] = {}
        keys = metrics_list[0].keys()
        for key in keys:
            vals = np.array([m[key] for m in metrics_list])
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals))
        aggregated[label] = agg

        # Save aggregated learning curve across seeds:
        # mean line + min/max band (train and val).
        if results_dir is not None and len(histories) > 0:
            max_epochs = max(len(h["train_loss"]) for h in histories)

            train_matrix = []
            val_matrix = []
            for h in histories:
                # Training loss
                train = np.asarray(h["train_loss"], dtype=float)
                if len(train) < max_epochs:
                    train = np.pad(train, (0, max_epochs - len(train)), mode="edge")
                train_matrix.append(train)

                # Validation loss
                val_hist = h.get("val_loss", [])
                if val_hist:
                    val = np.asarray(val_hist, dtype=float)
                    if len(val) < max_epochs:
                        val = np.pad(val, (0, max_epochs - len(val)), mode="edge")
                    val_matrix.append(val)

            train_matrix = np.stack(train_matrix, axis=0)  # (n_seeds, max_epochs)
            train_mean = train_matrix.mean(axis=0)
            train_min = train_matrix.min(axis=0)
            train_max = train_matrix.max(axis=0)

            val_mean = val_min = val_max = None
            if len(val_matrix) > 0:
                val_matrix = np.stack(val_matrix, axis=0)
                val_mean = val_matrix.mean(axis=0)
                val_min = val_matrix.min(axis=0)
                val_max = val_matrix.max(axis=0)

            history_for_plot: Dict[str, List[float]] = {
                "train_mean": train_mean.tolist(),
                "train_min": train_min.tolist(),
                "train_max": train_max.tolist(),
            }
            if val_mean is not None:
                history_for_plot.update(
                    {
                        "val_mean": val_mean.tolist(),
                        "val_min": val_min.tolist(),
                        "val_max": val_max.tolist(),
                    }
                )

            save_learning_curve(
                history=history_for_plot,
                title=f"{label} (average over seeds)",
                output_path=Path(results_dir) / f"learning_curve_{label}_avg.png",
            )

    # Save ONLY the single best MLP model, if requested
    if save_models:
        if best_mlp_model is not None:
            if results_dir is None:
                results_dir = "./"
            model_dir = Path(results_dir) / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            artefact_path = model_dir / f"mlp_best_{best_mlp_label}_seed{best_mlp_seed}.joblib"
            save_model_artifact(model=best_mlp_model, scaler=scaler, output_path=artefact_path)
            print(
                f"[run_mlp_experiments] Saved BEST MLP model '{best_mlp_label}' "
                f"(seed={best_mlp_seed}, test_rmse={best_mlp_rmse:.6f}) to {artefact_path}"
            )
        else:
            logger.warning(
                "save_models=True but no eligible 'S3_WD_Low' MLP configs were found; skipping model persistence."
            )

    return aggregated


def save_learning_curve(history: Dict[str, List[float]], title: str, output_path: Path) -> None:
    """Plot training/validation loss versus epoch and save to a PNG file.

    Two modes:
    - Aggregated mode: if history contains keys
        'train_mean', 'train_min', 'train_max'
      (and optionally 'val_mean', 'val_min', 'val_max'), we plot mean ±
      min/max band.
    - Simple mode: if only 'train_loss' / 'val_loss' are present, plot
      plain lines (backwards-compatible).
    """
    import matplotlib
    matplotlib.use("Agg")  # Use a non-interactive backend for file output
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    if "train_mean" in history:
        # Aggregated mode with band
        train_mean = np.asarray(history["train_mean"], dtype=float)
        train_min = np.asarray(history["train_min"], dtype=float)
        train_max = np.asarray(history["train_max"], dtype=float)
        epochs = np.arange(1, len(train_mean) + 1)

        ax.plot(epochs, train_mean, label="Training loss")
        ax.fill_between(epochs, train_min, train_max, alpha=0.2, label="Train range")

        if "val_mean" in history:
            val_mean = np.asarray(history["val_mean"], dtype=float)
            val_min = np.asarray(history["val_min"], dtype=float)
            val_max = np.asarray(history["val_max"], dtype=float)
            ax.plot(epochs, val_mean, label="Validation loss")
            ax.fill_between(epochs, val_min, val_max, alpha=0.2, label="Val range")

    else:
        # Simple mode: original behaviour
        train_loss = np.asarray(history.get("train_loss", []), dtype=float)
        val_loss = np.asarray(history.get("val_loss", []), dtype=float)
        epochs = np.arange(1, len(train_loss) + 1)

        ax.plot(epochs, train_loss, label="Training loss")
        if len(val_loss) > 0:
            ax.plot(epochs, val_loss, label="Validation loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_latex_results_table(
    baseline_results: dict,
    mlp_results: dict,
    row_specs: list[dict],
    output_path: Path,
    caption: str = "Regression performance on the STS-B dataset.",
    label: str = "tab:regression_results",
) -> None:
    """Create an Overleaf-ready LaTeX table of metrics."""
    lines: list[str] = []

    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{llcccccccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Model Type & Architecture / Config & RMSE ($\downarrow$) & "
        r"MAE ($\downarrow$) & Pearson $r$ ($\uparrow$) & Spearman $\rho$ ($\uparrow$) & "
        r"Precision$_{\text{macro}}$ ($\uparrow$) & Recall$_{\text{macro}}$ ($\uparrow$) & "
        r"F1$_{\text{macro}}$ ($\uparrow$) & F1$_{\text{micro}}$ ($\uparrow$) & "
        r"F1$_{\text{weighted}}$ ($\uparrow$) & Accuracy ($\uparrow$) \\"
    )
    lines.append(r"\midrule")

    prev_type: str | None = None

    for spec in row_specs:
        model_type = spec["model_type"]
        name = spec["name"]
        source = spec["source"]
        key = spec["key"]

        # Decide where to fetch metrics from
        if source == "baseline":
            metrics = baseline_results[key]
            rmse = metrics["test_rmse"]
            mae = metrics["test_mae"]
            pearson = metrics["test_pearson"]
            spearman = metrics["test_spearman"]
            cls_prec_macro = metrics.get("test_cls_precision_macro", float("nan"))
            cls_rec_macro = metrics.get("test_cls_recall_macro", float("nan"))
            cls_f1_macro = metrics.get("test_cls_f1_macro", float("nan"))
            cls_f1_micro = metrics.get("test_cls_f1_micro", float("nan"))
            cls_f1_weighted = metrics.get("test_cls_f1_weighted", float("nan"))
            cls_acc = metrics.get("test_cls_accuracy", float("nan"))
        elif source == "mlp":
            metrics = mlp_results[key]
            # We aggregate over seeds; use the mean values
            rmse = metrics["test_rmse_mean"]
            mae = metrics["test_mae_mean"]
            pearson = metrics["test_pearson_mean"]
            spearman = metrics["test_spearman_mean"]
            cls_prec_macro = metrics.get("test_cls_precision_macro_mean", float("nan"))
            cls_rec_macro = metrics.get("test_cls_recall_macro_mean", float("nan"))
            cls_f1_macro = metrics.get("test_cls_f1_macro_mean", float("nan"))
            cls_f1_micro = metrics.get("test_cls_f1_micro_mean", float("nan"))
            cls_f1_weighted = metrics.get("test_cls_f1_weighted_mean", float("nan"))
            cls_acc = metrics.get("test_cls_accuracy_mean", float("nan"))
        else:
            raise ValueError(f"Unknown source '{source}' in row_specs")

        # Only show the Model Type text on the first row of each block
        model_type_cell = model_type if model_type != prev_type else ""
        prev_type = model_type

        line = (
            f"{model_type_cell} & {name} & "
            f"{rmse:.3f} & {mae:.3f} & {pearson:.3f} & {spearman:.3f} & "
            f"{cls_prec_macro:.3f} & {cls_rec_macro:.3f} & {cls_f1_macro:.3f} & "
            f"{cls_f1_micro:.3f} & {cls_f1_weighted:.3f} & {cls_acc:.3f} \\\\"
        )
        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table}")
    lines.append("")  # final newline

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_error_distribution(y_true: np.ndarray, y_pred: np.ndarray, title: str, output_path: Path) -> None:
    """Save a histogram of prediction errors (residuals)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    residuals = y_true - y_pred
    mean_error = float(np.mean(residuals))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(residuals, bins=50, density=True, alpha=0.6, label="Residuals", color="#80cffa")
    ax.axvline(mean_error, linestyle="dashed", linewidth=1, label=f"Mean Error: {mean_error:.4f}")

    ax.set_xlabel("Residual (True - Predicted)", fontweight="bold")
    ax.set_ylabel("Density", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_prediction_scatter(y_true: np.ndarray, y_pred: np.ndarray, title: str, output_path: Path) -> None:
    """Save a density scatter plot of ground truth vs predictions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Calculate point density for the 'heatmap' effect
    xy = np.vstack([y_true, y_pred])
    z = gaussian_kde(xy)(xy)

    # Sort points by density so dense points are plotted on top
    idx = z.argsort()
    x, y, z = y_true[idx], y_pred[idx], z[idx]

    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(x, y, c=z, s=20, alpha=0.6)

    # Set axis limits to [0, 1] for both x and y
    lims = [0.0, 1.0]
    ax.plot(lims, lims, linestyle="--", alpha=0.75, zorder=0, label="Ideal (x=y)")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel("Ground Truth", fontweight="bold")
    ax.set_ylabel("Prediction", fontweight="bold")
    ax.legend()

    # Create colorbar with same height as the plot
    fig.tight_layout()
    cbar = fig.colorbar(scatter, ax=ax, orientation="vertical", shrink=0.8, aspect=20)
    cbar.set_label("Point Density", rotation=270, fontweight="bold", labelpad=20)
    fig.savefig(output_path)
    plt.close(fig)