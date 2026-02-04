"""
main.py
-------

This script orchestrates the end-to-end regression experiments on the
Semantic Textual Similarity Benchmark (STS-B) dataset.  It loads the
dataset, converts sentences to embeddings, splits the data into training,
validation and test sets, scales the features, trains baseline models and
multiple multilayer perceptrons (MLPs) with different hyperparameters, and
logs the results.  Learning curves are saved to disk for later analysis.

To run the script from the command line:

.. code-block:: bash

    python -m regression_project.main

By default, the script processes a subset of the dataset (e.g. 2000 examples)
to reduce runtime.  Adjust ``max_samples`` in the call to
``load_stsb_embeddings`` to use the full dataset.

Results (metrics and plots) will be stored in the ``results`` directory
relative to the project root.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Dict

import numpy as np
from sklearn.preprocessing import StandardScaler

from .train_utils import (
    split_dataset,
    run_baseline_models,
    run_mlp_experiments,
    save_latex_results_table,
    save_error_distribution,
    save_prediction_scatter,
)
from .mlp import MLPRegressor


def setup_logging() -> None:
    """Configure the logging format and level for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


# --- New function: evaluate_regression_models ---
def evaluate_regression_models(
    X: np.ndarray,
    y: np.ndarray,
    results_dir: str | Path = "./...",
    mlp_configs: List[dict] | None = None,
    seeds: List[int] | None = None,
    baseline_configs: Dict[str, dict] | None = None,

    tag: str = "regression",
    save_models: bool = False,
) -> None:
    """Run the full regression pipeline (scaling, splitting, baselines, MLPs,
    metrics and LaTeX table) on a given feature matrix X and target vector y.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input embeddings or feature vectors.
    y : ndarray of shape (n_samples,)
        Regression targets in [0, 1].
    results_dir : str or Path, optional
        Directory where metrics, learning curves and LaTeX tables are saved.
    tag : str, optional
        Name of the dataset (used only for logging).
    """
    logger = logging.getLogger("evaluation")
    logger.info("Starting regression experiment on dataset '%s'", tag)

    # 1) Standardise features to zero mean and unit variance.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logger.info("Standardised features: mean ~ 0, var ~ 1 (per feature)")

    # 2) Split into train/val/test sets
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset(
        X_scaled, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42
    )
    logger.info(
        "Data split into %d train, %d val, %d test examples",
        len(X_train),
        len(X_val),
        len(X_test),
    )

    # 3) Define baseline model configurations
    if baseline_configs is None:
        print(
            f"[evaluate_regression_models] No baseline configurations provided, "
            f"using default configurations for {tag}"
        )
        baseline_configs = {
            # Constant baselines: always return 0 or 1 as rank score
            "zero": {},
            "one": {},
            # Unsupervised baseline: cosine similarity between sentence embeddings
            "cosine": {},
            "ridge": {},
            # Non-linear tree-based baseline: XGBoost
            "xgboost": {
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "n_jobs": 1,
            },
            # # Kernel baseline: SVR with RBF kernel
            # "svr_rbf": {
            #     "C": 1.0,
            #     "gamma": "scale",
            #     "epsilon": 0.1,
            # },
        }
        print(f"[evaluate_regression_models] loaded {len(baseline_configs)} baseline configurations")

    # Train and evaluate baselines
    baseline_results = run_baseline_models(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        baseline_configs,
        scaler=scaler,
        save_models=save_models,
        results_dir=results_dir,
    )
    logger.info("Baseline results computed")

    # 4) Define MLP configurations
    # Each configuration must have a unique 'label'. Remaining keys correspond
    # to arguments accepted by MLPRegressor. The input dimension is inferred.
    if mlp_configs is None:
        print(
            f"[evaluate_regression_models] No MLP configurations provided, "
            f"using default configurations for {tag}"
        )
        mlp_configs = [
            # ==============================================================
            # PHASE 1: Architecture Search (Fix Loss=MSE, Dropout=0.2)
            # ==============================================================
            {
                "label": "mlp_P1_Arch_Shallow",
                "hidden_dims": (128, 64),
                "dropout": 0.2,
                "loss": "mse",
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "epochs": 50,
                "batch_size": 128,
            }
        ]
    else:
        print(f"[evaluate_regression_models] Using provided {len(mlp_configs)} MLP configurations for {tag}")

    if seeds is None:
        print(
            f"[evaluate_regression_models] No seeds provided, using default seeds for {tag}"
        )
        seeds = [42, 43, 44]

    # Directory where learning curves and aggregated results will be stored
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)

    # Train and evaluate MLPs
    mlp_results = run_mlp_experiments(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        mlp_configs,
        seeds=seeds,
        results_dir=results_dir,
        scaler=scaler,
        save_models=save_models,
    )
    logger.info("MLP experiments completed")

    # 5) Save or update results to disk (JSON + LaTeX table)
    metrics_path = results_dir / "metrics.json"

    # Load existing metrics if present, else start fresh.
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}

    # Ensure dictionaries exist for baselines and mlps
    all_metrics.setdefault("baselines", {})
    all_metrics.setdefault("mlps", {})

    # Update/add new entries to baselines
    for key, value in baseline_results.items():
        all_metrics["baselines"][key] = value

    # Update/add new entries to mlps
    for key, value in mlp_results.items():
        all_metrics["mlps"][key] = value

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4)
    logger.info("Saved metrics to %s", metrics_path)

    # Define the rows for the LaTeX table automatically for *all* configs.
    # We create one row per baseline (using its key) and one row per MLP config.
    # The keys must match the names used in `baseline_configs` and the `label`
    # fields in `mlp_configs`.
    row_specs: list[dict] = []

    # Add all baselines
    for key in sorted(baseline_results.keys()):
        row_specs.append(
            {
                "model_type": "Baselines",
                "name": key,          # use the config key as the display name
                "source": "baseline",
                "key": key,
            }
        )

    # Add all MLP configs
    for key in sorted(mlp_results.keys()):
        row_specs.append(
            {
                "model_type": "MLPs",
                "name": key,          # use the label as the display name
                "source": "mlp",
                "key": key,
            }
        )

    latex_table_path = results_dir / "metrics_table.tex"
    save_latex_results_table(
        baseline_results=baseline_results,
        mlp_results=mlp_results,
        row_specs=row_specs,
        output_path=latex_table_path,
        caption=f"Regression performance on the {tag} regression task.",
        label=f"tab:{tag}_regression",
    )
    logger.info("Saved LaTeX table to %s", latex_table_path)

    # 6) Fit the best MLP once more and generate diagnostic plots
    try:
        # Select best config by lowest test_rmse_mean
        best_label = min(
            mlp_results.keys(),
            key=lambda k: mlp_results[k].get("test_rmse_mean", float("inf")),
        )
        logger.info(
            "Selected best MLP configuration '%s' based on test_rmse_mean",
            best_label,
        )

        # Find its config dict
        best_cfg_dict: dict | None = None
        for cfg in mlp_configs:
            if cfg.get("label") == best_label:
                best_cfg_dict = dict(cfg)
                break

        if best_cfg_dict is None:
            logger.warning(
                "Best label '%s' not found in mlp_configs; skipping diagnostic plots",
                best_label,
            )
        else:
            best_cfg_dict.pop("label", None)
            best_cfg_dict.setdefault("input_dim", X_train.shape[1])
            best_seed = seeds[0] if len(seeds) > 0 else 42

            best_mlp = MLPRegressor(seed=best_seed, **best_cfg_dict)
            best_mlp.fit(X_train, y_train, X_val, y_val)
            y_pred_best = best_mlp.predict(X_test)

            err_path = results_dir / f"best_{tag}_error_dist.png"
            scat_path = results_dir / f"best_{tag}_scatter.png"

            save_error_distribution(
                y_true=y_test,
                y_pred=y_pred_best,
                title=f"{tag}: {best_label}",
                output_path=err_path,
            )
            save_prediction_scatter(
                y_true=y_test,
                y_pred=y_pred_best,
                title=f"{tag}: {best_label}",
                output_path=scat_path,
            )
            logger.info(
                "Saved error distribution and scatter plots for best model '%s'",
                best_label,
            )
    except Exception as exc:  # keep plotting failures from breaking the run
        logger.warning("Failed to generate best-model diagnostic plots: %s", exc)

    logger.info("Experiment complete for dataset '%s'", tag)