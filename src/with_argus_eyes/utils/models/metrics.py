"""
metrics.py
-----------

This module collects common regression evaluation metrics used throughout the
experiments.  Each function accepts NumPy arrays of true and predicted
values and returns a scalar.  By consolidating all metrics in one place we
avoid duplicating code across scripts and ensure consistent behaviour.
"""

from __future__ import annotations

import numpy as np

# Thresholds used to bucketise retrieval scores into categorical labels.
LOW_THRESHOLD = 0.33
HIGH_THRESHOLD = 0.66

# Mapping from class ids to human readable names (used for metric keys).
CLASS_ID_TO_NAME = {0: "low", 1: "medium", 2: "high"}


def _bucketise_scores(scores: np.ndarray) -> np.ndarray:
    """Convert continuous retrieval scores into discrete class ids.

    Scores in [0, 0.33) -> 0 (low), [0.33, 0.66) -> 1 (medium), [0.66, 1] -> 2 (high).
    Values outside [0, 1] are clipped before bucketing.
    """
    arr = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
    labels = np.zeros_like(arr, dtype=int)
    high_mask = arr >= HIGH_THRESHOLD
    medium_mask = (~high_mask) & (arr >= LOW_THRESHOLD)
    labels[high_mask] = 2
    labels[medium_mask] = 1
    # Remaining entries stay 0 (low)
    return labels


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator != 0 else 0.0


def _classification_metrics_from_labels(y_true_cls: np.ndarray, y_pred_cls: np.ndarray) -> dict[str, float]:
    """Compute an extensive suite of classification metrics."""
    classes = np.array(sorted(CLASS_ID_TO_NAME.keys()))
    n_classes = len(classes)

    # Confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for i, cls_true in enumerate(classes):
        true_mask = y_true_cls == cls_true
        for j, cls_pred in enumerate(classes):
            cm[i, j] = int(np.sum(true_mask & (y_pred_cls == cls_pred)))

    row_sums = cm.sum(axis=1)  # true class counts (support)
    col_sums = cm.sum(axis=0)  # predicted counts
    total = cm.sum()

    per_class_stats: dict[str, float] = {}
    precisions = []
    recalls = []
    f1s = []
    weighted_precisions = []
    weighted_recalls = []
    weighted_f1s = []

    for idx, cls in enumerate(classes):
        tp = cm[idx, idx]
        fp = col_sums[idx] - tp
        fn = row_sums[idx] - tp

        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2 * precision * recall, precision + recall)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        support = row_sums[idx]
        weight = _safe_divide(support, total)
        weighted_precisions.append(precision * weight)
        weighted_recalls.append(recall * weight)
        weighted_f1s.append(f1 * weight)

        name = CLASS_ID_TO_NAME[cls]
        per_class_stats[f"cls_{name}_precision"] = precision
        per_class_stats[f"cls_{name}_recall"] = recall
        per_class_stats[f"cls_{name}_f1"] = f1
        per_class_stats[f"cls_{name}_support"] = int(support)

    accuracy = _safe_divide(np.trace(cm), total)

    # Micro metrics (same for precision/recall/f1 in single-label setup)
    micro_precision = accuracy
    micro_recall = accuracy
    micro_f1 = accuracy

    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    weighted_precision = float(np.sum(weighted_precisions))
    weighted_recall = float(np.sum(weighted_recalls))
    weighted_f1 = float(np.sum(weighted_f1s))

    # Matthews correlation coefficient (multi-class)
    numerator = (np.trace(cm) * total) - np.sum(row_sums * col_sums)
    denominator = np.sqrt(
        (total**2 - np.sum(col_sums**2)) * (total**2 - np.sum(row_sums**2))
    )
    mcc = _safe_divide(numerator, denominator)

    # Cohen's kappa
    expected_acc = _safe_divide(np.sum(row_sums * col_sums), total**2 if total else 1.0)
    kappa = _safe_divide(accuracy - expected_acc, 1 - expected_acc) if total else 0.0

    metrics = {
        "cls_accuracy": accuracy,
        "cls_precision_macro": macro_precision,
        "cls_recall_macro": macro_recall,
        "cls_f1_macro": macro_f1,
        "cls_precision_micro": micro_precision,
        "cls_recall_micro": micro_recall,
        "cls_f1_micro": micro_f1,
        "cls_precision_weighted": weighted_precision,
        "cls_recall_weighted": weighted_recall,
        "cls_f1_weighted": weighted_f1,
        "cls_mcc": mcc,
        "cls_cohen_kappa": kappa,
    }
    metrics.update(per_class_stats)

    # Flatten confusion matrix entries
    for i, cls_true in enumerate(classes):
        for j, cls_pred in enumerate(classes):
            true_name = CLASS_ID_TO_NAME[cls_true]
            pred_name = CLASS_ID_TO_NAME[cls_pred]
            metrics[f"cls_confusion_{true_name}_pred_{pred_name}"] = int(cm[i, j])

    metrics["cls_total_samples"] = int(total)
    return metrics


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute classification metrics after discretising retrieval scores."""
    y_true_cls = _bucketise_scores(y_true)
    y_pred_cls = _bucketise_scores(y_pred)
    return _classification_metrics_from_labels(y_true_cls, y_pred_cls)


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the mean squared error (MSE) between ``y_true`` and ``y_pred``.

    The MSE is the average of the squared differences between actual and
    predicted values and penalises larger errors more severely than smaller
    ones.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the root mean squared error (RMSE) between ``y_true`` and
    ``y_pred``.

    RMSE is the square root of the mean squared error and has the same
    units as the target variable, making it more interpretable than MSE.
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the mean absolute error (MAE) between ``y_true`` and
    ``y_pred``.

    MAE is the average of the absolute differences between actual and
    predicted values and is more robust to outliers than MSE.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the coefficient of determination (R²) between ``y_true`` and
    ``y_pred``.

    The R² score measures how well the predictions approximate the true
    values relative to a trivial baseline that predicts the mean of
    ``y_true``.  An R² of 1.0 indicates perfect prediction, whereas 0.0
    indicates that the model performs no better than predicting the mean.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot != 0 else 0.0


def pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the Pearson correlation coefficient between ``y_true`` and
    ``y_pred``.

    The Pearson correlation measures the linear correlation between two
    variables, returning a value between -1 and 1.  It is often used to
    assess whether the model preserves the relative ordering of examples.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size < 2:
        return 0.0
    cov = np.cov(y_true, y_pred, ddof=0)
    stds = np.sqrt(np.diag(cov))
    return float(cov[0, 1] / (stds[0] * stds[1])) if stds[0] > 0 and stds[1] > 0 else 0.0


def spearman_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return Spearman's rank correlation coefficient between ``y_true`` and
    ``y_pred``.

    Spearman's rho measures the monotonic relationship between two
    variables by computing the Pearson correlation of the rank variables
    rather than the raw values.  This is useful when the absolute values
    matter less than their relative ordering.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # Obtain ranks with ties handled by averaging the ranks of tied elements
    def _ranks(x: np.ndarray) -> np.ndarray:
        # Compute ranks similar to scipy.stats.rankdata(method='average')
        temp = x.argsort()
        ranks = np.empty_like(temp, dtype=float)
        # Assign initial ranks based on sorted order
        ranks[temp] = np.arange(len(x))
        # Handle ties: average the rank positions for equal values
        _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
        # sum_ranks accumulates the sum of ranks for each unique value
        sum_ranks = np.bincount(inverse, weights=ranks)
        # average rank for each group
        avg_ranks = sum_ranks / counts
        return avg_ranks[inverse]

    rank_true = _ranks(y_true)
    rank_pred = _ranks(y_pred)
    return pearson_correlation(rank_true, rank_pred)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute a suite of regression metrics and return them in a dictionary.

    This convenience function bundles several useful metrics together.  If
    additional metrics are needed, they can be added here so that all
    evaluation remains centralised.
    """
    metrics = {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "pearson": pearson_correlation(y_true, y_pred),
        "spearman": spearman_correlation(y_true, y_pred),
    }
    metrics.update(classification_metrics(y_true, y_pred))
    return metrics