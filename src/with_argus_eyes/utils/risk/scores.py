from __future__ import annotations
from typing import Dict, List, Callable, Optional, Union
import numpy as np


def _extract_ranks(item: dict, phrase_type: str) -> List[int]:
    stats = item.get(f"{phrase_type}_related_tags_ranks", {}) or {}
    arr = list(stats.values())
    if len(arr) == 0:
        raise ValueError(f"No ranks found for {phrase_type} in {item}")
    # keep only valid ints
    vals = []
    for v in arr:
        try:
            vals.append(int(v))
        except Exception:
            pass
    return vals


def _extract_unrelevant_ranks(item: dict, order: Union[str, int]) -> List[int]:
    """Return ranks stored under rank_among_unrelevant_tags_{order}."""
    if order is None:
        raise ValueError("order must be provided to extract unrelevant ranks")
    order_str = str(order)
    key = f"rank_among_unrelevant_tags_{order_str}"
    raw = item[key]

 
    return raw


def normalized_score(x: List[float], *, method: str, context: Optional[Dict[str, float]] = None) -> List[float]:
    """
    method in {"none","max","zscore","minmax"}.

    If `context` is provided, it overrides statistics as follows:
      - "max":     expects {"max_possible": float}
      - "zscore":  expects {"mean": float, "std": float}
      - "minmax":  expects {"min": float, "max": float}
    """
    m = method.lower()
    ctx = context or {}

    if m == "none":
        return [float(xi) for xi in x]

    if m == "max":
        denom = float(ctx.get("max_possible", 1.0))
        return [float(xi) / denom if denom > 0 else float(xi) for xi in x]

    if m == "zscore":
        if "mean" in ctx and "std" in ctx:
            mu = float(ctx["mean"])
            sd = float(ctx["std"])
        else:
            mu = float(np.mean(x))
            sd = float(np.std(x))
        sd = sd if sd > 1e-12 else 1.0
        return [(float(xi) - mu) / sd for xi in x]

    if m == "minmax":
        if "min" in ctx and "max" in ctx:
            mn = float(ctx["min"])
            mx = float(ctx["max"])
        else:
            mn = float(np.min(x))
            mx = float(np.max(x))
        if mx <= mn + 1e-12:  # avoid division by zero
            return [0.0 for _ in x]
        return [(float(xi) - mn) / (mx - mn) for xi in x]

    raise ValueError(f"Unknown normalization method: {method}")


def risk_worst_rank(item: dict, phrase_type: str) -> float:
    """Risk = max(rank)."""
    ranks = _extract_ranks(item, phrase_type)
    raw = max(ranks) if ranks else 0.0
    return raw


def risk_mean_rank(item: dict, phrase_type: str) -> float:
    """Risk = mean(rank)."""
    ranks = _extract_ranks(item, phrase_type)
    raw = float(np.mean(ranks)) if ranks else 0.0
    return raw


def risk_percentile_rank(item: dict, phrase_type: str, *, p: float = 0.9) -> float:
    """Risk = percentile(rank, p). Example: p=0.9 (90th percentile)."""
    ranks = _extract_ranks(item, phrase_type)
    raw = float(np.percentile(ranks, 100 * p)) if ranks else 0.0
    return raw


def risk_minmax_combo(item: dict, phrase_type: str, *, alpha: float = 0.7) -> float:
    """Risk = alpha * max(rank) + (1-alpha) * mean(rank)."""
    ranks = _extract_ranks(item, phrase_type)
    if not ranks:
        raw = 0.0
    else:
        raw = alpha * max(ranks) + (1.0 - alpha) * float(np.mean(ranks))
    return raw


def risk_dev_log_num_tags(item: dict, phrase_type: str) -> float:
    """Risk ~ decreasing function of #tags: 1 / log(#tags + 1)."""
    ranks = _extract_ranks(item, phrase_type)
    if not ranks:
        raw = 1.0
    else:
        raw = 1.0 / np.log(len(ranks) + 1)
    return raw


def risk_dev_num_tags_inverse_power(item: dict, phrase_type: str, *, power: float = 0.5) -> float:
    """Risk ~ decreasing function of #tags: 1 / ( #tags + 1 )^power."""
    ranks = _extract_ranks(item, phrase_type)
    if not ranks:
        raw = 1.0
    else:
        raw = 1.0 / ((len(ranks) + 1) ** power)
    return raw


def risk_combined(
    item: dict,
    phrase_type: str,
    *,
    risk_names: List[str] = ["mean", "dev_num_tags_inverse_power"],
    risk_weights: List[float] = [0.5, 0.5],
) -> float:
    """
    Risk = sum_i ( risk_weights[i] * risk_i_raw ).
    Sub-risk components are computed raw.
    """
    if len(risk_weights) != len(risk_names):
        raise ValueError("risk_weights and risk_names must have the same length")

    # raw components
    risk_scores = [get_risk_fn(name)(item, phrase_type) for name in risk_names]
    raw_sum = sum(risk_weights[i] * risk_scores[i] for i in range(len(risk_names)))
    return raw_sum

def anti_risk_portion_below_k(item: dict, phrase_type: str, *, k: int = 1000) -> float:
    """
    Risk = proportion of tags whose rank is strictly less than k.
    Returns a value in [0, 1]. If no ranks are present, returns 0.0.
    """
    ranks = _extract_ranks(item, phrase_type)
    if not ranks:
        return 0.0
    count = sum(1 for r in ranks if int(r) < int(k))
    return float(count) / float(len(ranks))

def risk_exists_in_top_k(item: dict, phrase_type: str, *, k: int = 1000) -> float:
    """
    Risk = (top-rank)/k if the item exists in the top k tags, 1 otherwise.
    Returns a value in [0, 1]. If no ranks are present, returns 1.0.
    """
    ranks = _extract_ranks(item, phrase_type)
    if not ranks:
        return 1.0
    return float(min(ranks)) / float(k) if ranks[0] <= k else 1.0


def risk_ratio_unrelevant_below_k(
    item: dict,
    *,
    order: Union[str, int],
    k: int = 50,
) -> float:
    """Risk = fraction of unrelevant-tag ranks strictly below k.

    Ranks are read from ``rank_among_unrelevant_tags_{order}`` on the item.
    Returns a value in ``[0, 1]``. If no ranks are present, returns ``0.0``.
    """

    ranks = _extract_unrelevant_ranks(item, order)
    if not ranks:
        return 0.0
    cutoff = int(k)
    count = sum(1 for r in ranks if int(r) < cutoff)
    return float(count) / float(len(ranks))


# ---------------- Registry ----------------
_RISK_REGISTRY: Dict[str, Callable[..., float]] = {
    "worst": risk_worst_rank,
    "mean": risk_mean_rank,
    "pctl": risk_percentile_rank,     # requires kwarg p
    "minmax": risk_minmax_combo,      # requires kwarg alpha
    "dev_log_num_tags": risk_dev_log_num_tags,
    "dev_num_tags_inverse_power": risk_dev_num_tags_inverse_power,
    "anti_portion_below_k": anti_risk_portion_below_k,    # requires kwarg k
    "exists_in_top_k": risk_exists_in_top_k,    # requires kwarg k
    "ratio_unrelevant_below_k": risk_ratio_unrelevant_below_k,  # requires kwarg order, optional k
    "combined": risk_combined,
}


def get_risk_fn(name: str) -> Callable[..., float]:
    key = name.strip().lower()
    if key not in _RISK_REGISTRY:
        raise ValueError(f"Unknown risk function: {name}. Options: {list(_RISK_REGISTRY)}")
    return _RISK_REGISTRY[key]