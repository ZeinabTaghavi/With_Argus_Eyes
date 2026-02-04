import numpy as np
from sklearn.model_selection import train_test_split
from typing import Optional, Tuple, List


def split_data(X, y, 
    test_size, random_state, stratify):

    idx = np.arange(len(X))
    return train_test_split(X, y, idx, test_size=test_size, random_state=random_state, stratify=stratify)


def sample_by_histogram_bins_fn(
    X: np.ndarray,
    y: np.ndarray,
    *,
    bins: int | str = 50,                 # e.g. 50, "auto", "fd", etc.
    count_threshold: float = 100,         # absolute count when sampling_mode="count"; fraction (0,1] when "percentage"
    sampling_mode: str = "count",         # "count" | "average" | "percentage" | "equal" | "equal_tails"
    random_state: Optional[int] = 42,
    bin_edges: Optional[np.ndarray] = None,   # pass custom edges to reuse across calls
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Subsample by histogram bins over scores (y), with five modes:

      sampling_mode="count":
        - For each bin, sample up to `count_threshold` items uniformly at random.
        - This is a fixed cap per bin.

      sampling_mode="average":
        - Compute N = total number of items (len(y)).
        - Let B_nz = number of bins that have at least one item.
        - Set per-bin cap = ceil(N / B_nz).
        - For each non-empty bin, sample up to this cap (without replacement).

      sampling_mode="percentage":
        - Let M = max bin size across y (using the common binning).
        - For each bin, sample up to ceil(`count_threshold` * M) items.
          (e.g., count_threshold=0.2 → 20% of the largest bin size)

      sampling_mode="equal":
        - Requires `count_threshold` (pivot_bins) (int, 1..n_bins-1).
        - Include **all items from the first `count_threshold` bins** (lowest-score bins).
        - Let S = total count of items in those first bins.
        - From the remaining bins pooled together, sample approximately S items total (uniformly).

      sampling_mode="equal_tails":
        - Requires `count_threshold` (pivot_bins) (int, 1..n_bins-1).
        - Build two tails using `pivot_bins`:
            * LOW tail  = first  `pivot_bins` bins (lowest scores)
            * HIGH tail = last   `pivot_bins` bins (highest scores)
        - Let S_low  = size of LOW tail, S_high = size of HIGH tail.
        - Let k = min(S_low, S_high).
        - Randomly sample **k** items from LOW and **k** items from HIGH (without replacement),
          concatenate, shuffle, and return. (Perfectly balances both sides.)
        - If k == 0 (one tail empty), returns empty selections.

    Common steps:
      - Build bin edges over y (or use provided `bin_edges`).
      - Randomly select within bins without replacement.
      - Last bin is right-inclusive.

    Returns:
      X_sub, y_sub
    """
    # Ensure arrays
    X = np.asarray(X); y = np.asarray(y).ravel()

    # Basic checks
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y length mismatch: {X.shape[0]} vs {y.shape[0]}")
    if not np.all(np.isfinite(y)):
        raise ValueError("y contain non-finite values (NaN/Inf).")

    # Build bin edges if not provided
    if bin_edges is None:
        smin, smax = float(np.min(y)), float(np.max(y))
        if smax <= smin + 1e-12:
            smax = smin + 1.0
        if isinstance(bins, int):
            bin_edges = np.linspace(smin, smax, bins + 1)
        else:
            bin_edges = np.histogram_bin_edges(y, bins=bins)
    bin_edges = np.asarray(bin_edges)
    n_bins = bin_edges.size - 1
    if n_bins <= 0:
        raise ValueError("Invalid bin_edges computed (need at least 1 bin).")

    sampling_mode = str(sampling_mode).lower().strip()
    if sampling_mode not in ("count", "average", "percentage", "equal", "equal_tails"):
        raise ValueError(
            f"sampling_mode must be 'count', 'average', 'percentage', 'equal', or 'equal_tails', "
            f"got {sampling_mode!r}"
        )

    rng = np.random.default_rng(random_state)

    # Precompute indices for all bins once (used by all modes)
    bin_indices = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            mask = (y >= lo) & (y < hi)
        else:
            mask = (y >= lo) & (y <= hi)
        bin_idx = np.flatnonzero(mask)
        bin_indices.append(bin_idx)

    # --- "equal_tails" mode: balance low vs high sides ---
    if sampling_mode == "equal_tails":
        if count_threshold is None:
            raise ValueError("sampling_mode='equal_tails' requires count_threshold (pivot bins).")
        pivot_bins = int(count_threshold)
        if not (1 <= pivot_bins <= n_bins - 1):
            raise ValueError(f"pivot_bins must be in [1, {n_bins-1}], got {pivot_bins!r}")

        # Define low and high tails
        low_bins = list(range(0, pivot_bins))
        high_bins = list(range(n_bins - pivot_bins, n_bins))

        low_pool = np.concatenate([bin_indices[i] for i in low_bins]) if low_bins else np.empty((0,), dtype=int)
        high_pool = np.concatenate([bin_indices[i] for i in high_bins]) if high_bins else np.empty((0,), dtype=int)

        # Balance the two sides: same number from each tail
        k_bal = min(low_pool.size, high_pool.size)
        if k_bal <= 0:
            return X[:0], y[:0]

        low_sel = rng.choice(low_pool, size=k_bal, replace=False)
        high_sel = rng.choice(high_pool, size=k_bal, replace=False)

        selected_idx = np.concatenate([low_sel, high_sel])
        selected_idx = rng.permutation(selected_idx)

        X_sub = X[selected_idx]
        y_sub = y[selected_idx]
        return X_sub, y_sub

    # --- "equal" mode: include all low bins; match from the rest ---
    if sampling_mode == "equal":
        if count_threshold is None:
            raise ValueError("sampling_mode='equal' requires count_threshold (pivot bins).")
        pivot_bins = int(count_threshold)
        if not (1 <= pivot_bins <= n_bins - 1):
            raise ValueError(f"pivot_bins must be in [1, {n_bins-1}], got {pivot_bins!r}")

        # All items from the first pivot bins
        first_bins_total = sum(int(len(bin_indices[i])) for i in range(pivot_bins))

        # Remaining bins and their pooled indices
        remaining_bins = list(range(pivot_bins, n_bins))
        rem_pool = np.concatenate([bin_indices[i] for i in remaining_bins]) if remaining_bins else np.empty((0,), dtype=int)

        # If no remaining bins or no items to match, just return the first part (could be empty)
        if rem_pool.size == 0 or first_bins_total == 0:
            selected_idx = np.concatenate([bin_indices[i] for i in range(pivot_bins)]) if first_bins_total > 0 else np.empty((0,), dtype=int)
        else:
            k_rem = min(first_bins_total, rem_pool.size)
            chosen_rem = rng.choice(rem_pool, size=k_rem, replace=False)

            selected_idx_parts = []
            # include all from the first pivot bins
            for i in range(pivot_bins):
                if bin_indices[i].size > 0:
                    selected_idx_parts.append(bin_indices[i])

            # add the uniformly sampled items from the remaining bins' pool
            selected_idx_parts.append(chosen_rem)

            selected_idx = np.concatenate(selected_idx_parts) if selected_idx_parts else np.empty((0,), dtype=int)

        # Shuffle to avoid ordering bias
        if selected_idx.size:
            selected_idx = rng.permutation(selected_idx)

        X_sub = X[selected_idx] if selected_idx.size else X[:0]
        y_sub = y[selected_idx] if selected_idx.size else y[:0]
        return X_sub, y_sub

    # --- "count", "average", and "percentage" modes ---
    if sampling_mode == "count":
        # User-specified fixed cap per bin:
        # each non-empty bin will contribute up to `count_threshold` items (without replacement).
        per_bin_cap = int(count_threshold)
        if per_bin_cap < 0:
            raise ValueError("count_threshold must be non-negative for 'count' mode.")

    elif sampling_mode == "average":
        # Automatically derive a per-bin cap based on the average number of items
        # over NON-ZERO bins:
        #   N      = total number of items
        #   B_nz   = number of bins with at least one item
        #   cap    = ceil(N / B_nz)
        # So if N=100000 and B_nz=25, cap=4000 and each non-empty bin
        # will contribute up to 4000 items.
        N_total = int(y.shape[0])
        non_empty_counts = np.array([len(bi) for bi in bin_indices], dtype=int)
        non_zero_mask = non_empty_counts > 0
        B_nz = int(non_zero_mask.sum())
        if B_nz <= 0:
            raise ValueError("No non-empty bins found for 'average' mode.")
        per_bin_cap = int(np.ceil(N_total / float(B_nz)))
    else:
        # "percentage" mode: derive M = max bin size (over all bins)
        # and use per-bin cap = ceil(count_threshold * M).
        h_all = np.array([len(bi) for bi in bin_indices], dtype=int)
        max_bin_cnt = int(h_all.max()) if h_all.size else 0
        if not (0 < float(count_threshold) <= 1.0):
            raise ValueError("For 'percentage' mode, count_threshold must be in (0, 1].")
        per_bin_cap = int(np.ceil(float(count_threshold) * max_bin_cnt)) if max_bin_cnt > 0 else 0

    idx_all = []
    for i in range(n_bins):
        bi = bin_indices[i]
        if bi.size == 0:
            continue
        k = min(per_bin_cap, bi.size)
        if k <= 0:
            raise ValueError(f"bin {i} size is 0, k: {k}")
        chosen = rng.choice(bi, size=k, replace=False)
        assert len(chosen) == k, f"chosen: {len(chosen)}, k: {k}"
        idx_all.append(chosen)

    if not idx_all:
        return X[:0], y[:0]

    idx_sel = np.concatenate(idx_all)
    idx_sel = rng.permutation(idx_sel)

    X_sub = X[idx_sel] if idx_sel.size else X[:0]
    y_sub = y[idx_sel] if idx_sel.size else y[:0]

    return X_sub, y_sub


def select_extreme_risk_samples(
    X: np.ndarray,
    y: np.ndarray,
    kept: List[dict],
    *,
    k: int = 10,
    frac: float = 0.10,
    index_map: Optional[np.ndarray] = None,
    phrase_type: str = "default",
    random_state: Optional[int] = 123,
) -> Tuple[
    List[str], np.ndarray, np.ndarray, List[dict],
    List[str], np.ndarray, np.ndarray, List[dict]
]:
    """
    Randomly select up to `k` samples from the lowest `frac` and highest `1-frac` tails
    of the risk-score distribution, and return labels/embeddings/scores plus the
    per-item tag-ranks for the given `phrase_type`.

    Args:
      X:           (N, d) embeddings aligned with `y` (local split matrix if using a split).
      y:           (N,)   risk scores aligned with X (local split vector if using a split).
      kept:        list of original item dicts (global list from which X/y were built).
      k:           number of samples to draw from EACH tail (min(k, available)).
      frac:        tail fraction (e.g., 0.10 → bottom 10% and top 10%).
      index_map:   optional array mapping local indices in (X,y) to global indices in `kept`.
                   - For global selection: leave as None (local==global).
                   - For train/test splits: pass i_tr / i_te so labels map to `kept`.
      phrase_type: which phrase field to read ranks from; expects keys like
                   f"{phrase_type}_related_tags_ranks" in each kept item.
      random_state: RNG seed.

    Returns:
      low_labels,  low_embeds,  low_scores,  low_tag_ranks,
      high_labels, high_embeds, high_scores, high_tag_ranks
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y length mismatch: {X.shape[0]} vs {y.shape[0]}")
    if not np.all(np.isfinite(y)):
        raise ValueError("y contains non-finite values (NaN/Inf).")
    if not (0.0 < frac < 0.5):
        raise ValueError("`frac` must be in (0, 0.5).")

    rng = np.random.default_rng(random_state)
    y = y.ravel()

    q_low, q_high = np.quantile(y, [frac, 1.0 - frac])
    low_cand  = np.flatnonzero(y <= q_low)
    high_cand = np.flatnonzero(y >= q_high)

    # Sample without replacement (handle if fewer than k available)
    k_low  = min(k, low_cand.size)
    k_high = min(k, high_cand.size)

    low_sel  = rng.choice(low_cand,  size=k_low,  replace=False) if k_low  > 0 else np.array([], dtype=int)
    high_sel = rng.choice(high_cand, size=k_high, replace=False) if k_high > 0 else np.array([], dtype=int)

    ranks_key = f"{phrase_type}_related_tags_ranks"

    def _labels_embeds_scores_ranks(local_idx: np.ndarray):
        labels, embeds, scores, ranks_list = [], [], [], []
        for li in local_idx:
            gi = int(index_map[li]) if index_map is not None else int(li)  # global index in `kept`
            item = kept[gi] if 0 <= gi < len(kept) else {}
            label = item.get("label") or item.get("item_label")
            ranks = item.get(ranks_key) or {}
            labels.append(label if label is not None else "")
            embeds.append(X[li])
            scores.append(y[li])
            ranks_list.append(ranks)
        if embeds:
            embeds_arr = np.stack(embeds, axis=0)
            scores_arr = np.asarray(scores, dtype=float)
        else:
            embeds_arr = X[:0]
            scores_arr = y[:0]
        return labels, embeds_arr, scores_arr, ranks_list

    low_labels,  low_embeds,  low_scores,  low_tag_ranks  = _labels_embeds_scores_ranks(low_sel)
    high_labels, high_embeds, high_scores, high_tag_ranks = _labels_embeds_scores_ranks(high_sel)

    return low_labels, low_embeds, low_scores, low_tag_ranks, high_labels, high_embeds, high_scores, high_tag_ranks