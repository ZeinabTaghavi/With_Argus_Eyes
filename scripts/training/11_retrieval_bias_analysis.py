"""
Retrieval Bias Analysis Script
------------------------------

This script measures how a trained retrieval‐quality predictor (risk score)
correlates with social bias dimensions across two datasets from the
Hugging Face Hub: **HateXplain** and **CrowS-Pairs**.  For a fixed
retrieval backend (e.g. ``contriever``) and a regression model trained
on Wikipedia entity spans (see ``utils/models/evaluating_models.py``),
the script performs the following high‑level steps:

1. **Load the datasets** using the `datasets` library.  For
   HateXplain we read all posts and compute a majority label
   (hatespeech/offensive vs. normal) along with the union of target
   communities annotated in each post.  For CrowS‑Pairs we iterate
   over minimal pairs, each annotated with a bias type.

2. **Load the retriever and regression model**.  The retriever is
   constructed via ``utils.embeddings.build_retriever`` and is used
   to encode sentences into fixed‑size embeddings.  The regression
   model and its associated scaler are loaded from a Joblib artefact
   produced by ``evaluate_regression_models``; the artefact must
   contain the keys ``'model'`` and ``'scaler'``.

3. **Compute risk scores** for every text sample in the datasets.
   Sentences are batched and passed through the retriever's
   ``encode_texts`` method.  If a scaler is provided, features are
   standardised before being fed to the model.  Predictions yield
   floating‑point risk scores in the interval ``[0, 1]``.

4. **Aggregate statistics**.  For HateXplain the script tracks
   statistics per target community (e.g. ``African``, ``Women``) and
   distinguishes between ``biased`` (hatespeech or offensive) and
   ``neutral`` (normal) posts.  It reports the average risk score,
   the number of examples, and the number of examples exceeding a
   threshold (default 0.5) for both classes.  For CrowS‑Pairs the
   script records per bias type the average risk for the
   stereotypical and anti‑stereotypical sentences, counts of
   examples, counts above the threshold, and the average difference
   (stereotypical minus anti‑stereotypical).

5. **Write results**.  All computed statistics are dumped to a JSON
   file for downstream analysis and visualisation.

Usage:

.. code:: bash

   python retrieval_bias_analysis.py \
       --retriever contriever \
       --model_path path/to/best_model.joblib \
       --output_path bias_stats.json \
       [--threshold 0.5]

The script assumes that the current working directory is the root of
the project repository so that imports from ``utils`` succeed.  If
necessary, you can adjust the Python path via the ``--repo_root``
argument.
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import joblib
import numpy as np
from tqdm import tqdm

try:
    # The datasets library is used to fetch HateXplain and CrowS‑Pairs.
    from datasets import load_dataset
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The `datasets` library is required to run this script. "
        "Please install it via pip (e.g. `pip install datasets`) before executing."
    ) from exc

import sys

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
workspace_root = project_root
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# Optional: set HF caches
BASE = "../../../../data/proj/zeinabtaghavi"
os.environ["HF_HOME"]           = BASE
os.environ["HF_HUB_CACHE"]      = f"{BASE}/hub"
os.environ["HF_DATASETS_CACHE"] = f"{BASE}/datasets"

print("HF_HOME:", os.environ.get("HF_HOME"))
print("HF_HUB_CACHE:", os.environ.get("HF_HUB_CACHE"))
print("HF_DATASETS_CACHE:", os.environ.get("HF_DATASETS_CACHE"))
print("TRANSFORMERS_CACHE:", os.environ.get("TRANSFORMERS_CACHE"))



def get_retriever(name: str):
    """Instantiate a retriever by name using the project's factory.

    Parameters
    ----------
    name : str
        One of the valid retriever identifiers accepted by
        ``utils.embeddings.factory.build_retriever`` (e.g. ``contriever``).

    Returns
    -------
    BaseRetriever
        A retriever capable of encoding texts via ``encode_texts``.
    """
    from with_argus_eyes.utils.embeddings.factory import build_retriever

    return build_retriever(name)


def load_regression_model(path: str):
    """Load a regression model and optional scaler from a Joblib / pickle file.

    This loader is intentionally tolerant of several artefact formats:

    1. The recommended format used by ``save_model_artifact``:
       ``{"model": <estimator>, "scaler": <transformer_or_None>}``.

    2. A variant with different key names, e.g.:
       ``{"regressor": ..., "transformer": ...}``.

    3. A raw estimator object saved directly, e.g.:
       ``joblib.dump(model, path)``,
       in which case no scaler is returned (``None``).

    Parameters
    ----------
    path : str
        Path to the ``.joblib`` / ``.pkl`` file saved by the training
        pipeline.

    Returns
    -------
    Tuple[object, object]
        A tuple ``(model, scaler)`` where ``model`` implements
        ``predict(X) -> np.ndarray`` and ``scaler`` implements
        ``transform(X) -> np.ndarray`` or is ``None``.
    """
    print(f"[Loading the regression model] Loading regression model from '{path}'...")
    artefact = joblib.load(path)

    # Case 1: dict with explicit 'model' / 'scaler' keys (preferred)
    if isinstance(artefact, dict):
        if "model" in artefact:
            model = artefact["model"]
            scaler = artefact.get("scaler")
            print(f"[Loading the regression model] Model and scaler loaded. Model: {type(model).__name__}, Scaler: {type(scaler).__name__ if scaler else 'None'}")
            return model, scaler

        # Case 2: alternative key names sometimes used in other pipelines
        # (e.g., 'regressor' / 'transformer').
        if "regressor" in artefact or "transformer" in artefact:
            model = artefact.get("model", artefact.get("regressor"))
            scaler = artefact.get("scaler", artefact.get("transformer"))
            if model is None:
                raise ValueError(
                    f"Could not identify model in artefact at {path!r} "
                    f"(keys: {list(artefact.keys())})"
                )
            print(f"[Loading the regression model] Model and scaler loaded. Model: {type(model).__name__}, Scaler: {type(scaler).__name__ if scaler else 'None'}")
            return model, scaler

        raise ValueError(
            f"Unsupported artefact dict format in {path!r}; expected keys "
            f"like 'model'/'scaler' or 'regressor'/'transformer', "
            f"found keys: {list(artefact.keys())}"
        )

    # Case 3: direct estimator object (no scaler stored)
    # e.g., joblib.dump(model, path)
    model = artefact
    scaler = None
    print(f"[Loading the regression model] Model loaded. Model: {type(model).__name__}, Scaler: None")
    return model, scaler


def encode_batch(
    retriever, texts: List[str], *, batch_size: int = 32
) -> np.ndarray:
    """Encode a list of texts into a 2‑D numpy array of embeddings.

    This helper wraps the retriever's ``encode_texts`` method and
    concatenates the resulting list of vectors into a single
    ``float32`` array.

    Parameters
    ----------
    retriever : BaseRetriever
        The underlying model used to produce embeddings.
    texts : List[str]
        A list of raw strings to encode.
    batch_size : int, default 32
        How many sentences to encode per call to ``encode_texts``.

    Returns
    -------
    np.ndarray
        An array of shape ``(len(texts), D)`` where ``D`` is the
        embedding dimension.
    """
    print(f"[Encoding batch] Encoding {len(texts)} texts using retriever '{type(retriever).__name__}'")
    embeddings: List[np.ndarray] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding batch"):
        batch = texts[start : start + batch_size]
        vectors = retriever.encode_texts(batch, batch_size=len(batch))
        embeddings.extend(vectors)
    return np.asarray(embeddings, dtype=np.float32)


def predict_scores(
    retriever, model, scaler, sentences: List[str], *, batch_size: int = 32
) -> np.ndarray:
    """Compute risk scores for a list of sentences.

    Parameters
    ----------
    retriever : BaseRetriever
        The embedding backend used to convert sentences to feature
        vectors.
    model : object
        The regression model with a ``predict`` method accepting
        ``(N, D)`` arrays.
    scaler : object or None
        Optional scaler with a ``transform`` method to standardise
        features.  If ``None`` no scaling is applied.
    sentences : List[str]
        Raw text sentences to score.
    batch_size : int, default 32
        Number of sentences per encoding call.

    Returns
    -------
    np.ndarray
        One risk score per input sentence.
    """
    print(f"[Computing risk scores] Computing risk scores for {len(sentences)} sentences...")
    X = encode_batch(retriever, sentences, batch_size=batch_size)
    if scaler is not None:
        print(f"[Computing risk scores] Applying scaler '{type(scaler).__name__}' to features...")
        X = scaler.transform(X)
    y_pred = model.predict(X)
    # Ensure output is a 1‑D array of floats
    print(f"[Computing risk scores] Finished risk score computation for batch of size {len(sentences)}.")
    return np.asarray(y_pred, dtype=np.float32).flatten()



def analyze_hatexplain(
    retriever, model, scaler, *, threshold: float = 0.5, split: str = 'train'
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute retrieval statistics over the HateXplain dataset.

    Each entry in HateXplain contains multiple annotations.  We take
    the majority vote over labels (0: hatespeech, 2: offensive, 1:
    normal) to classify a post as ``biased`` (hatespeech/offensive) or
    ``neutral`` (normal).  The union of all target communities
    annotated by the annotators is used to index into per‑group
    statistics.

    Parameters
    ----------
    retriever : BaseRetriever
        The encoder used to produce sentence embeddings.
    model : object
        The trained regression model.
    scaler : object or None
        The feature scaler used with the regression model.
    threshold : float, optional
        Risk score threshold for counting high‑risk examples.  Default
        is 0.5.
    split : str, optional
        Which split of HateXplain to use.  Should be one of
        ``'train'``, ``'validation'``, or ``'test'``.  Default is
        ``'train'``.

    Returns
    -------
    dict
        A nested dictionary ``{group: {subset: stats, ...}, ...}``
        where ``subset`` is one of ``'overall'``, ``'biased'`` or
        ``'neutral'``.  Each ``stats`` contains ``mean`` (average
        score), ``count`` (number of examples) and ``above_threshold``
        (how many exceed ``threshold``).
    """
    print(f"[Step 1: Loading dataset] Loading HateXplain split '{split}'...")
    dataset = load_dataset('Hate-speech-CNERG/hatexplain', split=split, trust_remote_code=True)

    texts: List[str] = []
    labels: List[int] = []
    targets: List[List[str]] = []
    # Preprocess entries
    print(f"[Step 1: Processing entries] Preprocessing {len(dataset)} HateXplain entries...")
    for entry in tqdm(dataset):
        # Majority vote among annotators
        anns = entry['annotators']
        ann_labels = [ann for ann in anns['label']]
        # Determine the most common label (ties resolved by lowest label value)
        label_counts = {}
        for lab in ann_labels:
            label_counts[lab] = label_counts.get(lab, 0) + 1
        majority_label = min(
            (-count, lab) for lab, count in label_counts.items()
        )[1]
        labels.append(majority_label)
        # Collect union of target communities for this post
        tg = set()
        for ann in anns['target']:
            for tgt in ann:
                if tgt:
                    tg.add(tgt)
        if not tg:
            tg.add('None')
        targets.append(list(tg))
        # Reconstruct the post as a plain string
        tokens = entry.get('post_tokens')
        if tokens:
            text = ' '.join(tokens)
        else:
            # Fallback to raw 'post' field if present
            text = entry.get('post', '')
        texts.append(text)

    print(f"[Step 3: Computing risk scores] Computing risk scores for HateXplain split '{split}'...")
    scores = predict_scores(retriever, model, scaler, texts)

    # Aggregate statistics per group
    print(f"[Step 4: Aggregating statistics] Aggregating risk statistics for target communities in HateXplain...")
    group_stats: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
        lambda: {
            'overall': {'mean': 0.0, 'count': 0, 'above_threshold': 0},
            'biased': {'mean': 0.0, 'count': 0, 'above_threshold': 0},
            'neutral': {'mean': 0.0, 'count': 0, 'above_threshold': 0},
        }
    )


    for score, lbl, tgs in zip(scores, labels, targets):
        # Determine class type
        subset = 'biased' if lbl in (0, 2) else 'neutral'
        for group in tgs:
            for name in ('overall', subset):
                gs = group_stats[group][name]
                gs['count'] += 1
                gs['mean'] += float(score)
                if score > threshold:
                    gs['above_threshold'] += 1
    # Finalise means
    for group, parts in group_stats.items():
        for name, stats in parts.items():
            if stats['count'] > 0:
                stats['mean'] /= stats['count']
    print(f"[Step 4: Aggregating statistics] Aggregation complete for HateXplain.")

    return group_stats




def analyze_crows_pairs(
    retriever, model, scaler, *, threshold: float = 0.5, split: str = 'validation'
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute retrieval statistics over the CrowS‑Pairs dataset.

    For each example in CrowS‑Pairs, there are two sentences: one
    stereotypical (``sent_more``) and one anti‑stereotypical
    (``sent_less``).  Each example is associated with a ``bias_type``.
    This function computes per–bias type statistics for the two
    sentences individually and also for their difference.

    Parameters
    ----------
    retriever : BaseRetriever
        The encoder used to produce sentence embeddings.
    model : object
        The trained regression model.
    scaler : object or None
        The feature scaler used with the regression model.
    threshold : float, optional
        Risk score threshold for counting high‑risk examples.  Default
        is 0.5.
    split : str, optional
        Which split of CrowS‑Pairs to use.  Options are ``'train'``,
        ``'validation'`` or ``'test'``.  The default is ``'validation'``
        because the official dataset uses this as the primary set for
        bias analysis.

    Returns
    -------
    dict
        A nested dictionary ``{bias_type: {subset: stats, ...}, ...}``
        where ``subset`` is one of ``'stereotypical'``, ``'anti'`` or
        ``'difference'``.  Each ``stats`` contains ``mean`` (average
        score), ``count`` (number of examples), ``above_threshold``
        (for the first two subsets) and ``mean_diff`` (for the
        difference subset).
    """
    print(f"[Step 1: Loading dataset] Loading CrowS-Pairs split '{split}'...")
    dataset = load_dataset('nyu-mll/crows_pairs', split=split, trust_remote_code=True)
    bias_types: List[str] = []
    sent_more: List[str] = []
    sent_less: List[str] = []
    # Extract fields
    print(f"[Step 1: Processing entries] Preprocessing {len(dataset)} CrowS-Pairs entries...")
    for entry in dataset:
        bias_types.append(entry['bias_type'])
        sent_more.append(entry['sent_more'])
        sent_less.append(entry['sent_less'])
    print(f"[Step 3: Computing risk scores] Computing risk scores for CrowS-Pairs (stereotypical and anti) split '{split}'...")
    scores_more = predict_scores(retriever, model, scaler, sent_more)
    scores_less = predict_scores(retriever, model, scaler, sent_less)
    # Aggregate statistics
    print(f"[Step 4: Aggregating statistics] Aggregating risk statistics for bias types in CrowS-Pairs...")
    group_stats: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
        lambda: {
            'stereotypical': {'mean': 0.0, 'count': 0, 'above_threshold': 0},
            'anti': {'mean': 0.0, 'count': 0, 'above_threshold': 0},
            'difference': {'mean': 0.0, 'count': 0},
        }
    )
    for bt, s_more, s_less in zip(bias_types, scores_more, scores_less):
        stats = group_stats[bt]
        # stereotypical sentence
        stats['stereotypical']['count'] += 1
        stats['stereotypical']['mean'] += float(s_more)
        if s_more > threshold:
            stats['stereotypical']['above_threshold'] += 1
        # anti‑stereotypical sentence
        stats['anti']['count'] += 1
        stats['anti']['mean'] += float(s_less)
        if s_less > threshold:
            stats['anti']['above_threshold'] += 1
        # difference
        diff = float(s_more) - float(s_less)
        stats['difference']['count'] += 1
        stats['difference']['mean'] += diff
    # Finalise means
    for bt, parts in group_stats.items():
        for subset, stats in parts.items():
            if stats['count'] > 0:
                stats['mean'] /= stats['count']
    print(f"[Step 4: Aggregating statistics] Aggregation complete for CrowS-Pairs.")
    return group_stats



def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval bias across datasets.")
    parser.add_argument(
        '--retriever',
        type=str,
        required=True,
        help="Name of the embedding retriever (e.g. contriever, reasonir, qwen3, jina, etc.).",
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help="Path to the Joblib file containing the regression model and scaler.",
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='outputs/',
        help="Where to write the resulting JSON statistics (default: bias_statistics.json).",
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help="Threshold for counting high‑risk scores (default: 0.5).",
    )
    parser.add_argument(
        '--repo_root',
        type=str,
        default=None,
        help=(
            "Optional path to the repository root.  If provided, this directory "
            "will be added to sys.path so that imports under 'utils' work even when "
            "running the script from a different location."
        ),
    )
    parser.add_argument(
        '--hatexplain_split',
        type=str,
        default='test',
        choices=['train', 'validation', 'test'],
        help="Which split of HateXplain to analyse (default: train).",
    )
    parser.add_argument(
        '--crowspairs_split',
        type=str,
        default='test',
        choices=['train', 'validation', 'test'],
        help="Which split of CrowS‑Pairs to analyse (default: validation).",
    )
    args = parser.parse_args()
    # Optionally extend sys.path
    if args.repo_root:
        import sys as _sys
        if args.repo_root not in _sys.path:
            _sys.path.insert(0, args.repo_root)

    print("[Step 2: Loading retriever] Loading retriever ...")
    retriever = get_retriever(args.retriever)
    print(f"[Step 2: Loading retriever] Retriever '{args.retriever}' loaded of type '{type(retriever).__name__}'.")

    print("[Step 2: Loading regression model and scaler] Loading regression model and scaler ...")
    model, scaler = load_regression_model(args.model_path)

    print(f"[Step 1: Loading datasets and computing statistics] Starting HateXplain analysis for split '{args.hatexplain_split}' ...")
    hate_stats = analyze_hatexplain(
        retriever, model, scaler, threshold=args.threshold, split=args.hatexplain_split
    )

    print(f"[Step 1: Loading datasets and computing statistics] Starting CrowS-Pairs analysis for split '{args.crowspairs_split}' ...")
    crows_stats = analyze_crows_pairs(
        retriever, model, scaler, threshold=args.threshold, split=args.crowspairs_split
    )

    # Combine and write output
    print("[Step 5: Writing results] Saving results to disk ...")
    result = {
        'HateXplain': hate_stats,
        'CrowS_Pairs': crows_stats,
    }
    os.makedirs(os.path.join(args.output_path, '11_bias_statistics'), exist_ok=True)
    output_path = os.path.join(args.output_path, '11_bias_statistics', f'{args.retriever}.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[Step 5: Writing results] Finished. Results written to '{output_path}'.")


if __name__ == '__main__':
    main()