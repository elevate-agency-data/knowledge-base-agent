"""
Information-retrieval evaluation metrics.

All functions accept:
- ``results``      : Ordered list of retrieved items (chunk dicts or IDs).
                     Items are matched against ``relevant_ids`` using the
                     ``"id"`` key if the item is a dict, or the item itself
                     if it is a string.
- ``relevant_ids`` : Set (or list) of ground-truth relevant item IDs.
- ``k``            : Cut-off rank (only the first ``k`` results are considered).
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_id(item) -> str:
    """Extract the string ID from a chunk dict or a plain string."""
    if isinstance(item, dict):
        return str(item.get("id", ""))
    return str(item)


def _is_relevant(item, relevant_set: set) -> bool:
    """Return True if the item is in the relevant set."""
    return _get_id(item) in relevant_set


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------


def mean_reciprocal_rank(
    results: list,
    relevant_ids: list[str],
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR) over a single query result list.

    MRR = 1 / rank_of_first_relevant_item.
    Returns 0.0 if no relevant item is found.

    Args:
        results:      Ordered list of retrieved items.
        relevant_ids: Ground-truth relevant item IDs.

    Returns:
        MRR value in [0, 1].
    """
    relevant_set = set(relevant_ids)
    for rank, item in enumerate(results, start=1):
        if _is_relevant(item, relevant_set):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    results: list,
    relevant_ids: list[str],
    k: int,
) -> float:
    """
    Compute Normalised Discounted Cumulative Gain at rank *k* (nDCG@k).

    Uses binary relevance (1 if relevant, 0 otherwise).

    Args:
        results:      Ordered list of retrieved items.
        relevant_ids: Ground-truth relevant item IDs.
        k:            Rank cut-off.

    Returns:
        nDCG@k value in [0, 1].
    """
    relevant_set = set(relevant_ids)
    top_k = results[:k]

    # Actual DCG
    dcg = 0.0
    for rank, item in enumerate(top_k, start=1):
        if _is_relevant(item, relevant_set):
            dcg += 1.0 / math.log2(rank + 1)

    # Ideal DCG: best possible DCG given the number of relevant items
    n_relevant = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_relevant))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def recall_at_k(
    results: list,
    relevant_ids: list[str],
    k: int,
) -> float:
    """
    Compute Recall at rank *k* (R@k).

    R@k = |relevant ∩ top-k| / |relevant|.
    Returns 0.0 if ``relevant_ids`` is empty.

    Args:
        results:      Ordered list of retrieved items.
        relevant_ids: Ground-truth relevant item IDs.
        k:            Rank cut-off.

    Returns:
        R@k value in [0, 1].
    """
    if not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    top_k_ids = {_get_id(item) for item in results[:k]}
    return len(relevant_set & top_k_ids) / len(relevant_set)


def precision_at_k(
    results: list,
    relevant_ids: list[str],
    k: int,
) -> float:
    """
    Compute Precision at rank *k* (P@k).

    P@k = |relevant ∩ top-k| / k.
    Returns 0.0 if *k* is 0.

    Args:
        results:      Ordered list of retrieved items.
        relevant_ids: Ground-truth relevant item IDs.
        k:            Rank cut-off.

    Returns:
        P@k value in [0, 1].
    """
    if k == 0:
        return 0.0
    relevant_set = set(relevant_ids)
    top_k = results[:k]
    hits = sum(1 for item in top_k if _is_relevant(item, relevant_set))
    return hits / k


def compute_all_metrics(
    results: list,
    relevant_ids: list[str],
    k: int = 10,
) -> dict:
    """
    Compute MRR, nDCG@k, Recall@k, and Precision@k in one call.

    Args:
        results:      Ordered list of retrieved items.
        relevant_ids: Ground-truth relevant item IDs.
        k:            Rank cut-off for @k metrics.

    Returns:
        Dict with keys ``"mrr"``, ``"ndcg"``, ``"recall"``, ``"precision"``.
        All values are floats in [0, 1].
    """
    return {
        "mrr":       mean_reciprocal_rank(results, relevant_ids),
        "ndcg":      ndcg_at_k(results, relevant_ids, k),
        "recall":    recall_at_k(results, relevant_ids, k),
        "precision": precision_at_k(results, relevant_ids, k),
    }
