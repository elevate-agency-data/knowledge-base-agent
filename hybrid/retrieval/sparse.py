"""
Sparse (BM25 / full-text) retrieval.

Delegates to the store's native full-text search and normalises raw BM25
scores to [0, 1] for downstream fusion with dense results.
"""

from hybrid.stores.base import BaseStore


def sparse_search(
    query: str,
    store: BaseStore,
    index_name: str = "",
    top_k: int = 10,
    filters: dict | None = None,
) -> list[dict]:
    """
    Retrieve the most keyword-relevant chunks for *query* via BM25.

    Flow:
    1. Forward the raw query string to ``store.sparse_search()``.
    2. Normalise raw BM25 scores to [0, 1].

    Args:
        query:      Raw query string (no preprocessing needed).
        store:      Initialised store backend.
        index_name: Index to search (routes to dedicated table).
        top_k:      Number of results to return.
        filters:    Optional metadata filters (langue, domaine, …).

    Returns:
        List of chunk dicts, each with a ``"score"`` key in [0, 1].
        Sorted by score descending.
    """
    filters = filters or {}

    # 1. BM25 search via the store backend — routed to index_name's table
    results = store.sparse_search(
        index_name=index_name,
        query=query,
        top_k=top_k,
        filters=filters,
    )

    # 2. Normalise BM25 scores to [0, 1]
    results = _normalise_scores(results)

    return results


def _normalise_scores(results: list[dict]) -> list[dict]:
    """
    Linearly normalise the ``"score"`` field across *results* to [0, 1].

    BM25 scores are non-negative but unbounded; this makes them comparable
    with the cosine similarity scores from dense retrieval.

    Args:
        results: List of chunk dicts with a ``"score"`` key.

    Returns:
        Same list with normalised ``"score"`` values.
    """
    if not results:
        return results

    scores = [r.get("score", 0.0) for r in results]
    min_s = min(scores)
    max_s = max(scores)

    if max_s == min_s:
        for r in results:
            r["score"] = 1.0 if max_s > 0 else 0.0
        return results

    for r in results:
        r["score"] = (r.get("score", 0.0) - min_s) / (max_s - min_s)

    return results
