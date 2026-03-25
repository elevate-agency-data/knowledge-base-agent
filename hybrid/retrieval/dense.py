"""
Dense (vector) retrieval.

Embeds the query with the given model, queries the store, and normalises
cosine similarity scores to [0, 1] for downstream fusion.
"""

from hybrid.stores.base import BaseStore
from hybrid.embeddings.base import BaseEmbedding


def dense_search(
    query: str,
    store: BaseStore,
    embedding_model: BaseEmbedding,
    index_name: str = "",
    top_k: int = 10,
    filters: dict | None = None,
) -> list[dict]:
    """
    Retrieve the most semantically similar chunks to *query*.

    Flow:
    1. Embed the query with *embedding_model*.
    2. Call ``store.dense_search()`` routed to *index_name*'s table.
    3. Normalise raw cosine scores to the [0, 1] range.

    Args:
        query:           Raw query string.
        store:           Initialised store backend.
        embedding_model: Model used to embed the query.
        index_name:      Index to search (routes to dedicated table).
        top_k:           Number of results to return.
        filters:         Optional metadata filters (langue, domaine, …).

    Returns:
        List of chunk dicts, each with a ``"score"`` key in [0, 1].
        Sorted by score descending.
    """
    filters = filters or {}

    # 1. Embed query
    query_vector = embedding_model.embed_query(query)

    # 2. Store search — routed to index_name's table, HNSW used at 100%
    # Scores are raw cosine similarities [0, 1] — no normalisation needed
    # since RRF fusion uses rank, not score value.
    return store.dense_search(
        index_name=index_name,
        embedding=query_vector,
        top_k=top_k,
        filters=filters,
    )


def _normalise_scores(results: list[dict]) -> list[dict]:
    """
    Linearly normalise the ``"score"`` field across *results* to [0, 1].

    If all scores are identical (or there is only one result), scores are
    left at their original value (or 1.0 for a single perfect match).

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
