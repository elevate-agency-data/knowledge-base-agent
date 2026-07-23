"""
ADK tool: hybrid_rag_query

Query a hybrid RAG index using dense, sparse, or combined (RRF) retrieval.
Returns the retrieved context chunks together with source file links, ready
for the LLM to synthesise an answer.
"""

from hybrid.config import (
    DEFAULT_EMBEDDING_MODEL,
    TOP_K,
    DENSE_WEIGHT,
    SPARSE_WEIGHT,
    RRF_K,
    RRF_DENSE_GATED,
    DEDUP_CONTENT,
)
from hybrid.stores import get_store
from hybrid.embeddings import get_embedding_model
from hybrid.retrieval.dense  import dense_search
from hybrid.retrieval.sparse import sparse_search
from hybrid.retrieval.fusion import reciprocal_rank_fusion, deduplicate_by_content
from hybrid.retrieval.filter import apply_post_filter


def hybrid_rag_query(
    index_name: str,
    query: str,
    retrieval_mode: str = "hybrid",
    top_k: int = TOP_K,
    filters: dict | None = None,
) -> dict:
    """
    Retrieve relevant document chunks and return context for answer generation.

    Supports three retrieval modes:
    - ``"hybrid"`` : dense + sparse → Reciprocal Rank Fusion (recommended)
    - ``"dense"``  : semantic vector similarity only
    - ``"sparse"`` : BM25 full-text search only

    Args:
        index_name:     The index to query (must have been created with
                        ``hybrid_create_index`` and populated with
                        ``hybrid_add_data``).
        query:          Natural-language question or search string.
        retrieval_mode: One of ``"hybrid"``, ``"dense"``, ``"sparse"``.
        top_k:          Number of chunks to return (default 10).
        filters:        Optional metadata filters dict.  Supported keys:
                        ``index_name``, ``file_type``, ``domaine``,
                        ``langue``, ``author``, ``date_from``, ``date_to``,
                        ``tags``.

    Returns:
        Dict with keys:
        - ``status``         : ``"success"`` or ``"error"``
        - ``query``          : Echo of the query
        - ``index_name``     : Echo of the index
        - ``retrieval_mode`` : Mode used
        - ``context``        : Concatenated text of the top chunks
                               (ready to be passed to an LLM)
        - ``chunks``         : List of scored chunk dicts (full metadata)
        - ``sources``        : Deduplicated list of source dicts with
                               ``file_name`` and ``source_url``
        - ``total_results``  : Number of chunks returned
    """
    index_name = index_name.strip().lower()
    if not index_name:
        return {"status": "error", "message": "index_name is required."}
    if not query or not query.strip():
        return {"status": "error", "message": "query must not be empty."}

    valid_modes = {"hybrid", "dense", "sparse"}
    if retrieval_mode not in valid_modes:
        return {
            "status":  "error",
            "message": (
                f"Unknown retrieval_mode '{retrieval_mode}'. "
                f"Supported: {sorted(valid_modes)}"
            ),
        }

    filters = filters or {}

    try:
        store = get_store()

        # ----------------------------------------------------------------
        # Dense path — embed query once, pass vector directly
        # ----------------------------------------------------------------
        dense_results: list[dict] = []
        if retrieval_mode in ("hybrid", "dense"):
            embedder = get_embedding_model(DEFAULT_EMBEDDING_MODEL)
            query_embedding = embedder.embed_query(query)
            dense_results = dense_search(
                query=query,
                store=store,
                embedding_model=embedder,
                index_name=index_name,
                top_k=top_k * 2,   # Retrieve more for fusion
                filters=filters,
                query_embedding=query_embedding,
            )

        # ----------------------------------------------------------------
        # Sparse path
        # ----------------------------------------------------------------
        sparse_results: list[dict] = []
        if retrieval_mode in ("hybrid", "sparse"):
            sparse_results = sparse_search(
                query=query,
                store=store,
                index_name=index_name,
                top_k=top_k * 2,
                filters=filters,
            )

        # ----------------------------------------------------------------
        # Merge / select final results
        # ----------------------------------------------------------------
        if retrieval_mode == "hybrid":
            results = reciprocal_rank_fusion(
                dense_results=dense_results,
                sparse_results=sparse_results,
                k=RRF_K,
                dense_weight=DENSE_WEIGHT,
                sparse_weight=SPARSE_WEIGHT,
                dense_gated=RRF_DENSE_GATED,
            )
        elif retrieval_mode == "dense":
            results = dense_results
        else:  # sparse
            results = sparse_results

        # Post-filter (handles tag filtering and any filter the store missed)
        results = apply_post_filter(results, filters)

        # Collapse the same passage duplicated across files
        if DEDUP_CONTENT:
            results = deduplicate_by_content(results)

        # Trim to top_k
        results = results[:top_k]

        # ----------------------------------------------------------------
        # Build response payload
        # ----------------------------------------------------------------
        context_parts: list[str] = []
        seen_sources: dict[str, dict] = {}

        for chunk in results:
            content = chunk.get("content", "")
            if content:
                context_parts.append(content)

            source_url = chunk.get("source_url", "")
            file_name  = chunk.get("file_name", "")
            if source_url and source_url not in seen_sources:
                seen_sources[source_url] = {
                    "file_name":  file_name,
                    "source_url": source_url,
                    "file_type":  chunk.get("file_type", ""),
                    "domaine":    chunk.get("domaine", ""),
                    "langue":     chunk.get("langue", ""),
                }

        context = "\n\n---\n\n".join(context_parts)
        sources = list(seen_sources.values())

        return {
            "status":          "success",
            "query":           query,
            "index_name":      index_name,
            "retrieval_mode":  retrieval_mode,
            "context":         context,
            "chunks":          [_sanitize(c) for c in results],
            "sources":         sources,
            "total_results":   len(results),
        }

    except Exception as exc:
        return {
            "status":  "error",
            "message": f"Query failed: {exc}",
            "query":   query,
        }


def _sanitize(chunk: dict) -> dict:
    """Convert non-JSON-serializable values (date, bytes, etc.) to strings."""
    import datetime
    result = {}
    for k, v in chunk.items():
        if isinstance(v, (datetime.date, datetime.datetime)):
            result[k] = v.isoformat()
        elif isinstance(v, bytes):
            result[k] = v.decode("utf-8", errors="replace")
        elif isinstance(v, float) and (v != v):  # NaN check
            result[k] = None
        else:
            result[k] = v
    return result
