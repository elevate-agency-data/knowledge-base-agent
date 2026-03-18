"""
ADK tool: hybrid_query

Unified entry point for hybrid RAG retrieval.
Routes automatically:
- 1 index  → hybrid_rag_query  (single-index, full top_k)
- 2+ indexes → hybrid_multi_query (multi-index, top_k_per_index)

Index auto-resolution:
- If index_names is empty or omitted, all available indexes are listed and
  Gemini Flash selects the relevant ones based on the query (fallback: all).
"""

from hybrid.tools.hybrid_rag_query   import hybrid_rag_query
from hybrid.tools.hybrid_multi_query import hybrid_multi_query
from shared.query_rewriter           import rewrite_query


def hybrid_query(
    index_names: list[str],
    query: str,
    retrieval_mode: str = "hybrid",
    top_k: int = 10,
    filters: dict | None = None,
    context: str = "",
) -> dict:
    """
    Query one or more hybrid indexes. Routing is handled automatically.

    - If **one** index name is provided → single-index retrieval (fast, full top_k).
    - If **multiple** index names are provided → multi-index retrieval
      (top_k split equally across indexes).
    - If **no** index name is provided → all available indexes are listed and
      Gemini Flash resolves which ones are relevant (fallback: all indexes).

    Args:
        index_names:    List of index names to query. Pass an empty list or
                        omit to trigger automatic index resolution.
                        Examples: ["celio"]  |  ["celio", "fnac"]  |  []
        query:          Natural-language question or search string.
        retrieval_mode: ``"hybrid"`` (default), ``"dense"``, or ``"sparse"``.
        top_k:          Total number of chunks to return.
                        - Single index : returns up to ``top_k`` chunks.
                        - Multi  index : returns up to ``top_k // len(index_names)``
                          chunks per index.
        filters:        Optional metadata filters applied to every index.
        context:        Optional conversation history for query rewriting.

    Returns:
        Same structure as ``hybrid_rag_query`` (single) or
        ``hybrid_multi_query`` (multi), plus:
        - ``routing``          : ``"single"`` or ``"multi"``
        - ``original_query``   : raw query before rewriting
        - ``retrieval_query``  : rewritten query used for retrieval
        - ``indexes_resolved`` : list of index names actually queried
    """
    index_names = [n.strip().lower() for n in (index_names or []) if n.strip()]

    # ── Auto-resolve indexes when none are specified ──────────────────────────
    if not index_names:
        from hybrid.tools.hybrid_list_indexes import hybrid_list_indexes
        from shared.index_resolver            import resolve_indexes

        list_result = hybrid_list_indexes()
        available   = [idx["index_name"] for idx in list_result.get("indexes", [])]

        if not available:
            return {"status": "error", "message": "No hybrid indexes found."}

        index_names = resolve_indexes(query, available)

    retrieval_query = rewrite_query(query, context=context)

    if len(index_names) == 1:
        result = hybrid_rag_query(
            index_name=index_names[0],
            query=retrieval_query,
            retrieval_mode=retrieval_mode,
            top_k=top_k,
            filters=filters,
        )
        result["routing"]          = "single"
        result["original_query"]   = query
        result["retrieval_query"]  = retrieval_query
        result["indexes_resolved"] = index_names
        return result

    # Multiple indexes → multi-query
    top_k_per_index = max(1, top_k // len(index_names))
    result = hybrid_multi_query(
        index_names=index_names,
        query=retrieval_query,
        retrieval_mode=retrieval_mode,
        top_k_per_index=top_k_per_index,
        filters=filters,
    )
    result["routing"]          = "multi"
    result["original_query"]   = query
    result["retrieval_query"]  = retrieval_query
    result["indexes_resolved"] = index_names
    return result
