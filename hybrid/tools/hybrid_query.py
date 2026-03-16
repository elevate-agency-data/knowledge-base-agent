"""
ADK tool: hybrid_query

Unified entry point for hybrid RAG retrieval.
Routes automatically:
- 1 index  → hybrid_rag_query  (single-index, full top_k)
- 2+ indexes → hybrid_multi_query (multi-index, top_k_per_index)

No LLM inference needed for routing — decision is purely code-based.
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

    Args:
        index_names:    List of index names to query. Provide a single-element
                        list for a single-index query, or multiple names for a
                        cross-index query.
                        Examples: ["celio"]  |  ["celio", "fnac", "aldi"]
        query:          Natural-language question or search string.
        retrieval_mode: ``"hybrid"`` (default), ``"dense"``, or ``"sparse"``.
        top_k:          Total number of chunks to return.
                        - Single index : returns up to ``top_k`` chunks.
                        - Multi  index : returns up to ``top_k // len(index_names)``
                          chunks per index.
        filters:        Optional metadata filters applied to every index.

    Returns:
        Same structure as ``hybrid_rag_query`` (single) or
        ``hybrid_multi_query`` (multi), plus a ``routing`` key indicating
        which path was taken (``"single"`` or ``"multi"``).
    """
    index_names = [n.strip().lower() for n in (index_names or []) if n.strip()]

    if not index_names:
        return {"status": "error", "message": "index_names must not be empty."}

    retrieval_query = rewrite_query(query, context=context)

    if len(index_names) == 1:
        result = hybrid_rag_query(
            index_name=index_names[0],
            query=retrieval_query,
            retrieval_mode=retrieval_mode,
            top_k=top_k,
            filters=filters,
        )
        result["routing"] = "single"
        result["original_query"] = query
        result["retrieval_query"] = retrieval_query
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
    result["routing"] = "multi"
    result["original_query"] = query
    result["retrieval_query"] = retrieval_query
    return result
