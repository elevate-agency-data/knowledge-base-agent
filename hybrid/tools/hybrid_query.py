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

import re
import time as _time

from hybrid.tools.hybrid_rag_query   import hybrid_rag_query
from hybrid.tools.hybrid_multi_query import hybrid_multi_query
from shared.query_rewriter           import rewrite_query

_DRIVE_URL_RE = re.compile(
    r"https?://(?:drive|docs)\.google\.com/\S+", re.IGNORECASE
)


def hybrid_query(
    index_names: list[str],
    query: str,
    retrieval_mode: str = "hybrid",
    top_k: int = 10,
    filters: dict | None = None,
    context: str = "",
    retrieval_query: str = "",
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
                        Examples: ["rh"]  |  ["rh", "marketing"]  |  []
        query:          Natural-language question or search string.
        retrieval_mode: ``"hybrid"`` (default), ``"dense"``, or ``"sparse"``.
        top_k:          Total number of chunks to return.
                        - Single index : returns up to ``top_k`` chunks.
                        - Multi  index : global cap across all indexes.
        filters:        Optional metadata filters applied to every index.
        context:        Optional conversation history for query rewriting.

    Returns:
        Same structure as ``hybrid_rag_query`` (single) or
        ``hybrid_multi_query`` (multi), plus:
        - ``routing``          : ``"single"`` or ``"multi"``
        - ``original_query``   : raw query before rewriting
        - ``retrieval_query``  : rewritten query used for retrieval
        - ``indexes_resolved`` : list of index names actually queried
        - ``timings``          : dict with per-step durations in seconds
    """
    t_total = _time.perf_counter()
    timings: dict[str, float] = {}

    index_names = [n.strip().lower() for n in (index_names or []) if n.strip()]

    # ── Validate & expand index names when explicitly provided ─────────────────
    if index_names:
        from hybrid.tools.hybrid_list_indexes import hybrid_list_indexes
        from shared.index_resolver import expand_company_indexes

        available = [i["index_name"] for i in hybrid_list_indexes().get("indexes", [])]
        # Expand bare company names: "celio" → ["celio__rh", "celio__commercial", ...]
        index_names = expand_company_indexes(index_names, available)
        unknown = [n for n in index_names if n not in available]
        if unknown:
            return {
                "status":  "error",
                "message": (
                    f"Index inconnu : {', '.join(unknown)}. "
                    f"Index disponibles : {', '.join(available) if available else 'aucun'}."
                ),
            }

    # ── Safety net: redirect Drive URLs to hybrid_find_similar ────────────────
    url_match = _DRIVE_URL_RE.search(query)
    if url_match:
        from hybrid.tools.hybrid_find_similar import hybrid_find_similar
        if not index_names:
            from hybrid.tools.hybrid_list_indexes import hybrid_list_indexes
            list_result = hybrid_list_indexes()
            index_names = [idx["index_name"] for idx in list_result.get("indexes", [])]
        return hybrid_find_similar(
            document_url=url_match.group(0),
            index_names=index_names,
        )

    # ── Auto-resolve indexes + rewrite query (parallel when possible) ────────
    needs_resolve = not index_names
    needs_rewrite = not retrieval_query

    if needs_resolve and needs_rewrite:
        # Both are independent LLM calls — run them in parallel to save ~3-5s
        from concurrent.futures import ThreadPoolExecutor
        from hybrid.tools.hybrid_list_indexes import hybrid_list_indexes
        from shared.index_resolver            import resolve_indexes

        list_result = hybrid_list_indexes()
        available   = [idx["index_name"] for idx in list_result.get("indexes", [])]
        if not available:
            return {"status": "error", "message": "No hybrid indexes found."}

        t0 = _time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as pool:
            future_resolve = pool.submit(resolve_indexes, query, available)
            future_rewrite = pool.submit(rewrite_query, query, context)
            index_names     = future_resolve.result()
            retrieval_query = future_rewrite.result()
        timings["resolve_and_rewrite"] = round(_time.perf_counter() - t0, 2)
    else:
        if needs_resolve:
            from hybrid.tools.hybrid_list_indexes import hybrid_list_indexes
            from shared.index_resolver            import resolve_indexes

            list_result = hybrid_list_indexes()
            available   = [idx["index_name"] for idx in list_result.get("indexes", [])]
            if not available:
                return {"status": "error", "message": "No hybrid indexes found."}
            t0 = _time.perf_counter()
            index_names = resolve_indexes(query, available)
            timings["resolve_indexes"] = round(_time.perf_counter() - t0, 2)

        if needs_rewrite:
            t0 = _time.perf_counter()
            retrieval_query = rewrite_query(query, context=context)
            timings["rewrite_query"] = round(_time.perf_counter() - t0, 2)

    if len(index_names) == 1:
        t0 = _time.perf_counter()
        result = hybrid_rag_query(
            index_name=index_names[0],
            query=retrieval_query,
            retrieval_mode=retrieval_mode,
            top_k=top_k,
            filters=filters,
        )
        timings["search"] = round(_time.perf_counter() - t0, 2)
        timings["total"] = round(_time.perf_counter() - t_total, 2)
        result["routing"]          = "single"
        result["original_query"]   = query
        result["retrieval_query"]  = retrieval_query
        result["indexes_resolved"] = index_names
        result["timings"]          = timings
        return result

    # Multiple indexes → multi-query with global top_k cap
    t0 = _time.perf_counter()
    result = hybrid_multi_query(
        index_names=index_names,
        query=retrieval_query,
        retrieval_mode=retrieval_mode,
        top_k=top_k,
        filters=filters,
    )
    timings["search"] = round(_time.perf_counter() - t0, 2)
    # Merge inner timings from multi_query (embedding, threads)
    inner_timings = result.pop("timings", {})
    timings.update(inner_timings)
    timings["total"] = round(_time.perf_counter() - t_total, 2)

    result["routing"]          = "multi"
    result["original_query"]   = query
    result["retrieval_query"]  = retrieval_query
    result["indexes_resolved"] = index_names
    result["timings"]          = timings
    return result
