"""
ADK tool: hybrid_multi_query

Query multiple hybrid indexes simultaneously and merge the results.
Useful for cross-client questions like "which clients mention GA4?"
or "compare the onboarding process across all indexes".
"""

from hybrid.config import (
    DEFAULT_EMBEDDING_MODEL,
    TOP_K,
    DENSE_WEIGHT,
    SPARSE_WEIGHT,
    RRF_K,
)
from hybrid.stores import get_store
from hybrid.embeddings import get_embedding_model
from hybrid.retrieval.dense  import dense_search
from hybrid.retrieval.sparse import sparse_search
from hybrid.retrieval.fusion import reciprocal_rank_fusion
from hybrid.retrieval.filter import apply_post_filter
from hybrid.tools.hybrid_rag_query import _sanitize


def hybrid_multi_query(
    index_names: list[str],
    query: str,
    retrieval_mode: str = "hybrid",
    top_k_per_index: int = 5,
    filters: dict | None = None,
) -> dict:
    """
    Query multiple hybrid indexes and merge the results.

    Runs the query against each index independently, then merges all
    results into a single ranked list using Reciprocal Rank Fusion.
    Results are grouped by index so the LLM can attribute answers to
    specific clients/sources.

    Args:
        index_names:     List of index names to query simultaneously.
                         Example: ["rh", "marketing", "juridique", "finance"]
        query:           Natural-language question.
        retrieval_mode:  ``"hybrid"``, ``"dense"``, or ``"sparse"``.
        top_k_per_index: Number of chunks to retrieve per index (default 5).
                         Total results = top_k_per_index × len(index_names).
        filters:         Optional metadata filters applied to every index.

    Returns:
        Dict with keys:
        - ``status``          : ``"success"`` or ``"error"``
        - ``query``           : Echo of the query
        - ``indexes_queried`` : List of indexes that were searched
        - ``indexes_empty``   : Indexes that returned no results
        - ``context``         : Merged context string for LLM (grouped by index)
        - ``results_by_index``: Dict mapping index_name → list of chunks
        - ``sources``         : Deduplicated source links across all indexes
        - ``total_results``   : Total chunks returned across all indexes
    """
    index_names = [n.strip().lower() for n in index_names if n.strip()]
    if not index_names:
        return {"status": "error", "message": "index_names must not be empty."}
    if not query or not query.strip():
        return {"status": "error", "message": "query must not be empty."}

    valid_modes = {"hybrid", "dense", "sparse"}
    if retrieval_mode not in valid_modes:
        return {
            "status":  "error",
            "message": f"Unknown retrieval_mode '{retrieval_mode}'. Supported: {sorted(valid_modes)}",
        }

    filters = filters or {}

    try:
        store   = get_store()
        embedder = get_embedding_model(DEFAULT_EMBEDDING_MODEL)
    except Exception as exc:
        return {"status": "error", "message": f"Initialisation failed: {exc}"}

    results_by_index: dict[str, list[dict]] = {}
    indexes_empty: list[str] = []
    seen_sources: dict[str, dict] = {}
    context_sections: list[str] = []

    for index_name in index_names:
        try:
            dense_results: list[dict] = []
            sparse_results: list[dict] = []

            if retrieval_mode in ("hybrid", "dense"):
                dense_results = dense_search(
                    query=query,
                    store=store,
                    embedding_model=embedder,
                    index_name=index_name,
                    top_k=top_k_per_index * 2,
                    filters=filters,
                )

            if retrieval_mode in ("hybrid", "sparse"):
                sparse_results = sparse_search(
                    query=query,
                    store=store,
                    index_name=index_name,
                    top_k=top_k_per_index * 2,
                    filters=filters,
                )

            if retrieval_mode == "hybrid":
                results = reciprocal_rank_fusion(
                    dense_results, sparse_results,
                    k=RRF_K,
                    dense_weight=DENSE_WEIGHT,
                    sparse_weight=SPARSE_WEIGHT,
                )
            elif retrieval_mode == "dense":
                results = dense_results
            else:
                results = sparse_results

            results = apply_post_filter(results, filters)[:top_k_per_index]

            if not results:
                indexes_empty.append(index_name)
                continue

            results_by_index[index_name] = [_sanitize(r) for r in results]

            # Build context section for this index
            chunks_text = "\n\n".join(r.get("content", "") for r in results if r.get("content"))
            if chunks_text:
                context_sections.append(f"### {index_name.upper()}\n\n{chunks_text}")

            # Collect sources
            for chunk in results:
                url = chunk.get("source_url", "")
                if url and url not in seen_sources:
                    seen_sources[url] = {
                        "index":      index_name,
                        "file_name":  chunk.get("file_name", ""),
                        "source_url": url,
                        "file_type":  chunk.get("file_type", ""),
                    }

        except Exception as exc:
            indexes_empty.append(f"{index_name} (error: {exc})")

    total = sum(len(v) for v in results_by_index.values())
    context = "\n\n---\n\n".join(context_sections)

    return {
        "status":           "success",
        "query":            query,
        "indexes_queried":  index_names,
        "indexes_empty":    indexes_empty,
        "context":          context,
        "results_by_index": results_by_index,
        "sources":          list(seen_sources.values()),
        "total_results":    total,
    }
