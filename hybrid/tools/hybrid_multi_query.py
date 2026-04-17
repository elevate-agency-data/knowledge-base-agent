"""
ADK tool: hybrid_multi_query

Query multiple hybrid indexes simultaneously and merge the results.
Useful for cross-client questions like "which clients mention GA4?"
or "compare the onboarding process across all indexes".

Performance: indexes are searched **in parallel** using threads, and the
query embedding is computed **once** and reused across all dense searches.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

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


def _search_one_index(
    index_name: str,
    query: str,
    query_embedding: list[float] | None,
    embedder,
    retrieval_mode: str,
    top_k_per_index: int,
    filters: dict,
) -> tuple[str, list[dict] | None, str | None]:
    """Search a single index. Returns (index_name, results, error).

    Each call opens its own store connection so threads never share a
    DuckDB connection (which is not thread-safe).
    """
    import threading, time as _time
    t0 = _time.perf_counter()
    tid = threading.current_thread().name
    print(f"[multi_query] thread {tid} START  index={index_name}")

    try:
        # Each thread needs its own DuckDB connection — get_store() is a
        # singleton and would return the same connection to all threads,
        # causing concurrent access to the same DuckDB handle → SIGSEGV.
        from hybrid.config import ENV, DUCKDB_PATH
        from hybrid.stores.duckdb_store import DuckDBStore
        from hybrid.stores.alloydb_store import AlloyDBStore
        thread_store = DuckDBStore(DUCKDB_PATH) if ENV != "gcp" else get_store()

        dense_results: list[dict] = []
        sparse_results: list[dict] = []

        if retrieval_mode in ("hybrid", "dense"):
            dense_results = dense_search(
                query=query,
                store=thread_store,
                embedding_model=embedder,
                index_name=index_name,
                top_k=top_k_per_index * 2,
                filters=filters,
                query_embedding=query_embedding,
            )

        if retrieval_mode in ("hybrid", "sparse"):
            sparse_results = sparse_search(
                query=query,
                store=thread_store,
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
        elapsed = round(_time.perf_counter() - t0, 2)
        print(f"[multi_query] thread {tid} DONE   index={index_name} -> {len(results)} chunks in {elapsed}s")
        return (index_name, results, None)

    except Exception as exc:
        elapsed = round(_time.perf_counter() - t0, 2)
        print(f"[multi_query] thread {tid} ERROR  index={index_name} -> {exc} in {elapsed}s")
        return (index_name, None, str(exc))


def hybrid_multi_query(
    index_names: list[str],
    query: str,
    retrieval_mode: str = "hybrid",
    top_k_per_index: int = 0,
    top_k: int = TOP_K,
    filters: dict | None = None,
) -> dict:
    """
    Query multiple hybrid indexes **in parallel** and merge the results.

    The query embedding is computed **once** and reused across all dense
    searches, saving one embedding call per additional index.  Each index
    is searched in its own thread for maximum throughput.

    A **global cap** of *top_k* chunks is enforced: after collecting
    results from all indexes, only the top-scoring chunks are kept.

    Args:
        index_names:     List of index names to query simultaneously.
        query:           Natural-language question.
        retrieval_mode:  ``"hybrid"``, ``"dense"``, or ``"sparse"``.
        top_k_per_index: Chunks to retrieve per index before global trim.
                         0 (default) = auto-computed from top_k.
        top_k:           **Global** maximum chunks to return (default 10).
        filters:         Optional metadata filters applied to every index.

    Returns:
        Dict with status, context, results_by_index, sources, etc.
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

    # Auto-compute per-index budget: retrieve enough per index to fill
    # the global top_k, with a small margin for fusion quality.
    if top_k_per_index <= 0:
        top_k_per_index = max(3, (top_k * 2) // len(index_names))
    print(f"[multi_query] top_k={top_k}, top_k_per_index={top_k_per_index}, indexes={len(index_names)}")

    try:
        embedder = get_embedding_model(DEFAULT_EMBEDDING_MODEL)
    except Exception as exc:
        return {"status": "error", "message": f"Initialisation failed: {exc}"}

    # ── Embed the query ONCE ─────────────────────────────────────────────────
    import time as _time
    inner_timings: dict[str, float] = {}
    t_embed = _time.perf_counter()
    query_embedding: list[float] | None = None
    if retrieval_mode in ("hybrid", "dense"):
        query_embedding = embedder.embed_query(query)
    inner_timings["embedding"] = round(_time.perf_counter() - t_embed, 2)
    print(f"[multi_query] query embedded in {inner_timings['embedding']}s")

    # ── Search all indexes in parallel ───────────────────────────────────────
    print(f"[multi_query] launching {len(index_names)} threads: {index_names}")
    results_by_index: dict[str, list[dict]] = {}
    indexes_empty: list[str] = []
    seen_sources: dict[str, dict] = {}
    context_sections: list[str] = []

    with ThreadPoolExecutor(max_workers=min(len(index_names), 8)) as pool:
        futures = {
            pool.submit(
                _search_one_index,
                idx, query, query_embedding, embedder,
                retrieval_mode, top_k_per_index, filters,
            ): idx
            for idx in index_names
        }

        # Collect results, preserving original index order
        t_search = _time.perf_counter()
        raw_results: dict[str, list[dict] | None] = {}
        for future in as_completed(futures):
            idx_name, results, error = future.result()
            if error:
                indexes_empty.append(f"{idx_name} (error: {error})")
            else:
                raw_results[idx_name] = results
        inner_timings["threads"] = round(_time.perf_counter() - t_search, 2)
        print(f"[multi_query] all threads done in {inner_timings['threads']}s")

    # ── Global ranking: merge all chunks, keep only top_k best ─────────────
    all_chunks: list[dict] = []
    for index_name in index_names:
        results = raw_results.get(index_name)
        if results is None or not results:
            if index_name not in [e.split(" ")[0] for e in indexes_empty]:
                indexes_empty.append(index_name)
            continue
        for r in results:
            r["_index"] = index_name  # tag with source index
            all_chunks.append(r)

    # Sort by score descending and trim to global top_k
    all_chunks.sort(key=lambda c: c.get("score", 0), reverse=True)
    all_chunks = all_chunks[:top_k]

    print(f"[multi_query] global trim: {len(all_chunks)}/{top_k} chunks kept")

    # Rebuild per-index results and context from the trimmed set
    for chunk in all_chunks:
        idx = chunk.pop("_index", "")
        results_by_index.setdefault(idx, []).append(_sanitize(chunk))

        url = chunk.get("source_url", "")
        if url and url not in seen_sources:
            seen_sources[url] = {
                "index":      idx,
                "file_name":  chunk.get("file_name", ""),
                "source_url": url,
                "file_type":  chunk.get("file_type", ""),
            }

    for index_name in index_names:
        idx_chunks = results_by_index.get(index_name, [])
        if idx_chunks:
            chunks_text = "\n\n".join(
                c.get("content", "") for c in idx_chunks if c.get("content")
            )
            if chunks_text:
                context_sections.append(f"### {index_name.upper()}\n\n{chunks_text}")

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
        "timings":          inner_timings,
    }
