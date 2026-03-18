"""
Hybrid RAG service — thin wrappers around the hybrid tool functions.

Adds timing, normalises return shapes, and surfaces useful sub-fields so
the UI can display them without parsing raw tool dicts.
"""

from __future__ import annotations

import time


# ── Index management ──────────────────────────────────────────────────────────

def list_indexes() -> list[dict]:
    """
    Return all known hybrid indexes.

    Returns:
        List of dicts with keys: index_name, embedding_model, chunk_strategy,
        created_at, total_chunks, total_files.
    """
    try:
        from hybrid.tools.hybrid_list_indexes import hybrid_list_indexes
        result = hybrid_list_indexes()
        if result.get("status") == "success":
            indexes = result.get("indexes", [])
            # Normalise: tool returns "name", UI expects "index_name"
            for idx in indexes:
                if "index_name" not in idx and "name" in idx:
                    idx["index_name"] = idx["name"]
            return indexes
        return []
    except Exception:
        return []


def get_index_info(index_name: str) -> dict:
    """Return detailed info for a single index (files, chunks, etc.)."""
    from hybrid.tools.hybrid_index_info import hybrid_index_info
    return hybrid_index_info(index_name)


def create_index(index_name: str, embedding_model: str = "mpnet-768",
                 chunk_strategy: str = "fixed") -> dict:
    from hybrid.tools.hybrid_create_index import hybrid_create_index
    return hybrid_create_index(
        index_name=index_name,
        embedding_model=embedding_model,
        chunk_strategy=chunk_strategy,
    )


def delete_index(index_name: str) -> dict:
    from hybrid.tools.hybrid_delete_index import hybrid_delete_index
    return hybrid_delete_index(index_name=index_name, confirm=True)


# ── Index resolution ──────────────────────────────────────────────────────────

def resolve_indexes(query: str, available: list[str]) -> list[str]:
    """
    Determine which indexes to query for a given user prompt.

    Delegates to shared.index_resolver so the same logic is used by
    hybrid_query (ADK tool), Simple Chat, and RAG Comparison.
    """
    from shared.index_resolver import resolve_indexes as _resolve
    return _resolve(query, available)


# ── Retrieval ─────────────────────────────────────────────────────────────────

def query(index_name: str, query_text: str,
          retrieval_mode: str = "hybrid", top_k: int = 10,
          context: str = "") -> dict:
    """
    Query a hybrid index and return a structured result with timing.

    Returns:
        Dict with keys:
        - status         : "success" | "error"
        - answer         : Concatenated chunk text (context for the LLM)
        - chunks         : Full chunk list with scores and metadata
        - sources        : Deduplicated source list
        - total_results  : Number of chunks returned
        - elapsed_s      : Wall-clock seconds
        - retrieval_mode : Mode used
        - index_name     : Echo of input
    """
    from hybrid.tools.hybrid_query import hybrid_query

    t0 = time.perf_counter()
    result = hybrid_query(
        index_names=[index_name],
        query=query_text,
        retrieval_mode=retrieval_mode,
        top_k=top_k,
        context=context,
    )
    elapsed = round(time.perf_counter() - t0, 2)

    result["elapsed_s"] = elapsed
    # Alias "context" → "answer" so comparison page uses the same key as vertex
    result["answer"] = result.get("context", "")
    return result


def multi_query(index_names: list[str], query_text: str,
                retrieval_mode: str = "hybrid", top_k_per_index: int = 5,
                context: str = "") -> dict:
    """Query multiple hybrid indexes simultaneously."""
    from hybrid.tools.hybrid_query import hybrid_query

    t0 = time.perf_counter()
    result = hybrid_query(
        index_names=index_names,
        query=query_text,
        retrieval_mode=retrieval_mode,
        top_k=top_k_per_index * len(index_names),
        context=context,
    )
    result["elapsed_s"] = round(time.perf_counter() - t0, 2)
    # Alias "context" → "answer" pour cohérence avec query() et vertex_service
    result["answer"] = result.get("context", "")
    return result


# ── Drive helpers ─────────────────────────────────────────────────────────────

def list_drive_folder(folder_name: str) -> dict:
    from hybrid.tools.hybrid_list_drive import hybrid_list_drive
    return hybrid_list_drive(folder_name=folder_name)


def add_data(index_name: str, folder_names: list[str],
             embedding_model: str = "mpnet-768",
             chunk_strategy: str = "fixed",
             max_files: int = 0) -> dict:
    from hybrid.tools.hybrid_add_data import hybrid_add_data
    return hybrid_add_data(
        index_name=index_name,
        folder_names=folder_names,
        embedding_model=embedding_model,
        chunk_strategy=chunk_strategy,
        max_files=max_files,
    )
