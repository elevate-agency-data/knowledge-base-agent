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
            return result.get("indexes", [])
        return []
    except Exception:
        return []


def list_indexes_for_user(role: str | None) -> list[dict]:
    """
    Return only the indexes a given role is allowed to query.

    The page Knowledge Base / Admin still calls plain `list_indexes()` so they
    see everything (admins can manage any index). For the **query path**
    (Simple Chat, Agent), this filtered list enforces role-based access.
    """
    from shared.role_permissions import filter_indexes_for_role

    indexes = list_indexes()
    if not indexes:
        return []
    allowed_names = set(
        filter_indexes_for_role([i.get("index_name", "") for i in indexes], role)
    )
    return [i for i in indexes if i.get("index_name", "") in allowed_names]


def list_indexes_grouped() -> dict[str, list[dict]]:
    """
    Return indexes grouped by company (two-level hierarchy).

    Returns:
        Dict mapping company name → list of index dicts belonging to that company.
        Indexes without ``__`` are grouped under their own name.
    """
    indexes = list_indexes()
    groups: dict[str, list[dict]] = {}
    for idx in indexes:
        name = idx.get("index_name", "")
        if "__" in name:
            company = name.split("__", 1)[0]
        else:
            company = name
        groups.setdefault(company, []).append(idx)
    return groups


def get_index_info(index_name: str) -> dict:
    """Return detailed info for a single index (files, chunks, etc.)."""
    from hybrid.tools.hybrid_index_info import hybrid_index_info
    return hybrid_index_info(index_name)


def create_index(index_name: str, embedding_model: str = "", chunk_strategy: str = "fixed") -> dict:
    from hybrid.tools.hybrid_create_index import hybrid_create_index
    return hybrid_create_index(
        index_name=index_name,
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
          context: str = "", retrieval_query: str = "") -> dict:
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
        retrieval_query=retrieval_query,
    )
    elapsed = round(time.perf_counter() - t0, 2)

    result["elapsed_s"] = elapsed
    # Alias "context" → "answer" so comparison page uses the same key as vertex
    result["answer"] = result.get("context", "")
    return result


def multi_query(index_names: list[str], query_text: str,
                retrieval_mode: str = "hybrid", top_k: int = 10,
                context: str = "", retrieval_query: str = "") -> dict:
    """Query multiple hybrid indexes simultaneously.

    *top_k* is the **global** maximum number of chunks returned (not per index).
    """
    from hybrid.tools.hybrid_query import hybrid_query

    t0 = time.perf_counter()
    result = hybrid_query(
        index_names=index_names,
        query=query_text,
        retrieval_mode=retrieval_mode,
        top_k=top_k,
        context=context,
        retrieval_query=retrieval_query,
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
             chunk_strategy: str = "fixed",
             max_files: int = 0) -> dict:
    from hybrid.tools.hybrid_add_data import hybrid_add_data
    return hybrid_add_data(
        index_name=index_name,
        folder_names=folder_names,
        chunk_strategy=chunk_strategy,
        max_files=max_files,
    )


def add_data_auto(
    company_filter: list[str] | None = None,
    chunk_strategy: str = "fixed",
    max_files_per_index: int = 0,
) -> dict:
    """Auto-ingest all companies/notions from the Drive tree."""
    from hybrid.tools.hybrid_add_data import hybrid_add_data_auto
    return hybrid_add_data_auto(
        company_filter=company_filter or None,
        chunk_strategy=chunk_strategy,
        max_files_per_index=max_files_per_index,
    )
