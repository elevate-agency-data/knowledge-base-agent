"""
ADK tool: hybrid_list_indexes

Lists all hybrid indexes present in the local store with their stats.
"""

from hybrid.stores import get_store


def _apply_role_filter(index_names: list[str]) -> list[str]:
    """Filter index names against the current user role's whitelist."""
    try:
        from rag_agent.runtime_context import get_user_role
        from shared.role_permissions import filter_indexes_for_role
    except Exception:
        # role module unavailable → no filtering (safe fallback for CLI/test contexts)
        return index_names
    return filter_indexes_for_role(index_names, get_user_role())


def hybrid_list_indexes() -> dict:
    """
    List all hybrid indexes available in the local store.

    Returns the name, embedding model, chunk strategy, creation date,
    and chunk/file counts for each index (including empty ones).

    Returns:
        Dict with keys:
        - ``status``  : ``"success"`` or ``"error"``
        - ``indexes`` : List of index info dicts
        - ``total``   : Number of indexes found
    """
    try:
        store = get_store()

        from hybrid.stores.duckdb_store import DuckDBStore
        if not isinstance(store, DuckDBStore):
            index_names = _apply_role_filter(store.list_indexes())
            return {
                "status":  "success",
                "indexes": [{"index_name": n} for n in index_names],
                "total":   len(index_names),
            }

        conn = store._get_conn()

        # Read registry
        registry_rows = conn.execute("""
            SELECT index_name, embedding_model, chunk_strategy, created_at
            FROM hybrid_indexes
            ORDER BY created_at DESC
        """).fetchall()

        if not registry_rows:
            return {
                "status":  "success",
                "message": "No indexes found. Create one with hybrid_create_index.",
                "indexes": [],
                "total":   0,
            }

        # Apply role-based whitelist before any per-index work.
        allowed = set(_apply_role_filter([r[0] for r in registry_rows]))
        registry_rows = [r for r in registry_rows if r[0] in allowed]

        indexes = []
        for row in registry_rows:
            index_name = row[0]
            table = store._tbl(index_name)

            # Count chunks and files from the per-index table
            total_chunks = 0
            total_files  = 0
            try:
                stats = conn.execute(f"""
                    SELECT COUNT(*), COUNT(DISTINCT source_url)
                    FROM {table}
                """).fetchone()
                if stats:
                    total_chunks = stats[0] or 0
                    total_files  = stats[1] or 0
            except Exception:
                pass  # Table may not exist yet (empty index)

            indexes.append({
                "index_name":      index_name,
                "embedding_model": row[1],
                "chunk_strategy":  row[2],
                "created_at":      str(row[3]) if row[3] else "",
                "total_chunks":    total_chunks,
                "total_files":     total_files,
            })

        # Group indexes by company for hierarchical view
        INDEX_SEP = "__"
        companies: dict[str, list[str]] = {}
        for idx in indexes:
            name = idx["index_name"]
            if INDEX_SEP in name:
                company, notion = name.split(INDEX_SEP, 1)
                companies.setdefault(company, []).append(notion)
            else:
                companies.setdefault(name, [])

        return {
            "status":    "success",
            "indexes":   indexes,
            "total":     len(indexes),
            "companies": companies,
        }

    except Exception as exc:
        return {"status": "error", "message": str(exc)}
