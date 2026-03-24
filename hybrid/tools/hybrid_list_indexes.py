"""
ADK tool: hybrid_list_indexes

Lists all hybrid indexes present in the local store with their stats.
"""

from hybrid.stores import get_store


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
            index_names = store.list_indexes()
            return {
                "status":  "success",
                "indexes": [{"name": n} for n in index_names],
                "total":   len(index_names),
            }

        conn = store._get_conn()

        # Ensure metadata table exists
        conn.execute("""
            CREATE TABLE IF NOT EXISTS hybrid_indexes (
                index_name      VARCHAR PRIMARY KEY,
                embedding_model VARCHAR,
                chunk_strategy  VARCHAR,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        rows = conn.execute("""
            SELECT
                i.index_name,
                i.embedding_model,
                i.chunk_strategy,
                i.created_at,
                COALESCE(c.total_chunks, 0)  AS total_chunks,
                COALESCE(c.total_files, 0)   AS total_files
            FROM hybrid_indexes i
            LEFT JOIN (
                SELECT
                    index_name,
                    COUNT(*)                   AS total_chunks,
                    COUNT(DISTINCT source_url) AS total_files
                FROM chunks
                GROUP BY index_name
            ) c ON i.index_name = c.index_name
            ORDER BY i.created_at DESC
        """).fetchall()

        if not rows:
            return {
                "status":  "success",
                "message": "No indexes found. Create one with hybrid_create_index.",
                "indexes": [],
                "total":   0,
            }

        indexes = []
        for row in rows:
            indexes.append({
                "index_name":      row[0],
                "embedding_model": row[1],
                "chunk_strategy":  row[2],
                "created_at":      str(row[3]) if row[3] else "",
                "total_chunks":    row[4],
                "total_files":     row[5],
            })

        return {
            "status":  "success",
            "indexes": indexes,
            "total":   len(indexes),
        }

    except Exception as exc:
        return {"status": "error", "message": str(exc)}
