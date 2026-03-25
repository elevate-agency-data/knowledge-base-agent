"""
ADK tool: hybrid_index_info

Shows detailed content of a hybrid index: list of indexed files,
chunk counts per file, languages, domains, and embedding info.
"""

from hybrid.stores import get_store


def hybrid_index_info(
    index_name: str,
) -> dict:
    """
    Show the detailed contents of a hybrid index.

    Lists every file indexed, with its chunk count, language, domain,
    and source URL. Useful for verifying what has been ingested and
    checking for missing or failed files.

    Args:
        index_name: Name of the index to inspect.

    Returns:
        Dict with keys:
        - ``status``         : ``"success"`` or ``"error"``
        - ``index_name``     : Echo of the index
        - ``total_chunks``   : Total number of chunks in the index
        - ``total_files``    : Number of distinct source files
        - ``embedding_model``: Model used for this index
        - ``chunk_strategy`` : Chunking strategy used
        - ``files``          : List of file info dicts, each with:
                               name, type, chunks, langue, domaine, url
    """
    index_name = index_name.strip().lower()
    if not index_name:
        return {"status": "error", "message": "index_name is required."}

    try:
        store = get_store()

        from hybrid.stores.duckdb_store import DuckDBStore
        if not isinstance(store, DuckDBStore):
            return {
                "status":  "error",
                "message": "hybrid_index_info is only supported with the DuckDB store.",
            }

        conn = store._get_conn()
        table = store._tbl(index_name)

        # Check index exists in registry
        reg = conn.execute(
            "SELECT embedding_model, chunk_strategy FROM hybrid_indexes WHERE index_name = ?",
            [index_name],
        ).fetchone()

        if not reg:
            return {
                "status":  "error",
                "message": f"Index '{index_name}' not found. Run hybrid_create_index first.",
            }

        embedding_model = reg[0]
        chunk_strategy  = reg[1]

        # Check table exists and has data
        try:
            total = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        except Exception:
            return {
                "status":     "error",
                "message":    f"Index '{index_name}' table not found. Run hybrid_create_index first.",
                "index_name": index_name,
            }

        if not total or total[0] == 0:
            return {
                "status":          "success",
                "message":         f"Index '{index_name}' exists but contains no chunks.",
                "index_name":      index_name,
                "total_chunks":    0,
                "total_files":     0,
                "embedding_model": embedding_model,
                "chunk_strategy":  chunk_strategy,
                "files":           [],
            }

        # Global stats from per-index table
        stats = conn.execute(f"""
            SELECT
                COUNT(*)                   AS total_chunks,
                COUNT(DISTINCT source_url) AS total_files
            FROM {table}
        """).fetchone()

        # Per-file breakdown
        file_rows = conn.execute(f"""
            SELECT
                file_name,
                file_type,
                source_url,
                langue,
                domaine,
                COUNT(*) AS chunk_count
            FROM {table}
            GROUP BY file_name, file_type, source_url, langue, domaine
            ORDER BY file_name
        """).fetchall()

        files = [
            {
                "name":    row[0],
                "type":    row[1],
                "url":     row[2],
                "langue":  row[3],
                "domaine": row[4],
                "chunks":  row[5],
            }
            for row in file_rows
        ]

        return {
            "status":          "success",
            "index_name":      index_name,
            "total_chunks":    stats[0],
            "total_files":     stats[1],
            "embedding_model": embedding_model,
            "chunk_strategy":  chunk_strategy,
            "files":           files,
        }

    except Exception as exc:
        return {"status": "error", "message": str(exc)}
