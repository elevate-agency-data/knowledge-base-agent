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
        - ``status``        : ``"success"`` or ``"error"``
        - ``index_name``    : Echo of the index
        - ``total_chunks``  : Total number of chunks in the index
        - ``total_files``   : Number of distinct source files
        - ``embedding_model``: Model used for this index
        - ``chunk_strategy``: Chunking strategy used
        - ``files``         : List of file info dicts, each with:
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

        # Check index exists
        try:
            total = conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE index_name = ?",
                [index_name],
            ).fetchone()
        except Exception:
            return {
                "status":  "error",
                "message": f"Index '{index_name}' not found. Run hybrid_create_index first.",
            }

        if not total or total[0] == 0:
            return {
                "status":     "error",
                "message":    f"Index '{index_name}' exists but contains no chunks.",
                "index_name": index_name,
            }

        # Global stats
        stats = conn.execute("""
            SELECT
                COUNT(*)                   AS total_chunks,
                COUNT(DISTINCT source_url) AS total_files,
                MAX(embedding_model)       AS embedding_model,
                MAX(chunk_strategy)        AS chunk_strategy
            FROM chunks
            WHERE index_name = ?
        """, [index_name]).fetchone()

        # Per-file breakdown
        file_rows = conn.execute("""
            SELECT
                file_name,
                file_type,
                source_url,
                langue,
                domaine,
                COUNT(*) AS chunk_count
            FROM chunks
            WHERE index_name = ?
            GROUP BY file_name, file_type, source_url, langue, domaine
            ORDER BY file_name
        """, [index_name]).fetchall()

        files = []
        for row in file_rows:
            files.append({
                "name":        row[0],
                "type":        row[1],
                "url":         row[2],
                "langue":      row[3],
                "domaine":     row[4],
                "chunks":      row[5],
            })

        return {
            "status":          "success",
            "index_name":      index_name,
            "total_chunks":    stats[0],
            "total_files":     stats[1],
            "embedding_model": stats[2],
            "chunk_strategy":  stats[3],
            "files":           files,
        }

    except Exception as exc:
        return {"status": "error", "message": str(exc)}
