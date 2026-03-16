"""
ADK tool: hybrid_delete_index

Deletes all chunks belonging to a hybrid index from the local store.
Requires explicit confirmation to prevent accidental data loss.
"""

from hybrid.stores import get_store


def hybrid_delete_index(
    index_name: str,
    confirm: bool = False,
) -> dict:
    """
    Delete a hybrid index and all its chunks from the store.

    This operation is irreversible. The data can only be recovered by
    re-ingesting the source documents with hybrid_add_data.

    Args:
        index_name: Name of the index to delete.
        confirm:    Must be True to proceed. Prevents accidental deletion.

    Returns:
        Dict with keys:
        - ``status``       : ``"success"`` or ``"error"``
        - ``message``      : Human-readable confirmation or error
        - ``index_name``   : Echo of the deleted index
        - ``chunks_deleted``: Number of chunks removed
    """
    index_name = index_name.strip().lower()
    if not index_name:
        return {"status": "error", "message": "index_name is required."}

    if not confirm:
        return {
            "status":  "error",
            "message": (
                f"Confirmation required to delete index '{index_name}'. "
                "Set confirm=True to proceed. This action is irreversible."
            ),
        }

    try:
        store = get_store()

        # Count chunks before deletion for reporting
        chunks_before = 0
        from hybrid.stores.duckdb_store import DuckDBStore
        if isinstance(store, DuckDBStore):
            conn = store._get_conn()
            try:
                result = conn.execute(
                    "SELECT COUNT(*) FROM chunks WHERE index_name = ?",
                    [index_name],
                ).fetchone()
                chunks_before = result[0] if result else 0
            except Exception:
                chunks_before = 0

        if chunks_before == 0:
            return {
                "status":  "error",
                "message": f"Index '{index_name}' not found or already empty.",
            }

        store.delete_index(index_name)

        return {
            "status":        "success",
            "message":       f"Index '{index_name}' deleted. {chunks_before} chunk(s) removed.",
            "index_name":    index_name,
            "chunks_deleted": chunks_before,
        }

    except Exception as exc:
        return {"status": "error", "message": str(exc)}
