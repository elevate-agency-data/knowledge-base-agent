"""
Store backends for the hybrid RAG pipeline.

The ``get_store()`` factory reads ``hybrid.config.ENV`` and returns
the appropriate backend automatically:

- ``"local"``  → :class:`DuckDBStore`  (offline, no services needed)
- ``"gcp"``    → :class:`AlloyDBStore` (requires AlloyDB connection string)
"""

from .base import BaseStore
from .duckdb_store import DuckDBStore
from .alloydb_store import AlloyDBStore
from hybrid.config import ENV, DUCKDB_PATH


def get_store() -> BaseStore:
    """
    Return the store backend configured for the current environment.

    Reads ``hybrid.config.ENV``:
    - ``"local"`` → DuckDB at ``hybrid.config.DUCKDB_PATH``
    - ``"gcp"``   → AlloyDB using ``hybrid.config.ALLOYDB_CONNECTION_STRING``

    Returns:
        A :class:`BaseStore` implementation ready to be initialised.
    """
    if ENV == "gcp":
        return AlloyDBStore()
    return DuckDBStore(DUCKDB_PATH)


__all__ = [
    "BaseStore",
    "DuckDBStore",
    "AlloyDBStore",
    "get_store",
]
