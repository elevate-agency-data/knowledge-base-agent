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


_store: BaseStore | None = None


def get_store() -> BaseStore:
    """
    Return the store backend configured for the current environment.
    Singleton — même instance réutilisée à chaque appel (connexion persistante).

    Reads ``hybrid.config.ENV``:
    - ``"local"`` → DuckDB at ``hybrid.config.DUCKDB_PATH``
    - ``"gcp"``   → AlloyDB using ``hybrid.config.ALLOYDB_CONNECTION_STRING``

    Returns:
        A :class:`BaseStore` implementation ready to be initialised.
    """
    global _store
    if _store is None:
        _store = AlloyDBStore() if ENV == "gcp" else DuckDBStore(DUCKDB_PATH)
    return _store


__all__ = [
    "BaseStore",
    "DuckDBStore",
    "AlloyDBStore",
    "get_store",
]
