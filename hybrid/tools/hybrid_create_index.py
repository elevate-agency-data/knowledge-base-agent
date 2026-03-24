"""
ADK tool: hybrid_create_index

Creates a new hybrid index (table namespace) in the configured store backend.
Must be called once before adding data or running queries against an index.
"""

from hybrid.config import (
    DEFAULT_EMBEDDING_MODEL,
    ENV,
    DUCKDB_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from hybrid.stores import get_store


def hybrid_create_index(
    index_name: str,
    chunk_strategy: str = "fixed",
) -> dict:
    """
    Create a new hybrid RAG index.

    Initialises the underlying store (DuckDB locally, AlloyDB on GCP) and
    records the configuration used.  Idempotent — safe to call multiple
    times for the same index.

    The embedding model is always read from ``hybrid/config.py``
    (``DEFAULT_EMBEDDING_MODEL``) and cannot be overridden per call —
    all indexes in the same store must use the same model dimension.

    Args:
        index_name:      Logical name for the index (e.g. ``"hr-docs-v1"``).
                         Must be non-empty and contain only alphanumeric
                         characters, hyphens, or underscores.
        chunk_strategy:  Chunking strategy: ``"fixed"``, ``"semantic"``,
                         or ``"hierarchical"``.  Defaults to ``"fixed"``.

    Returns:
        Dict with keys:
        - ``status``         : ``"success"`` or ``"error"``
        - ``message``        : Human-readable confirmation or error text
        - ``index_name``     : Echo of the created index name
        - ``embedding_model``: Model used
        - ``chunk_strategy`` : Strategy used
        - ``store_backend``  : ``"duckdb"`` or ``"alloydb"``
        - ``environment``    : Current ENV value
    """
    embedding_model = DEFAULT_EMBEDDING_MODEL  # always use configured default

    # -- Normalise index name to lowercase ------------------------------------
    index_name = index_name.strip().lower()

    # -- Validation ----------------------------------------------------------
    if not index_name:
        return {
            "status":  "error",
            "message": "index_name must be a non-empty string.",
        }

    import re
    if not re.match(r"^[a-zA-Z0-9_-]+$", index_name):
        return {
            "status":  "error",
            "message": (
                f"Invalid index_name '{index_name}'. "
                "Use only letters, digits, hyphens, and underscores."
            ),
        }

    valid_strategies = {"fixed", "semantic", "hierarchical"}
    if chunk_strategy not in valid_strategies:
        return {
            "status":  "error",
            "message": (
                f"Unknown chunk_strategy '{chunk_strategy}'. "
                f"Supported: {sorted(valid_strategies)}"
            ),
        }

    # -- Initialise store ----------------------------------------------------
    try:
        store = get_store()
        store.initialize(index_name, embedding_model=embedding_model, chunk_strategy=chunk_strategy)
    except Exception as exc:
        return {
            "status":  "error",
            "message": f"Store initialisation failed: {exc}",
        }

    backend = "alloydb" if ENV == "gcp" else "duckdb"

    return {
        "status":          "success",
        "message": (
            f"Index '{index_name}' created successfully in the {backend} store. "
            f"Embedding model: {embedding_model} | "
            f"Chunk strategy: {chunk_strategy} | "
            f"Environment: {ENV}."
        ),
        "index_name":      index_name,
        "embedding_model": embedding_model,
        "chunk_strategy":  chunk_strategy,
        "store_backend":   backend,
        "environment":     ENV,
    }
