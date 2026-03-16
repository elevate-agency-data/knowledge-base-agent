"""
DuckDB-backed store for local development.

Stores chunk embeddings in a local .duckdb file and supports:
- Dense search via Python/NumPy cosine similarity
- Sparse search via DuckDB's built-in FTS (BM25)
- Full metadata filtering

No external services required — works completely offline.
"""

import os
from typing import Optional

import numpy as np

from .base import BaseStore

_CREATE_INDEXES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS hybrid_indexes (
    index_name      VARCHAR PRIMARY KEY,
    embedding_model VARCHAR,
    chunk_strategy  VARCHAR,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS chunks (
    id              VARCHAR PRIMARY KEY,
    index_name      VARCHAR,
    content         TEXT,
    embedding       FLOAT[],
    source_url      VARCHAR,
    file_name       VARCHAR,
    file_type       VARCHAR,
    created_at      DATE,
    updated_at      DATE,
    author          VARCHAR,
    domaine         VARCHAR,
    langue          VARCHAR,
    tags            VARCHAR[],
    chunk_index     INTEGER,
    chunk_total     INTEGER,
    chunk_strategy  VARCHAR,
    parent_chunk_id VARCHAR,
    embedding_model VARCHAR,
    embedding_dim   INTEGER
)
"""

_INSERT_SQL = """
INSERT OR REPLACE INTO chunks VALUES (
    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
)
"""


class DuckDBStore(BaseStore):
    """
    Hybrid store backed by a local DuckDB file.

    Dense retrieval is computed with NumPy cosine similarity (suitable for
    local/dev workloads up to ~100 k chunks).  Sparse retrieval uses
    DuckDB's FTS extension (BM25).

    Args:
        db_path: Path to the ``.duckdb`` file.  Created automatically if
                 the directory exists; parent directories are created on
                 ``initialize()``.
    """

    def __init__(self, db_path: str) -> None:
        """Set up the store with the given file path."""
        self.db_path = db_path
        self._conn = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_conn(self):
        """
        Return (and lazily create) the DuckDB connection.

        Creates parent directories and loads the FTS extension.
        """
        if self._conn is None:
            import duckdb

            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            self._conn = duckdb.connect(self.db_path)
            # Load FTS extension (non-fatal if unavailable)
            try:
                self._conn.execute("INSTALL fts")
                self._conn.execute("LOAD fts")
            except Exception:
                pass
        return self._conn

    def _build_where(
        self, filters: dict, prefix: str = ""
    ) -> tuple[str, list]:
        """
        Translate a filters dict into a SQL WHERE fragment + param list.

        Args:
            filters: Dict with optional keys: index_name, file_type,
                     domaine, langue, author, date_from, date_to.
            prefix:  Optional table alias prefix (e.g. ``"c."``).

        Returns:
            Tuple of (clause_string, params_list).  The clause string
            starts with ``AND`` so it can be appended to an existing WHERE.
        """
        clauses: list[str] = []
        params: list = []

        mapping = {
            "index_name": "index_name",
            "file_type": "file_type",
            "domaine": "domaine",
            "langue": "langue",
            "author": "author",
        }
        for key, col in mapping.items():
            if filters.get(key):
                clauses.append(f"AND {prefix}{col} = ?")
                params.append(filters[key])

        if filters.get("date_from"):
            clauses.append(f"AND {prefix}created_at >= ?")
            params.append(filters["date_from"])
        if filters.get("date_to"):
            clauses.append(f"AND {prefix}created_at <= ?")
            params.append(filters["date_to"])

        return " ".join(clauses), params

    def _rows_to_dicts(self, rows, conn) -> list[dict]:
        """Convert fetchall() rows to list of dicts using cursor description."""
        columns = [d[0] for d in conn.description]
        return [dict(zip(columns, row)) for row in rows]

    def _rebuild_fts(self) -> None:
        """Rebuild the FTS index after inserts (non-fatal on failure)."""
        try:
            self._conn.execute(
                "PRAGMA create_fts_index('chunks', 'id', 'content', overwrite=1)"
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # BaseStore interface
    # ------------------------------------------------------------------

    def initialize(self, index_name: str, embedding_model: str = "", chunk_strategy: str = "") -> None:
        """
        Create the chunks table, metadata table, and FTS index if they do not exist.
        Registers the index in hybrid_indexes so it appears in list_indexes() even when empty.

        Idempotent — safe to call multiple times.

        Args:
            index_name:      Logical index name (stored in chunk metadata).
            embedding_model: Model used for this index (stored in metadata).
            chunk_strategy:  Chunking strategy used (stored in metadata).
        """
        conn = self._get_conn()
        conn.execute(_CREATE_INDEXES_TABLE_SQL)
        conn.execute(_CREATE_TABLE_SQL)
        # Register the index (INSERT OR IGNORE so idempotent)
        conn.execute(
            "INSERT OR IGNORE INTO hybrid_indexes (index_name, embedding_model, chunk_strategy) VALUES (?, ?, ?)",
            [index_name, embedding_model, chunk_strategy],
        )
        self._rebuild_fts()

    def insert_chunks(self, chunks: list[dict]) -> None:
        """
        Bulk-insert chunk dicts into the store.

        Args:
            chunks: List of chunk dicts with all required fields.
        """
        conn = self._get_conn()
        for chunk in chunks:
            conn.execute(
                _INSERT_SQL,
                [
                    chunk["id"],
                    chunk["index_name"],
                    chunk["content"],
                    chunk.get("embedding"),
                    chunk.get("source_url", ""),
                    chunk.get("file_name", ""),
                    chunk.get("file_type", ""),
                    chunk.get("created_at"),
                    chunk.get("updated_at"),
                    chunk.get("author", ""),
                    chunk.get("domaine", ""),
                    chunk.get("langue", ""),
                    chunk.get("tags", []),
                    chunk.get("chunk_index", 0),
                    chunk.get("chunk_total", 1),
                    chunk.get("chunk_strategy", "fixed"),
                    chunk.get("parent_chunk_id"),
                    chunk.get("embedding_model", ""),
                    chunk.get("embedding_dim", 0),
                ],
            )
        # Rebuild FTS after bulk insert
        self._rebuild_fts()

    def dense_search(
        self,
        embedding: list[float],
        top_k: int,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """
        Return the *top_k* most similar chunks using cosine similarity.

        Similarity is computed in Python/NumPy after fetching candidate
        chunks from DuckDB (optionally pre-filtered by index_name).

        Args:
            embedding: Query vector.
            top_k:     Number of results to return.
            filters:   Metadata filters dict.

        Returns:
            List of chunk dicts with an added ``"score"`` key.
        """
        conn = self._get_conn()
        filters = filters or {}
        where_clause, params = self._build_where(filters)

        sql = f"""
            SELECT * FROM chunks
            WHERE embedding IS NOT NULL
            {where_clause}
        """
        rows = conn.execute(sql, params).fetchall()
        columns = [d[0] for d in conn.description]

        query_vec = np.array(embedding, dtype=np.float32)
        query_norm = float(np.linalg.norm(query_vec))

        scored: list[dict] = []
        for row in rows:
            row_dict = dict(zip(columns, row))
            emb = row_dict.get("embedding")
            if not emb:
                continue
            vec = np.array(emb, dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            if norm > 0.0 and query_norm > 0.0:
                score = float(np.dot(query_vec, vec) / (query_norm * norm))
            else:
                score = 0.0
            row_dict["score"] = score
            scored.append(row_dict)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def sparse_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """
        Return the *top_k* most relevant chunks using DuckDB FTS (BM25).

        Falls back to ILIKE pattern matching if the FTS extension is
        unavailable.

        Args:
            query:   Full-text query string.
            top_k:   Number of results to return.
            filters: Metadata filters dict.

        Returns:
            List of chunk dicts with an added ``"score"`` key.
        """
        conn = self._get_conn()
        filters = filters or {}
        where_clause, params = self._build_where(filters, prefix="c.")

        try:
            sql = f"""
                SELECT c.*, fts_main_chunks.match_bm25(c.id, ?) AS score
                FROM chunks c
                WHERE score IS NOT NULL
                {where_clause}
                ORDER BY score DESC
                LIMIT ?
            """
            full_params = [query] + params + [top_k]
            rows = conn.execute(sql, full_params).fetchall()
            return self._rows_to_dicts(rows, conn)
        except Exception:
            # FTS unavailable — fall back to ILIKE
            sql = f"""
                SELECT *, 1.0 AS score
                FROM chunks
                WHERE content ILIKE ?
                {where_clause.replace('c.', '')}
                LIMIT ?
            """
            fallback_params = [f"%{query}%"] + params + [top_k]
            rows = conn.execute(sql, fallback_params).fetchall()
            return self._rows_to_dicts(rows, conn)

    def delete_index(self, index_name: str) -> None:
        """
        Delete all chunks associated with *index_name*.

        Args:
            index_name: Index to purge.
        """
        conn = self._get_conn()
        conn.execute("DELETE FROM chunks WHERE index_name = ?", [index_name])
        self._rebuild_fts()

    def list_indexes(self) -> list[str]:
        """
        Return a sorted list of all distinct index names in the store.

        Returns:
            Sorted list of index name strings.
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT DISTINCT index_name FROM chunks ORDER BY index_name"
        ).fetchall()
        return [row[0] for row in rows if row[0]]

    def get_chunk_by_id(self, chunk_id: str) -> dict:
        """
        Retrieve a single chunk by primary key.

        Args:
            chunk_id: UUID string.

        Returns:
            Chunk dict, or ``{}`` if not found.
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM chunks WHERE id = ?", [chunk_id]
        ).fetchall()
        if not rows:
            return {}
        return self._rows_to_dicts(rows, conn)[0]
