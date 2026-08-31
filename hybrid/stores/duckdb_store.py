"""
DuckDB-backed store — one table per index.

Each logical index gets its own ``chunks_{name}`` table with a dedicated
HNSW vector index and FTS index.  This guarantees:
- HNSW is used at 100% for dense search (no cross-index pollution)
- No post-filtering by index_name — table routing handles isolation
- DROP TABLE for instant index deletion (vs DELETE with full scan)

Performance tiers (dense search, in order):
- Primary    → array_cosine_distance ORDER BY + HNSW — O(log n) ANN
- Secondary  → array_cosine_similarity full scan — O(n)
- Last resort → NumPy vectorised cosine similarity
"""

import logging
import os
import re
import threading
from typing import Optional

import numpy as np

from .base import BaseStore

# Embedding vector dimension — must match DEFAULT_EMBEDDING_MODEL in hybrid/config.py
EMBEDDING_DIM: int = 768

_CREATE_INDEXES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS hybrid_indexes (
    index_name      VARCHAR PRIMARY KEY,
    embedding_model VARCHAR,
    chunk_strategy  VARCHAR,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

_CHUNK_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS {table} (
    id              VARCHAR PRIMARY KEY,
    index_name      VARCHAR,
    content         TEXT,
    embedding       FLOAT[{dim}],
    source_url      VARCHAR,
    file_name       VARCHAR,
    file_type       VARCHAR,
    created_at      DATE,
    updated_at      DATE,
    author          VARCHAR,
    domaine         VARCHAR,
    langue          VARCHAR,
    tags            VARCHAR[],
    ligne_produit   VARCHAR,
    zone            VARCHAR,
    audience_role   VARCHAR[],
    matiere         VARCHAR[],
    chunk_index     INTEGER,
    chunk_total     INTEGER,
    chunk_strategy  VARCHAR,
    parent_chunk_id VARCHAR,
    embedding_model VARCHAR,
    embedding_dim   INTEGER
)
"""

# Columns added after the original schema — kept here so existing tables
# (ingested before the atelier model) can be migrated in place with a cheap
# idempotent ALTER. Order matches the INSERT below.
_ATELIER_COLUMNS: tuple[tuple[str, str], ...] = (
    ("ligne_produit", "VARCHAR"),
    ("zone", "VARCHAR"),
    ("audience_role", "VARCHAR[]"),
    ("matiere", "VARCHAR[]"),
)

# Named columns (not positional VALUES) so the INSERT is independent of the
# physical column order — a table migrated with ALTER ADD COLUMN has the new
# columns appended at the end, a freshly-created table has them mid-schema.
_INSERT_COLUMNS = (
    "id", "index_name", "content", "embedding", "source_url", "file_name",
    "file_type", "created_at", "updated_at", "author", "domaine", "langue",
    "tags", "ligne_produit", "zone", "audience_role", "matiere",
    "chunk_index", "chunk_total", "chunk_strategy", "parent_chunk_id",
    "embedding_model", "embedding_dim",
)
_INSERT_SQL = (
    "INSERT OR REPLACE INTO {table} (" + ", ".join(_INSERT_COLUMNS) + ") VALUES ("
    + ", ".join("?" for _ in _INSERT_COLUMNS) + ")"
)


def _sanitize(name: str) -> str:
    """Convert an index_name to a safe SQL identifier suffix."""
    safe = re.sub(r"[^a-z0-9_]", "_", name.lower()).strip("_")
    return safe or "default"


class DuckDBStore(BaseStore):
    """
    Hybrid store backed by a local DuckDB file.

    One table per index: ``chunks_{index_name}``.
    Each table has its own HNSW vector index and FTS index.

    Args:
        db_path: Path to the ``.duckdb`` file.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._conn = None
        self._vss_available = False
        # One base connection per process, but a SEPARATE cursor per thread.
        #
        # A DuckDBPyConnection carries the pending result of the last execute():
        # `conn.execute(sql).fetchall()` is two steps over shared state. With
        # several users served concurrently (Streamlit runs one thread per
        # session, and the agent spawns another per request), one thread's
        # execute() overwrites another's pending result, which then fetches
        # None. Measured on this store: 2 of 3 concurrent readers failed with
        # "'NoneType' object is not subscriptable", and the error was swallowed
        # by the retrieval fallbacks — users saw silently empty answers.
        #
        # cursor() opens a lightweight connection onto the same database, so
        # each thread gets its own result slot. Extensions and registered
        # indexes are inherited from the parent connection.
        self._open_lock = threading.Lock()   # guards base-connection creation
        self._write_lock = threading.RLock()  # serialises DDL / inserts
        self._local = threading.local()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _base_conn(self):
        """Create (once) the process-wide connection with its extensions."""
        if self._conn is not None:
            return self._conn
        with self._open_lock:
            if self._conn is not None:      # another thread won the race
                return self._conn
            import duckdb

            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            conn = duckdb.connect(
                self.db_path,
                config={"hnsw_enable_experimental_persistence": True},
            )
            self._conn = conn
            # FTS extension
            try:
                self._conn.execute("LOAD fts")
            except Exception:
                try:
                    self._conn.execute("INSTALL fts")
                    self._conn.execute("LOAD fts")
                except Exception:
                    pass
            # VSS extension (HNSW)
            try:
                self._conn.execute("LOAD vss")
                self._vss_available = True
            except Exception:
                try:
                    self._conn.execute("INSTALL vss")
                    self._conn.execute("LOAD vss")
                    self._vss_available = True
                except Exception:
                    self._vss_available = False
            # Registry table only — chunk tables are created per-index in initialize()
            self._conn.execute(_CREATE_INDEXES_TABLE_SQL)
        return self._conn

    def _get_conn(self):
        """Return this thread's cursor onto the shared database.

        Every caller keeps using ``_get_conn()`` exactly as before; the only
        difference is that two threads no longer share one result slot.
        """
        base = self._base_conn()
        cur = getattr(self._local, "cur", None)
        if cur is None:
            cur = base.cursor()
            self._local.cur = cur
        return cur

    def _tbl(self, index_name: str) -> str:
        """Return the SQL table name for this index."""
        return f"chunks_{_sanitize(index_name)}"

    def _hnsw_idx(self, index_name: str) -> str:
        """Return the HNSW index name for this index."""
        return f"{self._tbl(index_name)}_hnsw_idx"

    def _rebuild_fts(self, index_name: str) -> None:
        """Rebuild the FTS index for this index's table (non-fatal)."""
        table = self._tbl(index_name)
        try:
            self._get_conn().execute(
                f"PRAGMA create_fts_index('{table}', 'id', 'content', overwrite=1)"
            )
        except Exception:
            pass

    def _ensure_hnsw(self, index_name: str) -> None:
        """Create the HNSW index if it does not already exist (non-fatal)."""
        if not self._vss_available:
            return
        table = self._tbl(index_name)
        idx = self._hnsw_idx(index_name)
        try:
            exists = self._get_conn().execute(
                f"SELECT COUNT(*) FROM duckdb_indexes() WHERE index_name='{idx}'"
            ).fetchone()[0]
            if exists:
                return
            count = self._get_conn().execute(
                f"SELECT COUNT(*) FROM {table} WHERE embedding IS NOT NULL"
            ).fetchone()[0]
            if count == 0:
                return
            self._get_conn().execute(
                f"CREATE INDEX {idx} ON {table} USING HNSW (embedding) WITH (metric='cosine')"
            )
        except Exception:
            pass

    def _rebuild_hnsw(self, index_name: str) -> None:
        """
        Drop and recreate the HNSW index after data changes.

        DuckDB VSS does not support incremental HNSW updates — the index
        must be rebuilt after every insert/delete.
        """
        if not self._vss_available:
            return
        table = self._tbl(index_name)
        idx = self._hnsw_idx(index_name)
        try:
            count = self._get_conn().execute(
                f"SELECT COUNT(*) FROM {table} WHERE embedding IS NOT NULL"
            ).fetchone()[0]
            if count == 0:
                return
            self._get_conn().execute(f"DROP INDEX IF EXISTS {idx}")
            self._get_conn().execute(
                f"CREATE INDEX {idx} ON {table} USING HNSW (embedding) WITH (metric='cosine')"
            )
        except Exception:
            pass

    def _build_where(self, filters: dict, prefix: str = "") -> tuple[str, list]:
        """
        Build a SQL WHERE fragment from metadata filters.

        index_name is excluded — table routing handles isolation.
        """
        clauses: list[str] = []
        params: list = []

        for key, col in (
            ("file_type", "file_type"),
            ("domaine",   "domaine"),
            ("langue",    "langue"),
            ("author",    "author"),
        ):
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
        columns = [d[0] for d in conn.description]
        return [dict(zip(columns, row)) for row in rows]

    # ------------------------------------------------------------------
    # BaseStore interface
    # ------------------------------------------------------------------

    def initialize(self, index_name: str, embedding_model: str = "", chunk_strategy: str = "") -> None:
        """
        Create the per-index table, FTS index, and HNSW index if absent.
        Idempotent — safe to call multiple times.
        """
        conn = self._get_conn()
        table = self._tbl(index_name)
        with self._write_lock:
            conn.execute(_CHUNK_TABLE_SQL.format(table=table, dim=EMBEDDING_DIM))
            conn.execute(
                "INSERT OR IGNORE INTO hybrid_indexes (index_name, embedding_model, chunk_strategy) VALUES (?, ?, ?)",
                [index_name, embedding_model, chunk_strategy],
            )
            self._rebuild_fts(index_name)
            self._rebuild_hnsw(index_name)

    def insert_chunks(self, chunks: list[dict]) -> None:
        """
        Bulk-insert chunks into their respective per-index tables.
        Rebuilds FTS and HNSW for each affected index after insert.
        """
        conn = self._get_conn()

        # Group by index_name so we rebuild indexes once per index
        from collections import defaultdict
        by_index: dict[str, list] = defaultdict(list)
        for chunk in chunks:
            by_index[chunk["index_name"]].append(chunk)

        for index_name, index_chunks in by_index.items():
            table = self._tbl(index_name)
            # One writer at a time: an insert or an index rebuild running while
            # another thread reads would race on the same tables.
            with self._write_lock:
                self._ensure_atelier_columns(table)   # migrate old tables in place
                sql = _INSERT_SQL.format(table=table)
                for chunk in index_chunks:
                    conn.execute(sql, [
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
                        chunk.get("ligne_produit", ""),
                        chunk.get("zone", ""),
                        chunk.get("audience_role", []),
                        chunk.get("matiere", []),
                        chunk.get("chunk_index", 0),
                        chunk.get("chunk_total", 1),
                        chunk.get("chunk_strategy", "fixed"),
                        chunk.get("parent_chunk_id"),
                        chunk.get("embedding_model", ""),
                        chunk.get("embedding_dim", 0),
                    ])
                self._rebuild_fts(index_name)
                self._rebuild_hnsw(index_name)

    def _ensure_atelier_columns(self, table: str) -> None:
        """Add the atelier metadata columns to *table* if missing (idempotent).

        Lets tables created before the atelier model accept the extended INSERT
        without a manual migration. DuckDB's ADD COLUMN IF NOT EXISTS is a no-op
        when the column already exists, so this is safe to call on every insert.
        """
        conn = self._get_conn()
        for col, sqltype in _ATELIER_COLUMNS:
            conn.execute(
                f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} {sqltype}"
            )

    def dense_search(
        self,
        index_name: str,
        embedding: list[float],
        top_k: int,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """
        Return the *top_k* most similar chunks using cosine similarity.

        Queries only ``chunks_{index_name}`` — no cross-index scan.
        HNSW is triggered at 100% (no WHERE index_name filter to bypass it).

        Strategy (in order):
        1. HNSW ANN via array_cosine_distance ORDER BY — O(log n)
        2. array_cosine_similarity full scan — O(n)
        3. NumPy vectorised fallback
        """
        from hybrid.config import DENSE_SCORE_THRESHOLD

        conn = self._get_conn()
        table = self._tbl(index_name)
        filters = filters or {}
        where_clause, params = self._build_where(filters)

        def _apply_threshold(results: list[dict]) -> list[dict]:
            return [r for r in results if r.get("score", 0) >= DENSE_SCORE_THRESHOLD]

        # ── Primary: HNSW ANN — O(log n) ──────────────────────────────────
        if self._vss_available:
            self._ensure_hnsw(index_name)
            try:
                sql = f"""
                    SELECT *, (1.0 - array_cosine_distance(
                        embedding, ?::FLOAT[{EMBEDDING_DIM}]
                    )) AS score
                    FROM {table}
                    WHERE embedding IS NOT NULL
                    {where_clause}
                    ORDER BY array_cosine_distance(embedding, ?::FLOAT[{EMBEDDING_DIM}])
                    LIMIT ?
                """
                rows = conn.execute(sql, [embedding] + params + [embedding, top_k]).fetchall()
                return _apply_threshold(self._rows_to_dicts(rows, conn))
            except Exception:
                pass

        # ── Secondary: full scan ───────────────────────────────────────────
        try:
            sql = f"""
                SELECT *, array_cosine_similarity(
                    embedding, ?::FLOAT[{EMBEDDING_DIM}]
                ) AS score
                FROM {table}
                WHERE embedding IS NOT NULL
                {where_clause}
                ORDER BY score DESC
                LIMIT ?
            """
            rows = conn.execute(sql, [embedding] + params + [top_k]).fetchall()
            return _apply_threshold(self._rows_to_dicts(rows, conn))
        except Exception:
            pass

        # ── Fallback: NumPy ────────────────────────────────────────────────
        sql = f"SELECT * FROM {table} WHERE embedding IS NOT NULL {where_clause}"
        rows = conn.execute(sql, params).fetchall()
        columns = [d[0] for d in conn.description]

        if not rows:
            return []

        emb_col = columns.index("embedding")
        all_dicts = [dict(zip(columns, row)) for row in rows]
        query_vec = np.array(embedding, dtype=np.float32)
        matrix    = np.array([row[emb_col] for row in rows], dtype=np.float32)
        norms     = np.linalg.norm(matrix, axis=1)
        q_norm    = float(np.linalg.norm(query_vec))

        with np.errstate(invalid="ignore", divide="ignore"):
            scores = matrix @ query_vec / (norms * q_norm + 1e-9)

        for d, s in zip(all_dicts, scores.tolist()):
            d["score"] = s

        all_dicts.sort(key=lambda x: x["score"], reverse=True)
        return _apply_threshold(all_dicts[:top_k])

    def sparse_search(
        self,
        index_name: str,
        query: str,
        top_k: int,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """
        Return the *top_k* most relevant chunks via BM25.

        Queries only ``chunks_{index_name}``.
        Falls back to ILIKE if FTS is unavailable.
        """
        conn = self._get_conn()
        table = self._tbl(index_name)
        filters = filters or {}
        where_clause, params = self._build_where(filters, prefix="c.")

        try:
            sql = f"""
                SELECT c.*, fts_main_{table}.match_bm25(c.id, ?) AS score
                FROM {table} c
                WHERE score IS NOT NULL
                {where_clause}
                ORDER BY score DESC
                LIMIT ?
            """
            rows = conn.execute(sql, [query] + params + [top_k]).fetchall()
            return self._rows_to_dicts(rows, conn)
        except Exception as exc:
            # FTS unavailable (extension missing, index never built, …).
            #
            # The previous fallback matched the WHOLE query string with a single
            # ILIKE, so any question longer than one word silently returned zero
            # rows — indistinguishable from "BM25 found nothing". It also
            # swallowed the error, leaving nothing to diagnose. Now the failure
            # is logged, and the fallback matches individual terms so keyword
            # search degrades instead of disappearing.
            logging.warning(
                "FTS unavailable on '%s' (%s) — falling back to term matching. "
                "Rebuild with PRAGMA create_fts_index if this persists.",
                table, exc,
            )
            terms = [t for t in re.split(r"\W+", query) if len(t) > 2][:8]
            if not terms:
                return []

            where_no_prefix = where_clause.replace("c.", "")
            # Rank by how many query terms a chunk contains — a poor man's BM25,
            # but it puts the most on-topic passages first instead of none.
            score_expr = " + ".join(
                "CASE WHEN content ILIKE ? THEN 1 ELSE 0 END" for _ in terms
            )
            like_params = [f"%{t}%" for t in terms]
            sql = f"""
                SELECT *, ({score_expr}) AS score
                FROM {table}
                WHERE ({" OR ".join("content ILIKE ?" for _ in terms)})
                {where_no_prefix}
                ORDER BY score DESC
                LIMIT ?
            """
            rows = conn.execute(
                sql, like_params + like_params + params + [top_k]
            ).fetchall()
            return self._rows_to_dicts(rows, conn)

    def delete_chunks_by_file_name(self, file_name: str, index_name: str) -> None:
        """Delete all chunks for a file within one index, then rebuild indexes."""
        conn = self._get_conn()
        table = self._tbl(index_name)
        with self._write_lock:
            conn.execute(f"DELETE FROM {table} WHERE file_name = ?", [file_name])
            self._rebuild_fts(index_name)
            self._rebuild_hnsw(index_name)

    def delete_index(self, index_name: str) -> None:
        """Drop the index table entirely and remove from registry."""
        conn = self._get_conn()
        table = self._tbl(index_name)
        with self._write_lock:
            conn.execute(f"DROP TABLE IF EXISTS {table}")
            conn.execute("DELETE FROM hybrid_indexes WHERE index_name = ?", [index_name])

    def list_indexes(self) -> list[str]:
        """Return all registered index names from the registry table."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT index_name FROM hybrid_indexes ORDER BY index_name"
        ).fetchall()
        return [row[0] for row in rows if row[0]]

    def get_chunk_by_id(self, chunk_id: str) -> dict:
        """Retrieve a chunk by ID, searching across all index tables."""
        conn = self._get_conn()
        for index_name in self.list_indexes():
            table = self._tbl(index_name)
            rows = conn.execute(
                f"SELECT * FROM {table} WHERE id = ?", [chunk_id]
            ).fetchall()
            if rows:
                return self._rows_to_dicts(rows, conn)[0]
        return {}
