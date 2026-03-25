"""
AlloyDB + pgvector store for GCP production.

Uses asyncpg for async I/O with a connection pool.  Dense search leverages
pgvector's ``<=>`` cosine operator; sparse search uses PostgreSQL's native
tsvector/tsquery full-text search.

Sync wrappers are provided so that the store can be used from synchronous
ADK tool functions via ``asyncio.run()``.
"""

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Optional

from .base import BaseStore
from hybrid.config import ALLOYDB_CONNECTION_STRING

_CREATE_EXTENSION_SQL = "CREATE EXTENSION IF NOT EXISTS vector"

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS chunks (
    id              VARCHAR PRIMARY KEY,
    index_name      VARCHAR,
    content         TEXT,
    embedding       vector(1024),
    source_url      VARCHAR,
    file_name       VARCHAR,
    file_type       VARCHAR,
    created_at      DATE,
    updated_at      DATE,
    author          VARCHAR,
    domaine         VARCHAR,
    langue          VARCHAR,
    tags            TEXT[],
    chunk_index     INTEGER,
    chunk_total     INTEGER,
    chunk_strategy  VARCHAR,
    parent_chunk_id VARCHAR,
    embedding_model VARCHAR,
    embedding_dim   INTEGER,
    content_tsv     TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED
)
"""

_CREATE_VECTOR_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100)
"""

_CREATE_FTS_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS chunks_tsv_idx
    ON chunks USING GIN (content_tsv)
"""


class AlloyDBStore(BaseStore):
    """
    Production store backed by AlloyDB (PostgreSQL-compatible) + pgvector.

    Requires ``ALLOYDB_CONNECTION_STRING`` in ``hybrid/config.py``.

    All async methods are exposed directly; sync wrappers (``initialize``,
    ``insert_chunks``, etc.) use ``asyncio.run()`` for compatibility with
    synchronous ADK tool functions.

    Args:
        connection_string: Override the config connection string (for tests).
        pool_min_size:     Minimum asyncpg pool size.
        pool_max_size:     Maximum asyncpg pool size.
    """

    def __init__(
        self,
        connection_string: str = "",
        pool_min_size: int = 2,
        pool_max_size: int = 10,
    ) -> None:
        """Initialise without connecting yet (lazy pool creation)."""
        self._dsn = connection_string or ALLOYDB_CONNECTION_STRING
        self._pool_min = pool_min_size
        self._pool_max = pool_max_size
        self._pool = None

    # ------------------------------------------------------------------
    # Pool management
    # ------------------------------------------------------------------

    async def _get_pool(self):
        """Create (or return existing) asyncpg connection pool."""
        if self._pool is None:
            import asyncpg

            self._pool = await asyncpg.create_pool(
                dsn=self._dsn,
                min_size=self._pool_min,
                max_size=self._pool_max,
            )
        return self._pool

    async def _close_pool(self) -> None:
        """Gracefully close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @asynccontextmanager
    async def _acquire(self):
        """Context manager that acquires a connection from the pool."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            yield conn

    # ------------------------------------------------------------------
    # Async implementations
    # ------------------------------------------------------------------

    async def _async_initialize(self, index_name: str) -> None:
        """Async implementation of initialize()."""
        async with self._acquire() as conn:
            await conn.execute(_CREATE_EXTENSION_SQL)
            await conn.execute(_CREATE_TABLE_SQL)
            await conn.execute(_CREATE_VECTOR_INDEX_SQL)
            await conn.execute(_CREATE_FTS_INDEX_SQL)

    async def _async_insert_chunks(self, chunks: list[dict]) -> None:
        """Async implementation of insert_chunks()."""
        async with self._acquire() as conn:
            rows = []
            for c in chunks:
                embedding_str = (
                    "[" + ",".join(str(v) for v in c["embedding"]) + "]"
                    if c.get("embedding")
                    else None
                )
                rows.append(
                    (
                        c["id"],
                        c["index_name"],
                        c["content"],
                        embedding_str,
                        c.get("source_url", ""),
                        c.get("file_name", ""),
                        c.get("file_type", ""),
                        c.get("created_at"),
                        c.get("updated_at"),
                        c.get("author", ""),
                        c.get("domaine", ""),
                        c.get("langue", ""),
                        c.get("tags", []),
                        c.get("chunk_index", 0),
                        c.get("chunk_total", 1),
                        c.get("chunk_strategy", "fixed"),
                        c.get("parent_chunk_id"),
                        c.get("embedding_model", ""),
                        c.get("embedding_dim", 0),
                    )
                )
            await conn.executemany(
                """
                INSERT INTO chunks (
                    id, index_name, content, embedding,
                    source_url, file_name, file_type,
                    created_at, updated_at, author, domaine, langue,
                    tags, chunk_index, chunk_total, chunk_strategy,
                    parent_chunk_id, embedding_model, embedding_dim
                ) VALUES (
                    $1,$2,$3,$4::vector,$5,$6,$7,$8,$9,$10,$11,$12,
                    $13,$14,$15,$16,$17,$18,$19
                )
                ON CONFLICT (id) DO UPDATE SET
                    content         = EXCLUDED.content,
                    embedding       = EXCLUDED.embedding,
                    source_url      = EXCLUDED.source_url,
                    updated_at      = EXCLUDED.updated_at
                """,
                rows,
            )

    async def _async_dense_search(
        self,
        index_name: str,
        embedding: list[float],
        top_k: int,
        filters: dict,
    ) -> list[dict]:
        """Async implementation of dense_search() using pgvector."""
        async with self._acquire() as conn:
            embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"
            where_clause, params = self._build_where(filters)
            sql = f"""
                SELECT *,
                    1 - (embedding <=> $1::vector) AS score
                FROM chunks
                WHERE embedding IS NOT NULL
                {where_clause}
                ORDER BY score DESC
                LIMIT ${len(params) + 2}
            """
            all_params = [embedding_str] + params + [top_k]
            rows = await conn.fetch(sql, *all_params)
            return [dict(row) for row in rows]

    async def _async_sparse_search(
        self,
        index_name: str,
        query: str,
        top_k: int,
        filters: dict,
    ) -> list[dict]:
        """Async implementation of sparse_search() using tsvector."""
        async with self._acquire() as conn:
            where_clause, params = self._build_where(filters)
            sql = f"""
                SELECT *,
                    ts_rank(content_tsv, plainto_tsquery('simple', $1)) AS score
                FROM chunks
                WHERE content_tsv @@ plainto_tsquery('simple', $1)
                {where_clause}
                ORDER BY score DESC
                LIMIT ${len(params) + 2}
            """
            all_params = [query] + params + [top_k]
            rows = await conn.fetch(sql, *all_params)
            return [dict(row) for row in rows]

    async def _async_delete_chunks_by_file_name(self, file_name: str, index_name: str) -> None:
        """Async implementation of delete_chunks_by_file_name()."""
        async with self._acquire() as conn:
            await conn.execute(
                "DELETE FROM chunks WHERE file_name = $1 AND index_name = $2",
                file_name,
                index_name,
            )

    async def _async_delete_index(self, index_name: str) -> None:
        """Async implementation of delete_index()."""
        async with self._acquire() as conn:
            await conn.execute(
                "DELETE FROM chunks WHERE index_name = $1", index_name
            )

    async def _async_list_indexes(self) -> list[str]:
        """Async implementation of list_indexes()."""
        async with self._acquire() as conn:
            rows = await conn.fetch(
                "SELECT DISTINCT index_name FROM chunks ORDER BY index_name"
            )
            return [row["index_name"] for row in rows if row["index_name"]]

    async def _async_get_chunk_by_id(self, chunk_id: str) -> dict:
        """Async implementation of get_chunk_by_id()."""
        async with self._acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM chunks WHERE id = $1", chunk_id
            )
            return dict(row) if row else {}

    # ------------------------------------------------------------------
    # Sync helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_where(filters: dict) -> tuple[str, list]:
        """
        Build a parameterised WHERE fragment from a filters dict.

        Parameters are numbered from $2 onward (since $1 is reserved for
        the primary query parameter in dense/sparse search).

        Args:
            filters: Dict with optional keys: index_name, file_type,
                     domaine, langue, author, date_from, date_to.

        Returns:
            Tuple of (clause_string, params_list).
        """
        clauses: list[str] = []
        params: list = []
        offset = 2  # $1 is used by the caller

        mapping = {
            "index_name": "index_name",
            "file_type": "file_type",
            "domaine": "domaine",
            "langue": "langue",
            "author": "author",
        }
        for key, col in mapping.items():
            if filters.get(key):
                clauses.append(f"AND {col} = ${offset}")
                params.append(filters[key])
                offset += 1

        if filters.get("date_from"):
            clauses.append(f"AND created_at >= ${offset}")
            params.append(filters["date_from"])
            offset += 1
        if filters.get("date_to"):
            clauses.append(f"AND created_at <= ${offset}")
            params.append(filters["date_to"])
            offset += 1

        return " ".join(clauses), params

    @staticmethod
    def _run(coro):
        """Run an async coroutine from synchronous context."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    future = ex.submit(asyncio.run, coro)
                    return future.result()
            return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)

    # ------------------------------------------------------------------
    # BaseStore interface (sync wrappers)
    # ------------------------------------------------------------------

    def initialize(self, index_name: str) -> None:
        """
        Create the chunks table, pgvector extension, and indexes.

        Idempotent — safe to call multiple times.

        Args:
            index_name: Logical index name (stored in chunk metadata).
        """
        self._run(self._async_initialize(index_name))

    def insert_chunks(self, chunks: list[dict]) -> None:
        """
        Upsert chunk dicts into AlloyDB.

        Args:
            chunks: List of chunk dicts with all required fields.
        """
        self._run(self._async_insert_chunks(chunks))

    def dense_search(
        self,
        index_name: str,
        embedding: list[float],
        top_k: int,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """
        Return top-k chunks by pgvector cosine similarity.

        Args:
            index_name: Index to search (used as WHERE filter in AlloyDB).
            embedding:  Query vector.
            top_k:      Number of results.
            filters:    Metadata filters.

        Returns:
            List of chunk dicts with ``"score"`` key.
        """
        merged = {**(filters or {}), "index_name": index_name}
        return self._run(self._async_dense_search(index_name, embedding, top_k, merged))

    def sparse_search(
        self,
        index_name: str,
        query: str,
        top_k: int,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """
        Return top-k chunks by PostgreSQL tsvector full-text ranking.

        Args:
            index_name: Index to search (used as WHERE filter in AlloyDB).
            query:      Query text.
            top_k:      Number of results.
            filters:    Metadata filters.

        Returns:
            List of chunk dicts with ``"score"`` key.
        """
        merged = {**(filters or {}), "index_name": index_name}
        return self._run(self._async_sparse_search(index_name, query, top_k, merged))

    def delete_chunks_by_file_name(self, file_name: str, index_name: str) -> None:
        """
        Delete all chunks for *file_name* within *index_name*.

        Args:
            file_name:  Exact file_name stored in the chunks table.
            index_name: Index to scope the deletion to.
        """
        self._run(self._async_delete_chunks_by_file_name(file_name, index_name))

    def delete_index(self, index_name: str) -> None:
        """
        Delete all chunks for *index_name*.

        Args:
            index_name: Index to purge.
        """
        self._run(self._async_delete_index(index_name))

    def list_indexes(self) -> list[str]:
        """
        Return a sorted list of all distinct index names.

        Returns:
            Sorted list of strings.
        """
        return self._run(self._async_list_indexes())

    def get_chunk_by_id(self, chunk_id: str) -> dict:
        """
        Retrieve a single chunk by primary key.

        Args:
            chunk_id: UUID string.

        Returns:
            Chunk dict, or ``{}`` if not found.
        """
        return self._run(self._async_get_chunk_by_id(chunk_id))
