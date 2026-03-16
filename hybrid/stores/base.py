"""
Abstract base class for all vector/hybrid stores.

Every store backend (DuckDB, AlloyDB, …) must implement this interface so
that the retrieval layer and tools remain backend-agnostic.
"""

from abc import ABC, abstractmethod


class BaseStore(ABC):
    """
    Contract that every store backend must satisfy.

    A store holds *chunks* — pre-embedded document fragments — and exposes
    both dense (vector cosine) and sparse (BM25 / full-text) search on them.
    """

    @abstractmethod
    def initialize(self, index_name: str) -> None:
        """
        Prepare the store to accept chunks for *index_name*.

        Creates tables, indexes, and FTS structures if they do not already
        exist.  Idempotent — calling it twice must not raise an error.

        Args:
            index_name: Logical name of the index (e.g. ``"hr-docs"``).
        """

    @abstractmethod
    def insert_chunks(self, chunks: list[dict]) -> None:
        """
        Bulk-insert a list of chunk dicts into the store.

        Each dict must contain at minimum the keys defined in the DuckDB
        schema: ``id``, ``index_name``, ``content``, ``embedding``, …

        Args:
            chunks: List of chunk metadata + embedding dicts.
        """

    @abstractmethod
    def dense_search(
        self,
        embedding: list[float],
        top_k: int,
        filters: dict,
    ) -> list[dict]:
        """
        Return the *top_k* chunks most similar to *embedding* (cosine).

        Args:
            embedding: Query vector produced by an embedding model.
            top_k:     Maximum number of results to return.
            filters:   Optional metadata filters (index_name, langue, …).

        Returns:
            List of chunk dicts, each augmented with a ``"score"`` key
            (cosine similarity in [−1, 1], higher is better).
        """

    @abstractmethod
    def sparse_search(
        self,
        query: str,
        top_k: int,
        filters: dict,
    ) -> list[dict]:
        """
        Return the *top_k* chunks matching *query* via BM25 / full-text.

        Args:
            query:   Raw text query.
            top_k:   Maximum number of results to return.
            filters: Optional metadata filters.

        Returns:
            List of chunk dicts, each augmented with a ``"score"`` key
            (BM25 score, higher is better).
        """

    @abstractmethod
    def delete_index(self, index_name: str) -> None:
        """
        Remove all chunks belonging to *index_name*.

        Args:
            index_name: Index to delete.
        """

    @abstractmethod
    def list_indexes(self) -> list[str]:
        """
        Return distinct index names present in the store.

        Returns:
            Sorted list of index name strings.
        """

    @abstractmethod
    def get_chunk_by_id(self, chunk_id: str) -> dict:
        """
        Retrieve a single chunk by its primary key.

        Args:
            chunk_id: UUID of the chunk.

        Returns:
            Chunk dict, or an empty dict if not found.
        """
