"""
E5-Base-768 embedding model.

Uses intfloat/multilingual-e5-base (768 dimensions, 512 max tokens).
Excellent FR/EN quality, same dimension as mpnet-768 — no schema migration.
Requires "query: " prefix for queries and "passage: " prefix for documents.
"""

from .base import BaseEmbedding

_MODEL_ID = "intfloat/multilingual-e5-base"
_DIMENSION = 768
_MAX_SEQ_LENGTH = 512
_QUERY_PREFIX = "query: "
_PASSAGE_PREFIX = "passage: "


class E5BaseEmbedding(BaseEmbedding):
    """
    Multilingual E5-Base embedding model (768 dimensions).

    Automatically prepends the required "query: " / "passage: " prefixes
    so callers do not need to handle this themselves.
    """

    def __init__(self) -> None:
        """Initialise without loading the model yet."""
        self._model = None

    def _load(self) -> None:
        """Load the SentenceTransformer model if not already loaded."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(_MODEL_ID)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of document passages.

        Prepends the required "passage: " prefix to each text.

        Args:
            texts: Raw passage strings (no prefix needed).

        Returns:
            List of 768-dimensional float vectors.
        """
        self._load()
        prefixed = [_PASSAGE_PREFIX + t for t in texts]
        embeddings = self._model.encode(prefixed, normalize_embeddings=True)
        return [emb.tolist() for emb in embeddings]

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query string.

        Prepends the required "query: " prefix.

        Args:
            text: Raw query string (no prefix needed).

        Returns:
            768-dimensional float vector.
        """
        self._load()
        embedding = self._model.encode(
            _QUERY_PREFIX + text, normalize_embeddings=True
        )
        return embedding.tolist()

    def get_dimension(self) -> int:
        """Return vector dimension: 768."""
        return _DIMENSION

    def get_model_name(self) -> str:
        """Return the HuggingFace model identifier."""
        return _MODEL_ID

    def get_max_seq_length(self) -> int:
        """Return max input tokens: 512."""
        return _MAX_SEQ_LENGTH
