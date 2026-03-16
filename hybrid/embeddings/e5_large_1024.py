"""
E5-Large-1024 embedding model.

Uses intfloat/multilingual-e5-large (1024 dimensions).
Requires "query: " prefix for queries and "passage: " prefix for documents
as per the original E5 training setup.  Best quality among the local models.
"""

from .base import BaseEmbedding

_MODEL_ID = "intfloat/multilingual-e5-large"
_DIMENSION = 1024
_QUERY_PREFIX = "query: "
_PASSAGE_PREFIX = "passage: "


class E5LargeEmbedding(BaseEmbedding):
    """
    Multilingual E5-Large embedding model (1024 dimensions).

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
            List of 1024-dimensional float vectors.
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
            1024-dimensional float vector.
        """
        self._load()
        embedding = self._model.encode(
            _QUERY_PREFIX + text, normalize_embeddings=True
        )
        return embedding.tolist()

    def get_dimension(self) -> int:
        """Return vector dimension: 1024."""
        return _DIMENSION

    def get_model_name(self) -> str:
        """Return the HuggingFace model identifier."""
        return _MODEL_ID
