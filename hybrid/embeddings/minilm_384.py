"""
MiniLM-384 embedding model.

Uses sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2.
Fast, lightweight, 384-dimensional.  Good for high-throughput scenarios
where quality can be traded for speed.
"""

from .base import BaseEmbedding

_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_DIMENSION = 384


class MiniLMEmbedding(BaseEmbedding):
    """
    Multilingual MiniLM-L12 embedding model (384 dimensions).

    Lazy-loads the SentenceTransformer on first use so that importing
    the class does not trigger a model download.
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

        Args:
            texts: Passages to embed.

        Returns:
            List of 384-dimensional float vectors.
        """
        self._load()
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return [emb.tolist() for emb in embeddings]

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query string.

        Args:
            text: Query text.

        Returns:
            384-dimensional float vector.
        """
        self._load()
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def get_dimension(self) -> int:
        """Return vector dimension: 384."""
        return _DIMENSION

    def get_model_name(self) -> str:
        """Return the HuggingFace model identifier."""
        return _MODEL_ID
