"""
MPNet-768 embedding model.

Uses sentence-transformers/paraphrase-multilingual-mpnet-base-v2.
768-dimensional, stronger than MiniLM for semantic similarity tasks
while still supporting 50+ languages.
"""

from .base import BaseEmbedding

_MODEL_ID = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
_DIMENSION = 768


class MPNetEmbedding(BaseEmbedding):
    """
    Multilingual MPNet-base embedding model (768 dimensions).

    Lazy-loads the SentenceTransformer on first use.
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
            List of 768-dimensional float vectors.
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
            768-dimensional float vector.
        """
        self._load()
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def get_dimension(self) -> int:
        """Return vector dimension: 768."""
        return _DIMENSION

    def get_model_name(self) -> str:
        """Return the HuggingFace model identifier."""
        return _MODEL_ID
