"""
Vertex AI text-embedding-005 embedding model.

Only active when ENV == "gcp".  Uses the Google Cloud AI Platform SDK
and handles rate-limiting via batch processing.
"""

import time
from .base import BaseEmbedding
from hybrid.config import (
    PROJECT_ID,
    LOCATION,
    SERVICE_ACCOUNT_PATH,
    VERTEX_EMBEDDING_MODEL,
    EMBEDDING_REQUESTS_PER_MIN,
)

_MODEL_ID = VERTEX_EMBEDDING_MODEL
_DIMENSION = 768
_MAX_SEQ_LENGTH = 2048


class VertexEmbedding(BaseEmbedding):
    """
    Google Vertex AI text-embedding-005 model (768 dimensions).

    Automatically batches requests to stay within the configured
    EMBEDDING_REQUESTS_PER_MIN rate limit.  Uses a service-account
    credential file for authentication.
    """

    def __init__(self) -> None:
        """Initialise without authenticating yet (lazy init)."""
        self._client = None

    def _load(self) -> None:
        """Authenticate and build the Vertex AI client if needed."""
        if self._client is not None:
            return

        import vertexai
        from google.oauth2 import service_account

        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_PATH,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=creds)

        from vertexai.language_models import TextEmbeddingModel
        self._client = TextEmbeddingModel.from_pretrained(_MODEL_ID)

    # ------------------------------------------------------------------
    # Rate-limit helpers
    # ------------------------------------------------------------------

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed one batch of texts and return vectors.

        Args:
            texts: Up to ~250 texts (Vertex batch limit).

        Returns:
            List of 768-dimensional float vectors.
        """
        embeddings = self._client.get_embeddings(texts)
        return [emb.values for emb in embeddings]

    def _embed_with_rate_limit(self, texts: list[str]) -> list[list[float]]:
        """
        Embed *texts* in batches, sleeping between batches to respect
        EMBEDDING_REQUESTS_PER_MIN.

        Args:
            texts: Arbitrary number of texts to embed.

        Returns:
            List of float vectors in the same order as *texts*.
        """
        self._load()
        batch_size = min(250, max(1, EMBEDDING_REQUESTS_PER_MIN // 4))
        results: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            results.extend(self._embed_batch(batch))
            if i + batch_size < len(texts):
                # Sleep to stay within requests-per-minute limit
                sleep_s = 60.0 / (EMBEDDING_REQUESTS_PER_MIN / batch_size)
                time.sleep(sleep_s)
        return results

    # ------------------------------------------------------------------
    # BaseEmbedding interface
    # ------------------------------------------------------------------

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of document passages via Vertex AI.

        Args:
            texts: Passages to embed.

        Returns:
            List of 768-dimensional float vectors.
        """
        return self._embed_with_rate_limit(texts)

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query string via Vertex AI.

        Args:
            text: Query string.

        Returns:
            768-dimensional float vector.
        """
        self._load()
        result = self._embed_batch([text])
        return result[0]

    def get_dimension(self) -> int:
        """Return vector dimension: 768."""
        return _DIMENSION

    def get_model_name(self) -> str:
        """Return the Vertex AI model resource path."""
        return _MODEL_ID

    def get_max_seq_length(self) -> int:
        """Return max input tokens: 2048."""
        return _MAX_SEQ_LENGTH
