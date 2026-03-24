"""
Abstract base class for all embedding backends.

Concrete implementations wrap sentence-transformers, FlagEmbedding,
or Vertex AI and expose a uniform interface to the rest of the pipeline.
"""

from abc import ABC, abstractmethod


class BaseEmbedding(ABC):
    """
    Uniform contract for every embedding model used in the pipeline.

    Implementations are expected to:
    - Lazy-load the underlying model on first use.
    - Normalise output vectors to unit length where applicable.
    - Handle batching internally to respect rate limits.
    """

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of document passages.

        Args:
            texts: List of passage strings to embed.

        Returns:
            List of float vectors, one per input text.
        """

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query string.

        Some models (e.g. E5) prepend a prefix for queries vs passages;
        this method encapsulates that distinction.

        Args:
            text: Query string.

        Returns:
            Float vector for the query.
        """

    @abstractmethod
    def get_dimension(self) -> int:
        """
        Return the output vector dimensionality.

        Returns:
            Integer dimension (e.g. 384, 768, 1024).
        """

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Return the canonical model identifier string.

        Returns:
            Model name string as stored in chunk metadata.
        """

    @abstractmethod
    def get_max_seq_length(self) -> int:
        """
        Return the maximum input sequence length in tokens.

        Used by the chunker to guarantee every chunk fits in the model's
        context window.  Must match the value reported by the underlying
        SentenceTransformer (``model.max_seq_length``).

        Returns:
            Maximum number of tokens the model can process per input.
        """
