"""
Embedding model factory.

Usage::

    from hybrid.embeddings import get_embedding_model
    model = get_embedding_model("bge-m3")
    vectors = model.embed_documents(["Hello world", "Bonjour monde"])
"""

from .base import BaseEmbedding
from .minilm_384 import MiniLMEmbedding
from .mpnet_768 import MPNetEmbedding
from .e5_large_1024 import E5LargeEmbedding
from .bge_m3 import BGEM3Embedding
from .vertex_embedding import VertexEmbedding

_REGISTRY: dict[str, type[BaseEmbedding]] = {
    "minilm-384": MiniLMEmbedding,
    "mpnet-768": MPNetEmbedding,
    "e5-large-1024": E5LargeEmbedding,
    "bge-m3": BGEM3Embedding,
    "vertex": VertexEmbedding,
}


def get_embedding_model(model_name: str) -> BaseEmbedding:
    """
    Instantiate and return an embedding model by name.

    Supported keys: ``"minilm-384"``, ``"mpnet-768"``, ``"e5-large-1024"``,
    ``"bge-m3"``, ``"vertex"``.

    Args:
        model_name: One of the supported model keys.

    Returns:
        A fresh, lazy-loaded ``BaseEmbedding`` instance.

    Raises:
        KeyError: If *model_name* is not in the registry.
    """
    if model_name not in _REGISTRY:
        raise KeyError(
            f"Unknown embedding model '{model_name}'. "
            f"Available: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[model_name]()


__all__ = [
    "BaseEmbedding",
    "MiniLMEmbedding",
    "MPNetEmbedding",
    "E5LargeEmbedding",
    "BGEM3Embedding",
    "VertexEmbedding",
    "get_embedding_model",
]
