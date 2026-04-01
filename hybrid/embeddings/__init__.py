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
from .e5_base_768 import E5BaseEmbedding
from .e5_large_1024 import E5LargeEmbedding
from .bge_m3 import BGEM3Embedding
from .vertex_embedding import VertexEmbedding

_REGISTRY: dict[str, type[BaseEmbedding]] = {
    "minilm-384": MiniLMEmbedding,
    "mpnet-768": MPNetEmbedding,
    "e5-base-768": E5BaseEmbedding,
    "e5-large-1024": E5LargeEmbedding,
    "bge-m3": BGEM3Embedding,
    "vertex": VertexEmbedding,
}


_CACHE: dict[str, BaseEmbedding] = {}


def get_embedding_model(model_name: str) -> BaseEmbedding:
    """
    Return a **cached** embedding model instance by name.

    The first call for a given *model_name* creates the instance and
    stores it in a module-level cache.  Subsequent calls return the
    same instance, avoiding the ~30 s SentenceTransformer reload.

    Supported keys: ``"minilm-384"``, ``"mpnet-768"``, ``"e5-base-768"``,
    ``"e5-large-1024"``, ``"bge-m3"``, ``"vertex"``.

    Args:
        model_name: One of the supported model keys.

    Returns:
        A cached ``BaseEmbedding`` instance (lazy-loaded internally).

    Raises:
        KeyError: If *model_name* is not in the registry.
    """
    if model_name in _CACHE:
        return _CACHE[model_name]
    if model_name not in _REGISTRY:
        raise KeyError(
            f"Unknown embedding model '{model_name}'. "
            f"Available: {sorted(_REGISTRY.keys())}"
        )
    instance = _REGISTRY[model_name]()
    _CACHE[model_name] = instance
    return instance


__all__ = [
    "BaseEmbedding",
    "MiniLMEmbedding",
    "MPNetEmbedding",
    "E5BaseEmbedding",
    "E5LargeEmbedding",
    "BGEM3Embedding",
    "VertexEmbedding",
    "get_embedding_model",
]
