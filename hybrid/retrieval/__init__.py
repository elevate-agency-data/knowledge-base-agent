"""
Retrieval layer — dense, sparse, and hybrid search.

Public API::

    from hybrid.retrieval.dense  import dense_search
    from hybrid.retrieval.sparse import sparse_search
    from hybrid.retrieval.fusion import reciprocal_rank_fusion
    from hybrid.retrieval.filter import build_filters, apply_post_filter
"""

from .dense  import dense_search
from .sparse import sparse_search
from .fusion import reciprocal_rank_fusion
from .filter import build_filters, apply_post_filter

__all__ = [
    "dense_search",
    "sparse_search",
    "reciprocal_rank_fusion",
    "build_filters",
    "apply_post_filter",
]
