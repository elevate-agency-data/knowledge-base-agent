"""
Retrieval evaluation benchmark.

Run the full benchmark from the project root::

    python -m hybrid.benchmark.eval_retrieval

Or with a custom test file::

    python -m hybrid.benchmark.eval_retrieval --test-file path/to/queries.json
"""

from .metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
    precision_at_k,
    compute_all_metrics,
)

__all__ = [
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "recall_at_k",
    "precision_at_k",
    "compute_all_metrics",
]
