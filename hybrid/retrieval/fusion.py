"""
Reciprocal Rank Fusion (RRF) for hybrid retrieval.

Merges dense and sparse result lists into a single ranked list using the
RRF formula.  The weighted variant used here allows tuning the relative
importance of each signal:

    score(chunk) = dense_weight  * 1/(k + rank_dense)
                 + sparse_weight * 1/(k + rank_sparse)

Where ``k`` (default 60) controls the influence of low-ranked results.
"""


def reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = 60,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
) -> list[dict]:
    """
    Merge dense and sparse results with Reciprocal Rank Fusion.

    Chunks that appear in both lists receive contributions from both;
    chunks only in one list still get a contribution from that list's rank.

    Args:
        dense_results:  Ranked list of chunk dicts from dense search.
                        Must each have an ``"id"`` key.
        sparse_results: Ranked list of chunk dicts from sparse search.
        k:              RRF smoothing constant (default 60).
        dense_weight:   Weight for dense rank contribution (default 0.7).
        sparse_weight:  Weight for sparse rank contribution (default 0.3).

    Returns:
        Merged, deduplicated list of ``ScoredChunk`` dicts sorted by
        descending RRF score.  Each dict contains:

        - All original chunk fields from the higher-scoring source.
        - ``"score"``        : final RRF score
        - ``"rank_dense"``   : 1-based rank in dense list (0 = absent)
        - ``"rank_sparse"``  : 1-based rank in sparse list (0 = absent)
        - ``"score_dense"``  : normalised dense score (0.0 if absent)
        - ``"score_sparse"`` : normalised sparse score (0.0 if absent)
    """
    # Build lookup: chunk_id → (rank, score, chunk_dict) for each list
    dense_index: dict[str, tuple[int, float, dict]] = {}
    for rank, chunk in enumerate(dense_results, start=1):
        chunk_id = chunk.get("id", "")
        if chunk_id:
            dense_index[chunk_id] = (rank, chunk.get("score", 0.0), chunk)

    sparse_index: dict[str, tuple[int, float, dict]] = {}
    for rank, chunk in enumerate(sparse_results, start=1):
        chunk_id = chunk.get("id", "")
        if chunk_id:
            sparse_index[chunk_id] = (rank, chunk.get("score", 0.0), chunk)

    # Only chunks validated by dense — sparse alone is not enough
    all_ids = set(dense_index.keys())

    scored: list[dict] = []
    for chunk_id in all_ids:
        rank_d, score_d, chunk_d = dense_index.get(chunk_id, (0, 0.0, {}))
        rank_s, score_s, chunk_s = sparse_index.get(chunk_id, (0, 0.0, {}))

        rrf_score = 0.0
        if rank_d > 0:
            rrf_score += dense_weight * (1.0 / (k + rank_d))
        if rank_s > 0:
            rrf_score += sparse_weight * (1.0 / (k + rank_s))

        # Use the chunk dict that actually has data
        base_chunk = chunk_d if chunk_d else chunk_s

        scored.append(
            {
                **base_chunk,
                "score":        rrf_score,
                "rank_dense":   rank_d,
                "rank_sparse":  rank_s,
                "score_dense":  score_d,
                "score_sparse": score_s,
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored
