"""
Reciprocal Rank Fusion (RRF) for hybrid retrieval.

Merges dense and sparse result lists into a single ranked list using the
RRF formula.  The weighted variant used here allows tuning the relative
importance of each signal:

    score(chunk) = dense_weight  * 1/(k + rank_dense)
                 + sparse_weight * 1/(k + rank_sparse)

Where ``k`` (default 60) controls the influence of low-ranked results.

Note — RRF is *rank fusion*, not reranking: it only combines rank positions
and never re-reads the chunk text against the query. A cross-encoder reranker
would be a separate, later stage.

``dense_gated`` controls whether sparse-only hits survive the merge. See
``reciprocal_rank_fusion`` for the trade-off.
"""

import hashlib
import re


def reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = 60,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    dense_gated: bool = False,
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
        dense_gated:    When True, only chunks retrieved by dense search are
                        kept — sparse-only hits are dropped. This suppresses
                        BM25 noise but also loses exact-match hits the
                        embedding missed (product references, serial numbers,
                        part names), which defeats much of the point of hybrid
                        search. Default False: sparse-only hits are kept and
                        ranked on their sparse contribution alone.

    Returns:
        Merged, deduplicated (by chunk id) list of ``ScoredChunk`` dicts sorted
        by descending RRF score.  Each dict contains:

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

    # Which chunks are eligible for the merged list.
    # Gated  → only dense-validated chunks (drops exact-match sparse-only hits).
    # Open   → union, so BM25 can surface what the embedding missed.
    all_ids = (
        set(dense_index.keys())
        if dense_gated
        else set(dense_index.keys()) | set(sparse_index.keys())
    )

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


# ── Content deduplication ─────────────────────────────────────────────────────
#
# RRF already dedupes by chunk *id*. It cannot catch the same passage stored
# twice under different ids — the same care instructions copied into two files,
# a procedure duplicated across an old and a new document. Those surface as
# several near-identical chunks and waste the LLM context window.

_WS_RE = re.compile(r"\s+")


def _content_key(text: str) -> str:
    """Normalised fingerprint of a chunk's text (case/whitespace-insensitive)."""
    normalised = _WS_RE.sub(" ", (text or "").strip().lower())
    return hashlib.sha1(normalised.encode("utf-8")).hexdigest()


def deduplicate_by_content(
    results: list[dict],
    content_key: str = "content",
) -> list[dict]:
    """
    Drop chunks whose normalised text was already seen, keeping the best-ranked.

    Input order is authoritative: the first occurrence wins, so pass an
    already-sorted list (e.g. straight out of ``reciprocal_rank_fusion``).
    Only exact matches after whitespace/case normalisation are removed — this
    is deliberately conservative: it never discards a chunk that merely *looks*
    similar, so no genuine information is lost.

    Each surviving chunk gains ``"duplicate_count"``: how many raw chunks
    collapsed into it (1 = no duplicate). Useful to show in a demo.

    Args:
        results:     Ranked list of chunk dicts.
        content_key: Dict key holding the chunk text.

    Returns:
        New list, same order, duplicates removed.
    """
    seen: dict[str, dict] = {}
    kept: list[dict] = []

    for chunk in results:
        key = _content_key(chunk.get(content_key, ""))
        if key in seen:
            seen[key]["duplicate_count"] += 1
            continue
        out = {**chunk, "duplicate_count": 1}
        seen[key] = out
        kept.append(out)

    return kept
