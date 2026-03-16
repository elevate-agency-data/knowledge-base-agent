"""
Retrieval benchmark script.

Evaluates all 45 combinations of:
  5 embedding models  × 3 chunk strategies × 3 retrieval modes

Run with::

    python -m hybrid.benchmark.eval_retrieval [--test-file PATH] [--output PATH]

The test file is a JSON array of query objects::

    [
      {
        "query":        "What is the onboarding process?",
        "relevant_ids": ["chunk-uuid-1", "chunk-uuid-2"]
      },
      ...
    ]

If no test file is provided, a built-in synthetic dataset is used so the
benchmark can run completely offline without Google Drive access.

Results are saved to ``hybrid/data/benchmark_results.json`` and a ranked
comparison table is printed to stdout.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import tempfile
import uuid
from itertools import product
from typing import Any

# Q&A file support
try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

from hybrid.config import (
    BENCHMARK_EMBEDDING_MODELS,
    BENCHMARK_CHUNK_SIZES,
    BENCHMARK_CHUNK_STRATEGIES,
    BENCHMARK_RETRIEVAL_MODES,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SEMANTIC_BREAKPOINT_THRESHOLD,
    PARENT_CHUNK_SIZE,
    CHILD_CHUNK_SIZE,
    TOP_K,
    DENSE_WEIGHT,
    SPARSE_WEIGHT,
    RRF_K,
)
from hybrid.stores.duckdb_store import DuckDBStore
from hybrid.embeddings import get_embedding_model
from hybrid.ingestion.chunker import chunk_text
from hybrid.retrieval.dense import dense_search
from hybrid.retrieval.sparse import sparse_search
from hybrid.retrieval.fusion import reciprocal_rank_fusion
from hybrid.benchmark.metrics import compute_all_metrics


# ---------------------------------------------------------------------------
# Synthetic test corpus (used when no external test file is provided)
# ---------------------------------------------------------------------------

_SYNTHETIC_DOCS = [
    {
        "text": (
            "The employee onboarding process begins on the first day of work. "
            "New employees receive access to company systems, a welcome kit, "
            "and an introduction to their team. HR provides mandatory training "
            "on compliance, workplace safety, and company values. "
            "The probationary period lasts 90 days and includes regular "
            "check-ins with the line manager."
        ),
        "file_name": "onboarding_guide.pdf",
        "file_type": "PDF",
    },
    {
        "text": (
            "Our retail pricing strategy uses dynamic pricing algorithms "
            "that adjust product prices based on demand, competitor prices, "
            "and inventory levels. The merchandising team updates the catalogue "
            "weekly. Promotions are managed via the promotion engine and require "
            "approval from the Retail Director. SKU rationalisation reviews "
            "happen quarterly."
        ),
        "file_name": "retail_strategy.doc",
        "file_type": "Doc",
    },
    {
        "text": (
            "Customer complaints must be acknowledged within 2 business hours. "
            "The customer care team uses a ticketing system to track all "
            "interactions. Escalation to a senior agent is triggered after "
            "3 failed resolution attempts. Refund requests above €500 require "
            "manager approval. CSAT surveys are sent automatically 24 hours "
            "after ticket closure."
        ),
        "file_name": "customer_care_sop.pdf",
        "file_type": "PDF",
    },
    {
        "text": (
            "Annual performance reviews are conducted in December for all "
            "full-time employees. The review process includes self-assessment, "
            "peer feedback, and a structured conversation with the direct manager. "
            "Performance ratings influence salary adjustments and promotion "
            "decisions. Employees rated below expectations must follow a "
            "performance improvement plan."
        ),
        "file_name": "performance_review_policy.doc",
        "file_type": "Doc",
    },
    {
        "text": (
            "The product catalogue contains over 50,000 active SKUs across "
            "10 categories. Out-of-stock items are automatically hidden from "
            "the website. The inventory team conducts bi-annual stocktaking. "
            "Supplier lead times range from 3 to 21 days depending on the "
            "product category. Safety stock levels are calculated using the "
            "EOQ formula adjusted for seasonal demand."
        ),
        "file_name": "inventory_management.sheet",
        "file_type": "Sheet",
    },
]

_SYNTHETIC_QUERIES = [
    {
        "query":       "What happens during employee onboarding?",
        "relevant_ids": [],  # Will be filled after chunking
        "source_doc":  0,    # Index into _SYNTHETIC_DOCS
    },
    {
        "query":       "How are customer complaints handled?",
        "relevant_ids": [],
        "source_doc":  2,
    },
    {
        "query":       "How does the retail pricing strategy work?",
        "relevant_ids": [],
        "source_doc":  1,
    },
    {
        "query":       "What is the performance review process?",
        "relevant_ids": [],
        "source_doc":  3,
    },
    {
        "query":       "How is inventory managed?",
        "relevant_ids": [],
        "source_doc":  4,
    },
]


# ---------------------------------------------------------------------------
# Core benchmark logic
# ---------------------------------------------------------------------------


def _build_synthetic_dataset() -> tuple[list[dict], list[dict]]:
    """
    Build a synthetic test corpus and annotated query set.

    Returns:
        Tuple of (documents, queries).  Each document has ``text``,
        ``file_name``, ``file_type`` keys.  Each query has ``query``
        and ``relevant_ids`` keys.
    """
    queries = [dict(q) for q in _SYNTHETIC_QUERIES]

    # Assign a stable chunk_id per document for ground-truth annotation
    for q in queries:
        doc_idx = q.pop("source_doc")
        # Use deterministic ID so relevant_ids can be pre-set
        q["relevant_ids"] = [f"synthetic-doc-{doc_idx}-chunk-0"]

    return list(_SYNTHETIC_DOCS), queries


def _ingest_docs(
    store: DuckDBStore,
    index_name: str,
    docs: list[dict],
    embedder: Any,
    strategy: str,
    emb_model_name: str,
) -> dict[int, list[str]]:
    """
    Chunk and embed all docs into *store*, returning doc_index → chunk_ids.

    Args:
        store:          DuckDB store instance.
        index_name:     Index namespace to use.
        docs:           List of document dicts with ``text`` key.
        embedder:       Loaded embedding model.
        strategy:       Chunking strategy name.
        emb_model_name: Model name string for metadata.

    Returns:
        Dict mapping document index → list of chunk IDs inserted.
    """
    chunk_params: dict = {}
    if strategy.startswith("fixed"):
        # strategy is "fixed" or "fixed-N" (e.g. "fixed-256")
        if "-" in strategy:
            size = int(strategy.split("-", 1)[1])
        else:
            size = CHUNK_SIZE
        chunk_params = {"size": size, "overlap": max(1, size // 5)}
        strategy = "fixed"
    elif strategy == "semantic":
        chunk_params = {"threshold": SEMANTIC_BREAKPOINT_THRESHOLD}
    elif strategy == "hierarchical":
        chunk_params = {"parent_size": PARENT_CHUNK_SIZE, "child_size": CHILD_CHUNK_SIZE}

    doc_to_chunks: dict[int, list[str]] = {}

    for doc_idx, doc in enumerate(docs):
        text = doc["text"]
        chunks = chunk_text(text, strategy=strategy, **chunk_params)

        # Force first chunk ID to synthetic deterministic ID for ground truth
        if chunks:
            chunks[0]["id"] = f"synthetic-doc-{doc_idx}-chunk-0"

        texts = [c["content"] for c in chunks]
        if not texts:
            doc_to_chunks[doc_idx] = []
            continue

        embeddings = embedder.embed_documents(texts)
        records = []
        chunk_ids = []
        for chunk, emb in zip(chunks, embeddings):
            record = {
                "id":              chunk["id"],
                "index_name":      index_name,
                "content":         chunk["content"],
                "embedding":       emb,
                "source_url":      f"synthetic://doc-{doc_idx}",
                "file_name":       doc.get("file_name", f"doc_{doc_idx}"),
                "file_type":       doc.get("file_type", "PDF"),
                "created_at":      None,
                "updated_at":      None,
                "author":          "",
                "domaine":         "",
                "langue":          "en",
                "tags":            [],
                "chunk_index":     chunk["chunk_index"],
                "chunk_total":     chunk["chunk_total"],
                "chunk_strategy":  chunk["chunk_strategy"],
                "parent_chunk_id": chunk.get("parent_chunk_id"),
                "embedding_model": emb_model_name,
                "embedding_dim":   len(emb),
            }
            records.append(record)
            chunk_ids.append(chunk["id"])

        store.insert_chunks(records)
        doc_to_chunks[doc_idx] = chunk_ids

    return doc_to_chunks


def _run_combination(
    docs: list[dict],
    queries: list[dict],
    emb_model_name: str,
    chunk_strategy: str,
    retrieval_mode: str,
    db_path: str,
) -> dict:
    """
    Evaluate one (embedding, chunking, retrieval) combination.

    Args:
        docs:            Document corpus.
        queries:         Annotated queries with ``relevant_ids``.
        emb_model_name:  Embedding model key.
        chunk_strategy:  Chunking strategy name.
        retrieval_mode:  ``"dense"``, ``"sparse"``, or ``"hybrid"``.
        db_path:         Path for a temporary DuckDB file.

    Returns:
        Dict with averaged metrics and combination metadata.
    """
    index_name = "bench_eval"
    store = DuckDBStore(db_path)
    store.initialize(index_name)

    try:
        embedder = get_embedding_model(emb_model_name)
    except Exception as exc:
        return {
            "embedding":  emb_model_name,
            "chunking":   chunk_strategy,
            "retrieval":  retrieval_mode,
            "mrr":        0.0,
            "ndcg":       0.0,
            "recall":     0.0,
            "precision":  0.0,
            "error":      str(exc),
        }

    _ingest_docs(
        store, index_name, docs, embedder, chunk_strategy, emb_model_name
    )

    query_metrics: list[dict] = []
    filters = {"index_name": index_name}

    for q in queries:
        query_text   = q["query"]
        relevant_ids = q["relevant_ids"]

        try:
            if retrieval_mode == "dense":
                results = dense_search(
                    query=query_text,
                    store=store,
                    embedding_model=embedder,
                    top_k=TOP_K,
                    filters=filters,
                )
            elif retrieval_mode == "sparse":
                results = sparse_search(
                    query=query_text,
                    store=store,
                    top_k=TOP_K,
                    filters=filters,
                )
            else:  # hybrid
                dense_r  = dense_search(
                    query=query_text,
                    store=store,
                    embedding_model=embedder,
                    top_k=TOP_K * 2,
                    filters=filters,
                )
                sparse_r = sparse_search(
                    query=query_text,
                    store=store,
                    top_k=TOP_K * 2,
                    filters=filters,
                )
                results = reciprocal_rank_fusion(
                    dense_r, sparse_r, k=RRF_K,
                    dense_weight=DENSE_WEIGHT,
                    sparse_weight=SPARSE_WEIGHT,
                )[:TOP_K]

            m = compute_all_metrics(results, relevant_ids, k=TOP_K)
            query_metrics.append(m)
        except Exception:
            query_metrics.append({"mrr": 0.0, "ndcg": 0.0, "recall": 0.0, "precision": 0.0})

    # Average across queries
    n = len(query_metrics) or 1
    return {
        "embedding":  emb_model_name,
        "chunking":   chunk_strategy,
        "retrieval":  retrieval_mode,
        "mrr":        round(sum(m["mrr"]       for m in query_metrics) / n, 4),
        "ndcg":       round(sum(m["ndcg"]      for m in query_metrics) / n, 4),
        "recall":     round(sum(m["recall"]    for m in query_metrics) / n, 4),
        "precision":  round(sum(m["precision"] for m in query_metrics) / n, 4),
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _print_table(rows: list[dict]) -> None:
    """
    Print a markdown-style comparison table sorted by nDCG descending.

    Args:
        rows: List of result dicts from ``_run_combination``.
    """
    sorted_rows = sorted(rows, key=lambda r: r["ndcg"], reverse=True)

    header = (
        f"| {'Embedding':<16} | {'Chunking':<14} | {'Retrieval':<10} "
        f"| {'MRR':<6} | {'NDCG':<6} | {'R@10':<6} |"
    )
    separator = (
        "|" + "-" * 18 + "|" + "-" * 16 + "|" + "-" * 12 +
        "|" + "-" * 8 + "|" + "-" * 8 + "|" + "-" * 8 + "|"
    )

    print("\n" + "=" * 80)
    print("  HYBRID RAG RETRIEVAL BENCHMARK RESULTS")
    print("=" * 80)
    print(header)
    print(separator)

    for row in sorted_rows:
        error = row.get("error", "")
        if error:
            status = f"ERROR: {error[:30]}"
            print(
                f"| {row['embedding']:<16} | {row['chunking']:<14} | "
                f"{row['retrieval']:<10} | {status:<40} |"
            )
        else:
            print(
                f"| {row['embedding']:<16} | {row['chunking']:<14} | "
                f"{row['retrieval']:<10} | "
                f"{row['mrr']:<6.4f} | {row['ndcg']:<6.4f} | "
                f"{row['recall']:<6.4f} |"
            )

    print(separator)
    if sorted_rows:
        best = sorted_rows[0]
        if not best.get("error"):
            print(
                f"\n  BEST COMBINATION: "
                f"{best['embedding']} + {best['chunking']} + {best['retrieval']}"
                f"  →  MRR={best['mrr']:.4f}  nDCG={best['ndcg']:.4f}  R@10={best['recall']:.4f}"
            )
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Q&A file loading and text-based relevance
# ---------------------------------------------------------------------------


def _load_qa_file(path: str) -> list[dict]:
    """
    Load a Q&A evaluation file (CSV or Excel).

    Expected columns (case-insensitive):
        - ``Q`` or ``question``            : the question
        - ``A`` or ``answer``              : the expected answer text
        - ``url``, ``link`` or ``document``: Google Drive URL of the source document

    Args:
        path: Path to the CSV or Excel file.

    Returns:
        List of dicts with keys ``query``, ``answer``, ``url``.

    Raises:
        ImportError : if pandas is not installed.
        ValueError  : if required columns are missing.
    """
    if not _PANDAS_AVAILABLE:
        raise ImportError("pandas is required for Q&A mode: pip install pandas openpyxl")

    if path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    col_q   = next((c for c in df.columns if c in ("q", "question")), None)
    col_a   = next((c for c in df.columns if c in ("a", "answer")), None)
    col_url = next((c for c in df.columns if c in ("url", "link", "document", "doc", "source")), None)

    missing = [name for name, col in [("Q", col_q), ("A", col_a), ("url", col_url)] if col is None]
    if missing:
        raise ValueError(
            f"Missing columns in Q&A file: {missing}. "
            f"Found: {list(df.columns)}"
        )

    rows = []
    for _, row in df.iterrows():
        q   = str(row[col_q]).strip()
        a   = str(row[col_a]).strip()
        url = str(row[col_url]).strip()
        if q and a and url:
            rows.append({"query": q, "answer": a, "url": url})

    return rows


def _word_overlap(text_a: str, text_b: str) -> float:
    """
    Compute the fraction of ``text_a`` words that appear in ``text_b``.

    Used to decide if a chunk contains the expected answer.
    A value of 1.0 means every word of text_a is present in text_b.

    Args:
        text_a: Reference text (the expected answer).
        text_b: Candidate text (the chunk content).

    Returns:
        Overlap ratio in [0, 1].
    """
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a:
        return 0.0
    return len(words_a & words_b) / len(words_a)


def _find_relevant_ids(
    store: Any,
    index_name: str,
    answer: str,
    source_url: str,
    threshold: float,
) -> list[str]:
    """
    Scan all chunks in the store and return IDs of chunks that match the answer.

    A chunk is considered relevant if:
      1. Its ``source_url`` matches the Drive URL exactly.
      2. Its word overlap with ``answer`` is >= ``threshold``.

    Args:
        store:      DuckDB store instance.
        index_name: Index to scan.
        answer:     Expected answer text from the Q&A file.
        source_url: Google Drive URL of the source document.
        threshold:  Minimum word overlap ratio to consider a chunk relevant.

    Returns:
        List of relevant chunk ID strings.
    """
    try:
        conn = store._get_conn()
        rows = conn.execute(
            "SELECT id, content FROM chunks WHERE index_name = ? AND source_url = ?",
            [index_name, source_url],
        ).fetchall()
    except Exception:
        return []

    relevant = []
    for chunk_id, content in rows:
        if _word_overlap(answer, content or "") >= threshold:
            relevant.append(str(chunk_id))

    return relevant


def _run_combination_qa(
    qa_rows: list[dict],
    docs: list[dict],
    emb_model_name: str,
    chunk_strategy: str,
    retrieval_mode: str,
    db_path: str,
    answer_threshold: float,
) -> dict:
    """
    Evaluate one combination using a Q&A file for ground-truth relevance.

    After ingestion, relevant chunk IDs are determined dynamically by
    comparing each expected answer against chunk content (word overlap).

    Args:
        qa_rows:          Loaded Q&A rows (query, answer, document).
        docs:             Document corpus to ingest.
        emb_model_name:   Embedding model key.
        chunk_strategy:   Chunking strategy name.
        retrieval_mode:   ``"dense"``, ``"sparse"``, or ``"hybrid"``.
        db_path:          Path for a temporary DuckDB file.
        answer_threshold: Minimum word overlap to count a chunk as relevant.

    Returns:
        Dict with averaged metrics and combination metadata.
    """
    index_name = "bench_eval"
    store = DuckDBStore(db_path)
    store.initialize(index_name)

    try:
        embedder = get_embedding_model(emb_model_name)
    except Exception as exc:
        return {
            "embedding": emb_model_name, "chunking": chunk_strategy,
            "retrieval": retrieval_mode, "mrr": 0.0, "ndcg": 0.0,
            "recall": 0.0, "precision": 0.0, "error": str(exc),
        }

    _ingest_docs(store, index_name, docs, embedder, chunk_strategy, emb_model_name)

    query_metrics: list[dict] = []
    filters = {"index_name": index_name}

    for row in qa_rows:
        query_text = row["query"]
        answer     = row["answer"]
        source_url = row["url"]

        # Dynamically find relevant chunk IDs via text overlap
        relevant_ids = _find_relevant_ids(
            store, index_name, answer, source_url, answer_threshold
        )

        if not relevant_ids:
            # No matching chunk found → score 0 for this query
            query_metrics.append({"mrr": 0.0, "ndcg": 0.0, "recall": 0.0, "precision": 0.0})
            continue

        try:
            if retrieval_mode == "dense":
                results = dense_search(
                    query=query_text, store=store,
                    embedding_model=embedder, top_k=TOP_K, filters=filters,
                )
            elif retrieval_mode == "sparse":
                results = sparse_search(
                    query=query_text, store=store, top_k=TOP_K, filters=filters,
                )
            else:
                dense_r  = dense_search(query=query_text, store=store,
                    embedding_model=embedder, top_k=TOP_K * 2, filters=filters)
                sparse_r = sparse_search(query=query_text, store=store,
                    top_k=TOP_K * 2, filters=filters)
                results  = reciprocal_rank_fusion(
                    dense_r, sparse_r, k=RRF_K,
                    dense_weight=DENSE_WEIGHT, sparse_weight=SPARSE_WEIGHT,
                )[:TOP_K]

            m = compute_all_metrics(results, relevant_ids, k=TOP_K)
            query_metrics.append(m)
        except Exception:
            query_metrics.append({"mrr": 0.0, "ndcg": 0.0, "recall": 0.0, "precision": 0.0})

    n = len(query_metrics) or 1
    return {
        "embedding": emb_model_name,
        "chunking":  chunk_strategy,
        "retrieval": retrieval_mode,
        "mrr":       round(sum(m["mrr"]       for m in query_metrics) / n, 4),
        "ndcg":      round(sum(m["ndcg"]      for m in query_metrics) / n, 4),
        "recall":    round(sum(m["recall"]    for m in query_metrics) / n, 4),
        "precision": round(sum(m["precision"] for m in query_metrics) / n, 4),
    }


def main() -> None:
    """
    Run the full retrieval benchmark and print a ranked comparison table.

    Two modes:
      synthetic  Use the built-in synthetic dataset (no files needed).
      qa         Use a Q&A CSV/Excel file (columns: Q, A, document).
                 Relevance is determined by word overlap between the expected
                 answer and retrieved chunk content.
    """
    parser = argparse.ArgumentParser(
        description="Hybrid RAG retrieval benchmark — 45 combinations"
    )
    parser.add_argument(
        "--mode",
        choices=["synthetic", "qa"],
        default="synthetic",
        help=(
            "Dataset mode: "
            "'synthetic' uses the built-in dataset (default); "
            "'qa' uses a Q&A file (requires --qa-file)."
        ),
    )
    parser.add_argument(
        "--qa-file",
        default=None,
        help=(
            "Path to the Q&A evaluation file (CSV or Excel). "
            "Required when --mode=qa. "
            "Expected columns: Q (or question), A (or answer), document (or source)."
        ),
    )
    parser.add_argument(
        "--answer-threshold",
        type=float,
        default=0.3,
        help=(
            "Minimum word overlap ratio between expected answer and a chunk "
            "for the chunk to be considered relevant (default: 0.3). "
            "Only used in --mode=qa."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="MODEL",
        help=(
            "Restrict benchmark to specific embedding models. "
            "Example: --models minilm-384 mpnet-768 "
            f"Available: {', '.join(BENCHMARK_EMBEDDING_MODELS)}"
        ),
    )
    parser.add_argument(
        "--output",
        default="hybrid/data/benchmark_results.json",
        help="Path to save full results JSON (default: hybrid/data/benchmark_results.json).",
    )
    args = parser.parse_args()

    # -- Load data -----------------------------------------------------------
    qa_rows: list[dict] = []

    if args.mode == "qa":
        if not args.qa_file:
            print("ERROR: --qa-file is required when --mode=qa.", file=sys.stderr)
            sys.exit(1)
        print(f"Mode: Q&A file  →  {args.qa_file}")
        print(f"Answer match threshold: {args.answer_threshold}\n")
        qa_rows = _load_qa_file(args.qa_file)
        # Build a minimal docs list from unique document names for ingestion
        # (actual text comes from the synthetic corpus if doc names match,
        #  or the user must pass their own documents in a companion JSON)
        docs = list(_SYNTHETIC_DOCS)
        print(f"Loaded {len(qa_rows)} Q&A pairs from file.\n")
    else:
        print("Mode: synthetic dataset\n")
        docs, _synthetic_queries = _build_synthetic_dataset()

    # Filtre les modèles si --models est passé
    active_models = args.models if args.models else BENCHMARK_EMBEDDING_MODELS
    unknown = [m for m in active_models if m not in BENCHMARK_EMBEDDING_MODELS]
    if unknown:
        print(f"WARNING: modèles inconnus ignorés : {unknown}", file=sys.stderr)
    active_models = [m for m in active_models if m in BENCHMARK_EMBEDDING_MODELS]

    # Expand "fixed" into one variant per chunk size, keep semantic/hierarchical as-is
    expanded_strategies = []
    for s in BENCHMARK_CHUNK_STRATEGIES:
        if s == "fixed":
            expanded_strategies.extend([f"fixed-{sz}" for sz in BENCHMARK_CHUNK_SIZES])
        else:
            expanded_strategies.append(s)

    total_combinations = (
        len(active_models)
        * len(expanded_strategies)
        * len(BENCHMARK_RETRIEVAL_MODES)
    )
    n_queries = len(qa_rows) if args.mode == "qa" else len(_synthetic_queries if args.mode == "synthetic" else [])
    n_fixed   = len(BENCHMARK_CHUNK_SIZES)
    n_other   = len([s for s in BENCHMARK_CHUNK_STRATEGIES if s != "fixed"])
    print(
        f"Running {total_combinations} combinations "
        f"({len(active_models)} embeddings × "
        f"({n_fixed} fixed sizes + {n_other} autres chunking) × "
        f"{len(BENCHMARK_RETRIEVAL_MODES)} retrieval) "
        f"on {n_queries} queries...\n"
    )

    results: list[dict] = []
    combo_num = 0

    for emb_model, chunk_strategy, retrieval_mode in product(
        active_models,
        expanded_strategies,
        BENCHMARK_RETRIEVAL_MODES,
    ):
        combo_num += 1
        print(
            f"  [{combo_num:>2}/{total_combinations}] "
            f"{emb_model:<16} + {chunk_strategy:<14} + {retrieval_mode:<7} ... ",
            end="",
            flush=True,
        )

        with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as tmp:
            tmp_path = tmp.name
        # DuckDB cannot open an existing empty file — delete it so DuckDB creates a fresh one
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

        try:
            t0 = time.perf_counter()

            if args.mode == "qa":
                row = _run_combination_qa(
                    qa_rows=qa_rows,
                    docs=docs,
                    emb_model_name=emb_model,
                    chunk_strategy=chunk_strategy,
                    retrieval_mode=retrieval_mode,
                    db_path=tmp_path,
                    answer_threshold=args.answer_threshold,
                )
            else:
                row = _run_combination(
                    docs=docs,
                    queries=_synthetic_queries,
                    emb_model_name=emb_model,
                    chunk_strategy=chunk_strategy,
                    retrieval_mode=retrieval_mode,
                    db_path=tmp_path,
                )

            elapsed = time.perf_counter() - t0
            results.append(row)
            if row.get("error"):
                print(f"SKIPPED ({row['error'][:50]})")
            else:
                print(
                    f"nDCG={row['ndcg']:.4f}  MRR={row['mrr']:.4f}  "
                    f"R@10={row['recall']:.4f}  ({elapsed:.1f}s)"
                )

        except MemoryError:
            elapsed = time.perf_counter() - t0
            print(f"SKIPPED (MemoryError — modèle trop lourd, {elapsed:.1f}s)")
            results.append({
                "embedding": emb_model, "chunking": chunk_strategy,
                "retrieval": retrieval_mode, "mrr": 0.0, "ndcg": 0.0,
                "recall": 0.0, "precision": 0.0, "error": "MemoryError",
            })

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"SKIPPED ({str(exc)[:60]}, {elapsed:.1f}s)")
            results.append({
                "embedding": emb_model, "chunking": chunk_strategy,
                "retrieval": retrieval_mode, "mrr": 0.0, "ndcg": 0.0,
                "recall": 0.0, "precision": 0.0, "error": str(exc)[:120],
            })

        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

            # Libère les modèles PyTorch entre chaque combo pour éviter
            # l'accumulation en RAM (les poids sentence-transformers ne sont
            # pas libérés immédiatement sans gc explicite)
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

            # Sauvegarde intermédiaire après chaque combo — permet de
            # récupérer les résultats partiels si le process est tué
            if results:
                try:
                    os.makedirs(os.path.dirname(args.output), exist_ok=True)
                    with open(args.output, "w", encoding="utf-8") as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
                except OSError:
                    pass

    # -- Save results --------------------------------------------------------
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to: {args.output}")

    # -- Print table ---------------------------------------------------------
    _print_table(results)


if __name__ == "__main__":
    main()
