"""
Text chunking strategies.

Three strategies:
- ``fixed``        : fixed-size token overlap via langchain-text-splitters
- ``semantic``     : sentence-level splitting on cosine similarity drops
- ``hierarchical`` : parent + child chunks linked by ID

All functions return a list of chunk dicts with a consistent schema.
"""

import re
import uuid
from typing import Any


# ---------------------------------------------------------------------------
# Chunk dict schema
# ---------------------------------------------------------------------------

# Each chunk is a dict with at minimum:
#   id              : str   — uuid4
#   content         : str
#   chunk_index     : int
#   chunk_total     : int
#   chunk_strategy  : str   — "fixed" | "semantic" | "hierarchical"
#   parent_chunk_id : str | None


# ---------------------------------------------------------------------------
# Fixed chunking
# ---------------------------------------------------------------------------


def chunk_fixed(text: str, size: int = 512, overlap: int = 100) -> list[dict]:
    """
    Split *text* into fixed-size overlapping chunks.

    Uses ``langchain_text_splitters.RecursiveCharacterTextSplitter`` for
    clean paragraph/sentence-aware splitting.

    Args:
        text:    Source text to split.
        size:    Target chunk size in characters.
        overlap: Overlap between consecutive chunks in characters.

    Returns:
        List of chunk dicts.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )
    texts = splitter.split_text(text)
    if not texts:
        texts = [text] if text.strip() else []

    chunks: list[dict] = []
    total = len(texts)
    for i, t in enumerate(texts):
        chunks.append(
            {
                "id":               str(uuid.uuid4()),
                "content":          t,
                "chunk_index":      i,
                "chunk_total":      total,
                "chunk_strategy":   "fixed",
                "parent_chunk_id":  None,
            }
        )
    return chunks


# ---------------------------------------------------------------------------
# Semantic chunking
# ---------------------------------------------------------------------------


def _split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences using simple regex heuristics.

    Args:
        text: Input text.

    Returns:
        List of sentence strings (non-empty).
    """
    # Split on sentence-ending punctuation followed by whitespace
    pattern = r"(?<=[.!?])\s+"
    sentences = re.split(pattern, text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in [-1, 1].
    """
    import math

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def chunk_semantic(
    text: str,
    threshold: float = 0.85,
    embedding_model: Any = None,
) -> list[dict]:
    """
    Split *text* into semantic chunks by detecting topic shifts.

    Algorithm:
    1. Split the text into sentences.
    2. Embed each sentence using a lightweight model (MiniLM-384 by default).
    3. Compute cosine similarity between consecutive sentence embeddings.
    4. Cut the text wherever similarity drops below *threshold*.

    Args:
        text:            Source text to split.
        threshold:       Cosine similarity threshold for breakpoints.
                         Lower → more chunks; higher → fewer, larger chunks.
        embedding_model: Optional pre-loaded ``BaseEmbedding`` instance.
                         If ``None``, MiniLMEmbedding is used.

    Returns:
        List of chunk dicts.
    """
    sentences = _split_into_sentences(text)
    if len(sentences) <= 1:
        return chunk_fixed(text)

    # Load a fast model for sentence-level embeddings
    if embedding_model is None:
        from hybrid.embeddings.minilm_384 import MiniLMEmbedding
        embedding_model = MiniLMEmbedding()

    embeddings = embedding_model.embed_documents(sentences)

    # Find breakpoints where topic shifts
    breakpoints: list[int] = []
    for i in range(len(embeddings) - 1):
        sim = _cosine_similarity(embeddings[i], embeddings[i + 1])
        if sim < threshold:
            breakpoints.append(i + 1)

    # Build segment groups
    segments: list[list[str]] = []
    start = 0
    for bp in breakpoints:
        segments.append(sentences[start:bp])
        start = bp
    segments.append(sentences[start:])

    # Convert segments to chunk dicts
    chunks: list[dict] = []
    total = len(segments)
    for i, seg in enumerate(segments):
        content = " ".join(seg).strip()
        if content:
            chunks.append(
                {
                    "id":               str(uuid.uuid4()),
                    "content":          content,
                    "chunk_index":      i,
                    "chunk_total":      total,
                    "chunk_strategy":   "semantic",
                    "parent_chunk_id":  None,
                }
            )

    return chunks if chunks else chunk_fixed(text)


# ---------------------------------------------------------------------------
# Hierarchical chunking
# ---------------------------------------------------------------------------


def chunk_hierarchical(
    text: str,
    parent_size: int = 1024,
    child_size: int = 256,
) -> list[dict]:
    """
    Produce parent + child chunks linked by ``parent_chunk_id``.

    Each parent chunk is split into several child chunks.  The child chunks
    are what gets embedded and searched; the parent is retrieved for context
    when presenting results.

    Args:
        text:        Source text to split.
        parent_size: Character size for parent (context) chunks.
        child_size:  Character size for child (indexed) chunks.

    Returns:
        List of chunk dicts — parents first, then children interleaved.
        Child chunks have ``parent_chunk_id`` set to their parent's ``id``.
    """
    parent_chunks = chunk_fixed(text, size=parent_size, overlap=0)
    all_chunks: list[dict] = []

    for parent in parent_chunks:
        # Re-tag parent as hierarchical
        parent["chunk_strategy"] = "hierarchical"
        all_chunks.append(parent)

        child_splits = chunk_fixed(parent["content"], size=child_size, overlap=50)
        for child in child_splits:
            child["chunk_strategy"] = "hierarchical"
            child["parent_chunk_id"] = parent["id"]
            all_chunks.append(child)

    return all_chunks


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def chunk_text(text: str, strategy: str = "fixed", **params) -> list[dict]:
    """
    Dispatch to the appropriate chunking strategy.

    Args:
        text:     Source text to split.
        strategy: One of ``"fixed"``, ``"semantic"``, ``"hierarchical"``.
        **params: Passed through to the specific chunker:

                  - ``fixed``        : ``size``, ``overlap``
                  - ``semantic``     : ``threshold``, ``embedding_model``
                  - ``hierarchical`` : ``parent_size``, ``child_size``

    Returns:
        List of chunk dicts.

    Raises:
        ValueError: If *strategy* is not recognised.
    """
    if strategy == "fixed":
        return chunk_fixed(
            text,
            size=params.get("size", 512),
            overlap=params.get("overlap", 100),
        )
    elif strategy == "semantic":
        return chunk_semantic(
            text,
            threshold=params.get("threshold", 0.85),
            embedding_model=params.get("embedding_model"),
        )
    elif strategy == "hierarchical":
        return chunk_hierarchical(
            text,
            parent_size=params.get("parent_size", 1024),
            child_size=params.get("child_size", 256),
        )
    else:
        raise ValueError(
            f"Unknown chunking strategy '{strategy}'. "
            "Supported: 'fixed', 'semantic', 'hierarchical'."
        )
