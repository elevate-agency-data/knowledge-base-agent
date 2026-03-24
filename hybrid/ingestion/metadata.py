"""
Metadata enrichment for ingested document chunks.

Provides:
- Language detection via langdetect
- Domain classification via keyword rules
- Tag extraction via TF-IDF top-N keywords
- Metadata assembly into the final chunk dict
"""

from __future__ import annotations

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------


def detect_language(text: str) -> str:
    """
    Detect the dominant language of *text*.

    Uses the ``langdetect`` library.  Returns ``"unknown"`` on failure.

    Args:
        text: Input text (at least ~50 characters for reliable results).

    Returns:
        ISO 639-1 language code string (e.g. ``"fr"``, ``"en"``),
        or ``"unknown"`` if detection fails.
    """
    if not text or len(text.strip()) < 20:
        return "unknown"
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Domain detection
# ---------------------------------------------------------------------------

_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "HR": [
        "recrutement", "embauche", "contrat", "salarié", "employé",
        "paie", "congé", "formation", "onboarding", "offboarding",
        "rh", "ressources humaines", "évaluation", "performance",
        "recruitment", "employee", "payroll", "leave", "training",
        "human resources", "hiring", "workforce",
    ],
    "Retail": [
        "produit", "catalogue", "stock", "inventaire", "magasin",
        "boutique", "prix", "promotion", "vente", "achat", "commande",
        "livraison", "fournisseur", "merchandising", "sku", "référence",
        "product", "store", "inventory", "pricing", "supplier",
        "merchandise", "order", "retail",
    ],
    "CustomerCare": [
        "client", "support", "ticket", "réclamation", "satisfaction",
        "nps", "csat", "service client", "aide", "remboursement",
        "retour", "litige", "contact", "assistance", "faq",
        "customer", "complaint", "refund", "return", "helpdesk",
        "support ticket", "customer service", "care",
    ],
}


def detect_domaine(text: str, file_name: str = "") -> str:
    """
    Classify the business domain of a document using keyword rules.

    Args:
        text:      Document text (or a representative excerpt).
        file_name: File name hint (may contain domain keywords).

    Returns:
        One of ``"HR"``, ``"Retail"``, ``"CustomerCare"``, or ``"Other"``.
    """
    combined = (text + " " + file_name).lower()
    scores: dict[str, int] = {}
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in combined)
        scores[domain] = count

    best_domain = max(scores, key=lambda d: scores[d])
    if scores[best_domain] == 0:
        return "Other"
    return best_domain


# ---------------------------------------------------------------------------
# Tag extraction
# ---------------------------------------------------------------------------


def extract_tags(text: str, top_n: int = 10) -> list[str]:
    """
    Extract the top-N most representative keywords from *text* via TF-IDF.

    Falls back to simple word frequency if scikit-learn is not installed.

    Args:
        text:  Input text to analyse.
        top_n: Number of tags to return.

    Returns:
        List of keyword strings (lowercase, de-duplicated).
    """
    if not text.strip():
        return []

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np

        vectorizer = TfidfVectorizer(
            max_features=200,
            stop_words=None,  # Multilingual — no built-in stop list
            ngram_range=(1, 2),
            min_df=1,
            token_pattern=r"(?u)\b[a-zA-ZÀ-ÿ][a-zA-ZÀ-ÿ]{2,}\b",
        )
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        top_indices = np.argsort(scores)[::-1][:top_n]
        return [feature_names[i] for i in top_indices if scores[i] > 0]

    except ImportError:
        # Fallback: top-N by word frequency
        words = re.findall(r"\b[a-zA-ZÀ-ÿ]{3,}\b", text.lower())
        freq: dict[str, int] = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        sorted_words = sorted(freq, key=lambda w: freq[w], reverse=True)
        return sorted_words[:top_n]


# ---------------------------------------------------------------------------
# Metadata assembly
# ---------------------------------------------------------------------------


def build_metadata(
    file_info: dict,
    chunk: dict,
    embedding_model: str,
    index_name: str = "",
    embedding_dim: int = 0,
    doc_language: str = "",
    doc_domaine: str = "",
) -> dict:
    """
    Assemble the full metadata dict for a chunk ready to be inserted.

    Merges FileInfo from the extractor, chunk data from the chunker, and
    the auto-detected language, domain, and tags.

    Args:
        file_info:       FileInfo dict from ``list_drive_folder()`` or similar.
        chunk:           Chunk dict from ``chunk_text()``.
        embedding_model: Model identifier string (e.g. ``"bge-m3"``).
        index_name:      Logical index name for the chunk.
        embedding_dim:   Dimension of the embedding vector.
        doc_language:    Pre-detected document-level language (skips per-chunk detection).
        doc_domaine:     Pre-detected document-level domain (skips per-chunk detection).

    Returns:
        Complete chunk metadata dict, ready for ``store.insert_chunks()``.
    """
    content = chunk.get("content", "")

    lang = doc_language or detect_language(content)
    domain = doc_domaine or detect_domaine(content, file_info.get("file_name", ""))
    tags = extract_tags(content)

    # Parse ISO dates to date-only strings (YYYY-MM-DD)
    created_at = _parse_date(file_info.get("created_at", ""))
    updated_at = _parse_date(file_info.get("updated_at", ""))

    return {
        # Primary key
        "id":               chunk["id"],
        # Index
        "index_name":       index_name,
        # Content
        "content":          content,
        "embedding":        chunk.get("embedding"),  # set later by embedder
        # Source
        "source_url":       file_info.get("source_url", ""),
        "file_name":        file_info.get("file_name", ""),
        "file_type":        file_info.get("file_type", ""),
        # Timestamps
        "created_at":       created_at,
        "updated_at":       updated_at,
        # Author
        "author":           file_info.get("author", ""),
        # Detected metadata
        "domaine":          domain,
        "langue":           lang,
        "tags":             tags,
        # Chunk position
        "chunk_index":      chunk.get("chunk_index", 0),
        "chunk_total":      chunk.get("chunk_total", 1),
        "chunk_strategy":   chunk.get("chunk_strategy", "fixed"),
        "parent_chunk_id":  chunk.get("parent_chunk_id"),
        # Embedding info
        "embedding_model":  embedding_model,
        "embedding_dim":    embedding_dim,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_date(dt_str: str) -> Optional[str]:
    """
    Convert an ISO 8601 datetime string to a YYYY-MM-DD date string.

    Args:
        dt_str: ISO 8601 string such as ``"2024-03-15T10:30:00Z"``.

    Returns:
        ``"YYYY-MM-DD"`` string, or ``None`` if parsing fails.
    """
    if not dt_str:
        return None
    match = re.match(r"(\d{4}-\d{2}-\d{2})", dt_str)
    return match.group(1) if match else None
