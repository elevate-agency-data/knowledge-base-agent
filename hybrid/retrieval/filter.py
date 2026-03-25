"""
Metadata filtering utilities for retrieval.

``build_filters()``   : Assemble a filters dict from individual parameters.
``apply_post_filter()``: Filter an already-retrieved result list in Python
                          (useful when the store does not support all filters).
"""

from __future__ import annotations

from datetime import date
from typing import Optional


def build_filters(
    file_type: Optional[str] = None,
    domaine: Optional[str] = None,
    langue: Optional[str] = None,
    author: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    tags: Optional[list[str]] = None,
    index_name: Optional[str] = None,
) -> dict:
    """
    Build a normalised filters dict from individual keyword arguments.

    ``None`` and empty-string values are excluded so downstream callers
    can check ``if filters`` reliably.

    Args:
        file_type:  File type filter (``"PDF"``, ``"Doc"``, ``"Sheet"``,
                    ``"Slide"``).
        domaine:    Domain filter (``"HR"``, ``"Retail"``, ``"CustomerCare"``,
                    ``"Other"``).
        langue:     ISO 639-1 language code (e.g. ``"fr"``, ``"en"``).
        author:     Author display name.
        date_from:  Inclusive lower bound — ``"YYYY-MM-DD"`` string.
        date_to:    Inclusive upper bound — ``"YYYY-MM-DD"`` string.
        tags:       List of tags; a chunk must contain ALL tags to match.
        index_name: Logical index name to restrict search to one index.

    Returns:
        Dict with only the non-None/non-empty filter fields.
    """
    filters: dict = {}

    if file_type:
        filters["file_type"] = file_type
    if domaine:
        filters["domaine"] = domaine
    if langue:
        filters["langue"] = langue
    if author:
        filters["author"] = author
    if date_from:
        filters["date_from"] = date_from
    if date_to:
        filters["date_to"] = date_to
    if tags:
        filters["tags"] = [t.lower() for t in tags if t]
    if index_name:
        filters["index_name"] = index_name

    return filters


def apply_post_filter(
    results: list[dict],
    filters: dict,
) -> list[dict]:
    """
    Filter a result list in Python against *filters*.

    Useful as a post-retrieval step when the store does not natively support
    all filter types (e.g. tag filtering in DuckDB without array functions).

    Filters are AND-combined.  All provided filters must match for a chunk
    to be kept.

    Args:
        results: List of chunk dicts (e.g. from ``dense_search``).
        filters: Filters dict produced by ``build_filters()``.

    Returns:
        Filtered list (original dicts, not copies).
    """
    if not filters:
        return results

    kept: list[dict] = []
    for chunk in results:
        if _matches(chunk, filters):
            kept.append(chunk)
    return kept


def _matches(chunk: dict, filters: dict) -> bool:
    """
    Return True if *chunk* satisfies all conditions in *filters*.

    Args:
        chunk:   Single chunk dict.
        filters: Filters dict from ``build_filters()``.

    Returns:
        Boolean match result.
    """
    # Simple equality filters (index_name excluded — handled by table routing)
    for field in ("file_type", "domaine", "langue", "author"):
        if field in filters:
            if chunk.get(field) != filters[field]:
                return False

    # Date range filters
    if "date_from" in filters or "date_to" in filters:
        created = chunk.get("created_at")
        if created:
            try:
                chunk_date = date.fromisoformat(str(created)[:10])
                if "date_from" in filters:
                    if chunk_date < date.fromisoformat(filters["date_from"]):
                        return False
                if "date_to" in filters:
                    if chunk_date > date.fromisoformat(filters["date_to"]):
                        return False
            except (ValueError, TypeError):
                pass  # Unparseable date — skip date check

    # Tags filter: chunk must contain ALL required tags
    if "tags" in filters:
        required_tags = set(t.lower() for t in filters["tags"])
        chunk_tags = set(t.lower() for t in (chunk.get("tags") or []))
        if not required_tags.issubset(chunk_tags):
            return False

    return True
