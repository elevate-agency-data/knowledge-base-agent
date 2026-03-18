"""
Shared utility: resolve which hybrid indexes to query for a given prompt.

Uses Gemini Flash to detect client/index names mentioned in the query.
Falls back to substring matching, then to all indexes if nothing is found.

Used by:
  - hybrid/tools/hybrid_query.py   (ADK tool — auto-resolution when index_names=[])
  - ui/services/hybrid_service.py  (UI services — Simple Chat, RAG Comparison)
"""

from __future__ import annotations


def resolve_indexes(query: str, available: list[str]) -> list[str]:
    """
    Determine which indexes to query for a given user prompt.

    Args:
        query:     Raw user question.
        available: List of all known index names.

    Returns:
        Subset of *available* to query, or the full list if nothing specific
        is identified. Never returns an empty list when *available* is non-empty.
    """
    if not available:
        return []
    if len(available) == 1:
        return available

    prompt = (
        f"Index disponibles : {', '.join(available)}\n"
        f"Requête : \"{query}\"\n\n"
        "Quels index faut-il interroger ?\n"
        "- Si la requête mentionne un ou plusieurs clients précis correspondant "
        "à des noms d'index, retourne uniquement ceux-là.\n"
        "- Si la requête est comparative, générale, ou ne cible aucun client "
        "précis, retourne TOUS les index.\n"
        "Réponds UNIQUEMENT avec les noms d'index séparés par des virgules, "
        "en minuscules, sans explication."
    )
    try:
        from vertexai.generative_models import GenerativeModel
        response = GenerativeModel("gemini-2.0-flash-001").generate_content(prompt)
        names    = [n.strip().lower() for n in response.text.strip().split(",")]
        resolved = [n for n in names if n in available]
        return resolved if resolved else available
    except Exception:
        # Substring fallback
        q       = query.lower()
        matched = [n for n in available if n.lower() in q]
        return matched if matched else available
