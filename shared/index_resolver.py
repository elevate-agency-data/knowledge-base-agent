"""
Shared utility: resolve which hybrid indexes to query for a given prompt.

Uses Gemini Flash to select thematically relevant indexes for a given query.
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
        "Sélectionne le ou les index thématiquement pertinents pour répondre à cette requête.\n"
        "- Analyse le sujet de la requête et choisis uniquement les index dont le contenu "
        "pourrait contenir la réponse.\n"
        "- Tu peux retourner un seul index ou plusieurs si la question couvre plusieurs domaines.\n"
        "- Ne retourne PAS tous les index par défaut : sois sélectif.\n"
        "Réponds UNIQUEMENT avec les noms d'index séparés par des virgules, "
        "en minuscules, sans explication."
    )
    try:
        from vertexai.generative_models import GenerativeModel
        response = GenerativeModel("gemini-2.0-flash-001").generate_content(prompt)
        print(f"#### la reponse de flash : {response.text} #####" )
        names    = [n.strip().lower() for n in response.text.strip().split(",")]
        resolved = [n for n in names if n in available]
        return resolved if resolved else available
    except Exception:
        # Substring fallback
        q       = query.lower()
        matched = [n for n in available if n.lower() in q]
        return matched if matched else available
