"""
Shared utility: resolve which hybrid indexes to query for a given prompt.

Uses Gemini Flash to select thematically relevant indexes for a given query.
Falls back to substring matching, then to all indexes if nothing is found.

Understands the hierarchical naming convention ``company__notion``:
  - If the user mentions a company name, all its sub-indexes are included.
  - If the user mentions a specific notion, only matching sub-indexes are returned.

Used by:
  - hybrid/tools/hybrid_query.py   (ADK tool — auto-resolution when index_names=[])
  - ui/services/hybrid_service.py  (UI services — Simple Chat, RAG Comparison)
"""

from __future__ import annotations

INDEX_SEP = "__"


def _is_catchall(notion: str) -> bool:
    """Return True if the notion name looks like a catch-all / miscellaneous index."""
    _CATCHALL_KEYWORDS = {"divers", "misc", "autre", "other", "general", "divers"}
    return any(kw in notion.lower() for kw in _CATCHALL_KEYWORDS)


def expand_company_indexes(names: list[str], available: list[str]) -> list[str]:
    """
    Expand company-level names into all their ``company__notion`` sub-indexes.

    If a name in *names* is not a full ``company__notion`` index but matches
    the company prefix of available indexes, it is expanded.  Already-valid
    index names are kept as-is.

    **Catch-all indexes** (containing "divers", "misc", "autre", etc.) are
    automatically included whenever any index of the same company is selected,
    because miscellaneous folders may contain misfiled documents.

    Args:
        names:     Index names returned by the LLM or substring match.
        available: All known index names.

    Returns:
        Expanded and deduplicated list of index names.
    """
    result: list[str] = []
    seen: set[str] = set()
    companies_seen: set[str] = set()

    for name in names:
        if name in available:
            # Exact match — keep it
            if name not in seen:
                result.append(name)
                seen.add(name)
            # Track company for catch-all inclusion
            if INDEX_SEP in name:
                companies_seen.add(name.split(INDEX_SEP, 1)[0])
        else:
            # Try as company prefix: "celio" → all "celio__*"
            prefix = name + INDEX_SEP
            expanded = [a for a in available if a.startswith(prefix)]
            if expanded:
                companies_seen.add(name)
                for idx in expanded:
                    if idx not in seen:
                        result.append(idx)
                        seen.add(idx)
            # Also try as notion suffix: "rh" → all "*__rh"
            if not expanded:
                suffix = INDEX_SEP + name
                expanded = [a for a in available if a.endswith(suffix)]
                for idx in expanded:
                    if idx not in seen:
                        result.append(idx)
                        seen.add(idx)
                    if INDEX_SEP in idx:
                        companies_seen.add(idx.split(INDEX_SEP, 1)[0])

    # Always include catch-all indexes for any company we're querying
    for company in companies_seen:
        prefix = company + INDEX_SEP
        for a in available:
            if a.startswith(prefix) and a not in seen:
                notion = a.split(INDEX_SEP, 1)[1]
                if _is_catchall(notion):
                    result.append(a)
                    seen.add(a)

    return result


def resolve_indexes(query: str, available: list[str]) -> list[str]:
    """
    Determine which indexes to query for a given user prompt.

    Understands hierarchical index names (``company__notion``). When the LLM
    or substring match returns a bare company name (e.g. ``"celio"``), all
    sub-indexes for that company are included.

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

    # Build a human-readable description of the hierarchy for the LLM
    companies = _describe_hierarchy(available)
    hierarchy_hint = (
        "Les index suivent la convention entreprise__notion.\n"
        f"Hiérarchie :\n{companies}\n\n"
        "Tu peux répondre avec un nom d'entreprise seul (ex: 'celio') pour "
        "sélectionner tous ses sous-index, ou un index précis (ex: 'celio__rh').\n\n"
    )

    # Extract company names for the prompt
    company_names = sorted({
        name.split(INDEX_SEP, 1)[0] for name in available if INDEX_SEP in name
    })

    prompt = (
        f"Index disponibles : {', '.join(available)}\n"
        f"{hierarchy_hint}"
        f"Entreprises connues : {', '.join(company_names)}\n\n"
        f"Requête : \"{query}\"\n\n"
        "Sélectionne le ou les index pertinents pour répondre à cette requête.\n\n"
        "Règles STRICTES (dans cet ordre) :\n"
        "1. La requête mentionne-t-elle EXPLICITEMENT le nom d'une entreprise connue "
        f"({', '.join(company_names)}) ?\n"
        "   - OUI → retourne les index pertinents de CETTE entreprise (domaine + divers)\n"
        "   - NON → retourne TOUS les index de TOUTES les entreprises\n"
        "2. Un nom de produit, un sujet ou un thème ne suffit PAS à identifier une entreprise. "
        "Seul le nom exact de l'entreprise compte.\n"
        "3. Les index contenant 'divers', 'misc' ou 'autre' doivent TOUJOURS être inclus "
        "dès qu'au moins un index de la même entreprise est sélectionné.\n"
        "4. En cas de doute, retourne TOUS les index — il vaut mieux chercher trop large "
        "que rater l'information.\n\n"
        "Réponds UNIQUEMENT avec les noms d'index séparés par des virgules, "
        "en minuscules, sans explication."
    )
    try:
        from vertexai.generative_models import GenerativeModel
        response = GenerativeModel("gemini-2.0-flash-001").generate_content(prompt)
        print(f"#### la reponse de flash : {response.text} #####")
        names    = [n.strip().lower() for n in response.text.strip().split(",")]
        resolved = expand_company_indexes(names, available)
        return resolved if resolved else available
    except Exception:
        # Substring fallback
        q       = query.lower()
        matched = [n for n in available if n.lower() in q]
        # Also try matching bare company/notion names from the query
        if not matched:
            for a in available:
                parts = a.split(INDEX_SEP)
                if any(p in q for p in parts):
                    matched.append(a)
        matched = expand_company_indexes(matched, available) if matched else []
        return matched if matched else available


def _describe_hierarchy(available: list[str]) -> str:
    """
    Build a readable hierarchy string from available index names.

    Example output::

        - celio: rh, commercial, formation
        - clientb: juridique, finance
    """
    groups: dict[str, list[str]] = {}
    for name in available:
        if INDEX_SEP in name:
            company, notion = name.split(INDEX_SEP, 1)
            groups.setdefault(company, []).append(notion)
        else:
            groups.setdefault(name, [])

    lines = []
    for company, notions in sorted(groups.items()):
        if notions:
            lines.append(f"  - {company}: {', '.join(sorted(notions))}")
        else:
            lines.append(f"  - {company} (index simple)")
    return "\n".join(lines)
