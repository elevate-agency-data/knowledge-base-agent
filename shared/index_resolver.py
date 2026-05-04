"""
Shared utility: resolve which hybrid indexes to query for a given prompt.

Uses Gemini Flash to select thematically relevant indexes for a given query.
Falls back to substring matching, then to all indexes if nothing is found.

Naming convention is generic: ``DOMAIN__SUBTOPIC`` (or just ``DOMAIN`` when
there is no L2 wrapper).
  - DOMAIN is the data domain — finances, RH, patrimoine, métier, divers,
    pilotage, etc. — typically the L1 Drive folder name.
  - SUBTOPIC, when present, is an optional refinement (which sub-zone /
    which sub-collection lives inside that domain).
  - The resolver maps French municipal vocabulary (budget, effectifs,
    bâtiment, indicateurs, …) to the right domain rather than to the
    sub-topic, to avoid bad matches when a label loosely overlaps a
    keyword.

Used by:
  - hybrid/tools/hybrid_query.py   (ADK tool — auto-resolution when index_names=[])
  - ui/services/hybrid_service.py  (UI services — Simple Chat)
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

    # Build a human-readable description of the hierarchy for the LLM.
    # Naming convention is generic: ``L1__L2`` where L1 is the **data domain**
    # (finances, RH, patrimoine, métier, divers, pilotage…) and L2 is an
    # optional sub-topic (which commune / which sub-domain). When L2 is
    # absent the index is single-segment.
    hierarchy = _describe_hierarchy(available)
    domain_names = sorted({
        name.split(INDEX_SEP, 1)[0] if INDEX_SEP in name else name
        for name in available
    })

    prompt = (
        f"You are an index resolver for a French city-hall knowledge base.\n\n"
        f"Available indexes:\n{hierarchy}\n\n"
        f"Index naming convention: 'DOMAIN__SUBTOPIC' (or just 'DOMAIN' when "
        f"there is no sub-topic). DOMAIN is the data domain — finances, "
        f"RH, patrimoine, métier, divers, pilotage, etc. — NOT a city name.\n\n"
        f"Known domains (L1): {', '.join(domain_names)}\n\n"
        f"Query: \"{query}\"\n\n"
        "Pick the index(es) whose DOMAIN matches the user's intent. Use "
        "common sense to map French municipal vocabulary to the right "
        "domain:\n"
        "- 'budget', 'dépenses', 'recettes', 'dotation', 'subvention', "
        "'compte administratif', 'fiscalité', 'trésor' → finance / comptable\n"
        "- 'effectifs', 'masse salariale', 'agents', 'fonctionnaires', "
        "'formation', 'paie', 'absentéisme', 'congés' → RH / personnel\n"
        "- 'bâtiment', 'équipement', 'inventaire', 'maintenance', "
        "'travaux', 'énergie', 'parcelle', 'voirie' → patrimoine / maintenance\n"
        "- 'matériel informatique', 'véhicules', 'opérationnel', "
        "'service', 'délégation' → métier divers\n"
        "- 'rapport', 'délibération', 'arrêté', 'orientation', "
        "'égalité' → divers / rapports administratifs\n"
        "- 'indicateurs', 'tableau de bord', 'pilotage', 'mandat' → pilotage / suivi\n\n"
        "Rules:\n"
        "1. Match on DOMAIN keywords first. The SUBTOPIC level (e.g. "
        "'données piscine saint-raphaël') is a refinement — pick it only "
        "if the query explicitly names that sub-topic; otherwise keep "
        "the broader DOMAIN.\n"
        "2. If the query is ambiguous between two domains, return both.\n"
        "3. Indexes containing 'divers', 'misc', 'autre', 'général' should "
        "always be added when their domain matches, because misfiled "
        "documents may live there.\n"
        "4. When in doubt, return MORE indexes — better to search too "
        "broadly than to miss information.\n"
        "5. NEVER pick an index just because a word loosely overlaps with "
        "its label (e.g. 'budget' must NOT route to 'patrimoine-maintenance-"
        "énergie' just because the label contains 'énergie').\n\n"
        "Respond ONLY with index names separated by commas, in lowercase, "
        "no explanation. You may answer with a domain (L1) alone to select "
        "all its sub-topics."
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

        - paris: finances, rh, patrimoine
        - lyon: finances, metier
    """
    groups: dict[str, list[str]] = {}
    for name in available:
        if INDEX_SEP in name:
            commune, category = name.split(INDEX_SEP, 1)
            groups.setdefault(commune, []).append(category)
        else:
            groups.setdefault(name, [])

    lines = []
    for commune, categories in sorted(groups.items()):
        if categories:
            lines.append(f"  - {commune}: {', '.join(sorted(categories))}")
        else:
            lines.append(f"  - {commune} (single index)")
    return "\n".join(lines)
