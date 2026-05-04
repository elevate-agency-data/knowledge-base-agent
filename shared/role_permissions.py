"""
Role-based access control for hybrid indexes.

A user has a single business role (stored in users.db).
Each role grants access to a subset of indexes, determined by
substring keywords matched against the **category** part of the
index name (the part after ``__``).

Example: an index named ``paris__finances`` exposes the category
``finances``. The ``comptable`` role's keyword list contains
``"finance"``, which is a substring of ``"finances"`` — so
``paris__finances`` is visible to a ``comptable`` user.

Roles with ``None`` keywords get full access (no filtering).

Edit this file to adjust the allowed keywords without touching
the rest of the code.
"""

from __future__ import annotations

INDEX_SEP = "__"

# ── Canonical roles ──────────────────────────────────────────────────────────

ROLES: list[str] = ["maire", "adjoint", "comptable", "rh", "agent"]

# ── Role → keyword whitelist ─────────────────────────────────────────────────
# None = full access. A list = substring matches (case-insensitive) on the
# category part of the index name.
ROLE_KEYWORDS: dict[str, list[str] | None] = {
    "maire":     None,
    "adjoint":   None,
    "comptable": ["finance", "compta", "budget", "tresor"],
    "rh":        ["rh", "personnel", "ressources humaines", "ressources_humaines"],
    "agent":     ["metier", "service"],
}


def _category_of(index_name: str) -> str:
    """Return the category portion of ``commune__category``, lowercased."""
    if INDEX_SEP in index_name:
        return index_name.split(INDEX_SEP, 1)[1].lower()
    return index_name.lower()


def filter_indexes_for_role(indexes: list[str], role: str | None) -> list[str]:
    """
    Filter a list of index names down to those visible to *role*.

    Args:
        indexes: All available index names.
        role:    Business role string (defaults to "agent" if falsy/unknown).

    Returns:
        Subset of *indexes* the role is allowed to query. Order preserved.
    """
    if not indexes:
        return []
    role = (role or "agent").lower()
    keywords = ROLE_KEYWORDS.get(role, ROLE_KEYWORDS["agent"])
    if keywords is None:
        return list(indexes)

    out: list[str] = []
    for idx in indexes:
        category = _category_of(idx)
        if any(kw in category for kw in keywords):
            out.append(idx)
    return out


def role_has_full_access(role: str | None) -> bool:
    """True if the role bypasses all index filtering."""
    return ROLE_KEYWORDS.get((role or "agent").lower()) is None
