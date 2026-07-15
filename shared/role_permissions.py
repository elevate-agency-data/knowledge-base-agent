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

from shared.brand import ACTIVE as _BRAND

INDEX_SEP = "__"

# ── Canonical roles (from the active brand profile) ───────────────────────────
# Each brand defines its own roles, default role and per-role access rules in
# shared/brand.py — this module just consumes the active profile.
ROLES: list[str] = list(_BRAND.roles)
DEFAULT_ROLE: str = _BRAND.default_role

# ── Role → keyword whitelist ─────────────────────────────────────────────────
# None = full access. A tuple/list = substring matches (case-insensitive) on
# the category part of the index name.
ROLE_KEYWORDS: dict[str, tuple[str, ...] | None] = dict(_BRAND.role_keywords)


def _category_of(index_name: str) -> str:
    """Return the category portion of ``domain__category``, lowercased.

    Single-segment index names (no ``__``) are returned whole — for brands
    whose L1 index *is* the category (e.g. hermes ``garantie``).
    """
    if INDEX_SEP in index_name:
        return index_name.split(INDEX_SEP, 1)[1].lower()
    return index_name.lower()


def filter_indexes_for_role(indexes: list[str], role: str | None) -> list[str]:
    """
    Filter a list of index names down to those visible to *role*.

    Args:
        indexes: All available index names.
        role:    Business role string (defaults to the brand's default role
                 if falsy/unknown).

    Returns:
        Subset of *indexes* the role is allowed to query. Order preserved.
    """
    if not indexes:
        return []
    role = (role or DEFAULT_ROLE).lower()
    keywords = ROLE_KEYWORDS.get(role, ROLE_KEYWORDS.get(DEFAULT_ROLE))
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
    return ROLE_KEYWORDS.get((role or DEFAULT_ROLE).lower()) is None
