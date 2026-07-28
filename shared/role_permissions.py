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


_MISSING = object()


def _keywords_for(role: str | None) -> tuple[str, ...] | None:
    """Resolve the keyword whitelist for *role*.

    A role that is absent from the map (e.g. a legacy role left over from
    another brand's user store) falls back to the brand's default role. This
    must NOT be confused with a role mapped to ``None``, which means full
    access — ``dict.get`` alone returns ``None`` in both cases, which would
    silently grant an unknown role the run of the whole store.
    """
    keywords = ROLE_KEYWORDS.get((role or DEFAULT_ROLE).lower(), _MISSING)
    if keywords is _MISSING:
        keywords = ROLE_KEYWORDS.get(DEFAULT_ROLE)
    return keywords  # type: ignore[return-value]


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
    keywords = _keywords_for(role)
    if keywords is None:
        return list(indexes)

    out: list[str] = []
    for idx in indexes:
        category = _category_of(idx)
        if any(kw in category for kw in keywords):
            out.append(idx)
    return out


def role_has_full_access(role: str | None) -> bool:
    """True if the role bypasses all index filtering.

    An unknown role resolves through the default role, so it can only be
    full-access if the default role itself is.
    """
    return _keywords_for(role) is None


# ── Chunk-level RBAC (atelier model) ──────────────────────────────────────────
# Finer than index-name filtering: a chunk carries an ``audience_role`` list
# (set at ingestion from _meta.yaml). A role sees a chunk when the brand maps
# that role to one of the chunk's audience_role values. Brands that don't define
# ``role_audience`` fall back to the index-name keyword model (no chunk gating).

ROLE_AUDIENCE: dict[str, tuple[str, ...] | None] = dict(_BRAND.role_audience)


def _audience_for(role: str | None) -> tuple[str, ...] | None | object:
    """Allowed audience_role values for *role*.

    Returns ``None`` for full access, a tuple of allowed values, or the
    ``_MISSING`` sentinel when the brand defines no chunk-level RBAC for this
    role (caller should then skip chunk gating).
    """
    if not ROLE_AUDIENCE:
        return _MISSING
    val = ROLE_AUDIENCE.get((role or DEFAULT_ROLE).lower(), _MISSING)
    if val is _MISSING:
        val = ROLE_AUDIENCE.get(DEFAULT_ROLE, _MISSING)
    return val


def chunk_visible_for_role(chunk: dict, role: str | None) -> bool:
    """Return True if *role* may see *chunk* under the atelier RBAC.

    - Brand has no ``role_audience`` → no chunk gating, always visible.
    - Role maps to ``None`` → full access.
    - Chunk has an empty ``audience_role`` → treated as open (visible).
    - Otherwise visible iff the role's allowed set intersects the chunk's
      ``audience_role``.
    """
    allowed = _audience_for(role)
    if allowed is _MISSING or allowed is None:
        return True
    chunk_roles = [r.lower() for r in (chunk.get("audience_role") or [])]
    if not chunk_roles:
        return True
    allowed_set = {a.lower() for a in allowed}
    return bool(allowed_set & set(chunk_roles))


def filter_chunks_for_role(chunks: list[dict], role: str | None) -> list[dict]:
    """Filter a retrieved chunk list down to those visible to *role*."""
    return [c for c in chunks if chunk_visible_for_role(c, role)]
