"""
Atelier ingestion mapping — shared by the local and Drive ingesters.

Both sources produce the SAME thing from a folder path: which index a file
belongs to, and the ligne_produit / zone dimensions carried as metadata. Keeping
this in one place guarantees the local test corpus and the Drive corpus are
projected identically.

Convention (depth-based):

    <root>/<L1>/<L2>/[<L3>/]<file>
            │    │     └ zone (optional)      -> "zone"
            │    └ domaine (leaf)             -> INDEX NAME (ascii, lowercase)
            └ ligne_produit                   -> "ligne_produit"
                (the literal "_Transverse" L1 carries NO ligne_produit)

_meta.yaml files (audience_role, matiere) are merged nearest-wins up the tree.
"""

from __future__ import annotations

import re
import unicodedata

META_FILENAME = "_meta.yaml"
TRANSVERSE = "_transverse"   # L1 folder whose files carry no ligne_produit


def sanitize_index(name: str) -> str:
    """Domain folder name → ascii, lowercase, underscore index name.

    Drops accents so the RBAC/keyword matching (which is accent-sensitive
    substring matching) works: ``Authenticité`` → ``authenticite``.
    """
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_ = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9_]+", "_", ascii_.lower()).strip("_")


def map_folders(folders: list[str]) -> dict | None:
    """Map the folder chain (root-relative, filename excluded) to dimensions.

    Args:
        folders: e.g. ``["Maroquinerie", "Garantie", "EU"]`` or
                 ``["_Transverse", "Pilotage"]``.

    Returns:
        ``{"index", "domaine", "ligne_produit", "zone"}`` or ``None`` when the
        path is too shallow (needs at least L1/L2).
    """
    if len(folders) < 2:
        return None
    ligne_raw = folders[0]
    ligne = "" if ligne_raw.lower() == TRANSVERSE else ligne_raw.lower()
    domaine_folder = folders[1]
    zone = folders[2].lower() if len(folders) >= 3 else ""
    return {
        "index":          sanitize_index(domaine_folder),
        "domaine":        domaine_folder,     # human label kept as-is
        "ligne_produit":  ligne,
        "zone":           zone,
    }


def as_list(v) -> list[str]:
    """Normalise a _meta.yaml value into a clean list of strings."""
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [str(x).strip() for x in v if str(x).strip()]
    return [str(v).strip()] if str(v).strip() else []


def merge_meta(base: dict, override: dict) -> dict:
    """Nearest-wins merge of two _meta.yaml dicts (override on top of base)."""
    out = dict(base or {})
    for k, v in (override or {}).items():
        out[k] = v
    return out
