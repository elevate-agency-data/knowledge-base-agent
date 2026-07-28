"""
Upload store for SAV photos.

An LLM tool call can only carry text, so an uploaded photo is written to disk
once and referenced by a short id (``img_a1b2c3d4``). The UI saves the file and
shows the ref; the agent passes that ref to the vision tools, which resolve it
back to a path. Refs are scoped per user so one session can't read another's
uploads.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

from .config import UPLOAD_DIR, UPLOAD_MAX_AGE_HOURS, UPLOAD_MAX_MB

_ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".webp"}
_REF_PREFIX = "img_"


def _dir() -> Path:
    """Upload directory, created on demand (no import-time side effect)."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return UPLOAD_DIR


def _safe_user(user_id: str) -> str:
    """Filesystem-safe fragment of a user id, used to scope refs."""
    keep = [c if c.isalnum() else "_" for c in (user_id or "anon")]
    return "".join(keep)[:40] or "anon"


def save_upload(data: bytes, file_name: str, user_id: str = "") -> dict:
    """Persist an uploaded image and return its ref.

    Args:
        data:      Raw file bytes.
        file_name: Original name — only its extension is trusted.
        user_id:   Owner, embedded in the stored filename for scoping.

    Returns:
        ``{"status", "ref", "path", "file_name", "message"}``.
    """
    ext = Path(file_name).suffix.lower()
    if ext not in _ALLOWED_EXT:
        return {
            "status": "error",
            "message": f"Unsupported image type '{ext}'. Use PNG, JPG or WEBP.",
        }
    if not data:
        return {"status": "error", "message": "Empty file."}

    ref = f"{_REF_PREFIX}{uuid.uuid4().hex[:8]}"
    dest = _dir() / f"{ref}__{_safe_user(user_id)}{ext}"
    try:
        dest.write_bytes(data)
    except OSError as exc:
        return {"status": "error", "message": f"Could not store the image: {exc}"}

    return {
        "status": "success",
        "ref": ref,
        "path": str(dest),
        "file_name": file_name,
    }


def resolve_ref(ref: str, user_id: str = "") -> Path | None:
    """Return the stored path for *ref*, or None if unknown.

    When *user_id* is given, a ref belonging to another user is not resolved.
    """
    if not ref:
        return None
    ref = ref.strip()
    # Tolerate the agent passing a full path or a decorated ref.
    if ref.startswith(_REF_PREFIX) is False:
        candidate = Path(ref)
        if candidate.is_file() and candidate.parent.resolve() == _dir().resolve():
            return candidate
        return None

    matches = sorted(_dir().glob(f"{ref}__*"))
    if not matches:
        return None
    path = matches[0]
    if user_id and f"__{_safe_user(user_id)}" not in path.name:
        return None
    return path


def list_uploads(user_id: str = "") -> list[dict]:
    """List stored uploads (newest first), optionally scoped to one user."""
    try:
        files = [f for f in _dir().iterdir() if f.is_file()]
    except OSError:
        return []

    out = []
    scope = f"__{_safe_user(user_id)}" if user_id else ""
    for f in files:
        if scope and scope not in f.name:
            continue
        ref = f.name.split("__", 1)[0]
        try:
            mtime = f.stat().st_mtime
        except OSError:
            continue
        out.append({"ref": ref, "path": str(f), "modified": mtime})
    out.sort(key=lambda e: e["modified"], reverse=True)
    return out


def prune_uploads() -> int:
    """Delete uploads older than the age cap, then oldest-first over the size cap."""
    directory = _dir()
    try:
        files = [f for f in directory.iterdir() if f.is_file()]
    except OSError:
        return 0

    entries = []
    for f in files:
        try:
            stat = f.stat()
        except OSError:
            continue
        entries.append((f, stat.st_mtime, stat.st_size))

    removed = 0
    cutoff = time.time() - UPLOAD_MAX_AGE_HOURS * 3600
    survivors = []
    for f, mtime, size in entries:
        if mtime < cutoff:
            try:
                f.unlink()
                removed += 1
            except OSError:
                survivors.append((f, mtime, size))
        else:
            survivors.append((f, mtime, size))

    budget = UPLOAD_MAX_MB * 1024 * 1024
    total = sum(size for _f, _m, size in survivors)
    if total > budget:
        survivors.sort(key=lambda e: e[1])       # oldest first
        for f, _mtime, size in survivors:
            if total <= budget:
                break
            try:
                f.unlink()
                total -= size
                removed += 1
            except OSError:
                continue

    if removed:
        logging.info("Upload store pruned: %d file(s) removed", removed)
    return removed
