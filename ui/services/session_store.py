"""
Persistent session storage for RAG Comparison sessions.

Sessions are saved as JSON files in ui/data/comparison_sessions/.
Agent Chat sessions are kept in-memory only (st.session_state) — they are
intentionally NOT stored here to keep the two namespaces fully separate.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

_SESSIONS_DIR = Path(__file__).parent.parent / "data" / "comparison_sessions"


def _dir() -> Path:
    _SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    return _SESSIONS_DIR


# ── CRUD ──────────────────────────────────────────────────────────────────────

def list_sessions() -> list[dict]:
    """
    Return all saved comparison sessions sorted by created_at desc.
    Returns metadata only (no history) for fast listing.
    """
    sessions = []
    for f in _dir().glob("*.json"):
        try:
            with open(f, encoding="utf-8") as fp:
                data = json.load(fp)
            sessions.append({
                "id":          data["id"],
                "name":        data["name"],
                "created_at":  data["created_at"],
                "query_count": len(data.get("history", [])),
            })
        except Exception:
            pass
    return sorted(sessions, key=lambda s: s["created_at"], reverse=True)


def load_session(session_id: str) -> dict | None:
    """Load a full session (including history) by id. Returns None if not found."""
    path = _dir() / f"{session_id}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_session(session: dict) -> None:
    """Persist a session dict to disk (creates or overwrites)."""
    path = _dir() / f"{session['id']}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(session, f, ensure_ascii=False, indent=2, default=str)


def new_session(name: str | None = None) -> dict:
    """Create a new empty session, persist it, and return it."""
    now = datetime.now()
    session = {
        "id":         str(uuid.uuid4())[:8],
        "name":       name or f"Session {now.strftime('%d/%m %H:%M')}",
        "created_at": now.isoformat(),
        "history":    [],
    }
    save_session(session)
    return session


def append_entry(session: dict, entry: dict) -> dict:
    """Add a comparison entry to a session and save. Returns the updated session."""
    session["history"].append(entry)
    save_session(session)
    return session


def rename_session(session_id: str, new_name: str) -> None:
    session = load_session(session_id)
    if session:
        session["name"] = new_name
        save_session(session)


def delete_session(session_id: str) -> None:
    path = _dir() / f"{session_id}.json"
    if path.exists():
        path.unlink()
