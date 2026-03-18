"""
Persistent session storage for RAG Comparison sessions — DuckDB backend.

Base de données : ui/data/comparaison/comparaison.duckdb
Deux tables cloisonnées :
  - comparison_sessions  : métadonnées d'une session (id, name, created_at)
  - comparison_entries   : un enregistrement par échange Q/R (lié à une session)

Les contextes Vertex et Hybrid sont stockés séparément dans vertex_result et
hybrid_result (JSON) — ils ne sont jamais fusionnés.

Agent Chat sessions are intentionally NOT stored here.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

import duckdb

_DB_DIR  = Path(__file__).parent.parent / "data" / "comparaison"
_DB_PATH = _DB_DIR / "comparaison.duckdb"

_conn: duckdb.DuckDBPyConnection | None = None


# ── Connexion & schéma ────────────────────────────────────────────────────────

def _get_conn() -> duckdb.DuckDBPyConnection:
    global _conn
    if _conn is None:
        _DB_DIR.mkdir(parents=True, exist_ok=True)
        _conn = duckdb.connect(str(_DB_PATH))
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS comparison_sessions (
                id         VARCHAR PRIMARY KEY,
                name       VARCHAR NOT NULL,
                created_at VARCHAR NOT NULL
            )
        """)
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS comparison_entries (
                entry_id      VARCHAR PRIMARY KEY,
                session_id    VARCHAR NOT NULL,
                position      INTEGER NOT NULL,
                created_at    VARCHAR NOT NULL,
                query         VARCHAR NOT NULL,
                hybrid_mode   VARCHAR,
                indexes_used  VARCHAR,
                vertex_result VARCHAR,
                hybrid_result VARCHAR
            )
        """)
        _migrate_json()
    return _conn


# ── Migration JSON → DuckDB (one-time) ───────────────────────────────────────

def _migrate_json() -> None:
    """Importe les fichiers JSON existants dans DuckDB puis les archive."""
    for jf in _DB_DIR.glob("*.json"):
        try:
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
            already = _conn.execute(
                "SELECT id FROM comparison_sessions WHERE id = ?", [data["id"]]
            ).fetchone()
            if already:
                jf.rename(jf.with_suffix(".json.migrated"))
                continue
            _conn.execute(
                "INSERT INTO comparison_sessions VALUES (?, ?, ?)",
                [data["id"], data["name"], data["created_at"]],
            )
            for pos, entry in enumerate(data.get("history", [])):
                _insert_entry_row(_conn, data["id"], pos, data["created_at"], entry)
            jf.rename(jf.with_suffix(".json.migrated"))
        except Exception:
            pass


# ── Helpers internes ──────────────────────────────────────────────────────────

def _insert_entry_row(
    conn: duckdb.DuckDBPyConnection,
    session_id: str,
    position: int,
    created_at: str,
    entry: dict,
) -> None:
    conn.execute(
        "INSERT INTO comparison_entries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            str(uuid.uuid4()),
            session_id,
            position,
            created_at,
            entry["query"],
            entry.get("hybrid_mode", "mono"),
            json.dumps(entry.get("indexes_used", []), ensure_ascii=False),
            json.dumps(entry.get("vertex", {}), ensure_ascii=False, default=str),
            json.dumps(entry.get("hybrid", {}), ensure_ascii=False, default=str),
        ],
    )


def _rows_to_history(rows) -> list[dict]:
    history = []
    for r in rows:
        history.append({
            "query":        r[0],
            "hybrid_mode":  r[1],
            "indexes_used": json.loads(r[2]) if r[2] else [],
            "vertex":       json.loads(r[3]) if r[3] else {},
            "hybrid":       json.loads(r[4]) if r[4] else {},
        })
    return history


# ── API publique ──────────────────────────────────────────────────────────────

def list_sessions() -> list[dict]:
    """Retourne les métadonnées de toutes les sessions (sans historique), triées par date desc."""
    conn = _get_conn()
    rows = conn.execute("""
        SELECT s.id, s.name, s.created_at, COUNT(e.entry_id) AS query_count
        FROM comparison_sessions s
        LEFT JOIN comparison_entries e ON e.session_id = s.id
        GROUP BY s.id, s.name, s.created_at
        ORDER BY s.created_at DESC, s.id
    """).fetchall()
    return [
        {"id": r[0], "name": r[1], "created_at": r[2], "query_count": r[3]}
        for r in rows
    ]


def load_session(session_id: str) -> dict | None:
    """Charge une session complète (métadonnées + historique). None si introuvable."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT id, name, created_at FROM comparison_sessions WHERE id = ?",
        [session_id],
    ).fetchone()
    if not row:
        return None
    entries = conn.execute(
        """SELECT query, hybrid_mode, indexes_used, vertex_result, hybrid_result
           FROM comparison_entries
           WHERE session_id = ?
           ORDER BY position ASC""",
        [session_id],
    ).fetchall()
    return {
        "id":         row[0],
        "name":       row[1],
        "created_at": row[2],
        "history":    _rows_to_history(entries),
    }


def save_session(session: dict) -> None:
    """Upsert des métadonnées d'une session (ne touche pas aux entrées)."""
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO comparison_sessions VALUES (?, ?, ?)",
        [session["id"], session["name"], session["created_at"]],
    )


def new_session(name: str | None = None) -> dict:
    """Crée une nouvelle session vide, la persiste et la retourne."""
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
    """Ajoute un échange Q/R à la session et insère directement la ligne en base."""
    conn = _get_conn()
    session["history"].append(entry)
    _insert_entry_row(
        conn,
        session["id"],
        len(session["history"]) - 1,
        datetime.now().isoformat(),
        entry,
    )
    return session


def rename_session(session_id: str, new_name: str) -> None:
    conn = _get_conn()
    conn.execute(
        "UPDATE comparison_sessions SET name = ? WHERE id = ?",
        [new_name, session_id],
    )


def delete_session(session_id: str) -> None:
    conn = _get_conn()
    conn.execute("DELETE FROM comparison_entries WHERE session_id = ?", [session_id])
    conn.execute("DELETE FROM comparison_sessions WHERE id = ?", [session_id])
