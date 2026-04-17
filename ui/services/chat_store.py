"""
Persistence des chats — backend DuckDB.

Base : ui/data/chat/chat.duckdb
Tables :
  sc_sessions    — sessions Simple Chat (avec état RAG inclus)
  sc_messages    — messages Simple Chat
  agent_messages — messages Agent Chat (remplace les fichiers JSON sidecar)

Toutes les tables contiennent user_id pour l'isolation par utilisateur.
Les écritures sont sérialisées via threading.Lock (contrainte single-writer DuckDB).
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime
from pathlib import Path

import duckdb

_DB_DIR  = Path(__file__).parent.parent / "data" / "chat"
_DB_PATH = _DB_DIR / "chat.duckdb"

_conn: duckdb.DuckDBPyConnection | None = None
_lock: threading.Lock = threading.Lock()


# ── Connexion & schéma ────────────────────────────────────────────────────────

def _get_conn() -> duckdb.DuckDBPyConnection:
    global _conn
    if _conn is None:
        _DB_DIR.mkdir(parents=True, exist_ok=True)
        _conn = duckdb.connect(str(_DB_PATH))
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS sc_sessions (
                id          VARCHAR PRIMARY KEY,
                user_id     VARCHAR NOT NULL,
                name        VARCHAR NOT NULL,
                rag_on      BOOLEAN NOT NULL DEFAULT FALSE,
                pipeline    VARCHAR,
                corpus      VARCHAR,
                created_at  VARCHAR NOT NULL,
                updated_at  VARCHAR NOT NULL
            )
        """)
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS sc_messages (
                id          VARCHAR PRIMARY KEY,
                session_id  VARCHAR NOT NULL,
                user_id     VARCHAR NOT NULL,
                position    INTEGER NOT NULL,
                role        VARCHAR NOT NULL,
                text        TEXT    NOT NULL,
                sources     VARCHAR,
                chunks      VARCHAR,
                pipeline    VARCHAR,
                elapsed_s   DOUBLE,
                timings     VARCHAR,
                created_at  VARCHAR NOT NULL
            )
        """)
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_messages (
                id          VARCHAR PRIMARY KEY,
                user_id     VARCHAR NOT NULL,
                session_id  VARCHAR NOT NULL,
                position    INTEGER NOT NULL,
                role        VARCHAR NOT NULL,
                text        TEXT    NOT NULL,
                tool_events VARCHAR,
                created_at  VARCHAR NOT NULL
            )
        """)
    return _conn


# ═══════════════════════════════════════════════════════════════════════════════
# Simple Chat — sessions
# ═══════════════════════════════════════════════════════════════════════════════

def sc_list_sessions(user_id: str) -> list[dict]:
    """Retourne toutes les sessions de l'utilisateur, triées par dernière activité."""
    conn = _get_conn()
    rows = conn.execute("""
        SELECT s.id, s.name, s.rag_on, s.pipeline, s.corpus,
               s.created_at, s.updated_at,
               COUNT(m.id) AS msg_count
        FROM sc_sessions s
        LEFT JOIN sc_messages m ON m.session_id = s.id
        WHERE s.user_id = ?
        GROUP BY s.id, s.name, s.rag_on, s.pipeline, s.corpus,
                 s.created_at, s.updated_at
        ORDER BY s.updated_at DESC
    """, [user_id]).fetchall()
    return [
        {
            "id":         r[0],
            "name":       r[1],
            "rag_on":     bool(r[2]),
            "pipeline":   r[3],
            "corpus":     r[4],
            "created_at": r[5],
            "updated_at": r[6],
            "msg_count":  r[7],
        }
        for r in rows
    ]


def sc_new_session(
    user_id: str,
    rag_on:   bool        = False,
    pipeline: str | None  = None,
    corpus:   str | None  = None,
) -> dict:
    """Crée une session vide, la persiste et la retourne."""
    now = datetime.now().isoformat()
    session = {
        "id":         str(uuid.uuid4())[:8],
        "user_id":    user_id,
        "name":       f"Chat {datetime.now().strftime('%d/%m %H:%M')}",
        "rag_on":     rag_on,
        "pipeline":   pipeline,
        "corpus":     corpus,
        "created_at": now,
        "updated_at": now,
    }
    with _lock:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO sc_sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [session["id"], user_id, session["name"],
             rag_on, pipeline, corpus, now, now],
        )
    return session


def sc_update_settings(
    session_id: str,
    rag_on:     bool,
    pipeline:   str | None,
    corpus:     str | None,
) -> None:
    """Met à jour les paramètres RAG d'une session."""
    with _lock:
        conn = _get_conn()
        conn.execute(
            "UPDATE sc_sessions SET rag_on=?, pipeline=?, corpus=?, updated_at=? WHERE id=?",
            [rag_on, pipeline, corpus, datetime.now().isoformat(), session_id],
        )


def sc_delete_session(session_id: str) -> None:
    with _lock:
        conn = _get_conn()
        conn.execute("DELETE FROM sc_messages WHERE session_id = ?", [session_id])
        conn.execute("DELETE FROM sc_sessions  WHERE id        = ?", [session_id])


# ── Simple Chat — messages ────────────────────────────────────────────────────

def sc_load_messages(session_id: str) -> list[dict]:
    """Charge tous les messages d'une session, ordonnés par position."""
    conn = _get_conn()
    rows = conn.execute("""
        SELECT role, text, sources, chunks, pipeline, elapsed_s, timings
        FROM sc_messages
        WHERE session_id = ?
        ORDER BY position ASC
    """, [session_id]).fetchall()
    result = []
    for r in rows:
        msg: dict = {"role": r[0], "text": r[1]}
        if r[2]: msg["sources"]  = json.loads(r[2])
        if r[3]: msg["chunks"]   = json.loads(r[3])
        if r[4]: msg["pipeline"] = r[4]
        if r[5] is not None: msg["elapsed_s"] = r[5]
        if r[6]: msg["timings"]  = json.loads(r[6])
        result.append(msg)
    return result


def sc_append_message(session_id: str, user_id: str, msg: dict) -> None:
    """Insère un message et met à jour la session (updated_at + nom auto)."""
    now = datetime.now().isoformat()
    with _lock:
        conn = _get_conn()
        row = conn.execute(
            "SELECT COALESCE(MAX(position), -1) FROM sc_messages WHERE session_id = ?",
            [session_id],
        ).fetchone()
        position = (row[0] if row else -1) + 1

        conn.execute(
            "INSERT INTO sc_messages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                str(uuid.uuid4()),
                session_id,
                user_id,
                position,
                msg["role"],
                msg["text"],
                json.dumps(msg["sources"], ensure_ascii=False)
                    if msg.get("sources") is not None else None,
                json.dumps(msg["chunks"],  ensure_ascii=False)
                    if msg.get("chunks")  is not None else None,
                msg.get("pipeline"),
                msg.get("elapsed_s"),
                json.dumps(msg["timings"], ensure_ascii=False)
                    if msg.get("timings") is not None else None,
                now,
            ],
        )
        # Nommage automatique : premier message user → nom de la session
        if msg["role"] == "user" and position == 0:
            text = msg["text"]
            name = (text[:45] + "…") if len(text) > 45 else text
            conn.execute(
                "UPDATE sc_sessions SET name=?, updated_at=? WHERE id=?",
                [name, now, session_id],
            )
        else:
            conn.execute(
                "UPDATE sc_sessions SET updated_at=? WHERE id=?",
                [now, session_id],
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Agent Chat — remplace les fichiers JSON sidecar
# ═══════════════════════════════════════════════════════════════════════════════

def agent_load_messages(user_id: str, session_id: str) -> list[dict]:
    """Charge les messages d'affichage d'une session agent."""
    conn = _get_conn()
    rows = conn.execute("""
        SELECT role, text, tool_events
        FROM agent_messages
        WHERE user_id = ? AND session_id = ?
        ORDER BY position ASC
    """, [user_id, session_id]).fetchall()
    result = []
    for r in rows:
        msg: dict = {"role": r[0], "text": r[1]}
        if r[2]:
            msg["tool_events"] = json.loads(r[2])
        result.append(msg)
    return result


def agent_append_message(user_id: str, session_id: str, msg: dict) -> None:
    """Insère un message d'affichage agent."""
    now = datetime.now().isoformat()
    with _lock:
        conn = _get_conn()
        row = conn.execute(
            "SELECT COALESCE(MAX(position), -1) "
            "FROM agent_messages WHERE user_id = ? AND session_id = ?",
            [user_id, session_id],
        ).fetchone()
        position = (row[0] if row else -1) + 1
        conn.execute(
            "INSERT INTO agent_messages VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                str(uuid.uuid4()),
                user_id,
                session_id,
                position,
                msg["role"],
                msg["text"],
                json.dumps(msg.get("tool_events"), ensure_ascii=False, default=str)
                    if msg.get("tool_events") is not None else None,
                now,
            ],
        )


def agent_sessions_meta(user_id: str) -> dict[str, dict]:
    """
    Retourne un dict {session_id: {name, message_count}} pour enrichir
    la liste de sessions ADK dans agent_runner.list_sessions().
    """
    conn = _get_conn()
    rows = conn.execute("""
        SELECT session_id,
               COUNT(*)                                        AS msg_count,
               MIN(CASE WHEN role = 'user' THEN text END)     AS first_user_msg
        FROM agent_messages
        WHERE user_id = ?
        GROUP BY session_id
    """, [user_id]).fetchall()
    result: dict[str, dict] = {}
    for r in rows:
        sid, count, first = r[0], r[1], r[2] or ""
        name = (first[:45] + "…") if len(first) > 45 else first
        result[sid] = {
            "name":          name or f"Session {sid[:8]}",
            "message_count": count,
        }
    return result
