"""
Module d'authentification — login / guard / logout / gestion admin.

Stockage : SQLite à ui/data/users.db
Mots de passe : PBKDF2-HMAC-SHA256 (stdlib, aucune dépendance externe).

Création de comptes : réservée aux admins (via page Admin ou script CLI).
Les utilisateurs normaux ne peuvent que se connecter.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st

_DB_DIR  = Path(__file__).parent / "data"
_DB_PATH = _DB_DIR / "users.db"

_conn: sqlite3.Connection | None = None


# ── Connexion & schéma ────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _DB_DIR.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            TEXT PRIMARY KEY,
                email         TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                display_name  TEXT NOT NULL,
                created_at    TEXT NOT NULL,
                is_active     INTEGER NOT NULL DEFAULT 1,
                is_admin      INTEGER NOT NULL DEFAULT 0
            )
        """)
        # Migration : ajoute is_admin si la table existait sans cette colonne
        try:
            _conn.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER NOT NULL DEFAULT 0")
        except sqlite3.OperationalError:
            pass  # Colonne déjà présente
        _conn.commit()
    return _conn


# ── Hachage mot de passe ──────────────────────────────────────────────────────

def _hash_password(password: str, salt: bytes | None = None) -> str:
    if salt is None:
        salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 260_000)
    return f"{salt.hex()}:{dk.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, _ = stored.split(":", 1)
        return _hash_password(password, bytes.fromhex(salt_hex)) == stored
    except Exception:
        return False


# ── API publique ──────────────────────────────────────────────────────────────

def create_user(
    email: str,
    password: str,
    display_name: str,
    is_admin: bool = False,
) -> dict | None:
    """
    Crée un utilisateur (réservé aux admins ou au script d'init).
    Retourne le dict user ou None si l'email est déjà pris.
    """
    conn    = _get_conn()
    user_id = str(uuid.uuid4())
    now     = datetime.now().isoformat()
    try:
        conn.execute(
            "INSERT INTO users (id, email, password_hash, display_name, created_at, is_admin) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [user_id, email.lower().strip(), _hash_password(password),
             display_name.strip(), now, int(is_admin)],
        )
        conn.commit()
        return {
            "id":           user_id,
            "email":        email.lower().strip(),
            "display_name": display_name.strip(),
            "is_admin":     is_admin,
        }
    except sqlite3.IntegrityError:
        return None  # Email déjà pris


def authenticate_user(email: str, password: str) -> dict | None:
    """
    Vérifie les identifiants.
    Retourne le dict user (avec is_admin) ou None si invalide / inactif.
    """
    conn = _get_conn()
    row  = conn.execute(
        "SELECT id, email, password_hash, display_name, is_active, is_admin "
        "FROM users WHERE email = ?",
        [email.lower().strip()],
    ).fetchone()
    if not row:
        return None
    user_id, user_email, pw_hash, display_name, is_active, is_admin = row
    if not is_active:
        return None
    if not _verify_password(password, pw_hash):
        return None
    return {
        "id":           user_id,
        "email":        user_email,
        "display_name": display_name,
        "is_admin":     bool(is_admin),
    }


def list_users() -> list[dict]:
    """Retourne tous les utilisateurs (usage admin)."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, email, display_name, created_at, is_active, is_admin "
        "FROM users ORDER BY created_at DESC"
    ).fetchall()
    return [
        {
            "id":           r[0],
            "email":        r[1],
            "display_name": r[2],
            "created_at":   r[3],
            "is_active":    bool(r[4]),
            "is_admin":     bool(r[5]),
        }
        for r in rows
    ]


def set_user_active(user_id: str, active: bool) -> None:
    """Active ou désactive un compte utilisateur."""
    conn = _get_conn()
    conn.execute(
        "UPDATE users SET is_active = ? WHERE id = ?",
        [int(active), user_id],
    )
    conn.commit()


def reset_password(user_id: str, new_password: str) -> None:
    """Réinitialise le mot de passe d'un utilisateur (usage admin)."""
    conn = _get_conn()
    conn.execute(
        "UPDATE users SET password_hash = ? WHERE id = ?",
        [_hash_password(new_password), user_id],
    )
    conn.commit()


# ── Session state ─────────────────────────────────────────────────────────────

def login_user(user: dict) -> None:
    """Stocke les infos de l'utilisateur dans st.session_state."""
    st.session_state.user_id      = user["id"]
    st.session_state.user_email   = user["email"]
    st.session_state.display_name = user["display_name"]
    st.session_state.is_admin     = user.get("is_admin", False)


def logout_user() -> None:
    """Efface l'état auth + chat et redirige vers la page de login."""
    keys = [
        "user_id", "user_email", "display_name", "is_admin",
        "agent_active_session_id", "agent_sessions_cache",
        "sc_active_session_id", "sc_sessions_cache",
        "sc_messages", "sc_rag_on", "sc_pipeline", "sc_corpus",
    ]
    for k in keys:
        st.session_state.pop(k, None)
    st.rerun()


# ── Guards ────────────────────────────────────────────────────────────────────

def require_auth() -> dict:
    """
    Guard — à appeler en haut de chaque page protégée.
    Redirige vers le login si non authentifié.
    """
    if not st.session_state.get("user_id"):
        st.switch_page("pages/0_Login.py")
        st.stop()
    return {
        "id":           st.session_state.user_id,
        "email":        st.session_state.get("user_email", ""),
        "display_name": st.session_state.get("display_name", ""),
        "is_admin":     st.session_state.get("is_admin", False),
    }


def require_admin() -> dict:
    """
    Guard — à appeler en haut des pages réservées aux admins.
    Redirige vers login si non authentifié, affiche une erreur si non admin.
    """
    user = require_auth()
    if not user["is_admin"]:
        st.error("Accès refusé — cette page est réservée aux administrateurs.")
        st.stop()
    return user
