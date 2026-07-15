"""
Module d'authentification — login / guard / logout / gestion admin.

Stockage : SQLite, chemin par marque via shared/brand.py (users_db_path).
Chaque marque a donc sa propre base de comptes (isolation auth).
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

from shared.brand import ACTIVE as _BRAND
from shared.role_permissions import DEFAULT_ROLE

# Per-brand user store. Resolve relative paths against the project root
# (parent of ui/), so each profile keeps its accounts in a separate SQLite file.
_PROJECT_ROOT = Path(__file__).parent.parent
_DB_PATH = Path(_BRAND.users_db_path)
if not _DB_PATH.is_absolute():
    _DB_PATH = _PROJECT_ROOT / _DB_PATH
_DB_DIR  = _DB_PATH.parent

# Owner role bootstrapped for admins on legacy DB migration (first listed role).
_OWNER_ROLE = _BRAND.roles[0] if _BRAND.roles else DEFAULT_ROLE

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
                is_admin      INTEGER NOT NULL DEFAULT 0,
                role          TEXT NOT NULL DEFAULT 'agent'
            )
        """)
        # Migration : ajoute is_admin si la table existait sans cette colonne
        try:
            _conn.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER NOT NULL DEFAULT 0")
        except sqlite3.OperationalError:
            pass  # Colonne déjà présente
        # Migration : ajoute role + bootstrap (is_admin → owner role de la marque)
        try:
            _conn.execute("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'agent'")
            _conn.execute("UPDATE users SET role = ? WHERE is_admin = 1", [_OWNER_ROLE])
        except sqlite3.OperationalError:
            pass  # Colonne déjà présente
        _conn.commit()
        _ensure_default_admin(_conn)
    return _conn


# ── Bootstrap : admin par défaut à la première création de la BDD ─────────────
#
# Les identifiants ne sont PAS en dur : ils viennent du .env (chargé par
# ui/path_setup.py) / des variables d'environnement Cloud Run.
#   DEFAULT_ADMIN_EMAIL     (défaut "elevate" — non secret)
#   DEFAULT_ADMIN_NAME      (défaut "Elevate" — non secret)
#   DEFAULT_ADMIN_PASSWORD  (AUCUN défaut — secret obligatoire, sinon pas de seed)
#
# Toute BDD de marque fraîchement créée reçoit ce compte admin (rôle owner de
# la marque), sinon personne ne peut se connecter. Si le mot de passe n'est pas
# défini dans l'environnement, le seed est ignoré (aucun secret codé en dur).


def _ensure_default_admin(conn: sqlite3.Connection) -> None:
    """Create the bootstrap admin on an empty user store (any brand).

    Credentials come from the environment (.env). No password → no seed.
    """
    count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    if count:
        return

    password = os.getenv("DEFAULT_ADMIN_PASSWORD")
    if not password:
        print(
            "[auth] DEFAULT_ADMIN_PASSWORD non défini — pas de seed admin. "
            "Renseigne-le dans rag_agent/.env (ou via env Cloud Run) puis "
            "recrée la base, ou lance scripts/seed_admin.py."
        )
        return

    email = os.getenv("DEFAULT_ADMIN_EMAIL", "elevate").lower().strip()
    name = os.getenv("DEFAULT_ADMIN_NAME", "Elevate").strip()
    conn.execute(
        "INSERT INTO users (id, email, password_hash, display_name, created_at, is_admin, role) "
        "VALUES (?, ?, ?, ?, ?, 1, ?)",
        [
            str(uuid.uuid4()),
            email,
            _hash_password(password),
            name,
            datetime.now().isoformat(),
            _OWNER_ROLE,
        ],
    )
    conn.commit()
    print(f"[auth] Admin par défaut créé ({email}) pour la marque active.")


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
    role: str | None = None,
) -> dict | None:
    """
    Crée un utilisateur (réservé aux admins ou au script d'init).
    Retourne le dict user ou None si l'email est déjà pris.
    Le rôle par défaut est celui de la marque active (``DEFAULT_ROLE``).
    """
    role    = role or DEFAULT_ROLE
    conn    = _get_conn()
    user_id = str(uuid.uuid4())
    now     = datetime.now().isoformat()
    try:
        conn.execute(
            "INSERT INTO users (id, email, password_hash, display_name, created_at, is_admin, role) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [user_id, email.lower().strip(), _hash_password(password),
             display_name.strip(), now, int(is_admin), role],
        )
        conn.commit()
        return {
            "id":           user_id,
            "email":        email.lower().strip(),
            "display_name": display_name.strip(),
            "is_admin":     is_admin,
            "role":         role,
        }
    except sqlite3.IntegrityError:
        return None  # Email déjà pris


def authenticate_user(email: str, password: str) -> dict | None:
    """
    Vérifie les identifiants.
    Retourne le dict user (avec is_admin et role) ou None si invalide / inactif.
    """
    conn = _get_conn()
    row  = conn.execute(
        "SELECT id, email, password_hash, display_name, is_active, is_admin, role "
        "FROM users WHERE email = ?",
        [email.lower().strip()],
    ).fetchone()
    if not row:
        return None
    user_id, user_email, pw_hash, display_name, is_active, is_admin, role = row
    if not is_active:
        return None
    if not _verify_password(password, pw_hash):
        return None
    return {
        "id":           user_id,
        "email":        user_email,
        "display_name": display_name,
        "is_admin":     bool(is_admin),
        "role":         role or "agent",
    }


def list_users() -> list[dict]:
    """Retourne tous les utilisateurs (usage admin)."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, email, display_name, created_at, is_active, is_admin, role "
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
            "role":         r[6] or "agent",
        }
        for r in rows
    ]


def set_user_role(user_id: str, role: str) -> None:
    """Met à jour le rôle métier d'un utilisateur."""
    conn = _get_conn()
    conn.execute(
        "UPDATE users SET role = ? WHERE id = ?",
        [role, user_id],
    )
    conn.commit()


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
    st.session_state.role         = user.get("role") or DEFAULT_ROLE


def logout_user() -> None:
    """Efface l'état auth + chat et redirige vers la page de login."""
    keys = [
        "user_id", "user_email", "display_name", "is_admin", "role",
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
        "role":         st.session_state.get("role", DEFAULT_ROLE),
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
