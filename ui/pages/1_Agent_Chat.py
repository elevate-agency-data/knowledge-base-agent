"""
Page 1 — Agent Chat

Sessions ADK persistées via DatabaseSessionService (SQLite).
Messages d'affichage persistés dans chat.duckdb (table agent_messages).
Accès réservé aux utilisateurs authentifiés.
"""

import path_setup  # noqa: F401
import streamlit as st
from config import APP_TITLE
from auth import require_auth
from components.sidebar_auth import render_sidebar_nav, render_sidebar_user_info

st.set_page_config(
    page_title=f"Agent Chat — {APP_TITLE}",
    page_icon=None,
    layout="wide",
)

# ── Auth guard ────────────────────────────────────────────────────────────────

user     = require_auth()
USER_ID  = user["id"]
IS_ADMIN = user["is_admin"]

# ── Cached runner (one per role — admin gets write tools, users get read-only) ─

@st.cache_resource(show_spinner="Loading AI agent...")
def get_runner(is_admin: bool):
    from services.agent_runner import AgentRunner
    return AgentRunner(is_admin=is_admin)


# ── Session state (namespace : "agent_*") ─────────────────────────────────────

def _init():
    if "agent_active_session_id" not in st.session_state:
        runner = get_runner(IS_ADMIN)
        try:
            sessions = runner.list_sessions(USER_ID)
        except Exception:
            sessions = []
        if sessions:
            st.session_state.agent_active_session_id = sessions[0]["id"]
        else:
            st.session_state.agent_active_session_id = runner.create_session(USER_ID)
    if "agent_sessions_cache" not in st.session_state:
        _refresh_sessions()


def _refresh_sessions():
    runner = get_runner(IS_ADMIN)
    try:
        st.session_state.agent_sessions_cache = runner.list_sessions(USER_ID)
    except Exception:
        st.session_state.agent_sessions_cache = []


def _new_session():
    runner = get_runner(IS_ADMIN)
    sid    = runner.create_session(USER_ID)
    st.session_state.agent_active_session_id = sid
    _refresh_sessions()


def _active_sid() -> str:
    return st.session_state.agent_active_session_id


def _active_display() -> list[dict]:
    from services.chat_store import agent_load_messages
    return agent_load_messages(USER_ID, _active_sid())


_init()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    render_sidebar_nav()
    st.divider()
    st.header("Agent Chat")
    st.caption("Powered by **Gemini 2.5 Pro**")
    st.divider()

    if st.button("New conversation", use_container_width=True, type="primary"):
        _new_session()
        st.rerun()

    st.divider()
    st.markdown("**Sessions**")

    sessions   = st.session_state.get("agent_sessions_cache", [])
    active_sid = _active_sid()

    if not sessions:
        st.caption("_Aucune session_")

    for meta in sessions:
        sid       = meta["id"]
        is_active = sid == active_sid
        n_msg     = meta["message_count"]
        label     = f"{'▶ ' if is_active else ''}{meta['name']}"

        btn_c, info_c = st.columns([5, 1])
        with btn_c:
            if st.button(
                label,
                key=f"agent_sw_{sid}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
                help=f"{n_msg} message{'s' if n_msg != 1 else ''}",
            ):
                st.session_state.agent_active_session_id = sid
                st.rerun()
        with info_c:
            st.caption(str(n_msg))

    st.divider()
    if st.button("Refresh list", use_container_width=True):
        _refresh_sessions()
        st.rerun()

    st.caption(f"ADK session: `{active_sid[:16]}…`")
    st.divider()
    st.markdown("**Quick actions**")
    st.code("list all indexes")
    st.code("search [topic] about [question]")
    if IS_ADMIN:
        st.code("create an index [name]")
        st.code("add folder [X] to index [Y]")
        st.code("delete index [name]")

    render_sidebar_user_info()


# ── Page title ────────────────────────────────────────────────────────────────

sessions     = st.session_state.get("agent_sessions_cache", [])
active_meta  = next((s for s in sessions if s["id"] == _active_sid()), None)
session_name = active_meta["name"] if active_meta else "New conversation"

st.title(session_name)
st.caption("The AI agent automatically selects the best retrieval pipeline for your question.")

# ── Chat history ──────────────────────────────────────────────────────────────

from components.chat_message import (
    render_user_message,
    render_assistant_message,
    render_error_message,
)

for msg in _active_display():
    if msg["role"] == "user":
        render_user_message(msg["text"])
    elif msg["role"] == "assistant":
        render_assistant_message(
            text=msg["text"],
            tool_events=msg.get("tool_events", []),
        )
    elif msg["role"] == "error":
        render_error_message(msg["text"])

# ── Input ─────────────────────────────────────────────────────────────────────

user_input = st.chat_input("Ask a question or give an instruction...")

if user_input:
    from services.chat_store import agent_append_message

    sid = _active_sid()
    render_user_message(user_input)
    agent_append_message(USER_ID, sid, {"role": "user", "text": user_input})

    with st.spinner("Thinking..."):
        try:
            runner  = get_runner(IS_ADMIN)
            parsed  = runner.run(
                user_id=USER_ID,
                session_id=sid,
                message=user_input,
            )
            final_text  = runner.extract_final_text(parsed)
            tool_events = [e for e in parsed if e["type"] in ("tool_call", "tool_resp")]

            render_assistant_message(text=final_text, tool_events=tool_events)
            agent_append_message(USER_ID, sid, {
                "role":        "assistant",
                "text":        final_text,
                "tool_events": tool_events,
            })

        except Exception as exc:
            error_msg = str(exc)
            render_error_message(error_msg)
            agent_append_message(USER_ID, sid, {"role": "error", "text": error_msg})

    _refresh_sessions()
    st.rerun()
