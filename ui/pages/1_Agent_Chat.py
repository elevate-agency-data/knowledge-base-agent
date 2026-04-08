"""
Page 1 — Agent Chat

Sessions ADK persistées via DatabaseSessionService (SQLite).
Les sessions survivent aux redémarrages Streamlit et sont partagées
avec `adk web` (même base de données).

Affichage enrichi stocké dans des fichiers sidecar JSON séparés
(ui/data/agent_display/{session_id}.json) — séparé des sessions
de RAG Comparison (ui/data/comparison_sessions/).
"""

import path_setup  # noqa: F401
import streamlit as st
from config import APP_TITLE, ADK_USER_ID

st.set_page_config(
    page_title=f"Agent Chat — {APP_TITLE}",
    page_icon=None,
    layout="wide",
)

# ── Cached runner ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading AI agent...")
def get_runner():
    from services.agent_runner import AgentRunner
    return AgentRunner()


# ── Session state (namespace : "agent_*") ─────────────────────────────────────

def _init():
    if "agent_active_session_id" not in st.session_state:
        runner = get_runner()
        try:
            sessions = runner.list_sessions(ADK_USER_ID)
        except Exception:
            sessions = []
        if sessions:
            st.session_state.agent_active_session_id = sessions[0]["id"]
        else:
            st.session_state.agent_active_session_id = runner.create_session(ADK_USER_ID)
    if "agent_sessions_cache" not in st.session_state:
        _refresh_sessions()


def _refresh_sessions():
    runner = get_runner()
    try:
        st.session_state.agent_sessions_cache = runner.list_sessions(ADK_USER_ID)
    except Exception:
        st.session_state.agent_sessions_cache = []


def _new_session():
    runner = get_runner()
    sid    = runner.create_session(ADK_USER_ID)
    st.session_state.agent_active_session_id = sid
    _refresh_sessions()


def _active_sid() -> str:
    return st.session_state.agent_active_session_id


def _active_display() -> list[dict]:
    """Load display messages for the active session from sidecar file."""
    from services.agent_runner import load_display
    return load_display(_active_sid())


_init()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Agent Chat")
    st.caption("Powered by **Gemini 2.5 Pro**")
    st.divider()

    if st.button("New conversation", use_container_width=True, type="primary"):
        _new_session()
        st.rerun()

    st.divider()
    st.markdown("**Sessions**")

    sessions = st.session_state.get("agent_sessions_cache", [])
    active_sid = _active_sid()

    if not sessions:
        st.caption("_No sessions found_")

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
    st.markdown("**Quick actions**")
    st.code("list all indexes")
    st.code("create an index [name]")
    st.code("add folder [X] to index [Y]")
    st.code("Find documents similar to: https://drive.google.com/file/d/1q1d5AHjEgKfhOz3mkusT7azTLENStcn5/view?usp=sharing")
    st.code("compare [Index A] and [Index B]")

    st.divider()
    if st.button("Refresh list", use_container_width=True):
        _refresh_sessions()
        st.rerun()

    st.caption(f"ADK session: `{active_sid[:16]}…`")


# ── Page title ────────────────────────────────────────────────────────────────

# Name = first user message or default
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
    from services.agent_runner import append_display

    sid = _active_sid()
    render_user_message(user_input)
    append_display(sid, {"role": "user", "text": user_input})

    with st.spinner("Thinking..."):
        try:
            runner  = get_runner()
            parsed  = runner.run(
                user_id=ADK_USER_ID,
                session_id=sid,
                message=user_input,
            )
            final_text  = runner.extract_final_text(parsed)
            tool_events = [e for e in parsed if e["type"] in ("tool_call", "tool_resp")]

            render_assistant_message(text=final_text, tool_events=tool_events)
            append_display(sid, {
                "role":        "assistant",
                "text":        final_text,
                "tool_events": tool_events,
            })

        except Exception as exc:
            error_msg = str(exc)
            render_error_message(error_msg)
            append_display(sid, {"role": "error", "text": error_msg})

    # Refresh session list so name updates after first message
    _refresh_sessions()
    st.rerun()
