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

user      = require_auth()
USER_ID   = user["id"]
IS_ADMIN  = user["is_admin"]
USER_ROLE = user.get("role", "agent")

# Photo attachment is offered only when the brand defines an SAV taxonomy —
# same condition that registers the image tools on the agent.
from shared.brand import ACTIVE as _BRAND
_VISION_ENABLED = _BRAND.has_vision

# Shared visual language (tokens, saddle stitch, poinçon marks, chrome).
from components.lux_style import inject_lux_style, poincon
inject_lux_style()

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

    # The model is read from config rather than written here, so the label can
    # never drift from what actually answers.
    from rag_agent.config import MODEL as _AGENT_MODEL

    st.markdown(
        f'<div class="lux-eyebrow" style="margin-bottom:6px;">Agent</div>'
        f'<div style="font-size:.74rem;color:var(--lux-meta);">'
        f'{_AGENT_MODEL}</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    if st.button("Nouveau dossier", use_container_width=True, type="primary"):
        _new_session()
        st.rerun()

    st.divider()
    st.markdown(
        '<div class="lux-eyebrow" style="margin-bottom:8px;">Dossiers</div>',
        unsafe_allow_html=True,
    )

    sessions   = st.session_state.get("agent_sessions_cache", [])
    active_sid = _active_sid()

    if not sessions:
        st.caption("_Aucun dossier_")

    for meta in sessions:
        sid       = meta["id"]
        is_active = sid == active_sid
        n_msg     = meta["message_count"]

        # The active dossier is marked with a stitch in the margin, not with a
        # filled accent slab: the orange stays a signal.
        mark_c, btn_c = st.columns([1, 12], gap="small")
        with mark_c:
            if is_active:
                st.markdown(
                    '<div style="height:34px;width:5px;margin-top:4px;'
                    'background-image:repeating-linear-gradient(148deg,'
                    'var(--lux-accent) 0 2px,transparent 2px 5px);'
                    'background-size:5px 100%;background-repeat:repeat-y;"></div>',
                    unsafe_allow_html=True,
                )
        with btn_c:
            # Same rule as the header: an untouched dossier has no subject yet,
            # so it is not labelled with its generated id.
            _label = meta["name"] if n_msg else "Nouveau dossier"
            if st.button(
                _label,
                key=f"agent_sw_{sid}",
                use_container_width=True,
                help=f"{n_msg} échange{'s' if n_msg != 1 else ''}",
            ):
                st.session_state.agent_active_session_id = sid
                st.rerun()

    st.divider()
    if st.button("Actualiser la liste", use_container_width=True):
        _refresh_sessions()
        st.rerun()

    st.divider()
    st.markdown(
        '<div class="lux-eyebrow" style="margin-bottom:8px;">Raccourcis</div>',
        unsafe_allow_html=True,
    )
    # Phrased against this brand's own indexes rather than a leftover
    # `commune__category` placeholder from the city-hall profile.
    _example_index = (_BRAND.demo_indexes[0][0].lower()
                      if _BRAND.demo_indexes else "un domaine")
    st.code("lister les index disponibles", language=None)
    st.code(f"cherche dans {_example_index} : …", language=None)
    if IS_ADMIN:
        st.code("crée l'index [nom]", language=None)
        st.code("ajoute le dossier [X] à l'index [nom]", language=None)
        st.code("supprime l'index [nom]", language=None)

    render_sidebar_user_info()


# ── Page title ────────────────────────────────────────────────────────────────

sessions     = st.session_state.get("agent_sessions_cache", [])
active_meta  = next((s for s in sessions if s["id"] == _active_sid()), None)
_n_msg       = active_meta["message_count"] if active_meta else 0
# An untouched dossier has no subject yet, so showing its generated id would
# name the plumbing rather than the work. It takes its name from the first
# exchange instead.
session_name = (active_meta["name"] if active_meta and _n_msg
                else "Nouveau dossier")

# The header is a dossier heading, not a page title: what this conversation is
# about, then the marks that situate it (turns so far, the advisor's role).
_marks = [poincon(f"{_n_msg} échange{'s' if _n_msg != 1 else ''}")]
_marks.append(poincon(USER_ROLE.replace("_", " ")))
if _VISION_ENABLED:
    _marks.append(poincon("photo lisible", "accent"))

st.markdown(
    f"""
    <div class="lux-eyebrow">Dossier</div>
    <div class="lux-stitch"></div>
    <div class="lux-h2" style="margin-bottom:12px;">{session_name}</div>
    <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:6px;">
      {''.join(_marks)}
    </div>
    """,
    unsafe_allow_html=True,
)

# The onboarding line and the examples below are guidance for an empty dossier.
# They live in a placeholder so they clear the moment a question is sent, rather
# than lingering above the answer being written.
_intro_slot = st.empty()

st.markdown('<div class="lux-space-sm"></div>', unsafe_allow_html=True)

# ── Chat history ──────────────────────────────────────────────────────────────

from components.chat_message import (
    render_user_message,
    render_assistant_message,
    render_error_message,
)

# Tools that return retrieval payloads (sources + chunks) to surface in the UI.
_RETRIEVAL_TOOLS = {
    "hybrid_query", "hybrid_rag_query", "hybrid_multi_query",
    "hybrid_find_similar",
}


def _extract_retrieval_payload(tool_events: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Walk through tool_events, find the most recent successful retrieval
    response, and return (sources, chunks). Returns ([], []) if none found.
    """
    sources: list[dict] = []
    chunks: list[dict] = []
    for ev in reversed(tool_events or []):
        if ev.get("type") != "tool_resp":
            continue
        if ev.get("name") not in _RETRIEVAL_TOOLS:
            continue
        resp = ev.get("response", {}) or {}
        if resp.get("status") != "success":
            continue
        sources = resp.get("sources") or []
        chunks = resp.get("chunks") or []
        if not chunks and "results_by_index" in resp:
            chunks = [
                c for cs in (resp.get("results_by_index") or {}).values()
                for c in cs
            ]
        break
    return sources, chunks


_history = _active_display()

for msg in _history:
    if msg["role"] == "user":
        render_user_message(msg["text"])
    elif msg["role"] == "assistant":
        tool_events = msg.get("tool_events", [])
        sources, chunks = _extract_retrieval_payload(tool_events)
        render_assistant_message(
            text=msg["text"],
            tool_events=tool_events,
            sources=sources,
            chunks=chunks,
        )
    elif msg["role"] == "error":
        render_error_message(msg["text"])

# ── Empty dossier: an invitation, not a void ───────────────────────────────────
# The examples are real questions this corpus can answer, so a first-time user
# learns what the base actually holds instead of guessing.

if not _history:
    _examples = [
        ("Réparation",
         "Quel est le délai de réparation d'un Birkin 35 en cuir Togo ?"),
        ("Entretien",
         "Un carré de soie a pris une tache — que peut-on faire ?"),
        ("Garantie",
         "Le fermoir d'un Kelly est cassé : est-ce couvert par le service à vie ?"),
    ]
    _cards = "".join(
        f'<div style="border-top:1px solid var(--lux-hair);padding:14px 0;">'
        f'<div class="lux-poincon" style="margin-bottom:8px;">{dom}</div>'
        f'<div style="font-size:.95rem;line-height:1.6;color:var(--lux-body);">'
        f'« {q} »</div></div>'
        for dom, q in _examples
    )
    _photo_line = (
        '<div style="margin-top:22px;font-size:.86rem;color:var(--lux-meta);'
        'line-height:1.6;">Une pièce en main ? Joignez sa photo depuis le volet '
        'du composeur : elle sera lue avant la recherche.</div>'
        if _VISION_ENABLED else ""
    )
    _intro_slot.markdown(
        f"""
        <p class="lux-lead" style="font-size:.92rem;margin:14px 0 26px 0;">
          Posez la question comme vous la poseriez à l'atelier. Les index utiles
          sont choisis pour vous, et chaque réponse cite les documents dont elle
          vient.
        </p>
        <div style="max-width:56ch;">
          <div class="lux-eyebrow">Pour commencer</div>
          {_cards}
          {_photo_line}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Input ─────────────────────────────────────────────────────────────────────
# The photo rides in the composer itself, the way any chat handles an
# attachment. Streamlit returns an object carrying .text and .files, so the
# question and its piece arrive together — no staging state to keep in sync.

_submitted = st.chat_input(
    "Posez votre question…",
    accept_file=_VISION_ENABLED,
    file_type=["png", "jpg", "jpeg", "webp"] if _VISION_ENABLED else None,
)

user_input, _attached = "", []
if _submitted:
    if isinstance(_submitted, str):
        user_input = _submitted
    else:
        user_input = (getattr(_submitted, "text", "") or "").strip()
        _attached = list(getattr(_submitted, "files", None) or [])

# A photo on its own is a legitimate request — the advisor drops the piece and
# expects it read, without having to phrase a question.
if _attached and not user_input:
    user_input = "Analyse cette pièce."

if user_input:
    from services.chat_store import agent_append_message

    sid = _active_sid()
    _intro_slot.empty()   # the guidance has served its purpose

    # Store the attachment once and pass its ref: a tool call carries text only.
    photo_ref = ""
    if _attached and _VISION_ENABLED:
        from vision.uploads import save_upload
        _saved = save_upload(_attached[0].getvalue(), _attached[0].name,
                             user_id=USER_ID)
        if _saved["status"] == "success":
            photo_ref = _saved["ref"]
        else:
            st.error(_saved["message"])

    if photo_ref:
        agent_message = (
            f"{user_input}\n\n[Photo attached — image_ref: {photo_ref}. "
            f"Use sav_analyze_image with this ref.]"
        )
        display_text = f"{user_input}\n\n`pièce jointe · {photo_ref}`"
    else:
        agent_message = user_input
        display_text = user_input

    render_user_message(display_text)
    agent_append_message(USER_ID, sid, {"role": "user", "text": display_text})

    with st.spinner("Lecture de la base…"):
        try:
            runner  = get_runner(IS_ADMIN)
            parsed  = runner.run(
                user_id=USER_ID,
                session_id=sid,
                message=agent_message,
                user_role=USER_ROLE,
            )
            final_text  = runner.extract_final_text(parsed)
            tool_events = [e for e in parsed if e["type"] in ("tool_call", "tool_resp")]
            sources, chunks = _extract_retrieval_payload(tool_events)

            render_assistant_message(
                text=final_text,
                tool_events=tool_events,
                sources=sources,
                chunks=chunks,
            )
            agent_append_message(USER_ID, sid, {
                "role":        "assistant",
                "text":        final_text,
                "tool_events": tool_events,
            })

        except Exception as exc:
            error_msg = str(exc)
            render_error_message(error_msg)
            agent_append_message(USER_ID, sid, {"role": "error", "text": error_msg})

    # Nothing to detach: the composer clears its own attachment on submit, so
    # the photo cannot leak into the next message. The stored file stays
    # resolvable by ref, so the agent can still re-read it later in the dossier.
    _refresh_sessions()
    st.rerun()
