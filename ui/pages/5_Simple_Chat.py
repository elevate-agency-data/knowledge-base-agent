"""
Page 5 — Simple Chat (hybrid-only)

Direct chat without the agent layer. RAG toggle:
  - off → Gemini answers from its own knowledge
  - on  → Hybrid RAG retrieves chunks from local indexes and Gemini synthesises

Sessions are persisted per user (RAG on/off remembered per session).
Access reserved to authenticated users.
"""

import path_setup  # noqa: F401
import streamlit as st
from config import APP_TITLE
from auth import require_auth
from components.sidebar_auth import render_sidebar_nav, render_sidebar_user_info

st.set_page_config(
    page_title=f"Chat — {APP_TITLE}",
    page_icon=None,
    layout="wide",
)

# ── Auth guard ────────────────────────────────────────────────────────────────

user      = require_auth()
USER_ID   = user["id"]
USER_ROLE = user.get("role", "agent")

# ── Constants ─────────────────────────────────────────────────────────────────

_PIPELINE_HYBRID = "Hybrid RAG"


# ── Session helpers ───────────────────────────────────────────────────────────

def _load_session_into_state(sess: dict) -> None:
    """Load a DB session into st.session_state."""
    from services.chat_store import sc_load_messages
    st.session_state.sc_active_session_id = sess["id"]
    st.session_state.sc_rag_on            = bool(sess["rag_on"])
    st.session_state.sc_pipeline          = _PIPELINE_HYBRID
    st.session_state.sc_corpus            = None
    st.session_state.sc_messages          = sc_load_messages(sess["id"])


def _refresh_sessions() -> None:
    from services.chat_store import sc_list_sessions
    st.session_state.sc_sessions_cache = sc_list_sessions(USER_ID)


def _create_and_switch(rag_on: bool = False) -> None:
    """Create a new session with the given RAG flag and switch to it."""
    from services.chat_store import sc_new_session
    sess = sc_new_session(USER_ID, rag_on=rag_on, pipeline=_PIPELINE_HYBRID, corpus=None)
    _load_session_into_state({**sess, "messages": []})
    _refresh_sessions()


def _active_sid() -> str:
    return st.session_state.sc_active_session_id


def _build_context(n_pairs: int = 4) -> str:
    msgs  = st.session_state.sc_messages
    lines = []
    for m in msgs[-(n_pairs * 2):]:
        role = "Utilisateur" if m["role"] == "user" else "Assistant"
        lines.append(f"{role} : {m['text']}")
    return "\n".join(lines)


# ── Init ──────────────────────────────────────────────────────────────────────

def _init() -> None:
    if "sc_active_session_id" not in st.session_state:
        from services.chat_store import sc_list_sessions
        sessions = sc_list_sessions(USER_ID)
        if sessions:
            _load_session_into_state(sessions[0])
        else:
            _create_and_switch()
        _refresh_sessions()


_init()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    render_sidebar_nav()
    st.divider()

    if st.button("Nouvel entretien", use_container_width=True, type="primary"):
        _create_and_switch(rag_on=st.session_state.sc_rag_on)
        st.rerun()

    st.divider()

    # ── Retrieval on/off ──────────────────────────────────────────────────────
    st.markdown(
        '<div class="lux-eyebrow" style="margin-bottom:8px;">Recherche</div>',
        unsafe_allow_html=True,
    )
    rag_on = st.toggle("Consulter la base documentaire",
                       value=st.session_state.sc_rag_on)
    if rag_on != st.session_state.sc_rag_on:
        _create_and_switch(rag_on=rag_on)
        st.rerun()

    st.caption(
        "Les réponses citent les documents consultés."
        if rag_on else
        "Le modèle répond seul, sans consulter la base."
    )

    # ── Indexes reachable with this role (informational) ──────────────────────
    if rag_on:
        try:
            from services.hybrid_service import list_indexes_for_user
            _idx_names = [i["index_name"] for i in list_indexes_for_user(USER_ROLE)]
        except Exception:
            _idx_names = []

        st.divider()
        st.markdown(
            '<div class="lux-eyebrow" style="margin-bottom:8px;">'
            'Index accessibles</div>',
            unsafe_allow_html=True,
        )
        if _idx_names:
            st.markdown(
                "".join(
                    f'<div class="lux-row" style="padding:6px 0;">'
                    f'<span class="lux-row-name" style="font-size:.82rem;">'
                    f'{n}</span></div>'
                    for n in _idx_names
                ),
                unsafe_allow_html=True,
            )
            st.caption(
                f"Choisis d'après votre question, parmi ceux ouverts au rôle "
                f"« {USER_ROLE.replace('_', ' ')} »."
            )
        else:
            st.warning("Aucun index ouvert à votre rôle.")

    # ── Conversations ─────────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        '<div class="lux-eyebrow" style="margin-bottom:8px;">Entretiens</div>',
        unsafe_allow_html=True,
    )
    sessions_cache = st.session_state.get("sc_sessions_cache", [])
    active_sid     = _active_sid()

    if not sessions_cache:
        st.caption("_Aucun entretien_")

    for sess in sessions_cache:
        sid       = sess["id"]
        is_active = sid == active_sid
        n_msg     = sess["msg_count"]

        # Same marking as Agent Chat: a stitch in the margin, not a filled slab.
        mark_c, col_btn, col_del = st.columns([1, 10, 2], gap="small")
        with mark_c:
            if is_active:
                st.markdown(
                    '<div style="height:34px;width:5px;margin-top:4px;'
                    'background-image:repeating-linear-gradient(148deg,'
                    'var(--lux-accent) 0 2px,transparent 2px 5px);'
                    'background-size:5px 100%;background-repeat:repeat-y;"></div>',
                    unsafe_allow_html=True,
                )
        with col_btn:
            if st.button(
                sess["name"] if n_msg else "Nouvel entretien",
                key=f"sc_sw_{sid}",
                use_container_width=True,
                help=f"{n_msg} message{'s' if n_msg != 1 else ''}",
            ):
                _load_session_into_state(sess)
                st.rerun()
        with col_del:
            if st.button("✕", key=f"sc_del_{sid}", help="Supprimer"):
                from services.chat_store import sc_delete_session
                sc_delete_session(sid)
                if sid == active_sid:
                    _create_and_switch()
                else:
                    _refresh_sessions()
                st.rerun()

    render_sidebar_user_info()


# ── Page header ───────────────────────────────────────────────────────────────

rag_on = st.session_state.sc_rag_on

from components.lux_style import poincon

_sess = next((s for s in st.session_state.get("sc_sessions_cache", [])
              if s["id"] == _active_sid()), None)
_n_msg = _sess["msg_count"] if _sess else 0
_title = (_sess["name"] if _sess and _n_msg else "Nouvel entretien")

_marks = [poincon(f"{_n_msg} message{'s' if _n_msg != 1 else ''}"),
          poincon(USER_ROLE.replace("_", " "))]
_marks.append(poincon("base consultée", "accent") if rag_on
              else poincon("sans la base"))

st.markdown(
    f"""
    <div class="lux-eyebrow">Entretien</div>
    <div class="lux-stitch"></div>
    <div class="lux-h2" style="margin-bottom:12px;">{_title}</div>
    <div style="display:flex;gap:6px;flex-wrap:wrap;">{''.join(_marks)}</div>
    """,
    unsafe_allow_html=True,
)

_intro_slot = st.empty()
st.markdown('<div class="lux-space-sm"></div>', unsafe_allow_html=True)

# ── Chat history ──────────────────────────────────────────────────────────────

from components.chat_message import render_user_message, render_error_message
from components.detail_panel import render_detail_panel
from components.answer_renderer import render_answer

for msg in st.session_state.sc_messages:
    if msg["role"] == "user":
        render_user_message(msg["text"])

    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            sources  = msg.get("sources", [])
            chunks   = msg.get("chunks", [])
            elapsed  = msg.get("elapsed_s")
            render_answer(msg["text"], sources)
            # Same single fold-out as Agent Chat — no tool events here, so it
            # holds the documents consulted and the passages retained.
            render_detail_panel(None, sources, chunks)
            if elapsed is not None:
                st.markdown(
                    f'<div style="margin-top:6px;">{poincon(f"{elapsed}s")}</div>',
                    unsafe_allow_html=True,
                )

    elif msg["role"] == "error":
        render_error_message(msg["text"])


# ── Empty conversation: an invitation ─────────────────────────────────────────

if not st.session_state.sc_messages:
    _intro_slot.markdown(
        f"""
        <p class="lux-lead" style="font-size:.92rem;margin:14px 0 6px 0;">
          {"Posez votre question : la base est consultée et la réponse cite ses "
           "documents." if rag_on else
           "Le modèle répond de mémoire, sans consulter la base. Activez la "
           "recherche à gauche pour des réponses sourcées."}
        </p>
        """,
        unsafe_allow_html=True,
    )

# ── Input & routing ───────────────────────────────────────────────────────────

user_input = st.chat_input("Posez votre question…")

if user_input:
    from services.chat_store import sc_append_message
    from rag_agent.runtime_context import set_user_role

    _intro_slot.empty()   # the guidance has served its purpose

    # The dense pipeline reads runtime_context.user_role on the calling
    # thread to filter the candidate index list. Without this, the default
    # ("agent") would apply and block everything for users with broader roles.
    set_user_role(USER_ROLE)

    sid     = _active_sid()
    context = _build_context()

    st.session_state.sc_messages.append({"role": "user", "text": user_input})
    sc_append_message(sid, USER_ID, {"role": "user", "text": user_input})
    render_user_message(user_input)

    with st.spinner("Searching..."):
        try:

            # ── No RAG — direct Gemini ────────────────────────────────────────
            if not rag_on:
                from services.vertex_service import direct_query
                result   = direct_query(user_input, context=context)

                if result.get("status") == "error":
                    raise RuntimeError(result.get("message", "Error"))

                answer  = result.get("answer", "")
                elapsed = result.get("elapsed_s")

                with st.chat_message("assistant"):
                    st.markdown(answer)
                    if elapsed is not None:
                        st.caption(f"{elapsed}s")

                msg = {"role": "assistant", "text": answer,
                       "sources": [], "elapsed_s": elapsed}
                st.session_state.sc_messages.append(msg)
                sc_append_message(sid, USER_ID, msg)

            # ── Hybrid RAG ────────────────────────────────────────────────────
            else:
                import re
                import time
                from services.hybrid_service import (
                    list_indexes_for_user,
                    resolve_indexes,
                    multi_query as hybrid_multi_query,
                )

                all_indexes = [i["index_name"] for i in list_indexes_for_user(USER_ROLE)]
                if not all_indexes:
                    raise RuntimeError(
                        f"No Hybrid indexes accessible for role '{USER_ROLE}'."
                    )

                _drive_url = re.search(
                    r"https?://(?:drive|docs)\.google\.com/\S+",
                    user_input, re.IGNORECASE,
                )

                if _drive_url:
                    from hybrid.tools.hybrid_find_similar import hybrid_find_similar
                    t0     = time.perf_counter()
                    result = hybrid_find_similar(
                        document_url=_drive_url.group(0),
                        index_names=all_indexes,
                    )
                    elapsed = round(time.perf_counter() - t0, 2)
                    if result.get("status") == "error":
                        raise RuntimeError(result.get("message", "Error find_similar"))

                    similar_docs = result.get("results", [])
                    sources      = [
                        {"file_name": r["file_name"], "source_url": r["source_url"]}
                        for r in similar_docs
                    ]
                    answer = f"Here are the **{len(similar_docs)} most similar documents**:"

                    with st.chat_message("assistant"):
                        st.markdown(answer)
                        if similar_docs:
                            lines = [
                                f"- [{r.get('file_name', '')}]({r.get('source_url', '')}) "
                                f"· `{r.get('index_name', '')}` "
                                f"· {round(r.get('score', 0) * 100, 1)}%"
                                for r in similar_docs
                            ]
                            st.markdown("\n".join(lines))
                        st.caption(f"{elapsed}s · {len(all_indexes)} index")

                    msg = {
                        "role":      "assistant",
                        "text":      answer + "\n" + "\n".join(
                            f"- {r['file_name']}" for r in similar_docs
                        ),
                        "sources":   sources,
                        "pipeline":  "hybrid",
                        "elapsed_s": elapsed,
                    }
                    st.session_state.sc_messages.append(msg)
                    sc_append_message(sid, USER_ID, msg)

                else:
                    from services.vertex_service import synthesize_from_context
                    from shared.query_rewriter import rewrite_query
                    import time as _t
                    from concurrent.futures import ThreadPoolExecutor as _TPE

                    t_total = _t.perf_counter()
                    timings: dict[str, float] = {}

                    t0 = _t.perf_counter()
                    with _TPE(max_workers=2) as pool:
                        f_resolve = pool.submit(resolve_indexes, user_input, all_indexes)
                        f_rewrite = pool.submit(rewrite_query, user_input, context)
                        target_indexes = f_resolve.result()
                        rewritten      = f_rewrite.result()
                    timings["resolve+rewrite"] = round(_t.perf_counter() - t0, 2)

                    t0 = _t.perf_counter()
                    retrieval = hybrid_multi_query(
                        index_names=target_indexes,
                        query_text=rewritten,
                        context=context,
                        retrieval_query=rewritten,
                    )
                    timings["retrieval"] = round(_t.perf_counter() - t0, 2)

                    if retrieval.get("status") == "error":
                        raise RuntimeError(retrieval.get("message", "Hybrid error"))

                    timings.update(retrieval.get("timings", {}))

                    t0 = _t.perf_counter()
                    synth = synthesize_from_context(
                        query_text=user_input,
                        rag_context=retrieval.get("answer", ""),
                        conversation_context=context,
                    )
                    timings["generation"] = round(_t.perf_counter() - t0, 2)

                    if synth.get("status") == "error":
                        raise RuntimeError(synth.get("message", "Synthesis error"))

                    answer  = synth.get("answer", "")
                    sources = retrieval.get("sources", [])
                    chunks  = retrieval.get("chunks") or [
                        c for cs in retrieval.get("results_by_index", {}).values()
                        for c in cs
                    ]
                    elapsed = round(_t.perf_counter() - t_total, 2)
                    timings["total"] = elapsed

                    index_label = (
                        target_indexes[0]
                        if len(target_indexes) == 1
                        else f"{len(target_indexes)} indexes"
                    )

                    with st.chat_message("assistant"):
                        render_answer(answer, sources)
                        # Same single fold-out as the history replay above.
                        render_detail_panel(None, sources, chunks)
                        _m = [poincon(f"{elapsed}s"), poincon(index_label)]
                        st.markdown(
                            f'<div style="display:flex;gap:5px;flex-wrap:wrap;'
                            f'margin-top:6px;">{"".join(_m)}</div>',
                            unsafe_allow_html=True,
                        )
                        if rewritten and rewritten != user_input:
                            st.caption(f"Requête de recherche : _{rewritten}_")

                    msg = {
                        "role":      "assistant",
                        "text":      answer,
                        "sources":   sources,
                        "chunks":    chunks,
                        "pipeline":  "hybrid",
                        "elapsed_s": elapsed,
                        "timings":   timings,
                    }
                    st.session_state.sc_messages.append(msg)
                    sc_append_message(sid, USER_ID, msg)

        except Exception as exc:
            error_msg = str(exc)
            render_error_message(error_msg)
            err = {"role": "error", "text": error_msg}
            st.session_state.sc_messages.append(err)
            sc_append_message(sid, USER_ID, err)

    _refresh_sessions()
    st.rerun()
