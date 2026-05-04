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

user    = require_auth()
USER_ID = user["id"]

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
    st.header("Simple Chat")
    st.divider()

    # ── New conversation ──────────────────────────────────────────────────────
    if st.button("New conversation", use_container_width=True, type="primary"):
        _create_and_switch(rag_on=st.session_state.sc_rag_on)
        st.rerun()

    # ── RAG toggle ────────────────────────────────────────────────────────────
    rag_on = st.toggle(
        "RAG activation",
        value=st.session_state.sc_rag_on,
    )

    if rag_on != st.session_state.sc_rag_on:
        _create_and_switch(rag_on=rag_on)
        st.rerun()

    # ── Available indexes (info only) ─────────────────────────────────────────
    if rag_on:
        st.divider()
        try:
            from services.hybrid_service import list_indexes
            _idx_names = [i["index_name"] for i in list_indexes()]
        except Exception:
            _idx_names = []

        if _idx_names:
            st.markdown("**Available indexes**")
            for n in _idx_names:
                st.caption(f"· {n}")
            st.caption(
                "_Indexes detected automatically from your question. "
                "All queried by default._"
            )
        else:
            st.warning("No indexes available.")

    # ── Status badge ──────────────────────────────────────────────────────────
    st.divider()
    if not rag_on:
        st.info("Gemini chatbot\nWithout RAG enrichment")
    else:
        st.success("Hybrid RAG\nIndexes auto-detected")

    # ── Sessions list ─────────────────────────────────────────────────────────
    st.divider()
    st.markdown("**Sessions**")
    sessions_cache = st.session_state.get("sc_sessions_cache", [])
    active_sid     = _active_sid()

    if not sessions_cache:
        st.caption("_No sessions_")

    for sess in sessions_cache:
        sid       = sess["id"]
        is_active = sid == active_sid
        n_msg     = sess["msg_count"]
        label     = f"{'▶ ' if is_active else ''}{sess['name']}"

        col_btn, col_del = st.columns([5, 1])
        with col_btn:
            if st.button(
                label,
                key=f"sc_sw_{sid}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
                help=f"{n_msg} message{'s' if n_msg != 1 else ''}",
            ):
                _load_session_into_state(sess)
                st.rerun()
        with col_del:
            if st.button("✕", key=f"sc_del_{sid}", help="Delete"):
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

st.title("Chat")

if not rag_on:
    st.caption("Model response without document retrieval.")
else:
    st.caption(
        "Documents retrieved from Hybrid indexes — "
        "indexes automatically selected based on your question."
    )

# ── Chat history ──────────────────────────────────────────────────────────────

from components.chat_message import render_user_message, render_error_message
from components.source_card  import render_sources, render_chunks
from components.answer_renderer import render_answer

for msg in st.session_state.sc_messages:
    if msg["role"] == "user":
        render_user_message(msg["text"])

    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            sources  = msg.get("sources", [])
            chunks   = msg.get("chunks", [])
            elapsed  = msg.get("elapsed_s")
            timings  = msg.get("timings", {})
            render_answer(msg["text"], sources)
            if sources:
                with st.expander(f"Sources ({len(sources)})", expanded=False):
                    render_sources(sources, pipeline="hybrid")
            if chunks:
                with st.expander(f"Retrieved chunks ({len(chunks)})", expanded=False):
                    render_chunks(chunks)
            timing_parts: list[str] = []
            if timings:
                for k in ("resolve_and_rewrite", "rewrite_query", "embedding",
                          "threads", "search", "generation"):
                    if k in timings:
                        label = k.replace("_", " ").replace("and", "+")
                        timing_parts.append(f"{label} {timings[k]}s")
            if elapsed is not None:
                caption = f"{elapsed}s"
                if timing_parts:
                    caption += f" · {' · '.join(timing_parts)}"
                st.caption(caption)

    elif msg["role"] == "error":
        render_error_message(msg["text"])


# ── Input & routing ───────────────────────────────────────────────────────────

_placeholder = (
    "Ask your question..."
    if not rag_on
    else "Ask a question about your documents..."
)

user_input = st.chat_input(_placeholder)

if user_input:
    from services.chat_store import sc_append_message

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
                    list_indexes,
                    resolve_indexes,
                    multi_query as hybrid_multi_query,
                )

                all_indexes = [i["index_name"] for i in list_indexes()]
                if not all_indexes:
                    raise RuntimeError("No Hybrid indexes available.")

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
                        if sources:
                            with st.expander(f"Sources ({len(sources)})", expanded=False):
                                render_sources(sources, pipeline="hybrid")
                        if chunks:
                            with st.expander(f"Retrieved chunks ({len(chunks)})", expanded=False):
                                render_chunks(chunks)
                        timing_parts = []
                        for k in ("resolve+rewrite", "embedding", "threads", "generation"):
                            if k in timings:
                                timing_parts.append(f"{k} {timings[k]}s")
                        caption = f"{elapsed}s · {index_label}"
                        if timing_parts:
                            caption += f" · {' · '.join(timing_parts)}"
                        if rewritten and rewritten != user_input:
                            caption += f"\nretrieval query: _{rewritten}_"
                        st.caption(caption)

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
