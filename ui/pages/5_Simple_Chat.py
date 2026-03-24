"""
Page 5 — Simple Chat

Interface simplifiée pour utilisateurs non-techniques.
Un toggle active ou désactive le RAG. Quand le RAG est actif, le pipeline
(Vertex AI ou Hybrid) est sélectionné en sidebar.

Hybrid RAG : les index pertinents sont détectés automatiquement via Gemini Flash
             (fallback sur tous les index si aucun n'est identifié).
"""

import path_setup  # noqa: F401
import streamlit as st
from config import APP_TITLE

st.set_page_config(
    page_title=f"Chat — {APP_TITLE}",
    page_icon=None,
    layout="wide",
)

# ── Constantes ────────────────────────────────────────────────────────────────

_PIPELINE_VERTEX = "Vertex AI"
_PIPELINE_HYBRID = "Hybrid RAG"
_PIPELINES       = [_PIPELINE_VERTEX, _PIPELINE_HYBRID]

# ── Session state ─────────────────────────────────────────────────────────────

def _init() -> None:
    defaults = {
        "sc_messages":  [],
        "sc_rag_on":    False,
        "sc_pipeline":  _PIPELINE_HYBRID,
        "sc_corpus":    None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _clear() -> None:
    st.session_state.sc_messages = []


def _build_context(n_pairs: int = 4) -> str:
    msgs  = st.session_state.sc_messages
    lines = []
    for m in msgs[-(n_pairs * 2):]:
        role = "Utilisateur" if m["role"] == "user" else "Assistant"
        lines.append(f"{role} : {m['text']}")
    return "\n".join(lines)


_init()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Simple Chat")
    st.divider()

    # ── Toggle RAG ────────────────────────────────────────────────────────────
    rag_on = st.toggle(
        "RAG activé",
        value=st.session_state.sc_rag_on,
    )

    if rag_on != st.session_state.sc_rag_on:
        st.session_state.sc_rag_on = rag_on
        st.session_state.sc_corpus = None
        _clear()
        st.rerun()

    # ── Options RAG ───────────────────────────────────────────────────────────
    if rag_on:
        st.divider()
        st.markdown("**Pipeline**")
        pipeline = st.radio(
            label="Pipeline",
            options=_PIPELINES,
            index=_PIPELINES.index(st.session_state.sc_pipeline),
            label_visibility="collapsed",
        )

        if pipeline != st.session_state.sc_pipeline:
            st.session_state.sc_pipeline = pipeline
            st.session_state.sc_corpus   = None
            _clear()
            st.rerun()

        st.divider()

        if pipeline == _PIPELINE_VERTEX:
            st.markdown("**Corpus Vertex AI**")
            try:
                from services.vertex_service import list_corpora
                corpus_names = [c["display_name"] for c in list_corpora()]
            except Exception:
                corpus_names = []

            if corpus_names:
                current = st.session_state.sc_corpus
                default = corpus_names.index(current) if current in corpus_names else 0
                selected = st.selectbox("Corpus", corpus_names, index=default,
                                        label_visibility="collapsed")
                if selected != st.session_state.sc_corpus:
                    st.session_state.sc_corpus = selected
                    _clear()
                    st.rerun()
            else:
                st.warning("Aucun corpus disponible.")
                st.session_state.sc_corpus = None

        else:  # Hybrid RAG
            try:
                from services.hybrid_service import list_indexes
                _idx_names = [i["index_name"] for i in list_indexes()]
            except Exception:
                _idx_names = []

            if _idx_names:
                st.markdown("**Index disponibles**")
                for n in _idx_names:
                    st.caption(f"· {n}")
                st.caption(
                    "_Index détectés automatiquement dans votre question. "
                    "Tous interrogés par défaut._"
                )
            else:
                st.warning("Aucun index disponible.")

    st.divider()

    if st.button("Nouvelle conversation", use_container_width=True, type="primary"):
        _clear()
        st.rerun()

    # ── Status badge ──────────────────────────────────────────────────────────
    st.divider()
    if not rag_on:
        st.info("Chatbot Gemini\nSans enrichissement RAG")
    elif st.session_state.sc_pipeline == _PIPELINE_VERTEX:
        label = st.session_state.sc_corpus or "—"
        st.success(f"Vertex AI\nCorpus : **{label}**")
    else:
        st.success("Hybrid RAG\nIndex détectés automatiquement")


# ── Page header ───────────────────────────────────────────────────────────────

st.title("Chat")

if not rag_on:
    st.caption("Réponse du modèle sans récupération de documents.")
elif st.session_state.sc_pipeline == _PIPELINE_VERTEX:
    st.caption(
        f"Documents récupérés depuis le corpus Vertex AI "
        f"**{st.session_state.sc_corpus or '—'}**."
    )
else:
    st.caption(
        "Documents récupérés depuis les index Hybrid — "
        "index sélectionnés automatiquement selon votre question."
    )

# ── Chat history ──────────────────────────────────────────────────────────────

from components.chat_message import render_user_message, render_error_message
from components.source_card  import render_sources, render_chunks

for msg in st.session_state.sc_messages:
    if msg["role"] == "user":
        render_user_message(msg["text"])

    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(msg["text"])
            sources  = msg.get("sources", [])
            chunks   = msg.get("chunks", [])
            pipeline = msg.get("pipeline", "hybrid")
            elapsed  = msg.get("elapsed_s")
            if sources:
                with st.expander(f"Sources ({len(sources)})", expanded=False):
                    render_sources(sources, pipeline=pipeline)
            if chunks:
                with st.expander(f"Citations ({len(chunks)} chunks)", expanded=False):
                    render_chunks(chunks)
            if elapsed is not None:
                st.caption(f"{elapsed}s")

    elif msg["role"] == "error":
        render_error_message(msg["text"])


# ── Readiness check ───────────────────────────────────────────────────────────

_ready = True
if rag_on and st.session_state.sc_pipeline == _PIPELINE_VERTEX and not st.session_state.sc_corpus:
    st.warning("Sélectionnez un corpus Vertex AI dans la barre latérale.")
    _ready = False

_placeholder = (
    "Posez votre question…"
    if not rag_on
    else "Posez votre question sur les documents…"
)

# ── Input & routing ───────────────────────────────────────────────────────────

user_input = st.chat_input(_placeholder, disabled=not _ready)

if user_input:
    st.session_state.sc_messages.append({"role": "user", "text": user_input})
    render_user_message(user_input)

    context = _build_context()

    with st.spinner("Recherche en cours…"):
        try:

            # ── Sans RAG — Gemini direct ──────────────────────────────────────
            if not rag_on:
                from services.vertex_service import direct_query
                result   = direct_query(user_input, context=context)
                pipeline = "hybrid"

                if result.get("status") == "error":
                    raise RuntimeError(result.get("message", "Erreur"))

                answer  = result.get("answer", "")
                elapsed = result.get("elapsed_s")

                with st.chat_message("assistant"):
                    st.markdown(answer)
                    if elapsed is not None:
                        st.caption(f"{elapsed}s")

                st.session_state.sc_messages.append({
                    "role":      "assistant",
                    "text":      answer,
                    "sources":   [],
                    "elapsed_s": elapsed,
                })

            # ── Vertex AI RAG ─────────────────────────────────────────────────
            elif st.session_state.sc_pipeline == _PIPELINE_VERTEX:
                import re as _re
                _drive_url = _re.search(
                    r"https?://(?:drive|docs)\.google\.com/\S+", user_input, _re.IGNORECASE
                )

                if _drive_url:
                    # Drive URL → similarité documentaire via Vertex
                    from services.vertex_service import find_similar as vertex_find_similar
                    result = vertex_find_similar(
                        corpus_name=st.session_state.sc_corpus,
                        document_url=_drive_url.group(0),
                    )

                    if result.get("status") == "error":
                        raise RuntimeError(result.get("message", "Erreur Vertex find_similar"))

                    similar = result.get("results", [])
                    elapsed = result.get("elapsed_s")
                    answer  = f"Voici les **{len(similar)} documents** les plus similaires (Vertex AI RAG) :"

                    with st.chat_message("assistant"):
                        st.markdown(answer)
                        if similar:
                            lines = []
                            for r in similar:
                                url  = r.get("uri", "")
                                name = r.get("title", url)
                                lines.append(f"- [{name}]({url})")
                            st.markdown("\n".join(lines))
                        if elapsed is not None:
                            st.caption(f"{elapsed}s · vertex_find_similar · corpus {st.session_state.sc_corpus}")

                    sources = [{"title": r["title"], "uri": r["uri"]} for r in similar]
                    st.session_state.sc_messages.append({
                        "role":      "assistant",
                        "text":      answer,
                        "sources":   sources,
                        "pipeline":  "vertex",
                        "elapsed_s": elapsed,
                    })

                else:
                    from services.vertex_service import query as vertex_query
                    result = vertex_query(
                        corpus_name=st.session_state.sc_corpus,
                        query_text=user_input,
                        context=context,
                    )

                    if result.get("status") == "error":
                        raise RuntimeError(result.get("message", "Erreur Vertex"))

                    answer  = result.get("answer", "")
                    sources = result.get("sources", [])
                    elapsed = result.get("elapsed_s")

                    with st.chat_message("assistant"):
                        st.markdown(answer)
                        if sources:
                            with st.expander(f"Sources ({len(sources)})", expanded=False):
                                render_sources(sources, pipeline="vertex")
                        if elapsed is not None:
                            st.caption(f"{elapsed}s")

                    st.session_state.sc_messages.append({
                        "role":      "assistant",
                        "text":      answer,
                        "sources":   sources,
                        "pipeline":  "vertex",
                        "elapsed_s": elapsed,
                    })

            # ── Hybrid RAG — résolution automatique des index ─────────────────
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
                    raise RuntimeError("Aucun index Hybrid disponible.")

                _drive_url = re.search(
                    r"https?://(?:drive|docs)\.google\.com/\S+", user_input, re.IGNORECASE
                )

                # ── Similarité documentaire (URL Drive détectée) ──────────────
                if _drive_url:
                    from hybrid.tools.hybrid_find_similar import hybrid_find_similar
                    t0     = time.perf_counter()
                    result = hybrid_find_similar(
                        document_url=_drive_url.group(0),
                        index_names=all_indexes,
                    )
                    elapsed = round(time.perf_counter() - t0, 2)

                    if result.get("status") == "error":
                        raise RuntimeError(result.get("message", "Erreur find_similar"))

                    similar_docs = result.get("results", [])
                    sources = [
                        {"file_name": r["file_name"], "source_url": r["source_url"]}
                        for r in similar_docs
                    ]
                    answer = (
                        f"Voici les **{len(similar_docs)} documents** les plus similaires "
                        f"à votre document (recherche par similarité vectorielle) :"
                    )

                    with st.chat_message("assistant"):
                        st.markdown(answer)
                        if similar_docs:
                            lines = []
                            for r in similar_docs:
                                score = round(r.get("score", 0) * 100, 1)
                                url   = r.get("source_url", "")
                                name  = r.get("file_name", url)
                                idx   = r.get("index_name", "")
                                lines.append(f"- [{name}]({url}) · `{idx}` · {score}%")
                            st.markdown("\n".join(lines))
                        st.caption(f"{elapsed}s · {len(all_indexes)} index")

                    st.session_state.sc_messages.append({
                        "role":      "assistant",
                        "text":      answer + "\n" + "\n".join(
                            f"- {r['file_name']}" for r in similar_docs
                        ),
                        "sources":   sources,
                        "pipeline":  "hybrid",
                        "elapsed_s": elapsed,
                    })

                # ── Requête texte classique ───────────────────────────────────
                else:
                    from services.vertex_service import synthesize_from_context

                    with st.spinner("Détection des index pertinents…"):
                        target_indexes = resolve_indexes(user_input, all_indexes)

                    t0        = time.perf_counter()
                    retrieval = hybrid_multi_query(
                        index_names=target_indexes,
                        query_text=user_input,
                        context=context,
                    )

                    if retrieval.get("status") == "error":
                        raise RuntimeError(retrieval.get("message", "Erreur Hybrid"))

                    synth = synthesize_from_context(
                        query_text=user_input,
                        rag_context=retrieval.get("answer", ""),
                        conversation_context=context,
                    )

                    if synth.get("status") == "error":
                        raise RuntimeError(synth.get("message", "Erreur de synthèse"))

                    answer  = synth.get("answer", "")
                    sources = retrieval.get("sources", [])
                    # single-index → flat chunks list; multi-index → flatten results_by_index
                    chunks  = retrieval.get("chunks") or [
                        c for cs in retrieval.get("results_by_index", {}).values()
                        for c in cs
                    ]
                    elapsed = round(time.perf_counter() - t0, 2)
                    index_label = (
                        target_indexes[0]
                        if len(target_indexes) == 1
                        else f"{len(target_indexes)} index"
                    )

                    retrieval_query = retrieval.get("retrieval_query", "")
                    with st.chat_message("assistant"):
                        st.markdown(answer)
                        if sources:
                            with st.expander(f"Sources ({len(sources)})", expanded=False):
                                render_sources(sources, pipeline="hybrid")
                        if chunks:
                            with st.expander(f"Citations ({len(chunks)} chunks)", expanded=False):
                                render_chunks(chunks)
                        caption = f"{elapsed}s · {index_label}"
                        if retrieval_query and retrieval_query != user_input:
                            caption += f" · query : _{retrieval_query}_"
                        st.caption(caption)

                    st.session_state.sc_messages.append({
                        "role":      "assistant",
                        "text":      answer,
                        "sources":   sources,
                        "chunks":    chunks,
                        "pipeline":  "hybrid",
                        "elapsed_s": elapsed,
                    })

        except Exception as exc:
            error_msg = str(exc)
            render_error_message(error_msg)
            st.session_state.sc_messages.append(
                {"role": "error", "text": error_msg}
            )

    st.rerun()
