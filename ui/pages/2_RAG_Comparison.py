"""
Page 2 — RAG Comparison

Naive RAG (gauche) vs Hybrid RAG (droite).

Exécution : les deux pipelines tournent en parallèle (ThreadPoolExecutor).
Affichage : chaque colonne se met à jour dès que son pipeline est terminé
            (as_completed → with placeholder.container()).

Côté Hybrid :
  - Récupération chunks via hybrid_rag_query / hybrid_multi_query
  - Génération de réponse Gemini sur la base des chunks récupérés
  - Résolution automatique des index via Gemini Flash
"""

from __future__ import annotations

import path_setup  # noqa: F401
import concurrent.futures
import re
import streamlit as st

_DRIVE_URL_RE = re.compile(
    r"https?://(?:drive|docs)\.google\.com/\S+", re.IGNORECASE
)

from config import (
    APP_TITLE, VERTEX_COLOR, HYBRID_COLOR,
    VERTEX_LABEL, HYBRID_LABEL,
    DEFAULT_TOP_K, DEFAULT_RETRIEVAL_MODE, RETRIEVAL_MODES,
    GENERATION_MODEL, GENERATION_SYSTEM_PROMPT,
)

st.set_page_config(
    page_title=f"RAG Comparison — {APP_TITLE}",
    page_icon=None,
    layout="wide",
)

from auth import require_admin
from components.sidebar_auth import render_sidebar_nav, render_sidebar_user_info
require_admin()

# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_data(ttl=60, show_spinner=False)
def _load_corpora() -> list[dict]:
    try:
        from services.vertex_service import list_corpora
        return list_corpora()
    except Exception:
        return []


@st.cache_data(ttl=60, show_spinner=False)
def _load_indexes() -> list[str]:
    try:
        from services.hybrid_service import list_indexes
        return [i["index_name"] for i in list_indexes()]
    except Exception:
        return []


# ── Gemini generation (Hybrid side) ──────────────────────────────────────────

def _generate_answer(query: str, context: str) -> str:
    """
    Appelle Gemini pour synthétiser une réponse à partir des chunks.
    Flash est utilisé ici (et non Pro) pour éviter que le double appel Gemini
    (retrieval côté Hybrid + génération) ne soit plus lent que Vertex.
    """
    if not context.strip():
        return ""
    try:
        from vertexai.generative_models import GenerativeModel
        from shared.gemini_retry import generate_with_retry
        prompt = (
            f"{GENERATION_SYSTEM_PROMPT}\n\n"
            f"Documents:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
        return generate_with_retry(GenerativeModel(GENERATION_MODEL), prompt).text
    except Exception as exc:
        return f"_(Generation error: {exc})_"


from components.answer_renderer import render_answer as _render_answer


# ── Index resolver ────────────────────────────────────────────────────────────

def _resolve_indexes(query: str, available: list[str]) -> list[str]:
    from shared.index_resolver import resolve_indexes
    return resolve_indexes(query, available)


# ── Comparison session state (namespace: "cmp_*") ─────────────────────────────

def _init_cmp_state():
    if "cmp_active_session" not in st.session_state:
        from services.session_store import list_sessions, new_session, load_session
        saved = list_sessions()
        st.session_state.cmp_active_session = (
            load_session(saved[0]["id"]) if saved else new_session()
        )
    if "cmp_sessions_list" not in st.session_state:
        _refresh_sessions_list()


def _refresh_sessions_list():
    from services.session_store import list_sessions
    st.session_state.cmp_sessions_list = list_sessions()


def _active_cmp() -> dict:
    return st.session_state.cmp_active_session


def _new_cmp(name: str | None = None):
    from services.session_store import new_session
    st.session_state.cmp_active_session = new_session(name)
    _refresh_sessions_list()


def _load_cmp(session_id: str):
    from services.session_store import load_session
    loaded = load_session(session_id)
    if loaded:
        st.session_state.cmp_active_session = loaded


def _delete_cmp(session_id: str):
    from services.session_store import delete_session, list_sessions, new_session, load_session
    delete_session(session_id)
    remaining = list_sessions()
    st.session_state.cmp_active_session = (
        load_session(remaining[0]["id"]) if remaining else new_session()
    )
    _refresh_sessions_list()


_init_cmp_state()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    render_sidebar_nav()
    st.divider()
    st.header("RAG Comparison")
    st.divider()

    st.markdown("**Sessions**")
    if st.button("New session", use_container_width=True, type="primary"):
        _new_cmp()
        st.rerun()

    active_id     = _active_cmp()["id"]
    sessions_list = st.session_state.get("cmp_sessions_list", [])

    for meta in sessions_list:
        is_active = meta["id"] == active_id
        n         = meta["query_count"]
        label     = f"{'▶ ' if is_active else ''}{meta['name']}"
        btn_c, del_c = st.columns([5, 1])
        with btn_c:
            if st.button(
                label,
                key=f"cmp_load_{meta['id']}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
                help=f"{n} {'queries' if n != 1 else 'query'}",
            ):
                _load_cmp(meta["id"])
                st.rerun()
        with del_c:
            if st.button("X", key=f"cmp_del_{meta['id']}", help="Delete"):
                _delete_cmp(meta["id"])
                st.rerun()

    st.divider()

    st.markdown(
        f"<span style='color:{VERTEX_COLOR}'>**{VERTEX_LABEL}**</span>",
        unsafe_allow_html=True,
    )
    corpora        = _load_corpora()
    corpus_options = [c["display_name"] for c in corpora] if corpora else []
    if corpus_options:
        selected_corpus = st.selectbox("Corpus", corpus_options, key="sel_corpus")
    else:
        selected_corpus = st.text_input(
            "Corpus (name)", key="sel_corpus_text", placeholder="base-rag"
        )

    st.divider()

    st.markdown(
        f"<span style='color:{HYBRID_COLOR}'>**{HYBRID_LABEL}**</span>",
        unsafe_allow_html=True,
    )
    all_indexes = _load_indexes()
    if all_indexes:
        st.caption("Available indexes (auto-selected):")
        for idx in all_indexes:
            st.caption(f"• `{idx}`")
    else:
        st.warning("No hybrid indexes found.")

    st.divider()
    st.markdown("**Hybrid Options**")
    retrieval_mode = st.selectbox(
        "Retrieval mode", RETRIEVAL_MODES,
        index=RETRIEVAL_MODES.index(DEFAULT_RETRIEVAL_MODE),
        key="sel_mode",
    )
    top_k = st.slider(
        "Chunks per index", min_value=3, max_value=20,
        value=DEFAULT_TOP_K, key="sel_topk",
    )

    st.divider()
    if st.button("Refresh lists", use_container_width=True):
        _load_corpora.clear()
        _load_indexes.clear()
        _refresh_sessions_list()
        st.rerun()

    render_sidebar_user_info()


# ── Page header ───────────────────────────────────────────────────────────────

active_cmp = _active_cmp()
st.title(active_cmp['name'])
st.caption(
    "Both pipelines run in parallel. "
    "Each column updates as soon as its result is ready."
)

# ── Column headers ────────────────────────────────────────────────────────────

hdr_v, hdr_h = st.columns(2)
with hdr_v:
    st.markdown(
        f"<h3 style='color:{VERTEX_COLOR}'>{VERTEX_LABEL}</h3>",
        unsafe_allow_html=True,
    )
with hdr_h:
    st.markdown(
        f"<h3 style='color:{HYBRID_COLOR}'>{HYBRID_LABEL}</h3>",
        unsafe_allow_html=True,
    )
st.divider()

# ── Render helpers ────────────────────────────────────────────────────────────

from components.source_card  import render_sources, render_chunks
from components.chat_message import render_error_message as _render_err


def _show_error(message: str) -> None:
    """Show error or rate-limit warning depending on error type."""
    if "429" in message or "RESOURCE_EXHAUSTED" in message:
        st.warning(
            "The service is temporarily overloaded (API quota exceeded). "
            "Please try again in a few seconds."
        )
    else:
        st.error(message)
from services.session_store import append_entry


def _fn_badge_vertex(corpus: str, is_drive_url: bool = False) -> None:
    label = (
        f'vertex_find_similar(corpus="{corpus}")'
        if is_drive_url
        else f'rag_query(corpus="{corpus}")'
    )
    st.markdown(
        f"<span style='background:{VERTEX_COLOR}22;color:{VERTEX_COLOR};"
        f"padding:3px 8px;border-radius:4px;font-size:0.8em;font-family:monospace'>"
        f"{label}"
        f"</span>",
        unsafe_allow_html=True,
    )


def _fn_badge_hybrid(fn_name: str, indexes: list[str]) -> None:
    args = (
        f'index_name="{indexes[0]}"'
        if len(indexes) == 1
        else f'index_names={indexes}'
    )
    st.markdown(
        f"<span style='background:{HYBRID_COLOR}22;color:{HYBRID_COLOR};"
        f"padding:3px 8px;border-radius:4px;font-size:0.8em;font-family:monospace'>"
        # f"{fn_name}({args})"
        f"{'Multi indexes' if fn_name == 'hybrid_multi_query' else 'Single index'}({args})"
        f"</span>",
        unsafe_allow_html=True,
    )


def _render_vertex(result: dict, is_drive_url: bool = False) -> None:
    corpus_name = result.get("corpus_name", "")
    if corpus_name:
        _fn_badge_vertex(corpus_name, is_drive_url=is_drive_url)
        st.write("")
    if result.get("status") == "skipped":
        st.caption("_No corpus selected_")
        return
    if result.get("status") == "error":
        _show_error(result.get("message", "Error"))
        return

    elapsed = result.get("elapsed_s", "?")

    # Drive URL → résultats de similarité (pas de réponse générée)
    if is_drive_url:
        similar = result.get("results", [])
        st.caption(f"{elapsed}s · {len(similar)} documents")
        if not similar:
            st.info("No similar documents found.")
            return
        lines = []
        for r in similar:
            url  = r.get("uri", "")
            name = r.get("title", url)
            lines.append(f"- [{name}]({url})")
        st.markdown("\n".join(lines))
        return

    st.caption(f"{elapsed}s")
    _render_answer(result.get("answer", ""), result.get("sources", []))
    st.divider()
    render_sources(result.get("sources", []), pipeline="vertex")


def _format_timings(timings: dict) -> str:
    """Build a human-readable timing breakdown string from a timings dict."""
    parts: list[str] = []
    if "resolve_and_rewrite" in timings:
        parts.append(f"resolve+rewrite {timings['resolve_and_rewrite']}s")
    else:
        if "resolve_indexes" in timings:
            parts.append(f"resolve {timings['resolve_indexes']}s")
        if "rewrite_query" in timings:
            parts.append(f"rewrite {timings['rewrite_query']}s")
    if "embedding" in timings:
        parts.append(f"embedding {timings['embedding']}s")
    if "threads" in timings:
        parts.append(f"search {timings['threads']}s")
    elif "search" in timings:
        parts.append(f"search {timings['search']}s")
    if "generation" in timings:
        parts.append(f"generation {timings['generation']}s")
    return " · ".join(parts)


def _render_hybrid_mono(result: dict, indexes_used: list[str]) -> None:
    if indexes_used:
        _fn_badge_hybrid("hybrid_rag_query", indexes_used)
        st.write("")
    if result.get("status") == "skipped":
        st.caption("_No index selected_")
        return
    if result.get("status") == "error":
        _show_error(result.get("message", "Error"))
        return
    total_s        = result.get("elapsed_s", "?")
    retrieval_query = result.get("retrieval_query", "")
    timings        = result.get("timings", {})

    timing_detail = _format_timings(timings)

    st.caption(
        f"{total_s}s total · "
        f"{result.get('total_results', 0)} chunks · "
        f"mode `{result.get('retrieval_mode', '—')}`"
    )
    if timing_detail:
        st.caption(f"_{timing_detail}_")
    if retrieval_query:
        st.caption(f"retrieval query: _{retrieval_query}_")
    generated = result.get("generated_answer", "")
    _render_answer(generated, result.get("sources", []))
    st.divider()
    render_sources(result.get("sources", []), pipeline="hybrid")
    with st.expander("Retrieved chunks", expanded=False):
        render_chunks(result.get("chunks", []))


def _render_hybrid_similar(result: dict, indexes_used: list[str]) -> None:
    """Render hybrid_find_similar results (Drive URL similarity search)."""
    idx_str = ", ".join(f'"{i}"' for i in indexes_used)
    st.markdown(
        f"<span style='background:{HYBRID_COLOR}22;color:{HYBRID_COLOR};"
        f"padding:3px 8px;border-radius:4px;font-size:0.8em;font-family:monospace'>"
        f"hybrid_find_similar(index_names=[{idx_str}])"
        f"</span>",
        unsafe_allow_html=True,
    )
    st.write("")
    if result.get("status") == "error":
        _show_error(result.get("message", "Error"))
        return
    elapsed = result.get("elapsed_s", "?")
    similar = result.get("results", [])
    st.caption(f"{elapsed}s · {len(similar)} documents · {len(indexes_used)} index")

    # Résumé du document source
    doc_summary = result.get("doc_summary", "")
    if doc_summary:
        st.markdown(doc_summary)
        st.divider()

    if not similar:
        st.info("No similar documents found.")
        return
    lines = []
    for r in similar:
        score = round(r.get("score", 0) * 100, 1)
        url   = r.get("source_url", "")
        name  = r.get("file_name", url)
        idx   = r.get("index_name", "")
        lines.append(f"- [{name}]({url}) · `{idx}` · {score}%")
    st.markdown("\n".join(lines))


def _render_hybrid_multi(result: dict, indexes_used: list[str]) -> None:
    if indexes_used:
        _fn_badge_hybrid("hybrid_multi_query", indexes_used)
        st.write("")
    if result.get("status") == "skipped":
        st.caption("_No indexes available_")
        return
    if result.get("status") == "error":
        _show_error(result.get("message", "Error"))
        return
    empty          = result.get("indexes_empty", [])
    total          = result.get("total_results", 0)
    total_s        = result.get("elapsed_s", "?")
    retrieval_query = result.get("retrieval_query", "")
    timings        = result.get("timings", {})

    timing_detail = _format_timings(timings)

    st.caption(
        f"{total_s}s total · "
        f"{total} chunks · {len(indexes_used)} index"
        + (f" · empty: {', '.join(empty)}" if empty else "")
    )
    if timing_detail:
        st.caption(f"_{timing_detail}_")
    if retrieval_query:
        st.caption(f"retrieval query: _{retrieval_query}_")
    generated = result.get("generated_answer", "")
    all_sources = result.get("sources", [])
    _render_answer(generated, all_sources)
    st.divider()
    # Details by index in an expander
    with st.expander("Details by index", expanded=False):
        for idx_name, chunks in result.get("results_by_index", {}).items():
            st.markdown(
                f"<div style='border-left:3px solid {HYBRID_COLOR};"
                f"padding-left:8px;margin:6px 0'>"
                f"<strong>{idx_name.upper()}</strong> — {len(chunks)} chunks</div>",
                unsafe_allow_html=True,
            )
            render_chunks(chunks)
    if result.get("sources"):
        render_sources(result["sources"], pipeline="hybrid")


def _render_entry(entry: dict) -> None:
    st.markdown(f"**{entry['query']}**")
    c_v, c_h = st.columns(2)
    with c_v:
        _render_vertex(entry["vertex"], is_drive_url=entry.get("is_drive_url", False))
    with c_h:
        if entry.get("is_drive_url"):
            _render_hybrid_similar(entry["hybrid"], entry.get("indexes_used", []))
        elif entry.get("hybrid_mode") == "multi":
            _render_hybrid_multi(entry["hybrid"], entry.get("indexes_used", []))
        else:
            _render_hybrid_mono(entry["hybrid"], entry.get("indexes_used", []))
    st.divider()


# ── History ───────────────────────────────────────────────────────────────────

for entry in active_cmp.get("history", []):
    _render_entry(entry)

# ── New query ─────────────────────────────────────────────────────────────────

query_input = st.chat_input("Ask your question...")

if query_input:
    corpus = selected_corpus if isinstance(selected_corpus, str) else ""

    if not corpus and not all_indexes:
        st.warning("No Naive RAG corpus or Hybrid indexes available.")
        st.stop()

    # Auto-name session from first query
    if not active_cmp["history"] and active_cmp["name"].startswith("Session "):
        active_cmp["name"] = query_input[:45] + ("…" if len(query_input) > 45 else "")
        from services.session_store import save_session
        save_session(active_cmp)
        _refresh_sessions_list()

    st.markdown(f"**{query_input}**")

    # ── Contexte conversationnel — un contexte par pipeline (cloisonnés) ────────
    # Chaque pipeline n'utilise QUE ses propres réponses précédentes comme
    # contexte de conversation.  Mélanger les deux pipelines fausserait le
    # contexte fourni à chacun (ex. Vertex recevrait les réponses Hybrid).

    def _build_context_vertex(history: list[dict], n: int = 3) -> str:
        recent = history[-n:] if len(history) >= n else history
        lines = []
        for entry in recent:
            lines.append(f"Q: {entry['query']}")
            answer = entry.get("vertex", {}).get("answer") or ""
            if answer:
                lines.append(f"R: {str(answer)[:200]}")
        return "\n".join(lines)

    def _build_context_hybrid(history: list[dict], n: int = 3) -> str:
        recent = history[-n:] if len(history) >= n else history
        lines = []
        for entry in recent:
            lines.append(f"Q: {entry['query']}")
            answer = entry.get("hybrid", {}).get("generated_answer") or ""
            if answer:
                lines.append(f"R: {str(answer)[:200]}")
        return "\n".join(lines)

    history = active_cmp.get("history", [])
    conv_context_vertex = _build_context_vertex(history)
    conv_context_hybrid = _build_context_hybrid(history)

    # ── Drive URL detection (used by both placeholder and pipeline logic) ────
    _drive_url_match = _DRIVE_URL_RE.search(query_input)

    # ── Step 1: resolve indexes + rewrite query (parallel) ─────────────────
    import time as _t

    resolved: list[str] = []
    rewritten_query = query_input
    pre_timings: dict[str, float] = {}

    if _drive_url_match:
        # Drive URL — no resolve/rewrite needed
        if all_indexes:
            resolved = all_indexes
    else:
        # Run resolve + rewrite in parallel (both are Gemini Flash calls)
        with st.spinner("Resolving indexes + rewriting query..."):
            import concurrent.futures as _cf
            t_pre = _t.perf_counter()

            def _do_resolve():
                if not all_indexes:
                    return []
                t0 = _t.perf_counter()
                r = _resolve_indexes(query_input, all_indexes)
                pre_timings["resolve_indexes"] = round(_t.perf_counter() - t0, 2)
                return r

            def _do_rewrite():
                try:
                    from shared.query_rewriter import rewrite_query as _rw
                    shared_context = conv_context_hybrid or conv_context_vertex
                    t0 = _t.perf_counter()
                    r = _rw(query_input, context=shared_context)
                    pre_timings["rewrite_query"] = round(_t.perf_counter() - t0, 2)
                    return r
                except Exception:
                    return query_input

            with _cf.ThreadPoolExecutor(max_workers=2) as pre_pool:
                f_resolve = pre_pool.submit(_do_resolve)
                f_rewrite = pre_pool.submit(_do_rewrite)
                resolved = f_resolve.result()
                rewritten_query = f_rewrite.result()

            pre_timings["resolve_and_rewrite"] = round(_t.perf_counter() - t_pre, 2)

    hybrid_mode = "multi" if len(resolved) > 1 else "mono"

    # ── Step 2: colonnes avec placeholders ───────────────────────────────────
    col_v, col_h = st.columns(2)

    with col_v:
        v_ph = st.empty()
        v_ph.info(f"Naive RAG running... `{corpus}`")

    with col_h:
        if _drive_url_match:
            fn_label = "hybrid_find_similar"
        elif hybrid_mode == "multi":
            fn_label = "hybrid_multi_query"
        else:
            fn_label = "hybrid_rag_query"
        h_ph = st.empty()
        h_ph.info(f"`{fn_label}` running...")

    # ── Step 3: pipeline functions ────────────────────────────────────────────

    def _run_vertex():
        if not corpus:
            return {"status": "skipped", "answer": "", "sources": [],
                    "elapsed_s": 0, "corpus_name": ""}
        if _drive_url_match:
            from services.vertex_service import find_similar as vertex_find_similar
            return vertex_find_similar(
                corpus_name=corpus,
                document_url=_drive_url_match.group(0),
            )
        from services.vertex_service import query as vq
        return vq(corpus, rewritten_query, context=conv_context_vertex,
                  retrieval_query=rewritten_query)

    def _run_hybrid():
        import time
        t0 = time.perf_counter()

        # Drive URL → similarité documentaire vectorielle + résumé du doc source
        if _drive_url_match:
            from hybrid.tools.hybrid_find_similar import hybrid_find_similar
            from hybrid.ingestion.extractor import extract_from_url
            from hybrid.config import SERVICE_ACCOUNT_PATH

            # Extraction du texte pour le résumé Gemini
            doc_summary = ""
            try:
                from googleapiclient.discovery import build
                from google.oauth2 import service_account
                creds = service_account.Credentials.from_service_account_file(
                    SERVICE_ACCOUNT_PATH,
                    scopes=["https://www.googleapis.com/auth/drive.readonly"],
                )
                drive_service = build("drive", "v3", credentials=creds)
                doc_text = extract_from_url(_drive_url_match.group(0), drive_service)
                if doc_text:
                    doc_summary = _generate_answer(
                        "Summarize this document in 3-5 sentences: what is it about, "
                        "what is its purpose, who are the parties involved?",
                        doc_text[:6000],
                    )
            except Exception:
                pass

            result = hybrid_find_similar(
                document_url=_drive_url_match.group(0),
                index_names=resolved or all_indexes,
            )
            result["elapsed_s"] = round(time.perf_counter() - t0, 2)
            if result.get("status") == "success":
                result["doc_summary"] = doc_summary
                result["sources"] = [
                    {"file_name": r["file_name"], "source_url": r["source_url"]}
                    for r in result.get("results", [])
                ]
            return result

        if not resolved:
            return {"status": "skipped", "answer": "", "sources": [], "elapsed_s": 0}

        t_retrieval = time.perf_counter()
        if hybrid_mode == "multi":
            from services.hybrid_service import multi_query
            result = multi_query(resolved, rewritten_query, retrieval_mode, top_k,
                                 context=conv_context_hybrid, retrieval_query=rewritten_query)
        else:
            from services.hybrid_service import query as hq
            result = hq(resolved[0], rewritten_query, retrieval_mode, top_k,
                        context=conv_context_hybrid, retrieval_query=rewritten_query)
        retrieval_elapsed = round(time.perf_counter() - t_retrieval, 2)

        # Génération Gemini sur les chunks récupérés
        t_gen = time.perf_counter()
        if result.get("status") == "success":
            context = result.get("context") or result.get("answer", "")
            result["generated_answer"] = _generate_answer(rewritten_query, context)
        gen_elapsed = round(time.perf_counter() - t_gen, 2)

        # Merge all timings: pre-processing + retrieval inner + generation
        all_timings = dict(pre_timings)
        inner = result.get("timings", {})
        all_timings.update(inner)
        all_timings["generation"] = gen_elapsed
        all_timings["total"] = round(time.perf_counter() - t0, 2)
        result["timings"] = all_timings
        result["elapsed_s"] = all_timings["total"]
        return result

    # ── Step 4: exécution parallèle, affichage dès que prêt ──────────────────

    vertex_result = {"status": "skipped", "answer": "", "sources": [],
                     "elapsed_s": 0, "corpus_name": corpus}
    hybrid_result = {"status": "skipped", "answer": "", "sources": [], "elapsed_s": 0}

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        futures = {
            ex.submit(_run_vertex): "vertex",
            ex.submit(_run_hybrid): "hybrid",
        }

        for future in concurrent.futures.as_completed(futures):
            which = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"status": "error", "message": str(exc), "sources": [],
                          "elapsed_s": 0, "corpus_name": corpus}

            if which == "vertex":
                vertex_result = result
                with v_ph.container():
                    _render_vertex(result, is_drive_url=bool(_drive_url_match))
            else:
                hybrid_result = result
                with h_ph.container():
                    if _drive_url_match:
                        _render_hybrid_similar(result, resolved or all_indexes)
                    elif hybrid_mode == "multi":
                        _render_hybrid_multi(result, resolved)
                    else:
                        _render_hybrid_mono(result, resolved)

    # ── Step 5: persist ───────────────────────────────────────────────────────

    new_entry = {
        "query":        query_input,
        "vertex":       vertex_result,
        "hybrid":       hybrid_result,
        "hybrid_mode":  hybrid_mode,
        "indexes_used": resolved,
        "is_drive_url": bool(_drive_url_match),
    }
    st.session_state.cmp_active_session = append_entry(active_cmp, new_entry)
    _refresh_sessions_list()

    st.divider()
