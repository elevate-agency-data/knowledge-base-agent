"""
Page 2 — RAG Comparison

Vertex AI RAG (gauche) vs Hybrid RAG (droite).

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
)

st.set_page_config(
    page_title=f"Comparaison RAG — {APP_TITLE}",
    page_icon=None,
    layout="wide",
)

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
    Appelle Gemini Flash pour synthétiser une réponse à partir des chunks.
    Flash est utilisé ici (et non Pro) pour éviter que le double appel Gemini
    (retrieval côté Hybrid + génération) ne soit plus lent que Vertex.
    """
    if not context.strip():
        return ""
    try:
        from vertexai.generative_models import GenerativeModel
        prompt = (
            "Tu es l'assistant interne d'Elevate, société de conseil en Data & Analytics. "
            "Réponds EXCLUSIVEMENT à partir des documents internes Elevate fournis ci-dessous "
            "(propositions commerciales, analyses, offres rédigées par Elevate pour ses clients). "
            "N'utilise JAMAIS ta connaissance générale sur les entreprises ou les marques. "
            "Si l'information ne figure pas dans les documents, dis-le clairement.\n\n"
            f"Documents internes Elevate :\n{context}\n\n"
            f"Question : {query}\n\n"
            "Réponse :"
        )
        return GenerativeModel("gemini-2.0-flash-001").generate_content(prompt).text
    except Exception as exc:
        return f"_(Erreur de génération : {exc})_"


# ── Index resolver ────────────────────────────────────────────────────────────

def _resolve_indexes(query: str, available: list[str]) -> list[str]:
    if not available:
        return []
    if len(available) == 1:
        return available
    prompt = (
        f"Index disponibles : {', '.join(available)}\n"
        f"Requête : \"{query}\"\n\n"
        "Quels index faut-il interroger ?\n"
        "- Si la requête mentionne un ou plusieurs clients précis correspondant "
        "à des noms d'index, retourne uniquement ceux-là.\n"
        "- Si la requête est comparative, générale, ou ne cible aucun client "
        "précis, retourne TOUS les index.\n"
        "Réponds UNIQUEMENT avec les noms d'index séparés par des virgules."
    )
    try:
        from vertexai.generative_models import GenerativeModel
        response = GenerativeModel("gemini-2.0-flash-001").generate_content(prompt)
        names    = [n.strip() for n in response.text.strip().lower().split(",")]
        resolved = [n for n in names if n in available]
        return resolved if resolved else available
    except Exception:
        q       = query.lower()
        matched = [n for n in available if n.lower() in q]
        return matched if matched else available


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
    st.header("Comparaison RAG")
    st.divider()

    st.markdown("**Sessions**")
    if st.button("Nouvelle session", use_container_width=True, type="primary"):
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
                help=f"{n} requête{'s' if n != 1 else ''}",
            ):
                _load_cmp(meta["id"])
                st.rerun()
        with del_c:
            if st.button("X", key=f"cmp_del_{meta['id']}", help="Supprimer"):
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
            "Corpus (nom)", key="sel_corpus_text", placeholder="base-rag"
        )

    st.divider()

    st.markdown(
        f"<span style='color:{HYBRID_COLOR}'>**{HYBRID_LABEL}**</span>",
        unsafe_allow_html=True,
    )
    all_indexes = _load_indexes()
    if all_indexes:
        st.caption("Index disponibles (sélection automatique) :")
        for idx in all_indexes:
            st.caption(f"• `{idx}`")
    else:
        st.warning("Aucun index hybrid trouvé.")

    st.divider()
    st.markdown("**Options Hybrid**")
    retrieval_mode = st.selectbox(
        "Mode de retrieval", RETRIEVAL_MODES,
        index=RETRIEVAL_MODES.index(DEFAULT_RETRIEVAL_MODE),
        key="sel_mode",
    )
    top_k = st.slider(
        "Chunks par index", min_value=3, max_value=20,
        value=DEFAULT_TOP_K, key="sel_topk",
    )

    st.divider()
    if st.button("Actualiser les listes", use_container_width=True):
        _load_corpora.clear()
        _load_indexes.clear()
        _refresh_sessions_list()
        st.rerun()


# ── Page header ───────────────────────────────────────────────────────────────

active_cmp = _active_cmp()
st.title(active_cmp['name'])
st.caption(
    "Les deux pipelines s'exécutent en parallèle. "
    "Chaque colonne s'affiche dès que son résultat est prêt."
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

from components.source_card import render_sources, render_chunks
from services.session_store import append_entry


def _fn_badge_vertex(corpus: str) -> None:
    st.markdown(
        f"<span style='background:{VERTEX_COLOR}22;color:{VERTEX_COLOR};"
        f"padding:3px 8px;border-radius:4px;font-size:0.8em;font-family:monospace'>"
        f"rag_query(corpus=\"{corpus}\")"
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
        f"{'Multi indexes' if fn_name == 'hybrid_multi_query' else 'Index unique'}({args})"
        f"</span>",
        unsafe_allow_html=True,
    )


def _render_vertex(result: dict) -> None:
    corpus_name = result.get("corpus_name", "")
    if corpus_name:
        _fn_badge_vertex(corpus_name)
        st.write("")
    if result.get("status") == "skipped":
        st.caption("_Corpus non sélectionné_")
        return
    if result.get("status") == "error":
        st.error(result.get("message", "Erreur"))
        return
    st.caption(f"{result.get('elapsed_s', '?')}s")
    st.markdown(result.get("answer") or "_Aucune réponse générée._")
    st.divider()
    render_sources(result.get("sources", []), pipeline="vertex")


def _render_hybrid_mono(result: dict, indexes_used: list[str]) -> None:
    if indexes_used:
        _fn_badge_hybrid("hybrid_rag_query", indexes_used)
        st.write("")
    if result.get("status") == "skipped":
        st.caption("_Aucun index sélectionné_")
        return
    if result.get("status") == "error":
        st.error(result.get("message", "Erreur"))
        return
    retrieval_s    = result.get("elapsed_retrieval_s", result.get("elapsed_s", "?"))
    total_s        = result.get("elapsed_s", "?")
    retrieval_query = result.get("retrieval_query", "")
    st.caption(
        f"{total_s}s total (retrieval {retrieval_s}s + génération) · "
        f"{result.get('total_results', 0)} chunks · "
        f"mode `{result.get('retrieval_mode', '—')}`"
    )
    if retrieval_query:
        st.caption(f"query retrieval : _{retrieval_query}_")
    # Réponse générée par Gemini
    generated = result.get("generated_answer", "")
    st.markdown(generated if generated else "_Aucune réponse générée._")
    st.divider()
    render_sources(result.get("sources", []), pipeline="hybrid")
    with st.expander("Chunks récupérés", expanded=False):
        render_chunks(result.get("chunks", []))


def _render_hybrid_multi(result: dict, indexes_used: list[str]) -> None:
    if indexes_used:
        _fn_badge_hybrid("hybrid_multi_query", indexes_used)
        st.write("")
    if result.get("status") == "skipped":
        st.caption("_Aucun index disponible_")
        return
    if result.get("status") == "error":
        st.error(result.get("message", "Erreur"))
        return
    empty          = result.get("indexes_empty", [])
    total          = result.get("total_results", 0)
    total_s        = result.get("elapsed_s", "?")
    retrieval_s    = result.get("elapsed_retrieval_s", total_s)
    retrieval_query = result.get("retrieval_query", "")
    st.caption(
        f"{total_s}s total (retrieval {retrieval_s}s + génération) · "
        f"{total} chunks · {len(indexes_used)} index"
        + (f" · vides : {', '.join(empty)}" if empty else "")
    )
    if retrieval_query:
        st.caption(f"query retrieval : _{retrieval_query}_")
    # Réponse générée par Gemini (sur le contexte fusionné de tous les index)
    generated = result.get("generated_answer", "")
    st.markdown(generated if generated else "_Aucune réponse générée._")
    st.divider()
    # Détail par index dans un expander
    with st.expander("Détail par index", expanded=False):
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
        _render_vertex(entry["vertex"])
    with c_h:
        if entry.get("hybrid_mode") == "multi":
            _render_hybrid_multi(entry["hybrid"], entry.get("indexes_used", []))
        else:
            _render_hybrid_mono(entry["hybrid"], entry.get("indexes_used", []))
    st.divider()


# ── History ───────────────────────────────────────────────────────────────────

for entry in active_cmp.get("history", []):
    _render_entry(entry)

# ── New query ─────────────────────────────────────────────────────────────────

query_input = st.chat_input("Posez votre question…")

if query_input:
    corpus = selected_corpus if isinstance(selected_corpus, str) else ""

    if not corpus and not all_indexes:
        st.warning("Aucun corpus Vertex ni index Hybrid disponible.")
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

    # ── Step 1: resolve indexes (fast, avant l'exécution lourde) ─────────────
    resolved: list[str] = []
    if all_indexes:
        with st.spinner("Détection des index pertinents…"):
            resolved = _resolve_indexes(query_input, all_indexes)

    hybrid_mode = "multi" if len(resolved) > 1 else "mono"

    # ── Step 2: colonnes avec placeholders ───────────────────────────────────
    col_v, col_h = st.columns(2)

    with col_v:
        v_ph = st.empty()
        v_ph.info(f"Vertex en cours… `{corpus}`")

    with col_h:
        fn_label = "hybrid_multi_query" if hybrid_mode == "multi" else "hybrid_rag_query"
        h_ph = st.empty()
        h_ph.info(f"`{fn_label}` en cours…")

    # ── Step 3: pipeline functions ────────────────────────────────────────────

    _drive_url_match = _DRIVE_URL_RE.search(query_input)

    def _run_vertex():
        if not corpus:
            return {"status": "skipped", "answer": "", "sources": [],
                    "elapsed_s": 0, "corpus_name": ""}
        if _drive_url_match:
            from services.vertex_service import extract_query_from_drive_url, query as vq
            doc_query = extract_query_from_drive_url(_drive_url_match.group(0))
            return vq(corpus, doc_query, context=conv_context_vertex)
        from services.vertex_service import query as vq
        return vq(corpus, query_input, context=conv_context_vertex)

    def _run_hybrid():
        import time
        t0 = time.perf_counter()

        # Drive URL → similarité documentaire vectorielle
        if _drive_url_match:
            from hybrid.tools.hybrid_find_similar import hybrid_find_similar
            result = hybrid_find_similar(
                document_url=_drive_url_match.group(0),
                index_names=resolved or all_indexes,
            )
            result["elapsed_s"] = round(time.perf_counter() - t0, 2)
            if result.get("status") == "success":
                result["generated_answer"] = (
                    "Documents trouvés par similarité vectorielle "
                    f"(distance cosinus sur {len(result.get('results', []))} fichiers)."
                )
                # Normalise sources pour l'affichage
                result["sources"] = [
                    {"file_name": r["file_name"], "source_url": r["source_url"]}
                    for r in result.get("results", [])
                ]
            return result

        if not resolved:
            return {"status": "skipped", "answer": "", "sources": [], "elapsed_s": 0}
        if hybrid_mode == "multi":
            from services.hybrid_service import multi_query
            result = multi_query(resolved, query_input, retrieval_mode, top_k, context=conv_context_hybrid)
        else:
            from services.hybrid_service import query as hq
            result = hq(resolved[0], query_input, retrieval_mode, top_k, context=conv_context_hybrid)

        # Génération Gemini sur les chunks récupérés
        if result.get("status") == "success":
            context = result.get("context") or result.get("answer", "")
            result["generated_answer"] = _generate_answer(query_input, context)

        # Écrase elapsed_s partiel (retrieval seul) par le temps total réel
        result["elapsed_s"]      = round(time.perf_counter() - t0, 2)
        result["elapsed_retrieval_s"] = result.get("elapsed_s", 0)  # garder pour info
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
            which  = futures[future]
            result = future.result()

            if which == "vertex":
                vertex_result = result
                with v_ph.container():
                    _render_vertex(result)
            else:
                hybrid_result = result
                with h_ph.container():
                    if hybrid_mode == "multi":
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
    }
    st.session_state.cmp_active_session = append_entry(active_cmp, new_entry)
    _refresh_sessions_list()

    st.divider()
