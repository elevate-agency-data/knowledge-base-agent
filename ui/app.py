"""
Knowledge Base Agent — Streamlit home page.

Entry point: streamlit run ui/app.py
(from the project root so that rag_agent/ and hybrid/ are importable)
"""

import path_setup  # noqa: F401 — fixe sys.path et CWD avant tout import projet
import streamlit as st
from config import APP_TITLE, APP_ICON, LAYOUT, VERTEX_COLOR, HYBRID_COLOR

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=LAYOUT,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title(APP_TITLE)
st.caption("Interface de gestion et d'interrogation des bases de connaissances RAG")

st.divider()

# ── Pipeline overview ─────────────────────────────────────────────────────────
col_v, col_h = st.columns(2)

with col_v:
    st.markdown(
        f"""
        <div style="border-left: 4px solid {VERTEX_COLOR}; padding-left: 12px;">
        <h3 style="color:{VERTEX_COLOR}">Vertex AI RAG</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("""
    - Corpus global hébergé sur Google Cloud
    - Ingestion massive de dossiers Drive entiers
    - Recherche sémantique via `text-embedding-005`
    - Réponses générées par **Gemini 2.5 Pro**
    """)
    st.info("Idéal pour : ingestion volumineuse, recherche globale sans isolation client")

with col_h:
    st.markdown(
        f"""
        <div style="border-left: 4px solid {HYBRID_COLOR}; padding-left: 12px;">
        <h3 style="color:{HYBRID_COLOR}">Hybrid RAG (local)</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("""
    - Index locaux par client (DuckDB)
    - Recherche **dense** (vecteurs) + **sparse** (BM25) → fusion RRF
    - 5 modèles d'embedding disponibles (mpnet, e5, bge-m3…)
    - Multi-client en un seul appel via `hybrid_multi_query`
    """)
    st.info("Idéal pour : précision par client, comparaisons multi-index")

st.divider()

# ── Navigation cards ──────────────────────────────────────────────────────────
st.subheader("Navigation")

nav1, nav2, nav3 = st.columns(3)

with nav1:
    st.markdown("### Agent Chat")
    st.markdown(
        "Interface conversationnelle complète. "
        "L'agent choisit automatiquement le pipeline adapté "
        "et expose les appels d'outils."
    )

with nav2:
    st.markdown("### Comparaison RAG")
    st.markdown(
        "Pose la même question aux deux pipelines en parallèle. "
        "Compare les réponses, les sources et les temps de réponse côte-à-côte."
    )

with nav3:
    st.markdown("### Gestion des index")
    st.markdown(
        "Crée, inspecte et supprime des index hybrid. "
        "Lance l'ingestion depuis Google Drive et surveille l'état de chaque index."
    )

st.divider()

# ── Quick status ──────────────────────────────────────────────────────────────
st.subheader("État rapide")

status_col_v, status_col_h = st.columns(2)

with status_col_v:
    with st.spinner("Chargement des corpus Vertex…"):
        try:
            from services.vertex_service import list_corpora
            corpora = list_corpora()
            st.metric("Corpus Vertex AI", len(corpora))
            if corpora:
                for c in corpora:
                    st.caption(f"• {c['display_name']}")
        except Exception as exc:
            st.metric("Corpus Vertex AI", "—")
            st.warning(f"Vertex non disponible : {exc}")

with status_col_h:
    with st.spinner("Chargement des index hybrid…"):
        try:
            from services.hybrid_service import list_indexes
            indexes = list_indexes()
            st.metric("Index Hybrid", len(indexes))
            if indexes:
                for idx in indexes:
                    chunks = idx.get("total_chunks", "?")
                    st.caption(f"• {idx['index_name']}  ({chunks} chunks)")
        except Exception as exc:
            st.metric("Index Hybrid", "—")
            st.warning(f"Hybrid non disponible : {exc}")
