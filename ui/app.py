"""
Workshop Chatbot Knowledge Base — AI for Customer Care
Streamlit home page.

Entry point: streamlit run ui/app.py
"""

import path_setup  # noqa: F401
import streamlit as st
from config import (
    APP_TITLE, APP_SUBTITLE, APP_ICON, LAYOUT,
    VERTEX_COLOR, HYBRID_COLOR, VERTEX_LABEL, HYBRID_LABEL,
)

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=LAYOUT,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div style="padding: 20px 0 10px 0;">
    <h1 style="margin-bottom:4px;">{APP_TITLE}</h1>
    <p style="font-size:1.1em; color:#555; margin-top:0;">{APP_SUBTITLE}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# ── Pipeline overview ─────────────────────────────────────────────────────────
st.subheader("Two approaches, one goal: smarter customer support")
st.markdown(
    "This platform lets you compare two AI-powered knowledge retrieval strategies "
    "on your internal documents — so you can choose the best engine for your customer care chatbot."
)

col_v, col_h = st.columns(2)

with col_v:
    st.markdown(
        f"""
        <div style="border-left: 4px solid {VERTEX_COLOR};
             border-radius: 0 8px 8px 0; padding: 12px 16px;">
        <h3 style="color:{VERTEX_COLOR}; margin: 0">{VERTEX_LABEL}</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("""
    - All documents stored in a **single cloud corpus**
    - One-step ingestion from Google Drive
    - AI retrieves and answers in a **single call**
    - Fast setup, minimal configuration
    """)
    st.info("Best for: quick deployment, broad search across all documents at once")

with col_h:
    st.markdown(
        f"""
        <div style="border-left: 4px solid {HYBRID_COLOR};
             border-radius: 0 8px 8px 0; padding: 12px 16px;">
        <h3 style="color:{HYBRID_COLOR}; margin: 0">{HYBRID_LABEL}</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("""
    - Documents organized **by client and topic**
    - Combines keyword search + semantic understanding
    - Queries across multiple clients in one request
    - Higher precision, source-level traceability
    """)
    st.info("Best for: client-specific answers, precise sourcing, multi-tenant support")

st.divider()

# ── Navigation cards ──────────────────────────────────────────────────────────
st.subheader("Get started")

nav1, nav2, nav3 = st.columns(3)

with nav1:
    st.markdown("### Chat with the Agent")
    st.markdown(
        "Ask questions in natural language. "
        "The AI agent automatically picks the right pipeline "
        "and retrieves the most relevant answers from your knowledge base."
    )

with nav2:
    st.markdown("### Side-by-Side Comparison")
    st.markdown(
        "Send the same question to both pipelines at once. "
        "Compare answer quality, cited sources, and response time head-to-head."
    )

with nav3:
    st.markdown("### Knowledge Base Manager")
    st.markdown(
        "Import documents from Google Drive, organize them by client and topic, "
        "and monitor the health of your knowledge base."
    )

st.divider()

# ── Quick status ──────────────────────────────────────────────────────────────
st.subheader("System status")

@st.cache_data(ttl=120, show_spinner=False)
def _get_corpora():
    from services.vertex_service import list_corpora
    return list_corpora()

@st.cache_data(ttl=120, show_spinner=False)
def _get_indexes():
    from services.hybrid_service import list_indexes
    return list_indexes()

status_col_v, status_col_h = st.columns(2)

with status_col_v:
    try:
        corpora = _get_corpora()
        st.metric("Naive RAG Corpora", len(corpora))
        for c in corpora:
            st.caption(f"- {c['display_name']}")
    except Exception as exc:
        st.metric("Naive RAG Corpora", "---")
        st.warning(f"Naive RAG unavailable: {exc}")

with status_col_h:
    try:
        indexes = _get_indexes()
        st.metric("Hybrid RAG Indexes", len(indexes))
        for idx in indexes:
            chunks = idx.get("total_chunks", "?")
            st.caption(f"- {idx['index_name']}  ({chunks} chunks)")
    except Exception as exc:
        st.metric("Hybrid RAG Indexes", "---")
        st.warning(f"Hybrid RAG unavailable: {exc}")
