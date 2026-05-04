"""
Home page — overview of communes and indexed knowledge base.
"""

import path_setup  # noqa: F401
import streamlit as st
from config import (
    APP_TITLE, APP_SUBTITLE, APP_ICON, LAYOUT,
    ACCENT_COLOR, ACCENT_LIGHT,
)
from auth import require_auth
from components.sidebar_auth import render_sidebar_user

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=LAYOUT,
)

require_auth()

with st.sidebar:
    render_sidebar_user()

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

# ── Pitch ─────────────────────────────────────────────────────────────────────

st.subheader("Internal knowledge, instantly searchable")
st.markdown(
    "Indica indexes your commune's internal documents — finances, HR, business "
    "records, patrimony, maintenance — and lets mayors and municipal staff find "
    "answers in plain language, with sourced citations."
)

st.divider()

# ── Navigation cards ──────────────────────────────────────────────────────────

st.subheader("Get started")

nav1, nav2, nav3 = st.columns(3)

with nav1:
    st.markdown("### Chat with the Agent")
    st.markdown(
        "Ask questions in natural language. The AI agent automatically picks the right "
        "indexes and retrieves the most relevant answers from your knowledge base."
    )

with nav2:
    st.markdown("### Simple Chat")
    st.markdown(
        "Direct chat without the agent layer. Toggle RAG on or off to compare answers "
        "with and without document retrieval."
    )

with nav3:
    st.markdown("### Knowledge Base Manager")
    st.markdown(
        "Import documents from Google Drive, organize them by commune and category, "
        "and monitor the health of your knowledge base."
    )

st.divider()

# ── Quick status ──────────────────────────────────────────────────────────────

st.subheader("System status")


@st.cache_data(ttl=120, show_spinner=False)
def _get_indexes():
    from services.hybrid_service import list_indexes
    return list_indexes()


try:
    indexes = _get_indexes()
    communes = sorted({
        n.split("__", 1)[0] for n in (i.get("index_name", "") for i in indexes) if n
    })

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Communes", len(communes))
    col_b.metric("Indexes", len(indexes))
    col_c.metric(
        "Documents",
        sum(i.get("total_files", 0) for i in indexes),
    )

    if indexes:
        st.markdown("**Indexed:**")
        for idx in indexes:
            chunks = idx.get("total_chunks", "?")
            st.caption(f"- {idx['index_name']}  ({chunks} chunks)")
except Exception as exc:
    st.warning(f"Knowledge base unavailable: {exc}")
