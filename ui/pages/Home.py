"""
Home page — brand landing + knowledge base overview.

Layout follows luxury editorial codes (generous whitespace, display serif,
hairline rules, letter-spaced micro-labels, sparing accent). All colors come
from the active brand profile, so every brand stays visually coherent.
"""

import path_setup  # noqa: F401
import streamlit as st
from config import (
    APP_TITLE, APP_DESCRIPTION, APP_ICON, LAYOUT,
)
from auth import require_auth
from components.sidebar_auth import render_sidebar_user
from components.brand_header import brand_header
from shared.brand import ACTIVE

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=LAYOUT,
)

require_auth()

with st.sidebar:
    render_sidebar_user()

_ACCENT = ACTIVE.theme.accent

# ── Editorial styling ─────────────────────────────────────────────────────────

st.markdown(
    f"""
    <style>
      .lux-eyebrow {{
        font-size: .70rem; letter-spacing: .24em; text-transform: uppercase;
        color: #8f8b86; margin: 0 0 10px 0;
      }}
      .lux-h2 {{
        font-size: 1.85rem; font-weight: 400; line-height: 1.3;
        margin: 0 0 16px 0; letter-spacing: .01em;
      }}
      .lux-lead {{
        font-size: 1.02rem; line-height: 1.85; color: #55504b; max-width: 68ch;
      }}
      .lux-rule {{
        width: 44px; height: 2px; background: {_ACCENT}; margin: 0 0 26px 0;
      }}
      .lux-card {{
        border: 1px solid #e9e5e1; padding: 28px 24px; height: 100%;
      }}
      .lux-card-title {{
        font-size: .76rem; letter-spacing: .18em; text-transform: uppercase;
        margin: 0 0 12px 0;
      }}
      .lux-card-rule {{
        width: 26px; height: 2px; background: {_ACCENT}; margin: 0 0 16px 0;
      }}
      .lux-card-body {{
        font-size: .90rem; line-height: 1.75; color: #6a655f; margin: 0;
      }}
      .lux-metric-value {{
        font-size: 2.7rem; font-weight: 400; line-height: 1; letter-spacing: .01em;
      }}
      .lux-metric-label {{
        font-size: .68rem; letter-spacing: .20em; text-transform: uppercase;
        color: #8f8b86; margin-top: 10px;
      }}
      .lux-row {{
        display: flex; justify-content: space-between; align-items: baseline;
        border-bottom: 1px solid #f0edea; padding: 11px 0;
      }}
      .lux-row-name {{ font-size: .92rem; letter-spacing: .02em; }}
      .lux-row-meta {{ font-size: .76rem; color: #9a958f; letter-spacing: .06em; }}
      .lux-space {{ height: 60px; }}
      .lux-space-sm {{ height: 30px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────

brand_header()

st.markdown('<div class="lux-space"></div>', unsafe_allow_html=True)

# ── Manifesto ─────────────────────────────────────────────────────────────────

_HEADLINE = ACTIVE.headline or ACTIVE.subtitle

st.markdown(
    f"""
    <div class="lux-eyebrow">Overview</div>
    <div class="lux-rule"></div>
    <div class="lux-h2">{_HEADLINE}</div>
    <p class="lux-lead">{APP_DESCRIPTION}</p>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="lux-space"></div>', unsafe_allow_html=True)

# ── Services ──────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="lux-eyebrow">Services</div><div class="lux-rule"></div>',
    unsafe_allow_html=True,
)

_CARDS = [
    (
        "Agent chat",
        "Ask in natural language. The agent selects the relevant domains and "
        "retrieves a sourced answer from the knowledge base.",
    ),
    (
        "Simple chat",
        "A direct exchange without the agent layer. Toggle document retrieval "
        "on or off to compare answers.",
    ),
    (
        "Knowledge base",
        "Import documents from Drive, organize them by domain and monitor the "
        "state of the base.",
    ),
]

for col, (title, body) in zip(st.columns(3, gap="large"), _CARDS):
    with col:
        st.markdown(
            f"""
            <div class="lux-card">
              <div class="lux-card-title">{title}</div>
              <div class="lux-card-rule"></div>
              <p class="lux-card-body">{body}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown('<div class="lux-space"></div>', unsafe_allow_html=True)

# ── Knowledge base state ──────────────────────────────────────────────────────

st.markdown(
    '<div class="lux-eyebrow">Knowledge base</div><div class="lux-rule"></div>',
    unsafe_allow_html=True,
)


@st.cache_data(ttl=120, show_spinner=False)
def _get_indexes():
    from services.hybrid_service import list_indexes
    return list_indexes()


def _metric(col, value, label) -> None:
    col.markdown(
        f'<div class="lux-metric-value">{value}</div>'
        f'<div class="lux-metric-label">{label}</div>',
        unsafe_allow_html=True,
    )


try:
    indexes = _get_indexes()
    domains = sorted({
        (n.split("__", 1)[0] if "__" in n else n)
        for n in (i.get("index_name", "") for i in indexes) if n
    })

    col_a, col_b, col_c = st.columns(3)
    _metric(col_a, len(domains), "Domains")
    _metric(col_b, len(indexes), "Indexes")
    _metric(col_c, sum(i.get("total_files", 0) for i in indexes), "Documents")

    st.markdown('<div class="lux-space-sm"></div>', unsafe_allow_html=True)

    if indexes:
        rows = "".join(
            f'<div class="lux-row">'
            f'<span class="lux-row-name">{idx.get("index_name", "")}</span>'
            f'<span class="lux-row-meta">{idx.get("total_chunks", "?")} chunks</span>'
            f'</div>'
            for idx in indexes
        )
        st.markdown(rows, unsafe_allow_html=True)
    else:
        st.markdown(
            '<p class="lux-lead">The base holds no documents yet. '
            'Import from Drive via the Knowledge base page.</p>',
            unsafe_allow_html=True,
        )
except Exception as exc:
    st.warning(f"Knowledge base unavailable: {exc}")
