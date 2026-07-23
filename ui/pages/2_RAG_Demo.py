"""
RAG Demo — retrieval pipeline, made visible.

Workshop centrepiece: run the SAME query through dense, sparse and hybrid
retrieval side by side, with the chunks, their scores and their ranks on
screen. Live toggles show the effect of each component:

- dense vs sparse vs hybrid (RRF)
- dense-gating on/off (do BM25-only exact hits survive?)
- content dedup on/off
- dense/sparse weights and top_k

Not a chat page — this is the "look inside the box" page.
"""

import path_setup  # noqa: F401
import streamlit as st

from config import APP_TITLE, APP_ICON, LAYOUT
from auth import require_auth
from components.sidebar_auth import render_sidebar_user
from shared.brand import ACTIVE

st.set_page_config(page_title=f"RAG Demo — {APP_TITLE}", page_icon=APP_ICON, layout=LAYOUT)

user = require_auth()
_ACCENT = ACTIVE.theme.accent

# ── Lazy heavy imports (kept out of module import for fast page load) ──────────

@st.cache_resource(show_spinner=False)
def _engine():
    from hybrid.stores import get_store
    from hybrid.embeddings import get_embedding_model
    from hybrid.config import DEFAULT_EMBEDDING_MODEL
    store = get_store()
    embedder = get_embedding_model(DEFAULT_EMBEDDING_MODEL)
    return store, embedder


@st.cache_data(ttl=120, show_spinner=False)
def _indexes():
    from services.hybrid_service import list_indexes
    return [i.get("index_name", "") for i in list_indexes() if i.get("index_name")]


# ── Sidebar controls ──────────────────────────────────────────────────────────

with st.sidebar:
    render_sidebar_user()
    st.divider()
    st.markdown("**Retrieval controls**")

    all_idx = _indexes()
    picked = st.multiselect("Index(es)", all_idx, default=all_idx[:1] if all_idx else [])
    top_k = st.slider("Top K (per view)", 3, 20, 8)
    dense_w = st.slider("Dense weight", 0.0, 1.0, 0.7, 0.05)
    sparse_w = round(1.0 - dense_w, 2)
    st.caption(f"Sparse weight = {sparse_w}")
    dense_gated = st.toggle(
        "Dense-gating", value=False,
        help="ON: only chunks found by vector search survive the hybrid merge — "
             "exact keyword hits the embedding missed are dropped.",
    )
    dedup = st.toggle("Content dedup", value=True,
                      help="Collapse identical passages duplicated across files.")
    st.divider()
    st.markdown("**Business metadata**")
    tag_raw = st.text_input(
        "Filter by tags", placeholder="matiere:cuir, demande:reparation",
        help="Comma-separated. Convention: matiere: / produit: / demande: / cible:. "
             "A chunk must carry ALL listed tags.",
    )
    tag_filter = [t.strip().lower() for t in tag_raw.split(",") if t.strip()]

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown(
    f'<div style="font-size:.72rem;letter-spacing:.22em;text-transform:uppercase;'
    f'color:#8f8b86;">Retrieval pipeline</div>'
    f'<div style="width:44px;height:2px;background:{_ACCENT};margin:8px 0 18px 0;"></div>',
    unsafe_allow_html=True,
)

query = st.text_input("Question", placeholder="e.g. comment entretenir un sac en cuir Togo ?")
go = st.button("Search", type="primary")

# ── Rendering helpers ─────────────────────────────────────────────────────────

def _chunk_card(c: dict, *, tag: str = "") -> None:
    score = c.get("score", 0.0)
    fname = c.get("file_name", "?")
    idx = c.get("_index", c.get("index_name", ""))
    dup = c.get("duplicate_count", 1)
    rd, rs = c.get("rank_dense", 0), c.get("rank_sparse", 0)
    body = (c.get("content", "") or "")[:280]

    meta_bits = []
    if rd or rs:
        meta_bits.append(f"dense#{rd or '—'} · sparse#{rs or '—'}")
    if dup > 1:
        meta_bits.append(f"×{dup} dupes merged")
    if tag:
        meta_bits.append(tag)
    meta = "  |  ".join(meta_bits)

    tags = c.get("tags") or []
    tags_html = ""
    if tags:
        chips = "".join(
            f'<span style="background:{_ACCENT}14;color:{_ACCENT};font-size:.62rem;'
            f'padding:1px 6px;border-radius:3px;margin-right:4px;">{t}</span>'
            for t in tags[:6]
        )
        tags_html = f'<div style="margin:6px 0 2px;">{chips}</div>'

    st.markdown(
        f"""
        <div style="border:1px solid #e9e5e1;border-left:3px solid {_ACCENT};
                    padding:10px 12px;margin-bottom:8px;">
          <div style="display:flex;justify-content:space-between;align-items:baseline;">
            <span style="font-weight:600;font-size:.8rem;">{fname}</span>
            <span style="font-family:monospace;color:{_ACCENT};font-size:.82rem;">
              {score:.3f}</span>
          </div>
          <div style="font-size:.68rem;color:#9a958f;letter-spacing:.04em;margin:2px 0 6px;">
            {idx}{('  ·  ' + meta) if meta else ''}</div>
          {tags_html}
          <div style="font-size:.82rem;color:#55504b;line-height:1.5;">{body}…</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Run ───────────────────────────────────────────────────────────────────────

if go:
    if not query.strip():
        st.warning("Enter a question.")
        st.stop()
    if not picked:
        st.warning("Pick at least one index.")
        st.stop()

    from hybrid.retrieval.dense import dense_search
    from hybrid.retrieval.sparse import sparse_search
    from hybrid.retrieval.fusion import reciprocal_rank_fusion, deduplicate_by_content
    from hybrid.config import RRF_K

    store, embedder = _engine()

    with st.spinner("Retrieving…"):
        q_emb = embedder.embed_query(query)
        dense_all, sparse_all = [], []
        for idx in picked:
            for c in dense_search(query, store, embedder, index_name=idx,
                                  top_k=top_k * 2, query_embedding=q_emb):
                c["_index"] = idx
                dense_all.append(c)
            for c in sparse_search(query, store, index_name=idx, top_k=top_k * 2):
                c["_index"] = idx
                sparse_all.append(c)

        dense_all.sort(key=lambda c: c.get("score", 0), reverse=True)
        sparse_all.sort(key=lambda c: c.get("score", 0), reverse=True)

        # Business-metadata filter (tags convention) — applied before fusion so
        # the "improved by metadata" step is visible.
        if tag_filter:
            from hybrid.retrieval.filter import apply_post_filter
            f = {"tags": tag_filter}
            dense_all = apply_post_filter(dense_all, f)
            sparse_all = apply_post_filter(sparse_all, f)

        hybrid = reciprocal_rank_fusion(
            dense_all, sparse_all, k=RRF_K,
            dense_weight=dense_w, sparse_weight=sparse_w, dense_gated=dense_gated,
        )
        if dedup:
            hybrid = deduplicate_by_content(hybrid)

    dense_ids = {c.get("id") for c in dense_all}
    sparse_ids = {c.get("id") for c in sparse_all}

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown("##### Dense (vector)")
        st.caption("Semantic similarity — paraphrase-friendly, misses exact tokens.")
        for c in dense_all[:top_k]:
            _chunk_card(c)
    with c2:
        st.markdown("##### Sparse (BM25)")
        st.caption("Exact keywords — references, serials, part names.")
        for c in sparse_all[:top_k]:
            only = "sparse-only" if c.get("id") not in dense_ids else ""
            _chunk_card(c, tag=only)
    with c3:
        st.markdown("##### Hybrid (RRF)")
        gate = "gated" if dense_gated else "open"
        st.caption(f"Rank fusion · {gate}{' · deduped' if dedup else ''}")
        for c in hybrid[:top_k]:
            src = ""
            if c.get("id") in sparse_ids and c.get("id") not in dense_ids:
                src = "rescued by BM25"
            _chunk_card(c, tag=src)

    # Teaching callout: what dense-gating cost / saved
    rescued = [c for c in hybrid[:top_k]
               if c.get("id") in sparse_ids and c.get("id") not in dense_ids]
    st.divider()
    if dense_gated:
        st.info(
            "Dense-gating is **ON** — any exact-keyword chunk the vector search "
            "missed was dropped. Toggle it off to see what BM25 would rescue.",
            icon="🔒",
        )
    elif rescued:
        st.success(
            f"Dense-gating **OFF** — {len(rescued)} chunk(s) here were found only "
            "by BM25 and would vanish if gated. This is why exact-match hybrid matters.",
            icon="🔓",
        )
