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
    """All index names, sorted — the registry returns them in insertion order,
    which makes any positional default arbitrary."""
    from services.hybrid_service import list_indexes
    return sorted(
        i.get("index_name", "") for i in list_indexes() if i.get("index_name")
    )


# ── Sidebar controls ──────────────────────────────────────────────────────────

with st.sidebar:
    render_sidebar_user()
    st.divider()
    st.markdown(
        '<div class="lux-eyebrow" style="margin-bottom:8px;">Réglages</div>',
        unsafe_allow_html=True,
    )

    all_idx = _indexes()
    # Every index by default. Searching a single arbitrary one made BM25 look
    # broken: an exact reference lives in one index, so querying any other
    # returns nothing — while dense always returns its k nearest, however far.
    # That is the very comparison this page exists to show, so it must not be
    # sabotaged by the default selection.
    picked = st.multiselect("Domaines cherchés", all_idx, default=all_idx)
    top_k = st.slider("Résultats par colonne", 3, 20, 8)
    dense_w = st.slider("Poids du sens", 0.0, 1.0, 0.7, 0.05)
    sparse_w = round(1.0 - dense_w, 2)
    st.caption(f"Poids des mots exacts = {sparse_w}")
    dense_gated = st.toggle(
        "Ne garder que le sens", value=False,
        help="Activé : seuls les passages trouvés par la recherche de sens "
             "sont retenus. Les correspondances de mots exacts que le sens a "
             "manquées sont écartées.",
    )
    dedup = st.toggle("Fusionner les doublons", value=True,
                      help="Regroupe les passages identiques présents dans "
                           "plusieurs fichiers.")
    st.divider()
    st.markdown(
        '<div class="lux-eyebrow" style="margin-bottom:8px;">Métadonnées métier</div>',
        unsafe_allow_html=True,
    )
    tag_raw = st.text_input(
        "Filtrer par étiquette", placeholder="matiere:cuir, demande:reparation",
        help="Séparées par des virgules. Convention : matiere: / produit: / "
             "demande: / cible:. Un passage doit porter TOUTES les étiquettes.",
    )
    tag_filter = [t.strip().lower() for t in tag_raw.split(",") if t.strip()]

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="lux-eyebrow">Comparer les recherches</div>
    <div class="lux-stitch"></div>
    <div class="lux-h2" style="margin-bottom:12px;">Une question, trois façons de chercher</div>
    <p class="lux-lead" style="font-size:.92rem;">
      La même question est posée de trois manières : par le <b>sens</b>, par les
      <b>mots exacts</b>, puis en <b>combinant les deux</b>. Comparez ce que
      chacune remonte — c'est ce qui explique pourquoi l'assistant les associe.
    </p>
    """,
    unsafe_allow_html=True,
)

query = st.text_input(
    "Votre question",
    placeholder="ex. comment entretenir un sac en cuir Togo ?",
)
go = st.button("Rechercher", type="primary")

# ── Rendering helpers ─────────────────────────────────────────────────────────

def _fmt_date(v) -> str:
    """Best-effort short date from a DuckDB DATE / datetime / str."""
    if not v:
        return ""
    s = str(v)
    return s[:10]  # YYYY-MM-DD


def _chunk_card(c: dict, *, tag: str = "") -> None:
    score = c.get("score", 0.0)
    fname = c.get("file_name") or "?"
    idx = c.get("_index", c.get("index_name", ""))
    dup = c.get("duplicate_count", 1)
    rd, rs = c.get("rank_dense", 0), c.get("rank_sparse", 0)
    body = (c.get("content", "") or "")[:280]

    # retrieval-provenance line (rank / dedup / rescue tag)
    prov = []
    if rd or rs:
        prov.append(f"dense#{rd or '—'} · sparse#{rs or '—'}")
    if dup > 1:
        prov.append(f"×{dup} dupes merged")
    if tag:
        prov.append(tag)
    prov_html = (
        f'<div style="font-size:.66rem;color:{_ACCENT};letter-spacing:.03em;'
        f'margin:2px 0 6px;">{"  ·  ".join(prov)}</div>' if prov else ""
    )

    # business/document metadata carried on the chunk (real columns from the DB)
    ci, ct = c.get("chunk_index"), c.get("chunk_total")
    edim = c.get("embedding_dim")
    meta_pairs = [
        ("domaine", c.get("domaine")),
        ("type", c.get("file_type")),
        ("langue", c.get("langue")),
        ("auteur", c.get("author")),
        ("chunk", f"{ci}/{ct}" if ci is not None and ct else None),
        ("créé", _fmt_date(c.get("created_at"))),
        ("modèle", c.get("embedding_model")
         + (f" ({edim})" if edim else "") if c.get("embedding_model") else None),
    ]
    meta_rows = "".join(
        f'<div style="display:flex;gap:6px;"><span style="color:#b3aea8;'
        f'min-width:52px;">{k}</span><span style="color:#6a655f;">{v}</span></div>'
        for k, v in meta_pairs if v
    )
    meta_html = (
        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:1px 12px;'
        f'font-size:.66rem;margin:2px 0 6px;">{meta_rows}</div>' if meta_rows else ""
    )

    # source link (Drive / URL) when present
    src = c.get("source_url") or ""
    src_html = (
        f'<a href="{src}" target="_blank" style="font-size:.64rem;color:{_ACCENT};'
        f'text-decoration:none;">↗ source</a>' if src else ""
    )

    # tags — business facet tags (matiere:/demande:/…) get highlighted
    tags = c.get("tags") or []
    tags_html = ""
    if tags:
        chips = "".join(
            f'<span style="background:{_ACCENT}14;color:{_ACCENT};font-size:.6rem;'
            f'padding:1px 6px;border-radius:3px;margin:0 4px 3px 0;'
            f'display:inline-block;">{t}</span>'
            for t in tags[:10]
        )
        tags_html = f'<div style="margin:4px 0 2px;">{chips}</div>'

    st.markdown(
        f"""
        <div style="border:1px solid #e9e5e1;border-left:3px solid {_ACCENT};
                    padding:10px 12px;margin-bottom:8px;">
          <div style="display:flex;justify-content:space-between;align-items:baseline;gap:8px;">
            <span style="font-weight:600;font-size:.8rem;word-break:break-word;">{fname}</span>
            <span style="font-family:monospace;color:{_ACCENT};font-size:.82rem;">
              {score:.3f}</span>
          </div>
          <div style="font-size:.66rem;color:#9a958f;letter-spacing:.04em;margin:2px 0 4px;">
            {idx} {src_html}</div>
          {prov_html}
          {meta_html}
          {tags_html}
          <div style="font-size:.8rem;color:#55504b;line-height:1.5;">{body}…</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Run ───────────────────────────────────────────────────────────────────────

if go:
    if not query.strip():
        st.warning("Saisissez une question.")
        st.stop()
    if not picked:
        st.warning("Sélectionnez au moins un domaine.")
        st.stop()

    from hybrid.retrieval.dense import dense_search
    from hybrid.retrieval.sparse import sparse_search
    from hybrid.retrieval.fusion import reciprocal_rank_fusion, deduplicate_by_content
    from hybrid.config import RRF_K

    store, embedder = _engine()

    with st.spinner("Recherche en cours…"):
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
        st.markdown("##### Par le sens")
        st.caption("Comprend la reformulation, mais peut rater un mot exact.")
        for c in dense_all[:top_k]:
            _chunk_card(c)
    with c2:
        st.markdown("##### Par les mots exacts")
        st.caption("Références, numéros de série, noms de pièces — au mot près.")
        if not sparse_all:
            # An empty BM25 column is a result, not a failure — and it is one of
            # the sharpest lessons this page can teach, so it gets explained
            # rather than left as a blank column next to a full one.
            st.info(
                "**Aucun résultat — et c'est normal.**\n\n"
                "Aucun passage des index sélectionnés ne contient ces termes. "
                "BM25 ne retourne que ce qui contient littéralement les mots "
                "cherchés : pas de terme, pas de résultat.\n\n"
                "La colonne dense, elle, est pleine — elle retourne toujours ses "
                "k plus proches voisins, **même très éloignés**. Une colonne "
                "dense remplie ne veut donc pas dire qu'elle a trouvé quelque "
                "chose de pertinent.\n\n"
                "Élargissez la sélection d'index, ou essayez une référence "
                "exacte présente au catalogue.",
                icon="🔍",
            )
        for c in sparse_all[:top_k]:
            only = "mots exacts seuls" if c.get("id") not in dense_ids else ""
            _chunk_card(c, tag=only)
    with c3:
        st.markdown("##### Les deux combinés")
        gate = "sens seul" if dense_gated else "sens + mots"
        st.caption(f"Fusion des classements · {gate}"
                   f"{' · doublons fusionnés' if dedup else ''}")
        for c in hybrid[:top_k]:
            src = ""
            if c.get("id") in sparse_ids and c.get("id") not in dense_ids:
                src = "rattrapé par les mots exacts"
            _chunk_card(c, tag=src)

    # Teaching callout: what filtering on meaning alone costs / saves
    rescued = [c for c in hybrid[:top_k]
               if c.get("id") in sparse_ids and c.get("id") not in dense_ids]
    st.divider()
    if dense_gated:
        st.info(
            "**Seul le sens compte ici.** Les passages trouvés uniquement par "
            "leurs mots exacts ont été écartés. Désactivez l'option à gauche "
            "pour voir ce qu'ils apportaient.",
        )
    elif rescued:
        st.success(
            f"**{len(rescued)} passage(s) n'ont été trouvés que par leurs mots "
            f"exacts.** Ils disparaîtraient si l'on ne se fiait qu'au sens — "
            f"c'est précisément pour eux que les deux méthodes sont combinées.",
        )
