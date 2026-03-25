"""
Source card component.

Renders a compact list of source documents / chunks returned by either
the Vertex or Hybrid pipeline.
"""

from __future__ import annotations

import streamlit as st


def render_sources(sources: list[dict], pipeline: str = "hybrid") -> None:
    """
    Render a "Sources" section with linked document names.

    Args:
        sources:  List of source dicts.
                  Vertex shape: {"title": str, "uri": str}
                  Hybrid shape: {"file_name": str, "source_url": str,
                                 "file_type": str, "domaine": str, "langue": str}
        pipeline: "vertex" or "hybrid" — controls which keys to read.
    """
    if not sources:
        st.caption("_Aucune source disponible._")
        return

    st.markdown("**Sources**")
    for s in sources:
        if pipeline == "vertex":
            name = s.get("title") or "Document"
            url  = s.get("uri", "")
        else:
            name = s.get("file_name") or "Document"
            url  = s.get("source_url", "")
            domain = s.get("domaine", "")
            lang   = s.get("langue", "")
            meta   = " · ".join(filter(None, [domain, lang]))
            if meta:
                name = f"{name} _{meta}_"

        if url:
            st.markdown(f"- [{name}]({url})")
        else:
            st.markdown(f"- {name}")


def render_chunks(chunks: list[dict], collapsed: bool = True) -> None:
    """
    Render individual retrieved chunks with scores and metadata.

    Args:
        chunks:    List of chunk dicts from hybrid_rag_query.
        collapsed: Whether to start each chunk expander closed.
    """
    if not chunks:
        return

    st.markdown("**Chunks récupérés**")
    for i, chunk in enumerate(chunks):
        score        = chunk.get("rrf_score") or chunk.get("score", 0)
        score_dense  = chunk.get("score_dense")
        score_sparse = chunk.get("score_sparse")
        file_name    = chunk.get("file_name", f"chunk_{i+1}")
        content      = chunk.get("content", "")
        label        = f"#{i+1} — {file_name}  `RRF: {score:.4f}`"

        with st.expander(label, expanded=not collapsed):
            col_l, col_r = st.columns([3, 1])
            with col_l:
                st.text(content[:800] + ("…" if len(content) > 800 else ""))
            with col_r:
                st.caption(f"**Domaine** : {chunk.get('domaine', '—')}")
                st.caption(f"**Langue**  : {chunk.get('langue', '—')}")
                st.caption(f"**Type**    : {chunk.get('file_type', '—')}")
                st.caption(f"**RRF**     : {score:.4f}")
                if score_dense is not None:
                    st.caption(f"**Cosinus** : {score_dense:.4f}")
                if score_sparse is not None:
                    st.caption(f"**BM25**    : {score_sparse:.4f}")
                if chunk.get("source_url"):
                    st.markdown(f"[Ouvrir]({chunk['source_url']})")
