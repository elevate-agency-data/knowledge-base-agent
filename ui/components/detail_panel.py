"""
Pièces au dossier — the single fold-out behind an answer.

An answer used to be trailed by three separate widgets: one expander per tool
call, one for the sources, one for the retrieved chunks. That reads as debug
output stapled to a reply. Here they become **one** appendix, in the order an
advisor would consult it:

    sources consulted → passages retained → route taken (tools)

The summary line states what is inside before it is opened, so nobody expands
it just to find out whether it was worth expanding.
"""

from __future__ import annotations

import html

import streamlit as st


def _esc(v) -> str:
    return html.escape(str(v or ""))


def _summary(n_sources: int, n_chunks: int, n_tools: int) -> str:
    bits = []
    if n_sources:
        bits.append(f"{n_sources} source{'s' if n_sources > 1 else ''}")
    if n_chunks:
        bits.append(f"{n_chunks} extrait{'s' if n_chunks > 1 else ''}")
    if n_tools:
        bits.append(f"{n_tools} outil{'s' if n_tools > 1 else ''}")
    if not bits:
        return "Pièces au dossier"
    return "Pièces au dossier — " + " · ".join(bits)


def _pair_tool_events(events: list[dict]) -> list[tuple[dict, dict | None]]:
    """Pair each tool_call with its response, preserving call order."""
    pairs: list[tuple[dict, dict | None]] = []
    queue: list[dict] = []
    for ev in events:
        if ev["type"] == "tool_call":
            queue.append(ev)
        elif ev["type"] == "tool_resp" and queue:
            matched = next((c for c in queue if c["name"] == ev["name"]), queue[0])
            queue.remove(matched)
            pairs.append((matched, ev))
    pairs.extend((c, None) for c in queue)
    return pairs


def _render_sources(sources: list[dict]) -> None:
    rows = []
    for s in sources:
        name = s.get("file_name") or s.get("title") or "Document"
        url = s.get("source_url") or s.get("uri") or ""
        meta = " · ".join(filter(None, [s.get("domaine", ""), s.get("langue", "")]))
        label = (f'<a href="{_esc(url)}" target="_blank" '
                 f'style="color:var(--lux-ink);text-decoration:none;'
                 f'border-bottom:1px solid var(--lux-hair);">{_esc(name)}</a>'
                 if url else _esc(name))
        rows.append(
            f'<div class="lux-row"><span class="lux-row-name">{label}</span>'
            f'<span class="lux-row-meta">{_esc(meta)}</span></div>'
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def _render_chunks(chunks: list[dict]) -> None:
    for i, c in enumerate(chunks, start=1):
        score = c.get("rrf_score") or c.get("score", 0)
        marks = [f'<span class="lux-poincon">rrf {score:.4f}</span>']
        if c.get("score_dense") is not None:
            marks.append(f'<span class="lux-poincon">cos {c["score_dense"]:.3f}</span>')
        if c.get("score_sparse") is not None:
            marks.append(f'<span class="lux-poincon">bm25 {c["score_sparse"]:.3f}</span>')
        for key in ("ligne_produit", "zone"):
            if c.get(key):
                marks.append(f'<span class="lux-poincon">{_esc(c[key])}</span>')

        body = (c.get("content", "") or "")[:600]
        st.markdown(
            f"""
            <div style="border-top:1px solid var(--lux-hair);padding:14px 0;">
              <div style="display:flex;justify-content:space-between;
                          align-items:baseline;gap:10px;margin-bottom:6px;">
                <span style="font-size:.88rem;color:var(--lux-ink);">
                  {i:02d} · {_esc(c.get('file_name', 'extrait'))}</span>
              </div>
              <div style="display:flex;gap:5px;flex-wrap:wrap;margin-bottom:8px;">
                {''.join(marks)}
              </div>
              <div class="lux-seam" style="font-size:.86rem;line-height:1.65;
                          color:var(--lux-body);">{_esc(body)}…</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _trim(data, max_len: int = 300):
    """Shorten long values so the raw payload stays readable."""
    if isinstance(data, dict):
        return {k: _trim(v, max_len) for k, v in data.items()}
    if isinstance(data, list):
        return [_trim(v, max_len) for v in data[:12]]
    if isinstance(data, str) and len(data) > max_len:
        return data[:max_len] + f"… [{len(data)} car.]"
    return data


def render_detail_panel(
    tool_events: list[dict] | None,
    sources: list[dict] | None,
    chunks: list[dict] | None,
) -> None:
    """Render the single fold-out appendix under an answer."""
    tool_events = tool_events or []
    sources = sources or []
    chunks = chunks or []
    pairs = _pair_tool_events(tool_events)
    if not (sources or chunks or pairs):
        return

    with st.expander(_summary(len(sources), len(chunks), len(pairs)), expanded=False):
        if sources:
            st.markdown(
                '<div class="lux-eyebrow" style="margin-bottom:4px;">'
                'Documents consultés</div>', unsafe_allow_html=True)
            _render_sources(sources)

        if chunks:
            st.markdown(
                '<div class="lux-eyebrow" style="margin:22px 0 4px;">'
                'Passages retenus</div>', unsafe_allow_html=True)
            _render_chunks(chunks)

        if pairs:
            st.markdown(
                '<div class="lux-eyebrow" style="margin:22px 0 8px;">'
                'Chemin suivi</div>', unsafe_allow_html=True)
            for call, resp in pairs:
                st.markdown(
                    f'<div style="margin-bottom:4px;">'
                    f'<span class="lux-poincon lux-poincon--accent">'
                    f'{_esc(call["name"])}</span></div>',
                    unsafe_allow_html=True,
                )
                cols = st.columns(2, gap="medium")
                with cols[0]:
                    st.caption("Appel")
                    st.json(_trim(call.get("args", {})), expanded=False)
                with cols[1]:
                    st.caption("Retour")
                    if resp:
                        st.json(_trim(resp.get("response", {})), expanded=False)
                    else:
                        st.caption("_(sans réponse)_")
