"""
Fiche d'atelier — how a photo assessment is presented.

This is the signature moment of the interface. When the agent analyses a photo,
the result is not paraphrased into a paragraph: it is laid out as the sheet an
artisan fills at intake — the piece identified at the top, the observations
listed underneath with their severity, then the verdict per component.

The layout encodes the reading order of a real intake: *what is it → what is
wrong → what can be saved → what this photo cannot settle*. Severity is carried
by a saddle-stitch of varying weight rather than by a coloured pill, so the one
accent stays a signal.

Rendered from the raw ``sav_analyze_image`` tool response, so what you see is
what the model actually returned — no field is invented at display time.
"""

from __future__ import annotations

import html

import streamlit as st

# Severity → (label, stitch pitch). A tighter stitch reads as denser work.
_SEV = {
    "high":   ("majeur", 4),
    "medium": ("notable", 7),
    "low":    ("léger", 11),
}

_VERDICT_VARIANT = {
    "réutilisable": "solid",
    "à remplacer": "accent",
}


def _esc(v) -> str:
    return html.escape(str(v or ""))


def _stitch_span(pitch: int) -> str:
    """A short saddle-stitch used as a severity gauge."""
    return (
        f'<span style="display:inline-block;width:46px;height:7px;'
        f'vertical-align:middle;background-image:repeating-linear-gradient('
        f'58deg,var(--lux-accent) 0 2px,transparent 2px {pitch}px);"></span>'
    )


def render_sav_sheet(data: dict) -> None:
    """Render one ``sav_analyze_image`` response as an atelier intake sheet.

    Args:
        data: The tool response dict. A non-success payload renders its message
              instead, so a refusal stays visible rather than silently empty.
    """
    if not isinstance(data, dict):
        return
    if data.get("status") != "success":
        st.warning(data.get("message", "Analyse indisponible."))
        return

    product = data.get("product") or {}
    condition = data.get("condition") or {}
    damages = data.get("damages") or []
    salvageable = data.get("salvageable") or []
    limitations = data.get("limitations") or []

    conf = product.get("confidence", "")
    grade = condition.get("grade", "")

    # ── Header : the piece ────────────────────────────────────────────────────
    marks = []
    if product.get("line"):
        marks.append(f'<span class="lux-poincon">{_esc(product["line"])}</span>')
    if grade:
        marks.append(f'<span class="lux-poincon lux-poincon--accent">'
                     f'état : {_esc(grade)}</span>')
    if conf:
        marks.append(f'<span class="lux-poincon">confiance {_esc(conf)}</span>')

    ident = _esc(product.get("identification") or "Pièce non identifiée")
    model = product.get("model_guess") or ""

    spec_rows = []
    for label, value in (
        ("Matières", ", ".join(product.get("materials") or [])),
        ("Garniture", product.get("hardware")),
        ("Couleur", product.get("colour")),
        ("Modèle", model or "non déterminé d'après la photo"),
    ):
        if value:
            spec_rows.append(
                f'<div class="lux-row"><span class="lux-row-meta">{label}</span>'
                f'<span class="lux-row-name">{_esc(value)}</span></div>'
            )

    st.markdown(
        f"""
        <div style="border:1px solid var(--lux-hair);background:#fff;
                    padding:22px 24px 8px 24px;">
          <div class="lux-eyebrow">Fiche d'atelier — lecture de la photo</div>
          <div class="lux-stitch"></div>
          <div style="font-size:1.25rem;line-height:1.35;color:var(--lux-ink);
                      margin-bottom:10px;">{ident}</div>
          <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:16px;">
            {''.join(marks)}
          </div>
          {''.join(spec_rows)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if condition.get("overall"):
        st.markdown(
            f'<div class="lux-seam" style="margin:16px 0 4px 0;font-size:.95rem;'
            f'line-height:1.7;color:var(--lux-body);">'
            f'{_esc(condition["overall"])}</div>',
            unsafe_allow_html=True,
        )

    # ── Observations ──────────────────────────────────────────────────────────
    if damages:
        st.markdown(
            f'<div style="margin-top:26px;"><div class="lux-eyebrow">'
            f'Observations ({len(damages)})</div>'
            f'<div class="lux-stitch"></div></div>',
            unsafe_allow_html=True,
        )
        rows = []
        for d in damages:
            label, pitch = _SEV.get(d.get("severity", "medium"), _SEV["medium"])
            rows.append(
                f"""
                <div style="border-bottom:1px solid #f0edea;padding:13px 0;">
                  <div style="display:flex;justify-content:space-between;
                              align-items:baseline;gap:12px;">
                    <span style="font-size:.98rem;color:var(--lux-ink);">
                      {_esc(d.get('type'))}</span>
                    <span style="white-space:nowrap;">
                      {_stitch_span(pitch)}
                      <span class="lux-poincon" style="margin-left:8px;">
                        {label}</span></span>
                  </div>
                  <div style="font-size:.76rem;color:var(--lux-meta);
                              letter-spacing:.05em;margin:4px 0 6px;">
                    {_esc(d.get('location'))} · prise en charge :
                    {_esc(d.get('repairable'))}</div>
                  <div style="font-size:.90rem;line-height:1.65;
                              color:var(--lux-body);">
                    {_esc(d.get('observation'))}</div>
                </div>
                """
            )
        st.markdown("".join(rows), unsafe_allow_html=True)

    # ── Verdict per component ─────────────────────────────────────────────────
    if salvageable:
        st.markdown(
            f'<div style="margin-top:26px;"><div class="lux-eyebrow">'
            f'Éléments — ce qui se garde</div>'
            f'<div class="lux-stitch"></div></div>',
            unsafe_allow_html=True,
        )
        rows = []
        for s in salvageable:
            verdict = s.get("verdict", "")
            variant = _VERDICT_VARIANT.get(verdict, "")
            suffix = f" lux-poincon--{variant}" if variant else ""
            rows.append(
                f"""
                <div style="border-bottom:1px solid #f0edea;padding:11px 0;
                            display:flex;justify-content:space-between;
                            align-items:baseline;gap:14px;">
                  <span>
                    <span style="font-size:.95rem;color:var(--lux-ink);">
                      {_esc(s.get('component'))}</span>
                    <div style="font-size:.82rem;color:var(--lux-body);
                                line-height:1.6;margin-top:2px;">
                      {_esc(s.get('note'))}</div>
                  </span>
                  <span class="lux-poincon{suffix}">{_esc(verdict)}</span>
                </div>
                """
            )
        st.markdown("".join(rows), unsafe_allow_html=True)

    # ── What the photo cannot settle ──────────────────────────────────────────
    if limitations:
        items = "".join(
            f'<li style="margin-bottom:5px;">{_esc(l)}</li>' for l in limitations
        )
        st.markdown(
            f"""
            <div style="margin-top:26px;background:var(--lux-surface);
                        padding:16px 20px;">
              <div class="lux-eyebrow">Ce que cette photo ne permet pas d'établir</div>
              <ul style="margin:8px 0 0 0;padding-left:18px;font-size:.86rem;
                         line-height:1.6;color:var(--lux-body);">{items}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


def extract_sav_responses(tool_events: list[dict] | None) -> list[dict]:
    """Pull every ``sav_analyze_image`` response out of a tool-event list."""
    out = []
    for ev in tool_events or []:
        if ev.get("type") == "tool_resp" and ev.get("name") == "sav_analyze_image":
            resp = ev.get("response")
            if isinstance(resp, dict):
                out.append(resp)
    return out
