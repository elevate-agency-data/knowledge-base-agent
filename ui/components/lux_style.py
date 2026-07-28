"""
Shared visual language for the app — one stylesheet, driven by the brand profile.

Design brief this answers
-------------------------
The product is not a boutique window: it is the instrument a client advisor and
a workshop artisan use to answer for a client's piece. So the reference artifact
is the **fiche d'atelier** — the intake sheet filled in when a piece arrives —
not a luxury landing page.

Two structural devices carry the identity, both taken from the Maison's own
craft rather than from generic "luxury" styling:

``point sellier`` (saddle stitch)
    The two-needle hand stitch, rendered as slanted dashes. It replaces the
    plain accent rule, and it is used where something is genuinely *seamed*:
    an answer to its sources, a photo to its assessment. It is the one place
    boldness is spent.

``poinçon`` (blind stamp)
    The letterpressed date mark. Small, letter-spaced, monospaced, colourless —
    used for references, index names and scores, i.e. things that were *stamped*
    rather than written.

Everything else stays quiet: hairlines, generous whitespace, the accent used as
a signal and never as a wash.

Colours all come from ``ACTIVE.theme``, exposed as CSS custom properties, so
Activate / Indica / Lacoste inherit the same structure in their own palette.

Usage — call once per page, before rendering content::

    from components.lux_style import inject_lux_style
    inject_lux_style()
"""

from __future__ import annotations

import streamlit as st


_MONO = ('ui-monospace, "SF Mono", "JetBrains Mono", "IBM Plex Mono", '
         'Consolas, monospace')


def _css() -> str:
    from shared.brand import ACTIVE

    t = ACTIVE.theme
    accent = t.accent
    cuir = t.hybrid          # leather/bronze — the quiet secondary
    surface = t.user         # cream card surface
    # The brand's own faces when it names them; otherwise the generic stacks.
    mono = t.font_code or _MONO
    serif = t.font_body or "Georgia, serif"
    return f"""
    <style>
      :root {{
        --lux-accent:  {accent};
        --lux-cuir:    {cuir};
        --lux-ink:     #1C1917;
        --lux-surface: {surface};
        --lux-hair:    #E7E0D6;
        --lux-meta:    #8F8B86;
        --lux-body:    #55504B;
        --lux-mono:    {mono};
        --lux-serif:   {serif};
      }}

      /* ── Signature: point sellier ───────────────────────────────────────
         A saddle stitch, not a dashed border: slanted 2px dashes on a 9px
         pitch. Horizontal rule and vertical seam variants.                */
      .lux-stitch {{
        height: 5px;
        background-image: repeating-linear-gradient(
          58deg, var(--lux-accent) 0 2px, transparent 2px 6px);
        margin: 0 0 24px 0;
        width: 62px;
        opacity: .7;
      }}
      .lux-stitch--wide {{ width: 100%; opacity: .55; }}

      /* Vertical seam — used on blocks that are stitched to their sources. */
      .lux-seam {{
        padding-left: 18px;
        background-image: repeating-linear-gradient(
          148deg, var(--lux-accent) 0 2px, transparent 2px 5px);
        background-size: 7px 100%;
        background-repeat: repeat-y;
        background-position: left top;
      }}

      /* ── Signature: poinçon ─────────────────────────────────────────────
         A stamped mark. Reserved for references, index names, scores.      */
      .lux-poincon {{
        display: inline-block;
        font-family: var(--lux-mono);
        font-size: .62rem;
        letter-spacing: .14em;
        text-transform: uppercase;
        color: var(--lux-meta);
        border: 1px solid var(--lux-hair);
        border-radius: 2px;
        padding: 1px 7px;
        white-space: nowrap;
      }}
      .lux-poincon--accent {{
        color: var(--lux-accent);
        border-color: var(--lux-accent);
      }}
      .lux-poincon--solid {{
        color: #fff;
        background: var(--lux-cuir);
        border-color: var(--lux-cuir);
      }}

      /* ── Editorial primitives (shared by every page) ────────────────── */
      /* Micro-labels are set in the mark face, not the reading face: they are
         stamped captions, not prose. */
      .lux-eyebrow {{
        font-family: var(--lux-mono);
        font-size: .64rem; letter-spacing: .24em; text-transform: uppercase;
        color: var(--lux-meta); margin: 0 0 10px 0;
      }}
      .lux-h2 {{
        font-size: 1.85rem; font-weight: 400; line-height: 1.3;
        margin: 0 0 16px 0; letter-spacing: .01em; color: var(--lux-ink);
      }}
      .lux-lead {{
        font-size: 1.02rem; line-height: 1.85; color: var(--lux-body);
        max-width: 68ch;
      }}
      /* Kept for pages written against the earlier rule; now a stitch. */
      .lux-rule {{
        width: 62px; height: 5px;
        background-image: repeating-linear-gradient(
          58deg, var(--lux-accent) 0 2px, transparent 2px 6px);
        margin: 0 0 24px 0;
        opacity: .7;
      }}
      .lux-card {{
        border: 1px solid var(--lux-hair); padding: 28px 24px; height: 100%;
        background: #fff;
      }}
      .lux-card-title {{
        font-size: .76rem; letter-spacing: .18em; text-transform: uppercase;
        margin: 0 0 12px 0; color: var(--lux-ink);
      }}
      .lux-card-rule {{
        width: 44px; height: 5px;
        background-image: repeating-linear-gradient(
          58deg, var(--lux-accent) 0 2px, transparent 2px 6px);
        margin: 0 0 16px 0;
        opacity: .7;
      }}
      .lux-card-body {{
        font-size: .90rem; line-height: 1.75; color: #6a655f; margin: 0;
      }}
      .lux-metric-value {{
        font-size: 2.7rem; font-weight: 400; line-height: 1;
        letter-spacing: .01em; color: var(--lux-ink);
      }}
      .lux-metric-label {{
        font-size: .68rem; letter-spacing: .20em; text-transform: uppercase;
        color: var(--lux-meta); margin-top: 10px;
      }}
      .lux-row {{
        display: flex; justify-content: space-between; align-items: baseline;
        border-bottom: 1px solid #f0edea; padding: 11px 0;
      }}
      .lux-row-name {{ font-size: .92rem; letter-spacing: .02em; }}
      .lux-row-meta {{
        font-size: .76rem; color: var(--lux-meta); letter-spacing: .06em;
      }}
      .lux-space {{ height: 60px; }}
      .lux-space-sm {{ height: 30px; }}

      /* ── Streamlit chrome ───────────────────────────────────────────────
         Targeted through data-testid attributes only, and purely cosmetic:
         if Streamlit renames one, the page still lays out correctly.       */

      /* Chat turns become sheet entries. The default avatars are cartoon
         glyphs — nothing about a Maison's workshop reads that way, so they are
         removed entirely and the two voices are told apart by surface and
         indentation instead. */
      [data-testid="stChatMessageAvatarUser"],
      [data-testid="stChatMessageAvatarAssistant"] {{
        display: none !important;
      }}
      [data-testid="stChatMessage"] {{
        background: transparent;
        border: 0;
        border-top: 1px solid var(--lux-hair);
        border-radius: 0;
        padding: 24px 0 12px 0;
        gap: 0;
      }}
      [data-testid="stChatMessage"]:first-of-type {{ border-top: 0; }}
      /* The advisor's own turn: cream surface, seamed on its left edge. */
      [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{
        background: var(--lux-surface);
        border-top: 0;
        padding: 18px 22px 6px 22px;
        margin: 26px 0 4px 0;
        background-image: repeating-linear-gradient(
          148deg, var(--lux-accent) 0 2px, transparent 2px 5px);
        background-size: 7px 100%;
        background-repeat: repeat-y;
        background-position: left top;
        padding-left: 30px;
      }}

      /* Expanders read as folded appendices, not widgets. */
      [data-testid="stExpander"] details {{
        border: 1px solid var(--lux-hair);
        border-radius: 0;
        background: #fff;
      }}
      [data-testid="stExpander"] summary {{
        font-size: .74rem;
        letter-spacing: .12em;
        text-transform: uppercase;
        color: var(--lux-meta);
      }}
      [data-testid="stExpander"] summary:hover {{ color: var(--lux-accent); }}

      /* Buttons: squared, hairline, accent only on intent. */
      .stButton > button {{
        border-radius: 0;
        border: 1px solid var(--lux-hair);
        letter-spacing: .08em;
        font-size: .80rem;
        transition: border-color .18s ease, color .18s ease;
      }}
      .stButton > button:hover {{
        border-color: var(--lux-accent);
        color: var(--lux-accent);
      }}
      .stButton > button[kind="primary"] {{
        background: var(--lux-ink);
        border-color: var(--lux-ink);
        color: #fff;
      }}
      .stButton > button[kind="primary"]:hover {{
        background: var(--lux-accent);
        border-color: var(--lux-accent);
        color: #fff;
      }}

      /* ── Sidebar: a ledger margin, not a control panel ──────────────────
         Warm paper rather than Streamlit's cold grey, hairline rules instead
         of heavy dividers, and navigation set as an index — small caps, wide
         tracking, the active entry marked by a stitch rather than a fill.   */
      [data-testid="stSidebar"] {{
        background: #FBF9F6;
        border-right: 1px solid var(--lux-hair);
      }}
      [data-testid="stSidebar"] > div {{ padding-top: 8px; }}
      [data-testid="stSidebar"] hr {{
        border-color: var(--lux-hair);
        margin: 18px 0;
      }}

      /* Navigation entries */
      [data-testid="stSidebar"] [data-testid="stPageLink"] a,
      [data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"] {{
        border-radius: 0;
        padding: 7px 10px;
        font-size: .82rem;
        letter-spacing: .06em;
        color: var(--lux-body);
        border-left: 2px solid transparent;
        transition: border-color .18s ease, color .18s ease;
      }}
      [data-testid="stSidebar"] [data-testid="stPageLink"] a:hover {{
        background: transparent;
        color: var(--lux-ink);
        border-left-color: var(--lux-hair);
      }}
      /* Streamlit marks the current page link with aria-current */
      [data-testid="stSidebar"] a[aria-current] {{
        background: transparent !important;
        color: var(--lux-ink) !important;
        border-left-color: var(--lux-accent);
        font-weight: 600;
      }}

      /* Sidebar buttons sit quieter than in the main column */
      [data-testid="stSidebar"] .stButton > button {{
        background: transparent;
        font-size: .78rem;
        padding: 6px 10px;
      }}
      [data-testid="stSidebar"] .stButton > button[kind="primary"] {{
        background: var(--lux-ink);
      }}
      /* Captions and code samples in the margin stay discreet */
      [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
        color: var(--lux-meta);
      }}
      [data-testid="stSidebar"] code {{
        background: transparent;
        color: var(--lux-meta);
        font-size: .70rem;
        letter-spacing: .04em;
      }}
      [data-testid="stSidebar"] [data-testid="stCode"] {{
        background: transparent;
        border: 1px solid var(--lux-hair);
        border-radius: 0;
      }}

      /* Composer: squared, hairline, accent on focus. The attachment control
         lives inside it, so its button is toned down to match the send arrow
         rather than competing with the text. */
      [data-testid="stChatInput"] {{
        border-radius: 0;
        border: 1px solid var(--lux-hair);
        background: #fff;
      }}
      [data-testid="stChatInput"]:focus-within {{
        border-color: var(--lux-accent);
      }}
      [data-testid="stChatInput"] button {{
        border-radius: 0;
        color: var(--lux-meta);
      }}
      [data-testid="stChatInput"] button:hover {{
        color: var(--lux-accent);
        background: transparent;
      }}
      /* Attached-file chip shown above the composer */
      [data-testid="stChatInputFileName"],
      [data-testid="stChatInputFile"] {{
        font-family: var(--lux-mono);
        font-size: .68rem;
        letter-spacing: .06em;
        color: var(--lux-meta);
      }}
      .stTextInput input, .stTextArea textarea {{
        border-radius: 0;
      }}

      /* Quality floor: keyboard focus stays visible everywhere. */
      :focus-visible {{
        outline: 2px solid var(--lux-accent);
        outline-offset: 2px;
      }}

      @media (prefers-reduced-motion: reduce) {{
        * {{ transition: none !important; animation: none !important; }}
      }}
    </style>
    """


def inject_lux_style() -> None:
    """Inject the shared stylesheet. Safe to call once per page render."""
    st.markdown(_css(), unsafe_allow_html=True)


def stitch(wide: bool = False) -> None:
    """Render a saddle-stitch rule."""
    cls = "lux-stitch lux-stitch--wide" if wide else "lux-stitch"
    st.markdown(f'<div class="{cls}"></div>', unsafe_allow_html=True)


def poincon(label: str, variant: str = "") -> str:
    """Return the HTML for a stamped mark (to embed in a larger block).

    Args:
        label:   Text of the mark.
        variant: ``""``, ``"accent"`` or ``"solid"``.
    """
    suffix = f" lux-poincon--{variant}" if variant else ""
    return f'<span class="lux-poincon{suffix}">{label}</span>'


def eyebrow(text: str, with_stitch: bool = True) -> None:
    """Render a section eyebrow, optionally followed by a stitch rule."""
    html = f'<div class="lux-eyebrow">{text}</div>'
    if with_stitch:
        html += '<div class="lux-stitch"></div>'
    st.markdown(html, unsafe_allow_html=True)
