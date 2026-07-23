"""
Brand header component — logo + styled wordmark, driven by the active profile.

- ``brand_logo()``   : registers the profile logo via st.logo (top-left + sidebar)
                       when the asset file exists. No-op otherwise.
- ``brand_header()`` : a styled title block (wordmark + accent rule + subtitle)
                       for the Login and Home pages.

The logo asset itself is NOT shipped here — set ``logo_path`` per profile in
shared/brand.py and drop the official file at that path. With no file, a clean
text wordmark is shown instead.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from shared.brand import ACTIVE

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _logo_file() -> Path | None:
    if not ACTIVE.logo_path:
        return None
    p = Path(ACTIVE.logo_path)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    return p if p.exists() else None


def brand_logo() -> None:
    """Register the brand logo with Streamlit (top-left header + sidebar).

    Call once per run, e.g. in app.py before navigation. Safe no-op when the
    asset file is missing.
    """
    f = _logo_file()
    if f is not None:
        try:
            st.logo(str(f))
        except Exception:
            pass


def brand_header(centered: bool = False) -> None:
    """Render the brand header.

    With a logo asset: the logo is shown centered and the text wordmark is
    dropped (the logo already carries the name). Without one: a styled
    uppercase wordmark is used instead, aligned per *centered*.
    """
    accent = ACTIVE.theme.accent
    name = ACTIVE.name
    subtitle = ACTIVE.subtitle
    logo = _logo_file()

    # A logo always centers the header; the wordmark honours the caller.
    align = "center" if (centered or logo is not None) else "left"
    rule_margin = "0 auto" if align == "center" else "0"

    if logo is not None:
        import base64

        data = base64.b64encode(logo.read_bytes()).decode("ascii")
        ext = logo.suffix.lstrip(".").lower()
        mime = (
            "image/svg+xml" if ext == "svg"
            else f"image/{'jpeg' if ext in ('jpg', 'jpeg') else ext}"
        )
        mark_html = (
            f'<img src="data:{mime};base64,{data}" alt="{name}" '
            f'style="height:64px;display:block;margin:0 auto 4px auto;">'
        )
    else:
        mark_html = (
            '<div style="font-size:2.1rem; font-weight:700; letter-spacing:0.14em;'
            ' text-transform:uppercase; line-height:1.1; color:inherit;">'
            f'{name}</div>'
        )

    st.markdown(
        f"""
        <div style="text-align:{align}; padding: 24px 0 8px 0;">
          {mark_html}
          <div style="width:64px; height:3px; background:{accent}; margin:{rule_margin};
                      margin-top:14px; margin-bottom:10px; border-radius:2px;"></div>
          <div style="font-size:1.02rem; color:#6b6b6b; letter-spacing:0.02em;">
            {subtitle}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
