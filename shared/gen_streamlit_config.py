"""
Generate .streamlit/config.toml from the ACTIVE brand profile.

Run this BEFORE `streamlit run` (e.g. as a systemd ExecStartPre) so the native
Streamlit chrome (primary color, backgrounds, text) follows the brand selected
by BRAND_PROFILE — dynamically, with no per-brand values hardcoded anywhere but
shared/brand.py.

    python -m shared.gen_streamlit_config

Writes <project_root>/.streamlit/config.toml. Only the [theme] section is
generated; server flags stay on the CLI (--server.port, --server.address).
This file is generated (gitignored) — never edit it by hand.
"""

from __future__ import annotations

from pathlib import Path

from shared.brand import ACTIVE

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _PROJECT_ROOT / ".streamlit" / "config.toml"


def _theme_toml() -> str:
    t = ACTIVE.theme
    secondary = t.secondary_background or t.accent_light

    lines = [
        "# GENERATED from shared/brand.py — do not edit by hand.",
        f"# Brand profile: {ACTIVE.name}",
        "[theme]",
        f'base = "{t.base}"',
        f'primaryColor = "{t.accent}"',
        f'backgroundColor = "{t.background}"',
        f'secondaryBackgroundColor = "{secondary}"',
        f'textColor = "{t.text}"',
        # A profile that names real faces wins over the generic keyword.
        # TOML *literal* strings (single quotes) — a font stack contains double
        # quotes around multi-word family names, which would break a basic
        # double-quoted string.
        f"font = '{t.font_body or t.font}'",
    ]
    if t.font_heading:
        lines.append(f"headingFont = '{t.font_heading}'")
    if t.font_code:
        lines.append(f"codeFont = '{t.font_code}'")
    if t.base_font_size:
        lines.append(f"baseFontSize = {t.base_font_size}")
    return "\n".join(lines) + "\n"


def main() -> int:
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG_PATH.write_text(_theme_toml(), encoding="utf-8")
    print(
        f"[gen_streamlit_config] {ACTIVE.name} -> {_CONFIG_PATH} "
        f"(primary {ACTIVE.theme.accent})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
