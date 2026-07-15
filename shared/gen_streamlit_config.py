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
    return (
        "# GENERATED from shared/brand.py — do not edit by hand.\n"
        f"# Brand profile: {ACTIVE.name}\n"
        "[theme]\n"
        f'base = "{t.base}"\n'
        f'primaryColor = "{t.accent}"\n'
        f'backgroundColor = "{t.background}"\n'
        f'secondaryBackgroundColor = "{secondary}"\n'
        f'textColor = "{t.text}"\n'
    )


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
