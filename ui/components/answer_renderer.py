"""
Answer renderer component.

Converts Gemini markdown output to HTML with:
- Clickable source badges (green, linked to Drive)
- Proper table rendering
- Overflow protection

Used by RAG Comparison and Simple Chat pages.
"""

from __future__ import annotations

import re

import streamlit as st

from shared.brand import ACTIVE as _BRAND

# Accent colors follow the active brand profile (see shared/brand.py).
_ACCENT = _BRAND.theme.accent
_ACCENT_LIGHT = _BRAND.theme.accent_light


# ── Source badge highlighting ────────────────────────────────────────────────

def _highlight_sources(text: str, sources: list[dict]) -> str:
    """
    Post-process a generated answer to turn source citations in parentheses
    into highlighted green badges. Handles Gemini's actual output:
    - (INDEX__TOPIC - FileName, detail)
    - (INDEX__TOPIC - FileName ; INDEX__TOPIC - FileName2)
    - (FileName.pdf)
    """
    if not text:
        return text

    # Build lookup: file name → source_url
    source_map: dict[str, str] = {}
    for s in (sources or []):
        fname = s.get("file_name", "")
        url = s.get("source_url", "") or s.get("uri", "")
        if fname and url:
            source_map[fname.lower()] = url
            stem = fname.rsplit(".", 1)[0] if "." in fname else fname
            source_map[stem.lower()] = url

    _BADGE_LINK = (
        f'<a href="{{url}}" target="_blank" '
        f'style="background:{_ACCENT_LIGHT};color:{_ACCENT};padding:2px 8px;'
        f'border-radius:4px;font-size:0.82em;text-decoration:none;'
        f'border:1px solid {_ACCENT}30;white-space:normal;'
        f'margin:0 2px">{{label}}</a>'
    )
    _BADGE_SPAN = (
        f'<span style="background:{_ACCENT_LIGHT};color:{_ACCENT};padding:2px 8px;'
        f'border-radius:4px;font-size:0.82em;'
        f'border:1px solid {_ACCENT}30;white-space:normal;'
        f'margin:0 2px">{{label}}</span>'
    )

    def _find_url(citation: str) -> str:
        parts = citation.split(" - ", 1)
        file_part = parts[-1].strip() if len(parts) > 1 else citation.strip()
        file_core = file_part.split(",")[0].strip()
        for candidate in [file_core, file_part]:
            key = candidate.lower()
            if key in source_map:
                return source_map[key]
            stem = key.rsplit(".", 1)[0] if "." in key else key
            if stem in source_map:
                return source_map[stem]
        for known, u in source_map.items():
            if file_core.lower() in known or known in file_core.lower():
                return u
        return ""

    def _make_badge(citation: str) -> str:
        parts = citation.split(" - ", 1)
        label = parts[-1].strip() if len(parts) > 1 else citation.strip()
        url = _find_url(citation)
        if url:
            return _BADGE_LINK.format(url=url, label=label)
        return _BADGE_SPAN.format(label=label)

    def _replace_inner(inner: str) -> str:
        citations = [c.strip() for c in inner.split(";") if c.strip()]
        badges = [_make_badge(c) for c in citations]
        return " " + " ".join(badges)

    _CITATION_PATTERN = re.compile(
        r'__|\.(?:pdf|docx|xlsx|xls|pptx|ppt|doc|txt|csv|md)\b',
        re.IGNORECASE,
    )

    # Walk the string and replace each balanced ``(...)`` whose content
    # looks like a citation (contains ``__`` or a known file extension)
    # with rendered badges. Inner parentheses are tolerated as long as
    # the outer pair stays balanced — handles file names like
    # ``Compte administratif Piscine (Recettes) 2023.xlsx``.
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "(":
            out.append(text[i])
            i += 1
            continue
        depth = 1
        j = i + 1
        while j < n and depth > 0:
            if text[j] == "(":
                depth += 1
            elif text[j] == ")":
                depth -= 1
            j += 1
        if depth != 0:
            # Unbalanced — leave the rest as-is and stop scanning
            out.append(text[i:])
            break
        inner = text[i + 1: j - 1]
        if _CITATION_PATTERN.search(inner):
            out.append(_replace_inner(inner))
        else:
            out.append(text[i:j])
        i = j

    return "".join(out)


# ── Markdown to HTML converter ───────────────────────────────────────────────

def md_to_html(text: str) -> str:
    """Lightweight markdown to HTML for Gemini output.

    Handles: headers, bold, italic, bullet/numbered lists, tables, hr.
    """
    lines = text.split("\n")
    html_lines: list[str] = []
    in_ul = False
    in_ol = False
    in_table = False

    _TABLE_STYLE = (
        "border-collapse:collapse;width:100%;margin:8px 0;font-size:0.92em"
    )
    _TH_STYLE = (
        f"border:1px solid {_ACCENT}40;padding:6px 10px;background:{_ACCENT_LIGHT};"
        "text-align:left;font-weight:600"
    )
    _TD_STYLE = "border:1px solid #D4E8DC;padding:6px 10px"

    def _is_table_row(s: str) -> bool:
        return s.startswith("|") and s.endswith("|") and s.count("|") >= 3

    def _is_separator_row(s: str) -> bool:
        return bool(re.match(r'^\|[\s\-:|]+\|$', s))

    def _parse_cells(s: str) -> list[str]:
        return [c.strip() for c in s.strip("|").split("|")]

    for line in lines:
        stripped = line.strip()

        if in_ul and not stripped.startswith(("- ", "* ")):
            html_lines.append("</ul>")
            in_ul = False
        if in_ol and not re.match(r'^\d+[\.\)]\s', stripped):
            html_lines.append("</ol>")
            in_ol = False

        # Table handling
        if _is_table_row(stripped):
            if _is_separator_row(stripped):
                continue
            cells = _parse_cells(stripped)
            if not in_table:
                in_table = True
                html_lines.append(f"<table style='{_TABLE_STYLE}'>")
                html_lines.append("<thead><tr>")
                for c in cells:
                    html_lines.append(f"<th style='{_TH_STYLE}'>{c}</th>")
                html_lines.append("</tr></thead><tbody>")
            else:
                html_lines.append("<tr>")
                for c in cells:
                    html_lines.append(f"<td style='{_TD_STYLE}'>{c}</td>")
                html_lines.append("</tr>")
            continue

        if in_table:
            html_lines.append("</tbody></table>")
            in_table = False

        if stripped.startswith("#### "):
            html_lines.append(f"<h4>{stripped[5:]}</h4>")
        elif stripped.startswith("### "):
            html_lines.append(f"<h3>{stripped[4:]}</h3>")
        elif stripped.startswith("## "):
            html_lines.append(f"<h2>{stripped[3:]}</h2>")
        elif stripped.startswith("# "):
            html_lines.append(f"<h1>{stripped[2:]}</h1>")
        elif stripped.startswith(("- ", "* ")):
            if not in_ul:
                html_lines.append("<ul>")
                in_ul = True
            html_lines.append(f"<li>{stripped[2:]}</li>")
        elif re.match(r'^\d+[\.\)]\s', stripped):
            if not in_ol:
                html_lines.append("<ol>")
                in_ol = True
            content = re.sub(r'^\d+[\.\)]\s', '', stripped)
            html_lines.append(f"<li>{content}</li>")
        elif not stripped:
            html_lines.append("<br>")
        else:
            html_lines.append(f"<p style='margin:4px 0'>{stripped}</p>")

    if in_ul:
        html_lines.append("</ul>")
    if in_ol:
        html_lines.append("</ol>")
    if in_table:
        html_lines.append("</tbody></table>")

    result = "\n".join(html_lines)
    result = re.sub(r'\*\*\*', '<hr style="border:none;border-top:1px solid #D4E8DC;margin:16px 0">', result)
    result = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', result)
    result = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<em>\1</em>', result)

    # Markdown links: [label](https://...) → clickable label (preferred format)
    result = re.sub(
        r'\[([^\]]+)\]\((https?://[^\s\)]+)\)',
        rf'<a href="\2" target="_blank" style="color:{_ACCENT};text-decoration:underline">\1</a>',
        result,
    )

    # Bare URLs in parens: `(https://...)` → keep the parens, make the URL itself clickable.
    # Citation badges run on a later pass and match parens containing `__` or a known
    # file extension — neither matches a Drive URL, so they don't conflict.
    result = re.sub(
        r'\((https?://[^\s\)]+)\)',
        rf'(<a href="\1" target="_blank" style="color:{_ACCENT};text-decoration:underline">\1</a>)',
        result,
    )
    return result


# ── Main render function ─────────────────────────────────────────────────────

def render_answer(text: str, sources: list[dict] | None = None) -> None:
    """Render a generated answer with highlighted source badges.

    Uses st.html() so <a> links are preserved (st.markdown strips them).
    Falls back to st.markdown for empty/no-source answers.
    """
    if not text:
        st.markdown("_No answer generated._")
        return

    html_body = md_to_html(text)
    highlighted = _highlight_sources(html_body, sources or [])

    st.html(
        f'<div style="font-family:-apple-system,BlinkMacSystemFont,sans-serif;'
        f'font-size:15px;line-height:1.7;color:#1A1A1A;'
        f'overflow-x:hidden;overflow-wrap:break-word;word-break:break-word;'
        f'max-width:100%;box-sizing:border-box">'
        f'{highlighted}'
        f'</div>'
    )
