"""
Chat message rendering component.

Provides render_user_message() and render_assistant_message() which wrap
st.chat_message and apply consistent styling.
"""

from __future__ import annotations

import streamlit as st


def render_user_message(text: str) -> None:
    """Render a user chat bubble."""
    with st.chat_message("user"):
        st.markdown(text)


def render_assistant_message(
    text: str,
    tool_events: list[dict] | None = None,
    sources: list[dict] | None = None,
    chunks: list[dict] | None = None,
) -> None:
    """
    Render an assistant reply with citation badges + tool-call details.

    Args:
        text:        Final answer text (markdown supported). Citation patterns
                     like ``(commune__category - FileName)`` are converted to
                     clickable badges linked to the matching source URL.
        tool_events: List of parsed tool_call / tool_resp event dicts from
                     AgentRunner. If provided, renders an expander per tool.
        sources:     Deduplicated list of source dicts (file_name, source_url)
                     used to populate the citation badges and the Sources panel.
        chunks:      List of retrieved chunk dicts to display in a separate panel.
    """
    from components.answer_renderer import render_answer

    with st.chat_message("assistant"):
        # A photo assessment is laid out as an atelier intake sheet above the
        # prose answer: it is the evidence the answer rests on, so it belongs
        # in front of it rather than folded away with the machinery.
        from components.sav_sheet import extract_sav_responses, render_sav_sheet
        for _sheet in extract_sav_responses(tool_events):
            render_sav_sheet(_sheet)

        if text:
            render_answer(text, sources or [])
        elif not tool_events:
            st.caption("_(pas de réponse)_")
            return

        # Sources, passages and the route taken share one fold-out.
        from components.detail_panel import render_detail_panel
        render_detail_panel(tool_events, sources, chunks)


def render_error_message(message: str) -> None:
    """Render an inline error notice inside the chat flow."""
    with st.chat_message("assistant"):
        if "429" in message or "RESOURCE_EXHAUSTED" in message:
            st.warning(
                "Le quota du service est atteint. Renvoyez la question dans "
                "quelques secondes — votre dossier est conservé."
            )
        else:
            st.error(f"**Erreur** : {message}")
