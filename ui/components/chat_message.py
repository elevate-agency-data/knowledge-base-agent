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
        if tool_events:
            from components.tool_call_expander import render_tool_events
            render_tool_events(tool_events)

        if text:
            render_answer(text, sources or [])
        elif not tool_events:
            st.caption("_(no response)_")
            return

        if sources:
            from components.source_card import render_sources
            with st.expander(f"Sources ({len(sources)})", expanded=False):
                render_sources(sources, pipeline="hybrid")
        if chunks:
            from components.source_card import render_chunks
            with st.expander(f"Retrieved chunks ({len(chunks)})", expanded=False):
                render_chunks(chunks)


def render_error_message(message: str) -> None:
    """Render an inline error notice inside the chat flow."""
    with st.chat_message("assistant"):
        if "429" in message or "RESOURCE_EXHAUSTED" in message:
            st.warning(
                "The service is temporarily overloaded (API quota exceeded). "
                "Please try again in a few seconds."
            )
        else:
            st.error(f"**Error**: {message}")
