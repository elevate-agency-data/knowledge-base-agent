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
) -> None:
    """
    Render an assistant reply with optional tool-call details.

    Args:
        text:        Final answer text (markdown supported).
        tool_events: List of parsed tool_call / tool_resp event dicts from
                     AgentRunner.  If provided, renders an expander per tool.
    """
    with st.chat_message("assistant"):
        if tool_events:
            from components.tool_call_expander import render_tool_events
            render_tool_events(tool_events)

        if text:
            st.markdown(text)
        elif not tool_events:
            st.caption("_(no response)_")


def render_error_message(message: str) -> None:
    """Render an inline error notice inside the chat flow."""
    with st.chat_message("assistant"):
        st.error(f"**Erreur** : {message}")
