"""
Tool call expander component.

Renders a collapsible section for each tool invocation showing the
arguments sent and the response received.
"""

from __future__ import annotations

import json

import streamlit as st


# Labels per tool family
_TOOL_ICONS: dict[str, str] = {
    "hybrid_rag_query":    "",
    "hybrid_multi_query":  "",
    "hybrid_add_data":     "",
    "hybrid_create_index": "",
    "hybrid_delete_index": "",
    "hybrid_list_indexes": "",
    "hybrid_index_info":   "",
    "hybrid_list_drive":   "",
    "hybrid_find_similar": "",
    "compare_documents":   "",
    "query_document":      "",
    "get_document_content":"",
}


def _tool_icon(name: str) -> str:
    return _TOOL_ICONS.get(name, "")


def render_tool_events(events: list[dict]) -> None:
    """
    Render all tool_call / tool_resp events from a single agent turn.

    Pairs each tool_call with its matching tool_resp (by tool name order)
    and wraps them in a single expander.
    """
    # Build pairs: (call_event, resp_event | None)
    pairs: list[tuple[dict, dict | None]] = []
    call_queue: list[dict] = []

    for event in events:
        if event["type"] == "tool_call":
            call_queue.append(event)
        elif event["type"] == "tool_resp" and call_queue:
            # Match to the earliest unmatched call with the same name
            matched = next(
                (c for c in call_queue if c["name"] == event["name"]),
                call_queue[0],
            )
            call_queue.remove(matched)
            pairs.append((matched, event))

    # Unmatched calls (no response yet)
    for call in call_queue:
        pairs.append((call, None))

    for call, resp in pairs:
        label = f"`{call['name']}`"

        with st.expander(label, expanded=False):
            st.markdown("**Arguments**")
            st.json(call.get("args", {}), expanded=False)

            if resp:
                st.markdown("**Result**")
                response_data = resp.get("response", {})
                # Trim large fields for readability
                display = _trim_response(response_data)
                st.json(display, expanded=False)
            else:
                st.caption("_(waiting for response)_")


def _trim_response(data: dict, max_str_len: int = 400) -> dict:
    """
    Recursively truncate long string values so the expander stays readable.
    """
    if not isinstance(data, dict):
        return data
    result = {}
    for k, v in data.items():
        if isinstance(v, str) and len(v) > max_str_len:
            result[k] = v[:max_str_len] + f"… [{len(v)} chars]"
        elif isinstance(v, dict):
            result[k] = _trim_response(v, max_str_len)
        elif isinstance(v, list):
            result[k] = [
                _trim_response(i, max_str_len) if isinstance(i, dict) else i
                for i in v[:20]  # cap list length
            ]
        else:
            result[k] = v
    return result
