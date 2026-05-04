"""
Per-request runtime context for the RAG agent.

Carries the current user's business role into tool invocations so that
tools (notably hybrid_query / hybrid_list_indexes) can enforce role-based
index filtering without the agent prompt being able to bypass it.

Threading model:
    Streamlit's AgentRunner runs the ADK coroutine in a dedicated worker
    thread (see services.agent_runner._run_coroutine). The role is stored
    in a thread-local so each worker thread sees only its own user's role.
    The AgentRunner sets the role on the worker thread BEFORE the agent
    starts executing tools.
"""

from __future__ import annotations

import threading

_local = threading.local()


def set_user_role(role: str | None) -> None:
    _local.role = role or "agent"


def get_user_role() -> str:
    return getattr(_local, "role", "agent")


def clear_user_role() -> None:
    if hasattr(_local, "role"):
        del _local.role
