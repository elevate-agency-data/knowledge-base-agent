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

from shared.role_permissions import DEFAULT_ROLE

_local = threading.local()


def set_user_role(role: str | None) -> None:
    _local.role = role or DEFAULT_ROLE


def get_user_role() -> str:
    return getattr(_local, "role", DEFAULT_ROLE)


def clear_user_role() -> None:
    if hasattr(_local, "role"):
        del _local.role


# ── User identity ─────────────────────────────────────────────────────────────
# Carried the same way as the role: set by the runner on the worker thread, out
# of the agent prompt's reach. Used to scope uploaded-image refs so a session
# can only resolve its own uploads.

def set_user_id(user_id: str | None) -> None:
    _local.user_id = user_id or ""


def get_user_id() -> str:
    return getattr(_local, "user_id", "")


def clear_user_id() -> None:
    if hasattr(_local, "user_id"):
        del _local.user_id
