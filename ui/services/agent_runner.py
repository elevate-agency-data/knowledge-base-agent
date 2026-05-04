"""
ADK Agent Runner — wraps google.adk.runners.Runner pour Streamlit.

Persistence :
  - Sessions ADK     : même SQLite qu'`adk web` → rag_agent/.adk/session.db
                       app_name="rag_agent" pour partager les sessions.
  - Messages display : table agent_messages dans chat.duckdb (via chat_store)
                       remplace les anciens fichiers JSON sidecar.

Async → sync via thread dédié (compatible avec la boucle Tornado de Streamlit).
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

_APP_NAME = "rag_agent"
# Même SQLite qu'`adk web` — partagé entre les deux interfaces
_DB_PATH  = Path(__file__).parent.parent.parent / "rag_agent" / ".adk" / "session.db"


# ── Async helper ──────────────────────────────────────────────────────────────

def _run_coroutine(coro, user_role: str | None = None) -> Any:
    """
    Run *coro* in a fresh thread+event loop, propagating *user_role*
    into the worker thread's runtime context so tools can enforce RBAC.
    """
    result: list[Any]       = [None]
    error:  list[Exception] = [None]

    def _target():
        if user_role is not None:
            from rag_agent.runtime_context import set_user_role
            set_user_role(user_role)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result[0] = loop.run_until_complete(coro)
        except Exception as exc:
            error[0] = exc
        finally:
            loop.close()

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join()
    if error[0]:
        raise error[0]
    return result[0]


# ── AgentRunner ───────────────────────────────────────────────────────────────

class AgentRunner:
    """
    Wrapper synchrone autour de google.adk.runners.Runner.
    Utilise DatabaseSessionService pour que les sessions persistent
    comme avec `adk web`.
    Destiné à être caché via @st.cache_resource.
    """

    def __init__(self, is_admin: bool = False) -> None:
        from google.adk.runners import Runner
        from rag_agent.agent import root_agent, user_agent

        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)

        try:
            from google.adk.sessions import DatabaseSessionService
            self._session_service = DatabaseSessionService(
                db_url=f"sqlite:///{_DB_PATH}"
            )
        except Exception:
            from google.adk.sessions import InMemorySessionService
            self._session_service = InMemorySessionService()

        agent = root_agent if is_admin else user_agent
        self._runner = Runner(
            agent=agent,
            app_name=_APP_NAME,
            session_service=self._session_service,
        )

    # ── Session management ────────────────────────────────────────────────────

    def list_sessions(self, user_id: str) -> list[dict]:
        """
        Liste toutes les sessions ADK persistées pour cet utilisateur.
        Enrichit avec le nom et le compte de messages depuis chat_store.

        Returns list of dicts : {id, name, message_count}
        """
        from services.chat_store import agent_sessions_meta

        async def _list():
            resp = await self._session_service.list_sessions(
                app_name=_APP_NAME,
                user_id=user_id,
            )
            return getattr(resp, "sessions", []) or []

        try:
            adk_sessions = _run_coroutine(_list())
        except Exception:
            return []

        meta = agent_sessions_meta(user_id)

        result = []
        for s in adk_sessions:
            sid  = s.id
            info = meta.get(sid, {})
            result.append({
                "id":            sid,
                "name":          info.get("name", f"Session {sid[:8]}"),
                "message_count": info.get("message_count", 0),
            })

        return list(reversed(result))

    def create_session(self, user_id: str) -> str:
        """Crée une nouvelle session ADK et retourne son id."""
        async def _create():
            s = await self._session_service.create_session(
                app_name=_APP_NAME,
                user_id=user_id,
            )
            return s.id

        return _run_coroutine(_create())

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(
        self,
        user_id: str,
        session_id: str,
        message: str,
        user_role: str | None = None,
    ) -> list[dict]:
        """
        Envoie un message à l'agent. Retourne une liste d'événements parsés.
        Le contexte complet de la session est chargé automatiquement par ADK.

        *user_role* est propagé au thread d'exécution pour que les tools
        (hybrid_list_indexes, hybrid_query) appliquent le filtre RBAC.
        """
        from google.genai import types

        async def _run():
            raw = []
            async for event in self._runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=types.Content(
                    role="user",
                    parts=[types.Part(text=message)],
                ),
            ):
                raw.append(event)
            return raw

        return self._parse_events(_run_coroutine(_run(), user_role=user_role))

    # ── Event parsing ─────────────────────────────────────────────────────────

    @staticmethod
    def _parse_events(raw_events) -> list[dict]:
        parsed: list[dict] = []
        for event in raw_events:
            fn_calls = (
                event.get_function_calls()
                if hasattr(event, "get_function_calls") else []
            ) or []
            for fn in fn_calls:
                parsed.append({
                    "type": "tool_call",
                    "name": fn.name,
                    "args": dict(fn.args) if fn.args else {},
                })

            fn_resps = (
                event.get_function_responses()
                if hasattr(event, "get_function_responses") else []
            ) or []
            for fn in fn_resps:
                parsed.append({
                    "type":     "tool_resp",
                    "name":     fn.name,
                    "response": fn.response
                        if isinstance(fn.response, dict) else {"raw": str(fn.response)},
                })

            is_final = (
                event.is_final_response()
                if hasattr(event, "is_final_response") else False
            )
            if event.content and event.content.parts:
                for part in event.content.parts:
                    text = getattr(part, "text", None)
                    if text:
                        parsed.append({"type": "text", "text": text, "final": is_final})
        return parsed

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def extract_final_text(parsed_events: list[dict]) -> str:
        texts = [
            e["text"] for e in parsed_events
            if e["type"] == "text" and e.get("final")
        ]
        return texts[-1] if texts else ""
