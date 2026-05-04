"""
Vertex AI service — direct LLM calls (no RAG retrieval).

Used for two purposes:
  - `direct_query`: chat without retrieval (RAG-off mode in Simple Chat)
  - `synthesize_from_context`: final answer generation when Hybrid RAG
    returns raw chunks and we need an LLM to compose the response.
"""

from __future__ import annotations

import time


def _init_vertex() -> None:
    """
    Initialise le SDK Vertex AI avec le project ID string explicite.

    Sans cet appel, le SDK tente de résoudre le project ID depuis les
    credentials gcloud application-default (qui stockent le numéro de projet),
    ce qui déclenche un appel Cloud Resource Manager API inutile.
    """
    import vertexai
    from hybrid.config import PROJECT_ID, LOCATION
    from rag_agent.config import SERVICE_ACCOUNT_PATH
    try:
        from google.oauth2 import service_account
        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_PATH,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=creds)
    except Exception:
        vertexai.init(project=PROJECT_ID, location=LOCATION)


def direct_query(query_text: str, context: str = "") -> dict:
    """
    Query Gemini directly without any RAG retrieval.

    Args:
        query_text: User question.
        context:    Optional conversation history for multi-turn continuity.

    Returns:
        Dict with keys: status, answer, sources (empty), elapsed_s.
    """
    from vertexai.generative_models import GenerativeModel
    from rag_agent.config import MODEL

    t0 = time.perf_counter()
    try:
        model = GenerativeModel(model_name=MODEL)

        if context:
            prompt = (
                f"Historique de la conversation :\n{context}\n\n"
                f"Question : {query_text}"
            )
        else:
            prompt = query_text

        response = model.generate_content(prompt)
        return {
            "status":    "success",
            "answer":    response.text if hasattr(response, "text") else "",
            "sources":   [],
            "elapsed_s": round(time.perf_counter() - t0, 2),
        }
    except Exception as exc:
        return {
            "status":    "error",
            "message":   str(exc),
            "answer":    "",
            "sources":   [],
            "elapsed_s": round(time.perf_counter() - t0, 2),
        }


def synthesize_from_context(
    query_text: str,
    rag_context: str,
    conversation_context: str = "",
) -> dict:
    """
    Generate an LLM answer given pre-retrieved RAG context.

    Used by Simple Chat when the Hybrid pipeline returns raw chunks and
    we need a final synthesised response.

    Args:
        query_text:           Original user question.
        rag_context:          Retrieved chunks concatenated as text.
        conversation_context: Optional prior conversation history.

    Returns:
        Dict with keys: status, answer, elapsed_s.
    """
    _init_vertex()
    from vertexai.generative_models import GenerativeModel
    from rag_agent.config import MODEL, GENERATION_SYSTEM_PROMPT
    from shared.gemini_retry import generate_with_retry

    t0 = time.perf_counter()
    try:
        model = GenerativeModel(model_name=MODEL)

        parts = [GENERATION_SYSTEM_PROMPT]
        if conversation_context:
            parts.append(f"Historique de la conversation :\n{conversation_context}")
        parts.append(f"Documents récupérés :\n{rag_context}")
        parts.append(f"Question : {query_text}")
        parts.append("Réponds de manière claire et concise en te basant UNIQUEMENT sur les documents fournis.")

        response = generate_with_retry(model, "\n\n".join(parts))
        return {
            "status":    "success",
            "answer":    response.text if hasattr(response, "text") else "",
            "elapsed_s": round(time.perf_counter() - t0, 2),
        }
    except Exception as exc:
        return {
            "status":    "error",
            "message":   str(exc),
            "answer":    "",
            "elapsed_s": round(time.perf_counter() - t0, 2),
        }
