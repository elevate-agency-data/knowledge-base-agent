"""
Vertex AI RAG service — direct calls without ADK ToolContext.

Used by the comparison page to query Vertex AI independently of the agent.
Replicates the core logic of rag_agent/tools/rag_query.py without the
ToolContext dependency so it can be called from plain Streamlit code.
"""

from __future__ import annotations

import time
from typing import Any


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


def extract_query_from_drive_url(document_url: str) -> str:
    """
    Extract a retrieval-optimized text query from a Google Drive document URL.

    Flow:
      1. Extract raw text content from the Drive document.
      2. If the text is short (< 1500 chars), use it directly.
         If long, ask Gemini Flash to produce a ~150-word thematic summary
         that captures the document's key topics and entities.
      3. Return the summary/text — caller passes it to query() as query_text.

    Args:
        document_url: Full Google Drive URL.

    Returns:
        A text string suitable as a semantic query.
        Falls back to the raw URL on extraction failure.
    """
    _init_vertex()
    try:
        from rag_agent.tools.get_document_content import get_document_content
        result = get_document_content(document_url)
        if result.get("status") != "success":
            return document_url

        text  = result.get("content", "").strip()
        title = result.get("title", "")
        if not text:
            return title or document_url

        if len(text) <= 1500:
            return f"{title}\n{text}" if title else text

        # Long document → summarise with Gemini Flash
        from vertexai.generative_models import GenerativeModel
        model = GenerativeModel("gemini-2.0-flash-001")
        prompt = (
            f"Voici le contenu d'un document intitulé « {title} ».\n\n"
            f"{text[:8000]}\n\n"
            "Résume en 150 mots maximum les thèmes principaux, entités clés "
            "(noms de clients, projets, technologies) et sujets abordés. "
            "Ne commence pas par « Ce document » — donne directement les thèmes."
        )
        response = model.generate_content(prompt)
        summary  = response.text.strip() if hasattr(response, "text") else ""
        return summary if summary else text[:1500]
    except Exception:
        return document_url


def list_corpora() -> list[dict]:
    """
    Return all available Vertex AI RAG corpora.

    Returns:
        List of dicts with keys: resource_name, display_name, create_time, update_time.
    """
    try:
        _init_vertex()
        from vertexai import rag
        corpora = rag.list_corpora()
        return [
            {
                "resource_name": c.name,
                "display_name":  c.display_name,
                "create_time":   str(getattr(c, "create_time", "")),
                "update_time":   str(getattr(c, "update_time", "")),
            }
            for c in corpora
        ]
    except Exception as exc:
        return []


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
    from rag_agent.config import MODEL

    t0 = time.perf_counter()
    try:
        model = GenerativeModel(model_name=MODEL)

        parts = [
            "Tu es l'assistant interne d'Elevate, une société de conseil en Data & Analytics. "
            "Tu réponds EXCLUSIVEMENT à partir des documents internes fournis ci-dessous "
            "(propositions commerciales, analyses, offres d'accompagnement rédigées par Elevate "
            "pour ses clients). "
            "N'utilise JAMAIS ta connaissance générale sur les entreprises ou les marques. "
            "Si l'information ne figure pas dans les documents, dis-le clairement."
        ]
        if conversation_context:
            parts.append(f"Historique de la conversation :\n{conversation_context}")
        parts.append(f"Documents internes Elevate récupérés :\n{rag_context}")
        parts.append(f"Question : {query_text}")
        parts.append(
            "Réponds de manière claire et concise en te basant UNIQUEMENT sur les documents "
            "internes fournis. Ne complète pas avec ta connaissance générale."
        )

        response = model.generate_content("\n\n".join(parts))
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


def query(corpus_name: str, query_text: str, context: str = "") -> dict:
    """
    Query a Vertex AI RAG corpus and return a structured result.

    Args:
        corpus_name: Display name or full resource name of the corpus.
        query_text:  Natural-language question.

    Returns:
        Dict with keys:
        - status        : "success" | "error"
        - answer        : Generated answer text
        - sources       : List of {title, uri} dicts
        - elapsed_s     : Wall-clock time in seconds
        - corpus_name   : Echo of input
    """
    _init_vertex()
    from vertexai import rag
    from vertexai.generative_models import GenerativeModel, Tool
    from rag_agent.config import DEFAULT_TOP_K, DEFAULT_DISTANCE_THRESHOLD, MODEL
    from rag_agent.tools.utils import get_corpus_resource_name

    t0 = time.perf_counter()
    try:
        corpus_resource_name = get_corpus_resource_name(corpus_name)

        rag_store = rag.VertexRagStore(
            rag_resources=[rag.RagResource(rag_corpus=corpus_resource_name)],
            rag_retrieval_config=rag.RagRetrievalConfig(
                top_k=DEFAULT_TOP_K,
                filter=rag.utils.resources.Filter(
                    vector_distance_threshold=DEFAULT_DISTANCE_THRESHOLD
                ),
            ),
        )

        rag_retrieval_tool = Tool.from_retrieval(
            retrieval=rag.Retrieval(source=rag_store)
        )
        from shared.query_rewriter import rewrite_query
        retrieval_query = rewrite_query(query_text, context=context)

        model = GenerativeModel(model_name=MODEL, tools=[rag_retrieval_tool])
        response = model.generate_content(retrieval_query)

        answer = response.text if hasattr(response, "text") else ""

        sources: list[dict] = []
        if response.candidates and response.candidates[0].grounding_metadata:
            seen_uris: set[str] = set()
            for chunk in response.candidates[0].grounding_metadata.grounding_chunks:
                ctx   = getattr(chunk, "retrieved_context", None)
                uri   = getattr(ctx, "uri",   None) if ctx else None
                title = getattr(ctx, "title", "Document sans titre") if ctx else "Document sans titre"
                if uri and uri not in seen_uris:
                    seen_uris.add(uri)
                    sources.append({"title": title, "uri": uri})

        return {
            "status":      "success",
            "answer":      answer,
            "sources":     sources,
            "elapsed_s":   round(time.perf_counter() - t0, 2),
            "corpus_name": corpus_name,
        }

    except Exception as exc:
        return {
            "status":      "error",
            "message":     str(exc),
            "answer":      "",
            "sources":     [],
            "elapsed_s":   round(time.perf_counter() - t0, 2),
            "corpus_name": corpus_name,
        }
