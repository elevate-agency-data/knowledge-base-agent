"""
Vertex AI RAG service — direct calls without ADK ToolContext.

Used by the comparison page to query Vertex AI independently of the agent.
Replicates the core logic of rag_agent/tools/rag_query.py without the
ToolContext dependency so it can be called from plain Streamlit code.
"""

from __future__ import annotations

import time
from typing import Any


def list_corpora() -> list[dict]:
    """
    Return all available Vertex AI RAG corpora.

    Returns:
        List of dicts with keys: resource_name, display_name, create_time, update_time.
    """
    try:
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
    from vertexai.generative_models import GenerativeModel
    from rag_agent.config import MODEL

    t0 = time.perf_counter()
    try:
        model = GenerativeModel(model_name=MODEL)

        parts = ["Tu es un assistant qui répond en t'appuyant sur les documents fournis."]
        if conversation_context:
            parts.append(f"Historique de la conversation :\n{conversation_context}")
        parts.append(f"Documents récupérés :\n{rag_context}")
        parts.append(f"Question : {query_text}")
        parts.append("Réponds de manière claire et concise en te basant uniquement sur les documents.")

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
    from vertexai import rag
    from vertexai.preview.generative_models import GenerativeModel, Tool
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
