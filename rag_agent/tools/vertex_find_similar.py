"""
ADK tool: vertex_find_similar

Find documents similar to a Google Drive document using Vertex AI RAG.

Flow:
  1. Extract text content from the Drive document.
  2. If long (> 1500 chars), summarise with Gemini Flash (~150 words).
  3. Use that text as a retrieval query against the Vertex RAG corpus.
  4. Deduplicate results by URI and return a ranked list of similar docs.

This is a text-query-based similarity search — Vertex handles vectorisation
internally. Unlike hybrid_find_similar, the signal is bounded by the summary
quality and the corpus' own embedding model.
"""

import time

from ..config import DEFAULT_TOP_K, DEFAULT_DISTANCE_THRESHOLD
from .utils import get_corpus_resource_name


def vertex_find_similar(
    corpus_name: str,
    document_url: str,
) -> dict:
    """
    Find documents similar to a Google Drive document in a Vertex AI RAG corpus.

    Extracts the source document text, optionally summarises it with Gemini Flash,
    then runs a retrieval query against the corpus — no LLM generation, pure retrieval.

    Args:
        corpus_name:  Display name or resource name of the Vertex corpus.
        document_url: Full Google Drive URL of the source document.
                      Supported: Google Docs, Slides, Sheets, PDF.

    Returns:
        Dict with keys:
        - ``status``          : ``"success"`` or ``"error"``
        - ``source_document`` : Echo of document_url
        - ``corpus_name``     : Echo of corpus_name
        - ``results``         : List of similar docs, each with ``title``, ``uri``, ``score``
        - ``total_results``   : Number of unique documents found
        - ``elapsed_s``       : Wall-clock time in seconds
    """
    from vertexai import rag
    from .get_document_content import get_document_content

    t0 = time.perf_counter()

    try:
        # -- Extract & optionally summarise source document ------------------
        result = get_document_content(document_url)
        if result.get("status") != "success":
            return {
                "status":    "error",
                "message":   "Impossible d'extraire le contenu du document.",
                "elapsed_s": round(time.perf_counter() - t0, 2),
            }

        text  = result.get("content", "").strip()
        title = result.get("title", "")

        if not text:
            return {
                "status":    "error",
                "message":   "Le document est vide ou illisible.",
                "elapsed_s": round(time.perf_counter() - t0, 2),
            }

        if len(text) <= 1500:
            doc_query = f"{title}\n{text}" if title else text
        else:
            from vertexai.generative_models import GenerativeModel
            model  = GenerativeModel("gemini-2.0-flash-001")
            prompt = (
                f"Voici le contenu d'un document intitulé « {title} ».\n\n"
                f"{text[:8000]}\n\n"
                "Résume en 150 mots maximum les thèmes principaux, entités clés "
                "(noms de clients, projets, technologies) et sujets abordés. "
                "Ne commence pas par « Ce document » — donne directement les thèmes."
            )
            response  = model.generate_content(prompt)
            doc_query = response.text.strip() if hasattr(response, "text") and response.text else text[:1500]

        # -- Retrieval query against the corpus ------------------------------
        corpus_resource_name = get_corpus_resource_name(corpus_name)

        response = rag.retrieval_query(
            rag_resources=[rag.RagResource(rag_corpus=corpus_resource_name)],
            text=doc_query,
            rag_retrieval_config=rag.RagRetrievalConfig(
                top_k=DEFAULT_TOP_K,
                filter=rag.utils.resources.Filter(
                    vector_distance_threshold=DEFAULT_DISTANCE_THRESHOLD
                ),
            ),
        )

        # -- Deduplicate by URI, keep best score per document ----------------
        seen: dict[str, dict] = {}
        for chunk in response.contexts.contexts:
            uri   = getattr(chunk, "source_uri",          None) or ""
            name  = getattr(chunk, "source_display_name", None) or "Document sans titre"
            score = float(getattr(chunk, "score", 0.0))
            if uri and uri != document_url:
                if uri not in seen or score > seen[uri]["score"]:
                    seen[uri] = {"title": name, "uri": uri, "score": round(score, 4)}

        results = sorted(seen.values(), key=lambda x: x["score"], reverse=True)

        return {
            "status":          "success",
            "source_document": document_url,
            "corpus_name":     corpus_name,
            "results":         results,
            "total_results":   len(results),
            "elapsed_s":       round(time.perf_counter() - t0, 2),
        }

    except Exception as exc:
        return {
            "status":          "error",
            "message":         str(exc),
            "source_document": document_url,
            "corpus_name":     corpus_name,
            "results":         [],
            "elapsed_s":       round(time.perf_counter() - t0, 2),
        }
