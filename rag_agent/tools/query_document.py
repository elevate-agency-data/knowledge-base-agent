"""
ADK tool: query_document

Extrait le contenu d'un document Drive et répond à une question dessus.
Utile quand l'utilisateur veut interroger un fichier spécifique plutôt que
de chercher dans toute la base de connaissances.
"""

from .get_document_content import get_document_content


def query_document(document_url: str, question: str) -> dict:
    """
    Answer a question about a specific Google Drive document.

    Extracts the document's text content and uses Gemini to answer the
    question based exclusively on that content — no RAG index involved.

    Args:
        document_url: Full Google Drive URL of the document.
        question:     The question to answer based on the document.

    Returns:
        Dict with keys:
        - status   : "success" | "error"
        - title    : Document title
        - answer   : Gemini's answer grounded in the document
        - url      : Echo of document_url
    """
    # ── Extract document ──────────────────────────────────────────────────────
    doc = get_document_content(document_url)
    if doc.get("status") != "success":
        return {
            "status":  "error",
            "message": f"Impossible de lire le document : {doc.get('message', '')}",
            "url":     document_url,
        }

    title   = doc.get("title", "Document")
    content = doc.get("content", "").strip()

    if not content:
        return {
            "status":  "error",
            "message": "Le document est vide ou son contenu n'a pas pu être extrait.",
            "title":   title,
            "url":     document_url,
        }

    # ── Build prompt ──────────────────────────────────────────────────────────
    prompt = (
        f"Document : **{title}**\n\n"
        f"{content[:8000]}\n\n"
        f"---\nQuestion : {question}\n\n"
        "Réponds en te basant EXCLUSIVEMENT sur le contenu du document ci-dessus. "
        "Si l'information n'est pas dans le document, dis-le clairement."
    )

    # ── Generate answer ───────────────────────────────────────────────────────
    try:
        from vertexai.generative_models import GenerativeModel
        from rag_agent.config import MODEL, GENERATION_SYSTEM_PROMPT
        from shared.gemini_retry import generate_with_retry

        model    = GenerativeModel(model_name=MODEL, system_instruction=GENERATION_SYSTEM_PROMPT)
        response = generate_with_retry(model, prompt)
        answer   = response.text if hasattr(response, "text") else ""
    except Exception as exc:
        return {
            "status":  "error",
            "message": f"Erreur Gemini : {exc}",
            "title":   title,
            "url":     document_url,
        }

    return {
        "status": "success",
        "title":  title,
        "answer": answer,
        "url":    document_url,
    }
