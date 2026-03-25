"""
ADK tool: compare_documents

Extrait le contenu de 2 à 10 documents Drive et demande à Gemini de les comparer
selon un aspect précis ou de manière générale.
"""

from .get_document_content import get_document_content

MAX_DOCUMENTS = 10
# Chars per document injected in the prompt — total stays within model context
CHARS_PER_DOC = 8000 // MAX_DOCUMENTS  # ~800 chars each at max, more when fewer docs


def compare_documents(
    document_urls: list[str],
    aspect: str = "",
) -> dict:
    """
    Compare 2 to 10 Google Drive documents using Gemini.

    Extracts the text content of each document, then asks Gemini to produce
    a structured comparison. If *aspect* is provided, the comparison focuses
    on that specific angle (e.g. "politique de télétravail", "tarifs",
    "obligations légales"). Otherwise a general comparison is produced.

    Args:
        document_urls: List of 2–10 full Google Drive URLs to compare.
                       Maximum 10 documents — excess URLs are ignored.
        aspect:        Optional focus for the comparison (free text).
                       Leave empty for a general comparison.

    Returns:
        Dict with keys:
        - status     : "success" | "error"
        - titles     : List of document titles in order
        - aspect     : Echo of the requested aspect (or "général")
        - comparison : Gemini's structured comparison text
        - errors     : List of URLs that could not be read (if any)
    """
    if not document_urls or len(document_urls) < 2:
        return {
            "status":  "error",
            "message": "Au moins 2 URLs sont nécessaires pour une comparaison.",
        }

    # Guardrail: max 10 documents
    if len(document_urls) > MAX_DOCUMENTS:
        document_urls = document_urls[:MAX_DOCUMENTS]

    # ── Extract all documents (allow partial failure) ─────────────────────────
    docs   = []
    errors = []
    for i, url in enumerate(document_urls):
        result = get_document_content(url)
        if result.get("status") == "success":
            docs.append({
                "title":   result.get("title", f"Document {i+1}"),
                "content": result.get("content", "").strip(),
                "url":     url,
            })
        else:
            errors.append(url)

    if len(docs) < 2:
        return {
            "status":  "error",
            "message": f"Impossible de lire suffisamment de documents ({len(docs)}/{ len(document_urls)}). Erreurs : {errors}",
            "errors":  errors,
        }

    # Distribute chars budget evenly across successfully loaded docs
    chars_each = max(500, 8000 // len(docs))

    # ── Build Gemini prompt ───────────────────────────────────────────────────
    focus     = aspect.strip() if aspect.strip() else "général"
    titles    = [d["title"] for d in docs]
    n         = len(docs)

    if aspect.strip():
        instruction = (
            f"Compare ces {n} documents en te concentrant sur : **{aspect}**.\n"
            "Pour chaque document, indique ce qu'il dit sur ce point, "
            "puis synthétise les similitudes et différences clés.\n"
            "Termine par une conclusion comparative."
        )
    else:
        instruction = (
            f"Compare ces {n} documents de manière structurée.\n"
            "1. Thème et objectif de chaque document\n"
            "2. Points communs\n"
            "3. Différences clés\n"
            "4. Conclusion"
        )

    docs_block = "\n\n".join(
        f"---\n**Document {i+1} — {d['title']}**\n{d['content'][:chars_each]}"
        for i, d in enumerate(docs)
    )

    prompt = f"{instruction}\n\n{docs_block}\n"

    # ── Generate comparison ───────────────────────────────────────────────────
    try:
        from vertexai.generative_models import GenerativeModel
        from rag_agent.config import MODEL, GENERATION_SYSTEM_PROMPT
        from shared.gemini_retry import generate_with_retry

        model    = GenerativeModel(model_name=MODEL, system_instruction=GENERATION_SYSTEM_PROMPT)
        response = generate_with_retry(model, prompt)
        comparison = response.text if hasattr(response, "text") else ""
    except Exception as exc:
        return {
            "status":  "error",
            "message": f"Erreur Gemini : {exc}",
            "titles":  titles,
            "errors":  errors,
        }

    return {
        "status":     "success",
        "titles":     titles,
        "aspect":     focus,
        "comparison": comparison,
        "errors":     errors,
    }
