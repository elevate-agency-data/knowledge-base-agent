"""
Shared utility: query_rewriter

Rewrites a user-facing query into a retrieval-optimized semantic query
before it hits any embedding model (Vertex AI RAG or Hybrid RAG).

Problem it solves:
    User queries are often action-oriented ("compare X et Y", "résume tous
    les clients", "liste les offres") — those phrases embed poorly and
    retrieve irrelevant chunks.  This module strips the action intent and
    returns a descriptive, content-focused query that the embedding model
    can match against document chunks.

Model choice:
    Uses gemini-2.0-flash for low latency and cost.  The main agent
    (gemini-2.5-pro) handles synthesis; this step just improves recall.
"""

from vertexai.generative_models import GenerativeModel
from shared.gemini_retry import generate_with_retry
from shared.brand import ACTIVE as _BRAND

_REWRITER_MODEL = "gemini-2.0-flash-001"

# Domain framing (corpus intro, terms to keep, few-shot examples) is injected
# from the active brand profile; the rewrite rules themselves are generic.
_SYSTEM_PROMPT = f"""You are a query optimizer for a knowledge base.

{_BRAND.rewriter_intro}

Your role: transform what a user types into a semantic query optimized
for vector search across these data domains.

Rules:
- KEEP proper nouns: project names, supplier names, person names, document references
- KEEP domain-specific terms: {_BRAND.rewriter_keep_terms}...
- ALWAYS KEEP numerical information (dates, durations, amounts) — e.g. "2024", "3 mois", "150k€"
- ALWAYS include time-related concepts if present: échéance, durée, période, délai
- Remove conversational fluff ("can you tell me", "I'd like to know", "please help")
- If the query references a data domain, keep it front and center
- If conversational context is provided, include key entities from it
- Formulate a short retrieval-oriented query (max 20 words)
- Respond ONLY with the rewritten query, no explanation

For vague questions, return broad terms covering likely topics.

Examples without context:
{_BRAND.rewriter_examples}
"""


def rewrite_query(user_query: str, context: str = "") -> str:
    """
    Rewrite a user query into a retrieval-optimized semantic query.

    Args:
        user_query: The raw user message or query string.
        context:    Optional conversation context (last N Q&A exchanges)
                    to resolve references like "la prestation", "ce client", etc.

    Returns:
        A rewritten query string better suited for embedding/retrieval.
        Falls back to the original query if the rewrite fails.
    """
    if not user_query or not user_query.strip():
        return user_query

    prompt = user_query.strip()
    if context and context.strip():
        prompt = f"Contexte conversationnel :\n{context.strip()}\n\nQuestion : {prompt}"

    try:
        model = GenerativeModel(
            model_name=_REWRITER_MODEL,
            system_instruction=_SYSTEM_PROMPT,
        )
        response = generate_with_retry(model, prompt)
        rewritten = response.text.strip()
        print(f"rewritten query : {rewritten}")
        return rewritten if rewritten else user_query
    except Exception:
        # Never block the retrieval pipeline on a rewrite failure
        return user_query
