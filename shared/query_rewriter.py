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

_REWRITER_MODEL = "gemini-2.0-flash-001"

_SYSTEM_PROMPT = """You are a query optimizer for RAG systems.

Your role: transform the user query into a semantic query optimized
for vector search in a document knowledge base.

Rules:
- Remove action verbs (compare, list, summarize, describe, explain...)
- ALWAYS KEEP proper nouns: domain names, topics, projects
  (e.g. HR, Marketing, Legal, Finance, GA4, Elevate...)
- Keep relevant concepts, themes, and entities
- If conversational context is provided, include key entities
  (domains, topics discussed) in the rewritten query
- Formulate a short description of the content to find (max 20 words)
- Respond ONLY with the rewritten query, no explanation
- For a general or vague request WITHOUT proper nouns, return
  "internal policy procedures rules documentation"

Examples without context:
  "compare hr and marketing"           → "HR Marketing policies procedures differences"
  "tell me about legal"                → "Legal contracts compliance legal procedures"
  "what do you know about hr"          → "HR human resources policies procedures"
  "summarize all indexes"              → "general overview domains policies procedures"
  "which domains mention GA4?"         → "GA4 analytics web tracking"
  "explain the remote work policy"     → "remote work policy rules allowed days"
  "list employee benefits"             → "employee benefits compensation perks"

Examples with context:
  context: "Q: what is the leave policy?"
  question: "and for remote work?"
  → "remote work policy rules allowed days"

  context: "Q: tell me about the marketing strategy"
  question: "and the tools used?"
  → "Marketing tools strategy digital"
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
