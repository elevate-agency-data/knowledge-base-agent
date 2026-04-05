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

_SYSTEM_PROMPT = """You are a query optimizer for a customer care knowledge base.

Your role: transform what an advisor types (often a customer question relayed as-is)
into a semantic query optimized for vector search in the internal documentation.

Rules:
- KEEP product names, sizes, references, model names, brand terms
- KEEP customer-facing concepts: return, refund, exchange, delivery, warranty, sizing, order status
- ALWAYS KEEP numerical information (dates, durations, quantities) — e.g. "3 days", "48h"
- ALWAYS include time-related concepts if present: delay, duration, processing time, preparation time
- ALWAYS include policy-related concepts when relevant: SLA, standard time, escalation, delay handling
- Remove conversational fluff ("the customer wants to know", "can you tell me", "please help")
- If the query mentions a delay or abnormal situation, include terms like: delay, SLA, escalation, issue
- If the query mentions a specific product or policy, keep it front and center
- If conversational context is provided, include key entities from it
- Formulate a short retrieval-oriented query (max 20 words)
- Respond ONLY with the rewritten query, no explanation

For vague questions, return broad terms covering likely topics:
"product policy return exchange delivery warranty sizing"

Examples without context:
  "customer wants to return a polo bought 3 weeks ago"
    → "return policy polo 3 weeks delay conditions refund"
  "what size should I recommend for someone who wears M in slim fit?"
    → "slim fit sizing guide medium size conversion size up"
  "is the ConnectWatch waterproof?"
    → "ConnectWatch water resistance rating specifications"
  "how long is the warranty on leather goods?"
    → "leather goods warranty duration conditions"
  "client asks about free shipping"
    → "shipping policy free delivery threshold conditions"
  "what is your exchange policy for online orders?"
    → "online order exchange policy conditions return"
  "order 3 days still in preparation"
    → "order preparation time SLA 3 days delay escalation warehouse"
  "I placed my order 3 days ago still in preparation"
    → "order preparation delay 3 days SLA 24-48 hours escalation warehouse"
  "customer received wrong item"
    → "wrong item received error shipping claim exchange procedure"
  "package lost in transit"
    → "lost package shipping claim carrier delay tracking"

Examples with context:
  context: "Q: what is the return policy?"
  question: "and if they lost the receipt?"
  → "return policy without receipt proof of purchase conditions"

  context: "Q: ConnectWatch features"
  question: "and the battery life?"
  → "ConnectWatch battery life autonomy specifications"

  context: "Q: order still in preparation after 3 days"
  question: "what should I tell the customer?"
  → "order delay escalation procedure advisor response SLA exceeded"
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
