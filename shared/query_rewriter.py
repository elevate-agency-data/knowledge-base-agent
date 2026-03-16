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

_REWRITER_MODEL = "gemini-2.0-flash-001"

_SYSTEM_PROMPT = """Tu es un optimiseur de requêtes pour systèmes RAG.

Ton rôle : transformer la requête utilisateur en une requête sémantique
optimisée pour la recherche vectorielle dans une base documentaire.

Règles :
- Supprime les verbes d'action (compare, liste, résume, décris, explique…)
- Garde uniquement les concepts, thèmes et entités pertinents
- Si un contexte conversationnel est fourni, inclus les entités clés
  (noms de clients, sujets abordés) dans la requête réécrite
- Formule une description courte du contenu à trouver (max 20 mots)
- Réponds UNIQUEMENT avec la requête réécrite, sans explication

Exemples sans contexte :
  "compare celio et fnac"           → "offres produits, services clients et positionnement commercial"
  "résume tous les index"           → "présentation générale, activités et offres"
  "quels clients parlent de GA4 ?"  → "GA4 analytics suivi web tracking"
  "explique la stratégie de celio"  → "stratégie commerciale positionnement marché"

Exemples avec contexte :
  contexte : "Q: quel est le chiffrage celio ?"
  question : "combien sera facturé la prestation ?"
  → "tarifs facturation prestation Celio"

  contexte : "Q: parle-moi de la stratégie fnac"
  question : "et les concurrents ?"
  → "concurrents positionnement marché Fnac"
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
        response = model.generate_content(prompt)
        rewritten = response.text.strip()
        return rewritten if rewritten else user_query
    except Exception:
        # Never block the retrieval pipeline on a rewrite failure
        return user_query
