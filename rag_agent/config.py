"""
Configuration settings for the RAG Agent.

These settings are used by the various RAG tools.
Vertex AI initialization is performed in the package's __init__.py
"""

# Vertex AI settings
PROJECT_ID = "knowledge-base-agent-485813"
LOCATION = "europe-west1"
SERVICE_ACCOUNT_PATH = "rag_agent/key.json" 

# RAG settings
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_TOP_K = 10
DEFAULT_DISTANCE_THRESHOLD = 0.5
DEFAULT_EMBEDDING_MODEL = "publishers/google/models/text-embedding-005"
DEFAULT_EMBEDDING_REQUESTS_PER_MIN = 1000
MODEL = "gemini-2.5-flash"  # gemini-2.5-pro | gemini-2.5-flash

# System prompt shared by all pipelines for final answer generation.
GENERATION_SYSTEM_PROMPT = (
    "Tu es un assistant expert en analyse documentaire. "
    "Tu peux analyser, comparer, synthétiser et croiser les informations "
    "issues des documents fournis. "
    "Quand tu compares ou croises des sources, structure ta réponse clairement "
    "(tableau, bullet points, sections par entreprise/thème). "
    "RÈGLE ABSOLUE : chaque affirmation doit être traçable à un document source. "
    "Après chaque information clé, cite la source avec ce format EXACT : "
    "[source: nom_du_fichier.ext] "
    "Exemples : "
    "\"Le télétravail est limité à 2 jours [source: Charte_RH.pdf]\" "
    "\"CA 2023 : 36M€ [source: Rapport_Annuel_2023.pdf]\" "
    "Ne mets JAMAIS de nom d'index dans la citation, uniquement le nom du fichier. "
    "N'utilise JAMAIS ta connaissance générale pour compléter ou enrichir. "
    "Si l'information ne figure pas dans les documents, dis-le explicitement — "
    "ne comble jamais les lacunes par des suppositions."
)