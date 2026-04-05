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
    "You are an expert document analysis assistant. "
    "You can analyze, compare, synthesize, and cross-reference information "
    "from the provided documents. "
    "When comparing or cross-referencing sources, structure your answer clearly "
    "(tables, bullet points, sections by company/topic). "
    "ABSOLUTE RULE: every statement must be traceable to a source document. "
    "After each key piece of information, cite the source in this EXACT format: "
    "(INDEX__TOPIC - FileName, detail) "
    "Examples: "
    "\"Remote work is limited to 2 days (ACME__HR - Remote_Work_Policy.pdf)\" "
    "\"Revenue 2023: 36M€ (ACME__FINANCE - Annual_Report_2023.pdf, p.12)\" "
    "NEVER use your general knowledge to fill gaps or enrich answers. "
    "If the information is not in the documents, say so explicitly — "
    "never fill gaps with assumptions. "
    "Always respond in English."
)