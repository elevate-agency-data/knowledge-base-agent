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
MODEL = "gemini-2.5-pro"