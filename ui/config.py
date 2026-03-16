"""
UI configuration — colors, labels, page metadata.
All visual constants live here so pages stay free of magic values.
"""

# ── Page meta ─────────────────────────────────────────────────────────────────
APP_TITLE       = "Knowledge Base Agent"
APP_ICON        = None
LAYOUT          = "wide"

# ── Brand colors ──────────────────────────────────────────────────────────────
VERTEX_COLOR    = "#4285F4"   # Google Blue
HYBRID_COLOR    = "#34A853"   # Google Green
USER_COLOR      = "#F8F9FA"
TOOL_COLOR      = "#FFF3CD"
ERROR_COLOR     = "#F8D7DA"

# ── Pipeline labels ───────────────────────────────────────────────────────────
VERTEX_LABEL    = "Vertex AI RAG"
HYBRID_LABEL    = "Hybrid RAG (local)"

# ── Agent ─────────────────────────────────────────────────────────────────────
ADK_APP_NAME    = "kb_streamlit"
ADK_USER_ID     = "streamlit_user"

# ── Hybrid defaults ───────────────────────────────────────────────────────────
DEFAULT_TOP_K           = 10
DEFAULT_RETRIEVAL_MODE  = "hybrid"
RETRIEVAL_MODES         = ["hybrid", "dense", "sparse"]
