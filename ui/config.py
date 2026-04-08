"""
UI configuration — colors, labels, page metadata.
All visual constants live here so pages stay free of magic values.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_agent.config import MODEL as GENERATION_MODEL, GENERATION_SYSTEM_PROMPT

# ── Page meta ─────────────────────────────────────────────────────────────────
APP_TITLE       = "AI for Customer Care"
APP_SUBTITLE    = "Workshop Chatbot Knowledge Base"
APP_ICON        = None
LAYOUT          = "wide"

# ── Brand colors ──────────────────────────────────────────────────────────────
VERTEX_COLOR    = "#4285F4"   # Google Blue
HYBRID_COLOR    = "#34A853"   # Google Green
ACCENT_COLOR    = "#4285F4"   # Primary accent
ACCENT_LIGHT    = "#EBF3FD"   # Light blue tint
USER_COLOR      = "#F8F9FA"
TOOL_COLOR      = "#FFF3CD"
ERROR_COLOR     = "#F8D7DA"

# ── Pipeline labels ───────────────────────────────────────────────────────────
VERTEX_LABEL    = "Naive RAG"
HYBRID_LABEL    = "Hybrid RAG"

# ── Agent ─────────────────────────────────────────────────────────────────────
ADK_APP_NAME    = "kb_streamlit"
ADK_USER_ID     = "streamlit_user"

# ── Hybrid defaults ───────────────────────────────────────────────────────────
DEFAULT_TOP_K           = 10
DEFAULT_RETRIEVAL_MODE  = "hybrid"
RETRIEVAL_MODES         = ["hybrid", "dense", "sparse"]
