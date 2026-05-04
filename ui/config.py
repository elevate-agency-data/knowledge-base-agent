"""
UI configuration — colors, labels, page metadata.
All visual constants live here so pages stay free of magic values.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_agent.config import MODEL as GENERATION_MODEL, GENERATION_SYSTEM_PROMPT

# ── Page meta ─────────────────────────────────────────────────────────────────
APP_TITLE       = "Indica"
APP_SUBTITLE    = "Knowledge base for the city hall"
APP_ICON        = None
LAYOUT          = "wide"

# ── Brand colors ──────────────────────────────────────────────────────────────
HYBRID_COLOR    = "#34A853"   # Hybrid pipeline accent
ACCENT_COLOR    = "#4285F4"   # Primary accent
ACCENT_LIGHT    = "#EBF3FD"   # Light accent tint
USER_COLOR      = "#F8F9FA"
TOOL_COLOR      = "#FFF3CD"
ERROR_COLOR     = "#F8D7DA"

# ── Pipeline label ────────────────────────────────────────────────────────────
HYBRID_LABEL    = "Hybrid RAG"

# ── Agent ─────────────────────────────────────────────────────────────────────
ADK_APP_NAME    = "kb_streamlit"

# ── Hybrid defaults ───────────────────────────────────────────────────────────
DEFAULT_TOP_K           = 10
DEFAULT_RETRIEVAL_MODE  = "hybrid"
RETRIEVAL_MODES         = ["hybrid", "dense", "sparse"]
