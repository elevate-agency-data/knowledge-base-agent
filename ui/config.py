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

# ── Brand colors (Lacoste green on white) ────────────────────────────────────
VERTEX_COLOR    = "#006A4E"   # Lacoste dark green
HYBRID_COLOR    = "#00A651"   # Lacoste bright green
ACCENT_COLOR    = "#006A4E"   # Primary accent
ACCENT_LIGHT    = "#E6F4ED"   # Light green tint for backgrounds
USER_COLOR      = "#F5FAF7"   # Soft green-white
TOOL_COLOR      = "#E6F4ED"   # Light green tint
ERROR_COLOR     = "#F8D7DA"

# ── Pipeline labels ───────────────────────────────────────────────────────────
VERTEX_LABEL    = "Naive RAG"
HYBRID_LABEL    = "Hybrid RAG"

# ── Agent ─────────────────────────────────────────────────────────────────────
ADK_APP_NAME    = "kb_streamlit"

# ── Hybrid defaults ───────────────────────────────────────────────────────────
DEFAULT_TOP_K           = 10
DEFAULT_RETRIEVAL_MODE  = "hybrid"
RETRIEVAL_MODES         = ["hybrid", "dense", "sparse"]
