"""
UI configuration — colors, labels, page metadata.
All visual constants live here so pages stay free of magic values.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_agent.config import MODEL as GENERATION_MODEL, GENERATION_SYSTEM_PROMPT
from shared.brand import ACTIVE as BRAND

# ── Page meta ─────────────────────────────────────────────────────────────────
APP_TITLE       = BRAND.name
APP_SUBTITLE    = BRAND.subtitle
APP_DESCRIPTION = BRAND.description
APP_ICON        = None
LAYOUT          = "wide"

# ── Brand colors (from the active profile theme) ──────────────────────────────
HYBRID_COLOR    = BRAND.theme.hybrid    # Hybrid pipeline accent
ACCENT_COLOR    = BRAND.theme.accent    # Primary accent
ACCENT_LIGHT    = BRAND.theme.accent_light  # Light accent tint
USER_COLOR      = BRAND.theme.user
TOOL_COLOR      = BRAND.theme.tool
ERROR_COLOR     = BRAND.theme.error

# ── Pipeline label ────────────────────────────────────────────────────────────
HYBRID_LABEL    = "Hybrid RAG"

# ── Agent ─────────────────────────────────────────────────────────────────────
ADK_APP_NAME    = "kb_streamlit"

# ── Hybrid defaults ───────────────────────────────────────────────────────────
DEFAULT_TOP_K           = 10
DEFAULT_RETRIEVAL_MODE  = "hybrid"
RETRIEVAL_MODES         = ["hybrid", "dense", "sparse"]
