"""
Configuration for the SAV image-analysis tools.

Vertex AI settings and the upload workdir. Values default to the ones already
used by the RAG agent (rag_agent/.env) so a single GCP setup covers both; every
value is overridable by environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent

# ── GCP / Vertex AI ───────────────────────────────────────────────────────────

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "knowledge-base-agent-485813")

# Same region as the RAG agent. gemini-2.5-flash serves vision there, so no
# separate location is needed (unlike image *generation*, which is global-only).
LOCATION = os.getenv("VISION_LOCATION", os.getenv("GOOGLE_CLOUD_LOCATION", "europe-west1"))

SERVICE_ACCOUNT_PATH = os.getenv(
    "GOOGLE_APPLICATION_CREDENTIALS", str(_PROJECT_ROOT / "rag_agent" / "key.json")
)

# ── Model ─────────────────────────────────────────────────────────────────────

# Vision + reasoning model used to analyse an SAV photo. flash is enough here:
# the task is description + classification against a supplied taxonomy, not
# spatial grounding.
VISION_MODEL = os.getenv("VISION_MODEL", "gemini-2.5-flash")

# Timeout per request (ms). Generous — a cold model call can take a while — but
# bounded so a silent request never hangs the UI.
REQUEST_TIMEOUT_MS = int(os.getenv("VISION_REQUEST_TIMEOUT_MS", str(3 * 60 * 1000)))

# ── Image handling ────────────────────────────────────────────────────────────

# The copy sent to the model is clamped into this range: tiny photos are
# upscaled, huge ones downscaled. Keeps cost and latency predictable without
# losing the detail damage assessment needs.
IMAGE_MIN_SIDE = int(os.getenv("VISION_IMAGE_MIN_SIDE", "768"))
IMAGE_MAX_SIDE = int(os.getenv("VISION_IMAGE_MAX_SIDE", "1536"))

# ── Upload workdir ────────────────────────────────────────────────────────────

# Photos uploaded from the Agent Chat page. Git-ignored, pruned on startup.
UPLOAD_DIR = Path(os.getenv("VISION_UPLOAD_DIR", str(_PROJECT_ROOT / "ui" / "data" / "uploads")))

UPLOAD_MAX_AGE_HOURS = float(os.getenv("VISION_UPLOAD_MAX_AGE_HOURS", "24"))
UPLOAD_MAX_MB = float(os.getenv("VISION_UPLOAD_MAX_MB", "200"))
