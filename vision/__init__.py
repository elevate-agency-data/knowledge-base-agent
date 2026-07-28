"""
SAV image analysis — vision tools for the agent.

Lets a user drop a photo of an item and get back a workshop-grade reading of
it: which product it is, what damage is visible, and which components can be
salvaged or must be replaced. The result is meant to be chained into the RAG
(repair procedures, warranty coverage, delays) by the agent.

The vocabulary is NOT hardcoded here — it comes from the active brand profile
(``shared/brand.py`` → ``vision_*`` fields), so each Maison names its products,
materials, damages and parts its own way.

Auth mirrors the RAG agent: the service-account key is exported as ADC so the
google-genai client picks it up, falling back to the machine's own ADC.
"""

import os
from pathlib import Path

from .config import SERVICE_ACCOUNT_PATH, PROJECT_ID, LOCATION

_key = Path(SERVICE_ACCOUNT_PATH)
if _key.is_file():
    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", str(_key))

os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", PROJECT_ID)
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", LOCATION)
