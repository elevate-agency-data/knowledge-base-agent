"""
Shared google-genai client (Vertex AI backend) + quota retry.

Auth relies on the ADC set up in ``vision/__init__.py``.
"""

from __future__ import annotations

import logging
import time

from .config import PROJECT_ID, LOCATION, REQUEST_TIMEOUT_MS

_MAX_RETRIES = 3
_BASE_DELAY_S = 4.0
_BACKOFF = 2.0

_client = None


def get_client():
    """Singleton client, bound to the configured project and region."""
    global _client
    if _client is None:
        from google import genai
        from google.genai import types

        _client = genai.Client(
            vertexai=True,
            project=PROJECT_ID,
            location=LOCATION,
            http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
        )
    return _client


def _is_quota_error(exc: Exception) -> bool:
    """True only for a real quota overrun (429 / RESOURCE_EXHAUSTED).

    Reads the error code carried by the exception rather than grepping the
    message for "429" — those digits also show up in request ids and image
    dimensions, which would retry permanently-failed calls for a minute.
    """
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code == 429:
        return True
    status = str(getattr(exc, "status", "") or "")
    return status.upper() == "RESOURCE_EXHAUSTED"


def generate_with_retry(*, model: str, contents, config=None):
    """generate_content with exponential backoff on quota overrun."""
    client = get_client()
    delay = _BASE_DELAY_S

    for attempt in range(_MAX_RETRIES + 1):
        try:
            return client.models.generate_content(
                model=model, contents=contents, config=config
            )
        except Exception as exc:  # noqa: BLE001
            if not _is_quota_error(exc) or attempt == _MAX_RETRIES:
                raise
            logging.warning(
                "Vertex quota hit (attempt %d/%d) — retrying in %.0fs",
                attempt + 1, _MAX_RETRIES, delay,
            )
            time.sleep(delay)
            delay *= _BACKOFF
