"""
Shared utility: gemini_retry

Wraps vertexai GenerativeModel.generate_content with automatic retry
and exponential backoff on 429 / RESOURCE_EXHAUSTED errors.
"""

from __future__ import annotations

import time
import logging

_MAX_RETRIES  = 4
_BASE_DELAY_S = 5.0   # seconds before first retry
_BACKOFF      = 2.0   # multiplier per attempt  (5s, 10s, 20s, 40s)


def generate_with_retry(model, prompt, **kwargs):
    """
    Call model.generate_content(prompt) with exponential backoff on quota errors.

    Args:
        model:   A vertexai GenerativeModel instance.
        prompt:  The prompt string or content parts.
        **kwargs: Extra kwargs forwarded to generate_content.

    Returns:
        The GenerativeModel response object.

    Raises:
        The last exception if all retries are exhausted.
    """
    delay = _BASE_DELAY_S
    last_exc: Exception | None = None

    for attempt in range(_MAX_RETRIES + 1):
        try:
            return model.generate_content(prompt, **kwargs)
        except Exception as exc:
            msg = str(exc)
            is_quota = "429" in msg or "RESOURCE_EXHAUSTED" in msg
            if not is_quota or attempt == _MAX_RETRIES:
                raise
            logging.warning(
                "Gemini quota hit (attempt %d/%d) — retrying in %.0fs",
                attempt + 1, _MAX_RETRIES, delay,
            )
            time.sleep(delay)
            delay *= _BACKOFF
            last_exc = exc

    raise last_exc  # never reached but satisfies type checkers
