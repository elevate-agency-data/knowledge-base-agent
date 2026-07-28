"""
ADK tools: SAV image analysis.

The user drops a photo in the Agent Chat page; it is stored once and given a
short ref (``img_a1b2c3d4``). These tools resolve that ref and read the photo:

- ``sav_analyze_image``  → what the item is, what is damaged, what can be saved
- ``sav_list_images``    → which photos this user has uploaded

The analysis is only half the answer. It returns ``recommended_queries``, which
the agent is expected to run through ``hybrid_query`` so the repair procedure,
warranty coverage and delays come from the knowledge base rather than from the
model's own idea of how a workshop operates.

Image refs are resolved against the caller's own uploads: the user id comes
from the runtime context set by the runner, not from the prompt.
"""

from __future__ import annotations


def sav_analyze_image(image_ref: str, client_note: str = "") -> dict:
    """
    Analyse a photo of a client's item for after-sales intake.

    Identifies the product, lists the visible damage, and states which
    components can be reused, which must be replaced, and which need a
    workshop evaluation. Also proposes knowledge-base questions.

    Use this whenever the user has uploaded a photo and asks what the item is,
    what is wrong with it, whether it can be repaired, or what a repair would
    involve. After calling it, run the returned ``recommended_queries`` through
    ``hybrid_query`` to ground procedure, warranty and delays in real documents
    — never answer those from the photo alone.

    Args:
        image_ref:   Reference of the uploaded photo, e.g. ``"img_a1b2c3d4"``.
                     Shown in the chat when the user uploads an image. Call
                     ``sav_list_images`` if you don't have it.
        client_note: Optional context from the client or advisor — what
                     happened to the item, what they are asking for. Improves
                     the assessment but never overrides what is visible.

    Returns:
        Dict with keys:
        - ``status``              : ``"success"`` or ``"error"``
        - ``product``             : line, identification, model_guess,
                                    materials, hardware, colour, confidence
        - ``condition``           : overall sentence + grade
        - ``damages``             : list of {type, location, severity,
                                    observation, repairable}
        - ``salvageable``         : list of {component, verdict, note}
        - ``recommended_queries`` : questions to run through ``hybrid_query``
        - ``limitations``         : what a single photo cannot establish
        - ``summary``             : advisor-facing summary
        - ``message``             : error message on failure
    """
    from rag_agent.runtime_context import get_user_id
    from vision.uploads import resolve_ref
    from vision.sav import analyze_sav_image

    ref = (image_ref or "").strip()
    if not ref:
        return {
            "status": "error",
            "message": "No image reference given. Ask the user to upload a "
                       "photo, or call sav_list_images to find an existing one.",
        }

    path = resolve_ref(ref, user_id=get_user_id())
    if path is None:
        return {
            "status": "error",
            "message": f"Unknown image reference '{ref}'. Call sav_list_images "
                       "to see the photos available for this user.",
        }

    return analyze_sav_image(str(path), client_note=client_note)


def sav_list_images() -> dict:
    """
    List the photos the current user has uploaded, newest first.

    Use it when the user mentions a photo but you don't have its reference, or
    to confirm an upload went through.

    Returns:
        Dict with keys:
        - ``status``  : ``"success"``
        - ``images``  : list of {ref, file_name} — pass ``ref`` to
                        ``sav_analyze_image``
        - ``count``   : number of photos available
    """
    from rag_agent.runtime_context import get_user_id
    from vision.uploads import list_uploads
    from pathlib import Path

    uploads = list_uploads(user_id=get_user_id())
    return {
        "status": "success",
        "count": len(uploads),
        "images": [
            {"ref": u["ref"], "file_name": Path(u["path"]).name}
            for u in uploads
        ],
    }
