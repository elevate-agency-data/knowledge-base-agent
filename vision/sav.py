"""
SAV image analysis — the vision pass behind the agent's image tools.

One call answers the three questions a workshop intake asks of a photo:

  1. *What is it?*        → product line, model guess, material, confidence
  2. *What's wrong?*      → damage list, each with severity and location
  3. *What's salvageable?*→ components to reuse, to replace, or beyond repair

The taxonomy comes from the active brand profile, so the model answers in the
Maison's own vocabulary instead of inventing generic labels. It also proposes
knowledge-base queries, which the agent runs through the RAG to ground the
repair procedure, warranty coverage and delays in real documents.

Deliberately NOT agent-driven: a single deterministic call, structured JSON out.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

_MAX_BYTES = 20 * 1024 * 1024   # refuse absurd inputs before touching the API


def _load_normalized(image_path: str):
    """Open an image with EXIF rotation applied, as RGB."""
    from PIL import Image, ImageOps

    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def _model_copy(img):
    """Clamp the copy sent to the model into the configured size range."""
    from PIL import Image

    from .config import IMAGE_MIN_SIDE, IMAGE_MAX_SIDE

    longest = max(img.size)
    if longest < IMAGE_MIN_SIDE:
        ratio = IMAGE_MIN_SIDE / longest
    elif longest > IMAGE_MAX_SIDE:
        ratio = IMAGE_MAX_SIDE / longest
    else:
        return img
    return img.resize(
        (max(1, round(img.size[0] * ratio)), max(1, round(img.size[1] * ratio))),
        Image.LANCZOS,
    )


def _png_bytes(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _bullets(values: tuple[str, ...] | list[str]) -> str:
    return "\n".join(f"- {v}" for v in values)


def _build_system_prompt(brand) -> str:
    """Assemble the analyst prompt from the brand's SAV taxonomy."""
    return (
        f"You are an after-sales (SAV) intake expert for {brand.name}, a luxury "
        "Maison. You receive ONE photo of a client's item and produce the "
        "workshop intake assessment.\n\n"
        "Work only from what is VISIBLE. Never invent a reference, a serial "
        "number or a date you cannot read in the photo. When you are unsure, "
        "say so through the confidence field and in `limitations` — an honest "
        "'not visible in this photo' is worth more than a confident guess.\n\n"
        "## Product lines (choose the closest one)\n"
        f"{_bullets(brand.vision_product_lines)}\n\n"
        "## Materials to recognise\n"
        f"{_bullets(brand.vision_materials)}\n\n"
        "## Damage vocabulary (use these terms)\n"
        f"{_bullets(brand.vision_damage_types)}\n\n"
        "## Components that can be salvaged or replaced\n"
        f"{_bullets(brand.vision_components)}\n\n"
        "## Output\n"
        "Return STRICT JSON only (no markdown fence), exactly this shape:\n"
        "{\n"
        '  "product": {\n'
        '    "line": "<one of the product lines above>",\n'
        '    "identification": "<what the item is, as precise as the photo '
        'allows, e.g. \'sac à main à rabat, format ~30 cm\'>",\n'
        '    "model_guess": "<model name IF genuinely recognisable, else \'\'>",\n'
        '    "materials": ["<materials actually visible>"],\n'
        '    "hardware": "<metal finish observed, or \'\'>",\n'
        '    "colour": "<dominant colour>",\n'
        '    "confidence": "high" | "medium" | "low"\n'
        "  },\n"
        '  "condition": {\n'
        '    "overall": "<one sentence on the general state>",\n'
        '    "grade": "excellent" | "bon" | "usage marqué" | "détérioré"\n'
        "  },\n"
        '  "damages": [\n'
        "    {\n"
        '      "type": "<a term from the damage vocabulary>",\n'
        '      "location": "<where on the item, e.g. \'angle inférieur droit\'>",\n'
        '      "severity": "high" | "medium" | "low",\n'
        '      "observation": "<what you actually see>",\n'
        '      "repairable": "atelier" | "entretien" | "irréversible" | "à évaluer"\n'
        "    }\n"
        "  ],\n"
        '  "salvageable": [\n'
        "    {\n"
        '      "component": "<a component from the list above>",\n'
        '      "verdict": "réutilisable" | "à remplacer" | "à évaluer en atelier",\n'
        '      "note": "<short justification from what is visible>"\n'
        "    }\n"
        "  ],\n"
        '  "recommended_queries": [\n'
        '    "<2 to 4 short questions to ask the knowledge base to ground the '
        'repair procedure, care instructions, warranty coverage or delays for '
        'THIS item and THESE damages>"\n'
        "  ],\n"
        '  "limitations": ["<what this single photo does not allow you to judge>"],\n'
        '  "summary": "<3-5 sentences: the item, its state, what the workshop '
        'would look at. Written for a client advisor.>"\n'
        "}\n\n"
        "Rules:\n"
        "- `damages` empty ONLY if the item is genuinely flawless.\n"
        "- List a component in `salvageable` only if it is visible or clearly "
        "implied by the product type.\n"
        "- Never quote a price or a firm delay — those come from the knowledge "
        "base, not from a photo.\n"
        "- Write every free-text field in French."
    )


def analyze_sav_image(image_path: str, client_note: str = "") -> dict:
    """Analyse one SAV photo against the active brand's taxonomy.

    Args:
        image_path:  Path to the image file.
        client_note: Optional context from the client/advisor (what happened,
                     what they are asking for). Steers the assessment.

    Returns:
        dict with ``status`` and, on success, ``product`` / ``condition`` /
        ``damages`` / ``salvageable`` / ``recommended_queries`` /
        ``limitations`` / ``summary``. On failure, ``message``.
    """
    from google.genai import types

    from shared.brand import ACTIVE
    from .config import VISION_MODEL
    from .client import generate_with_retry

    if not ACTIVE.has_vision:
        return {
            "status": "error",
            "message": (
                f"The '{ACTIVE.name}' profile defines no SAV image taxonomy "
                "(vision_* fields in shared/brand.py), so image analysis is "
                "disabled for this brand."
            ),
        }

    path = Path(image_path)
    if not path.is_file():
        return {"status": "error", "message": f"Image not found: {image_path}"}
    try:
        if path.stat().st_size > _MAX_BYTES:
            return {"status": "error", "message": "Image too large (max 20 MB)."}
    except OSError as exc:
        return {"status": "error", "message": f"Unreadable image: {exc}"}

    try:
        img = _load_normalized(str(path))
        img_bytes = _png_bytes(_model_copy(img))
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "message": f"Could not read the image: {exc}"}

    user_prompt = "Analyse the attached item and return the JSON assessment."
    if client_note.strip():
        user_prompt += (
            "\n\nContext given by the client / advisor (take it into account, "
            f"but never let it override what you actually see):\n{client_note.strip()}"
        )

    contents = [
        types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
        types.Part.from_text(text=user_prompt),
    ]
    config = types.GenerateContentConfig(
        system_instruction=_build_system_prompt(ACTIVE),
        response_mime_type="application/json",
        temperature=0.2,
    )

    try:
        response = generate_with_retry(
            model=VISION_MODEL, contents=contents, config=config
        )
        raw = response.text or "{}"
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "message": f"Vision model error: {exc}"}

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "status": "error",
            "message": f"Model returned non-JSON: {raw[:300]}",
        }

    product = data.get("product") or {}
    condition = data.get("condition") or {}
    return {
        "status": "success",
        "image_size": list(img.size),
        "product": {
            "line": product.get("line", ""),
            "identification": product.get("identification", ""),
            "model_guess": product.get("model_guess", ""),
            "materials": [str(m) for m in (product.get("materials") or [])],
            "hardware": product.get("hardware", ""),
            "colour": product.get("colour", ""),
            "confidence": product.get("confidence", "low"),
        },
        "condition": {
            "overall": condition.get("overall", ""),
            "grade": condition.get("grade", ""),
        },
        "damages": [
            {
                "type": d.get("type", ""),
                "location": d.get("location", ""),
                "severity": d.get("severity", "medium"),
                "observation": d.get("observation", ""),
                "repairable": d.get("repairable", "à évaluer"),
            }
            for d in (data.get("damages") or [])
        ],
        "salvageable": [
            {
                "component": s.get("component", ""),
                "verdict": s.get("verdict", "à évaluer en atelier"),
                "note": s.get("note", ""),
            }
            for s in (data.get("salvageable") or [])
        ],
        "recommended_queries": [str(q) for q in (data.get("recommended_queries") or [])],
        "limitations": [str(l) for l in (data.get("limitations") or [])],
        "summary": data.get("summary", ""),
    }
