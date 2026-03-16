"""
Ingestion pipeline — extract, chunk, and enrich document chunks.

Public API::

    from hybrid.ingestion.extractor import extract_from_url, list_drive_folder
    from hybrid.ingestion.chunker   import chunk_text
    from hybrid.ingestion.metadata  import build_metadata
"""

from .extractor import (
    extract_from_pdf,
    extract_from_google_doc,
    extract_from_google_sheet,
    extract_from_google_slide,
    extract_from_url,
    list_drive_folder,
)
from .chunker import chunk_text, chunk_fixed, chunk_semantic, chunk_hierarchical
from .metadata import detect_language, detect_domaine, extract_tags, build_metadata

__all__ = [
    # Extractor
    "extract_from_pdf",
    "extract_from_google_doc",
    "extract_from_google_sheet",
    "extract_from_google_slide",
    "extract_from_url",
    "list_drive_folder",
    # Chunker
    "chunk_text",
    "chunk_fixed",
    "chunk_semantic",
    "chunk_hierarchical",
    # Metadata
    "detect_language",
    "detect_domaine",
    "extract_tags",
    "build_metadata",
]
