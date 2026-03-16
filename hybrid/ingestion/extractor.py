"""
Document extraction utilities.

Supports:
- PDF files via pymupdf
- Google Docs, Sheets, and Slides via the Google Drive API
- Auto-dispatch by MIME type via ``extract_from_url()``
- Folder listing via ``list_drive_folder()``
"""

import io
from typing import Any


# ---------------------------------------------------------------------------
# FileInfo type alias
# ---------------------------------------------------------------------------

# Each file in a Drive folder is represented as a dict with these keys:
#
#   file_id    : str  — Google Drive file ID
#   file_name  : str  — Human-readable file name
#   file_type  : str  — "PDF" | "Doc" | "Sheet" | "Slide"
#   mime_type  : str  — Google MIME type string
#   source_url : str  — Canonical web URL to open the file
#   created_at : str  — ISO 8601 creation timestamp
#   updated_at : str  — ISO 8601 last-modified timestamp
#   author     : str  — Owner display name (best-effort)

_MIME_TO_TYPE: dict[str, str] = {
    "application/vnd.google-apps.document":     "Doc",
    "application/vnd.google-apps.spreadsheet":  "Sheet",
    "application/vnd.google-apps.presentation": "Slide",
    "application/pdf":                          "PDF",
}

_MIME_TO_EXPORT: dict[str, str] = {
    "application/vnd.google-apps.document":     "text/plain",
    "application/vnd.google-apps.spreadsheet":  "text/csv",
    "application/vnd.google-apps.presentation": "text/plain",
}


# ---------------------------------------------------------------------------
# Public extractors
# ---------------------------------------------------------------------------


def extract_from_pdf(file_path: str) -> str:
    """
    Extract plain text from a local PDF file using pymupdf.

    Args:
        file_path: Absolute or relative path to the PDF file.

    Returns:
        Concatenated text from all pages, separated by newlines.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If pymupdf cannot open the file.
    """
    import fitz  # pymupdf

    doc = fitz.open(file_path)
    pages: list[str] = []
    for page in doc:
        text = page.get_text("text")
        if text.strip():
            pages.append(text)
    doc.close()
    return "\n\n".join(pages)


def extract_from_google_doc(file_id: str, drive_service: Any) -> str:
    """
    Export and extract text from a Google Doc.

    Args:
        file_id:       Google Drive file ID.
        drive_service: Authenticated Google Drive API service object.

    Returns:
        Plain-text content of the document.
    """
    data = (
        drive_service.files()
        .export(fileId=file_id, mimeType="text/plain")
        .execute()
    )
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return str(data)


def extract_from_google_sheet(file_id: str, drive_service: Any) -> str:
    """
    Export and extract text from a Google Sheet (CSV export).

    Args:
        file_id:       Google Drive file ID.
        drive_service: Authenticated Google Drive API service object.

    Returns:
        CSV content of the first sheet as a string.
    """
    data = (
        drive_service.files()
        .export(fileId=file_id, mimeType="text/csv")
        .execute()
    )
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return str(data)


def extract_from_google_slide(file_id: str, drive_service: Any) -> str:
    """
    Export and extract speaker notes + slide text from a Google Slides file.

    Args:
        file_id:       Google Drive file ID.
        drive_service: Authenticated Google Drive API service object.

    Returns:
        Plain text content of all slides.
    """
    data = (
        drive_service.files()
        .export(fileId=file_id, mimeType="text/plain")
        .execute()
    )
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return str(data)


def extract_from_url(url: str, drive_service: Any) -> str:
    """
    Auto-extract text from a Google Drive file URL.

    Resolves the file ID from the URL, fetches its metadata to determine
    the MIME type, then dispatches to the appropriate extractor.

    Supported URL formats:
    - ``https://drive.google.com/file/d/<id>/view``
    - ``https://docs.google.com/document/d/<id>/edit``
    - ``https://docs.google.com/spreadsheets/d/<id>/edit``
    - ``https://docs.google.com/presentation/d/<id>/edit``

    Args:
        url:           Google Drive or Docs URL.
        drive_service: Authenticated Google Drive API service object.

    Returns:
        Extracted plain text.

    Raises:
        ValueError: If the file ID cannot be parsed from the URL.
    """
    file_id = _extract_file_id(url)
    if not file_id:
        raise ValueError(f"Cannot extract file ID from URL: {url!r}")

    meta = (
        drive_service.files()
        .get(fileId=file_id, fields="mimeType", supportsAllDrives=True)
        .execute()
    )
    mime = meta.get("mimeType", "")

    if mime == "application/vnd.google-apps.document":
        return extract_from_google_doc(file_id, drive_service)
    elif mime == "application/vnd.google-apps.spreadsheet":
        return extract_from_google_sheet(file_id, drive_service)
    elif mime == "application/vnd.google-apps.presentation":
        return extract_from_google_slide(file_id, drive_service)
    elif mime == "application/pdf":
        # Download and extract in-memory
        request = drive_service.files().get_media(fileId=file_id)
        buf = io.BytesIO(request.execute())
        import fitz

        doc = fitz.open(stream=buf, filetype="pdf")
        pages = [page.get_text("text") for page in doc]
        doc.close()
        return "\n\n".join(p for p in pages if p.strip())
    else:
        # Best-effort: try plain export
        try:
            data = (
                drive_service.files()
                .export(fileId=file_id, mimeType="text/plain")
                .execute()
            )
            if isinstance(data, bytes):
                return data.decode("utf-8", errors="replace")
            return str(data)
        except Exception as exc:
            raise ValueError(
                f"Unsupported MIME type '{mime}' for file {file_id}"
            ) from exc


def list_drive_folder(
    folder_name: str, drive_service: Any, recursive: bool = True
) -> list[dict]:
    """
    List all supported files inside a Google Drive folder.

    Searches Drive by folder name (first match wins if multiple exist).
    When *recursive* is True (default), descends into all subfolders
    automatically so that nested structures like CELIO/Livrables/file.pdf
    are discovered without extra calls.

    Args:
        folder_name:   Display name of the folder to search.
        drive_service: Authenticated Google Drive API service object.
        recursive:     If True, scan subfolders recursively (default: True).

    Returns:
        List of FileInfo dicts (see module docstring for schema).

    Raises:
        ValueError: If no folder with *folder_name* is found.
    """
    # Locate the folder — try multiple case variants for robustness
    candidates = _case_variants(folder_name)
    folders = []
    for candidate in candidates:
        escaped = candidate.replace("'", "\\'")
        folder_query = (
            f"name = '{escaped}' "
            "and mimeType = 'application/vnd.google-apps.folder' "
            "and trashed = false"
        )
        folder_resp = (
            drive_service.files()
            .list(
                q=folder_query,
                fields="files(id, name)",
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        folders = folder_resp.get("files", [])
        if folders:
            break

    if not folders:
        raise ValueError(f"Drive folder not found: {folder_name!r}")

    folder_id = folders[0]["id"]
    return _list_folder_by_id(folder_id, drive_service, recursive=recursive)


def _list_folder_by_id(
    folder_id: str, drive_service: Any, recursive: bool = True
) -> list[dict]:
    """
    List all supported files inside a folder given its Drive ID.

    Recursively descends into subfolders when *recursive* is True.

    Args:
        folder_id:     Google Drive folder ID.
        drive_service: Authenticated Google Drive API service object.
        recursive:     Descend into subfolders.

    Returns:
        Flat list of FileInfo dicts for all supported files found.
    """
    mime_filter = " or ".join(
        f"mimeType = '{m}'" for m in _MIME_TO_TYPE
    )
    # Query files + subfolders in one call
    all_query = (
        f"'{folder_id}' in parents "
        "and trashed = false"
    )
    resp = (
        drive_service.files()
        .list(
            q=all_query,
            fields=(
                "files(id, name, mimeType, webViewLink, "
                "createdTime, modifiedTime, owners)"
            ),
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            pageSize=1000,
        )
        .execute()
    )

    result: list[dict] = []
    for f in resp.get("files", []):
        mime = f.get("mimeType", "")

        # Recurse into subfolders
        if mime == "application/vnd.google-apps.folder" and recursive:
            result.extend(
                _list_folder_by_id(f["id"], drive_service, recursive=True)
            )
            continue

        if mime not in _MIME_TO_TYPE:
            continue

        file_type = _MIME_TO_TYPE.get(mime, "Other")
        owners = f.get("owners", [])
        author = owners[0].get("displayName", "") if owners else ""
        result.append(
            {
                "file_id":    f["id"],
                "file_name":  f.get("name", ""),
                "file_type":  file_type,
                "mime_type":  mime,
                "source_url": f.get("webViewLink", ""),
                "created_at": f.get("createdTime", ""),
                "updated_at": f.get("modifiedTime", ""),
                "author":     author,
            }
        )
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _case_variants(name: str) -> list[str]:
    """
    Return case variants of a folder name to handle Drive case sensitivity.

    Tries in order: original, UPPER, lower, Title Case, Title Case each word.

    Args:
        name: Folder name as provided by the user.

    Returns:
        Deduplicated list of variants to try, original first.
    """
    variants = [
        name,
        name.upper(),
        name.lower(),
        name.capitalize(),
        name.title(),
    ]
    # Deduplicate while preserving order
    seen = set()
    result = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            result.append(v)
    return result


def _extract_file_id(url: str) -> str:
    """
    Parse the Google Drive file ID from various URL formats.

    Args:
        url: Google Drive / Docs URL string.

    Returns:
        File ID string, or empty string if not found.
    """
    import re

    patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"/document/d/([a-zA-Z0-9_-]+)",
        r"/spreadsheets/d/([a-zA-Z0-9_-]+)",
        r"/presentation/d/([a-zA-Z0-9_-]+)",
        r"[?&]id=([a-zA-Z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return ""
