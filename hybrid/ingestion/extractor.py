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
    # Google native formats
    "application/vnd.google-apps.document":     "Doc",
    "application/vnd.google-apps.spreadsheet":  "Sheet",
    "application/vnd.google-apps.presentation": "Slide",
    # PDF
    "application/pdf":                          "PDF",
    # Office Open XML (uploaded natively, not converted to Google format)
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document":   "Docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":         "Xlsx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "Pptx",
    # Legacy Office formats
    "application/msword":       "Docx",
    "application/vnd.ms-excel": "Xlsx",
    "application/vnd.ms-powerpoint": "Pptx",
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


def extract_from_docx(file_id: str, drive_service: Any) -> str:
    """
    Download and extract text from a .docx (or legacy .doc) file.

    Args:
        file_id:       Google Drive file ID.
        drive_service: Authenticated Google Drive API service object.

    Returns:
        Plain text — one paragraph per line, tables included.
    """
    import docx

    request = drive_service.files().get_media(fileId=file_id, supportsAllDrives=True)
    buf = io.BytesIO(request.execute())
    doc = docx.Document(buf)
    lines: list[str] = []
    for para in doc.paragraphs:
        if para.text.strip():
            lines.append(para.text)
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                lines.append("\t".join(cells))
    return "\n\n".join(lines)


def extract_from_xlsx(file_id: str, drive_service: Any) -> str:
    """
    Download and extract text from a .xlsx (or legacy .xls) file.

    Each sheet is exported as tab-separated rows.

    Args:
        file_id:       Google Drive file ID.
        drive_service: Authenticated Google Drive API service object.

    Returns:
        Plain text with one row per line, sheets separated by headers.
    """
    import openpyxl

    request = drive_service.files().get_media(fileId=file_id, supportsAllDrives=True)
    buf = io.BytesIO(request.execute())
    wb = openpyxl.load_workbook(buf, read_only=True, data_only=True)
    lines: list[str] = []
    for sheet in wb.worksheets:
        lines.append(f"=== {sheet.title} ===")
        for row in sheet.iter_rows(values_only=True):
            cells = [str(c) for c in row if c is not None and str(c).strip()]
            if cells:
                lines.append("\t".join(cells))
    wb.close()
    return "\n".join(lines)


def extract_from_pptx(file_id: str, drive_service: Any) -> str:
    """
    Download and extract text from a .pptx (or legacy .ppt) file.

    Args:
        file_id:       Google Drive file ID.
        drive_service: Authenticated Google Drive API service object.

    Returns:
        Plain text — one block per slide.
    """
    from pptx import Presentation

    request = drive_service.files().get_media(fileId=file_id, supportsAllDrives=True)
    buf = io.BytesIO(request.execute())
    prs = Presentation(buf)
    blocks: list[str] = []
    for i, slide in enumerate(prs.slides, 1):
        texts = []
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                texts.append(shape.text.strip())
        if texts:
            blocks.append(f"=== Slide {i} ===\n" + "\n".join(texts))
    return "\n\n".join(blocks)


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
        request = drive_service.files().get_media(fileId=file_id, supportsAllDrives=True)
        buf = io.BytesIO(request.execute())
        import fitz
        doc = fitz.open(stream=buf, filetype="pdf")
        pages = [page.get_text("text") for page in doc]
        doc.close()
        return "\n\n".join(p for p in pages if p.strip())
    elif mime in (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
    ):
        return extract_from_docx(file_id, drive_service)
    elif mime in (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
    ):
        return extract_from_xlsx(file_id, drive_service)
    elif mime in (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.ms-powerpoint",
    ):
        return extract_from_pptx(file_id, drive_service)
    else:
        raise ValueError(f"Unsupported MIME type '{mime}' for file {file_id}")


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
    folder_id = _resolve_folder_id(folder_name, drive_service)
    return _list_folder_by_id(folder_id, drive_service, recursive=recursive)


def _list_folder_by_id(
    folder_id: str, drive_service: Any, recursive: bool = True
) -> list[dict]:
    """
    List all supported files inside a folder given its Drive ID.

    Recursively descends into subfolders when *recursive* is True.
    Handles Drive API pagination — nextPageToken is followed until all
    items in every folder level have been retrieved.

    Args:
        folder_id:     Google Drive folder ID.
        drive_service: Authenticated Google Drive API service object.
        recursive:     Descend into subfolders.

    Returns:
        Flat list of FileInfo dicts for all supported files found.
    """
    all_query = (
        f"'{folder_id}' in parents "
        "and trashed = false"
    )

    result: list[dict] = []
    page_token = None

    while True:
        kwargs: dict = dict(
            q=all_query,
            # nextPageToken MUST be in fields or Drive won't return it
            fields=(
                "nextPageToken, "
                "files(id, name, mimeType, webViewLink, "
                "createdTime, modifiedTime, owners)"
            ),
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            pageSize=1000,
        )
        if page_token:
            kwargs["pageToken"] = page_token

        resp = drive_service.files().list(**kwargs).execute()

        for f in resp.get("files", []):
            mime = f.get("mimeType", "")

            # Recurse into subfolders (full pagination applied at every level)
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

        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_folder_by_name(
    name: str, drive_service: Any, parent_id: str | None = None
) -> str | None:
    """
    Return the Drive ID of the first folder matching *name*.

    If *parent_id* is given, the search is restricted to direct children of
    that folder — this prevents picking up a same-named folder elsewhere in
    Drive.

    Args:
        name:          Folder display name (exact match).
        drive_service: Authenticated Drive API service.
        parent_id:     Optional parent folder ID to scope the search.

    Returns:
        Folder ID string, or None if not found.
    """
    escaped = name.replace("'", "\\'")
    q = (
        f"name = '{escaped}' "
        "and mimeType = 'application/vnd.google-apps.folder' "
        "and trashed = false"
    )
    if parent_id:
        q += f" and '{parent_id}' in parents"

    resp = drive_service.files().list(
        q=q,
        fields="files(id, name)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
        pageSize=10,
    ).execute()
    folders = resp.get("files", [])
    return folders[0]["id"] if folders else None


def _resolve_folder_id(folder_name: str, drive_service: Any) -> str:
    """
    Resolve *folder_name* to a Drive folder ID, scoped to DRIVE_ROOT_FOLDER.

    Strategy:
    1. Locate DRIVE_ROOT_FOLDER (the shared root "Insight Factory - RAG").
    2. Search for *folder_name* as a **direct child** of that root — this
       ensures we never pick up a same-named folder from another Drive.
    3. Try case variants (original → UPPER → lower → Capitalize → Title).
    4. If the root itself cannot be found, fall back to a global search
       (permissive mode, emits a warning).

    Args:
        folder_name:   Client folder name (e.g. "CELIO").
        drive_service: Authenticated Drive API service.

    Returns:
        Drive folder ID string.

    Raises:
        ValueError: If the folder cannot be found even after all fallbacks.
    """
    from hybrid.config import DRIVE_ROOT_FOLDER

    # Step 1 — find root
    root_id = _find_folder_by_name(DRIVE_ROOT_FOLDER, drive_service)

    # Step 2+3 — find client folder within root (with case variants)
    for candidate in _case_variants(folder_name):
        fid = _find_folder_by_name(candidate, drive_service, parent_id=root_id)
        if fid:
            return fid

    # Step 4 — global fallback (root not shared with service account, etc.)
    if root_id is None:
        print(
            f"[extractor] WARN: root folder '{DRIVE_ROOT_FOLDER}' not found — "
            "falling back to global Drive search."
        )
    for candidate in _case_variants(folder_name):
        fid = _find_folder_by_name(candidate, drive_service, parent_id=None)
        if fid:
            return fid

    raise ValueError(
        f"Drive folder not found: {folder_name!r} "
        f"(searched inside '{DRIVE_ROOT_FOLDER}' and globally)"
    )


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
