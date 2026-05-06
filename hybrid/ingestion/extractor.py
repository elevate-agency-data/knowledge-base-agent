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


# ── Tabular helpers (xlsx structure-aware chunking) ─────────────────────────


def _clean_cell(value: Any) -> str:
    """Render a cell value as plain text — empty cells become ``""``."""
    if value is None:
        return ""
    s = str(value).strip()
    # Markdown table-cell guard: pipes break the row layout, newlines split the row
    return s.replace("|", "/").replace("\n", " ").replace("\r", " ")


def _detect_header_row(rows: list[list[Any]], max_scan: int = 20) -> int:
    """
    Heuristically detect the most likely header row in ``rows``.

    A row is considered a header if:
    - at least 50% of its cells are non-empty, AND
    - at least 70% of those non-empty cells are strings (not numbers/dates).

    Bonus: if the next row is majority numeric, the match is confirmed.

    Returns:
        0-based index of the detected header row, or 0 if no row qualifies.
    """
    for i, row in enumerate(rows[: max_scan]):
        n_total = len(row)
        non_empty = [c for c in row if c is not None and str(c).strip()]
        if n_total == 0 or len(non_empty) < max(2, n_total * 0.5):
            continue
        text_count = sum(1 for c in non_empty if isinstance(c, str))
        if text_count < len(non_empty) * 0.7:
            continue
        # Bonus check on the next row
        if i + 1 < len(rows):
            next_non_empty = [c for c in rows[i + 1] if c is not None]
            if next_non_empty:
                numeric = sum(
                    1 for c in next_non_empty if isinstance(c, (int, float))
                )
                if numeric / max(len(next_non_empty), 1) > 0.5:
                    return i
        return i
    return 0


def _split_regions(
    rows: list[list[Any]], blank_threshold: int = 2
) -> list[tuple[int, list[list[Any]]]]:
    """
    Split a sheet into contiguous regions separated by ``blank_threshold``
    or more consecutive blank rows. Each region is a sub-table.

    Returns:
        List of ``(offset_in_original, region_rows)`` tuples.
    """
    regions: list[tuple[int, list[list[Any]]]] = []
    current: list[list[Any]] = []
    blank_count = 0
    region_start = 0
    for i, row in enumerate(rows):
        is_blank = all(c is None or str(c).strip() == "" for c in row)
        if is_blank:
            blank_count += 1
            if blank_count >= blank_threshold and current:
                regions.append((region_start, current))
                current = []
        else:
            if not current:
                region_start = i
            blank_count = 0
            current.append(row)
    if current:
        regions.append((region_start, current))
    return regions


def _forward_fill(header: list[str]) -> list[str]:
    """
    Forward-fill empty header cells with the previous non-empty value.
    Handles merged-cell headers where openpyxl returns ``None`` for the
    second cell of the merge.
    """
    out: list[str] = []
    last = ""
    for h in header:
        if h.strip():
            last = h
        out.append(last or h)
    return out


def _format_chunk_md(
    sheet_name: str,
    header: list[str],
    block_rows: list[list[Any]],
    row_start_excel: int,
    row_end_excel: int,
) -> str:
    """
    Render a header + data block as a list of key-value records.

    Each row becomes one line of the form::

        ligne 12 — Service: Stade Nautique | Catégorie: Charges patronales | 2022: 12345

    This format is friendlier to embeddings than a markdown ASCII table:
    the column label sits right next to its value in the same short
    context, so semantic queries like
    ``"charges patronales du Service Stade Nautique 2022"`` retrieve
    the right chunk reliably. The leading ``=== Sheet (lignes X-Y) ===``
    keeps location info visible in both the chunk panel and the LLM
    prompt.
    """
    safe_header = [_clean_cell(c) for c in header]
    n_cols = len(safe_header)

    lines = [f"=== {sheet_name} (lignes {row_start_excel}-{row_end_excel}) ==="]
    if n_cols and any(h.strip() for h in safe_header):
        lines.append("Colonnes : " + ", ".join(h for h in safe_header if h.strip()))
    lines.append("")

    for i, row in enumerate(block_rows):
        cells = [_clean_cell(c) for c in row]
        if n_cols:
            cells = (cells + [""] * n_cols)[: n_cols]
        if not any(c.strip() for c in cells):
            continue
        excel_row = row_start_excel + i

        if n_cols and any(h.strip() for h in safe_header):
            pairs = [
                f"{h.strip()}: {c}"
                for h, c in zip(safe_header, cells)
                if h.strip() and c.strip()
            ]
            if not pairs:
                continue
            lines.append(f"ligne {excel_row} — " + " | ".join(pairs))
        else:
            lines.append(
                f"ligne {excel_row} — " + " | ".join(c for c in cells if c.strip())
            )

    return "\n".join(lines)


_XLSX_MIMES: set[str] = {
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel",
}


def extract_chunks(
    file_info: dict,
    drive_service: Any,
    chunk_strategy: str = "fixed",
    chunk_params: dict | None = None,
    max_chars: int = 0,
) -> tuple[list[dict], str]:
    """
    Unified extraction + chunking dispatcher for the ingestion pipeline.

    - ``.xlsx`` / ``.xls`` → structure-aware chunks (one chunk per row block,
      header always repeated, location prefix in the content).
    - All other formats → text extract + ``chunk_text`` with the requested
      chunk strategy.

    Args:
        file_info:      FileInfo dict (must contain ``source_url``, ``mime_type``,
                        ``file_id``).
        drive_service:  Authenticated Drive API service.
        chunk_strategy: Chunk strategy for non-tabular files (passed to chunker).
        chunk_params:   Chunk strategy parameters.
        max_chars:      Optional truncation cap for non-tabular text. ``0`` = no cap.

    Returns:
        ``(chunks, sample_text)``. ``sample_text`` is a small excerpt suitable
        for language / domain detection (empty if no chunks were produced).
    """
    from hybrid.ingestion.chunker import chunk_text

    mime = file_info.get("mime_type", "")
    file_id = file_info.get("file_id", "")
    url = file_info.get("source_url", "")

    if mime in _XLSX_MIMES and file_id:
        chunks = extract_xlsx_chunks(file_id, drive_service, mime_type=mime)
        if not chunks:
            return [], ""
        # Inject the file name into both the LLM-facing body AND the
        # embedding-only text so semantic queries that mention a topic
        # only present in the file title still retrieve the chunk.
        fname = file_info.get("file_name", "")
        if fname:
            head = f"Fichier : {fname}"
            for c in chunks:
                c["content"] = f"{head}\n\n{c['content']}"
                if c.get("embedding_text"):
                    c["embedding_text"] = f"{head}\n{c['embedding_text']}"
        return chunks, chunks[0].get("content", "")[:5000]

    text = extract_from_url(url, drive_service)
    if not text.strip():
        return [], ""
    if max_chars and len(text) > max_chars:
        text = text[:max_chars]
    chunks = chunk_text(text, strategy=chunk_strategy, **(chunk_params or {}))
    return chunks, text[:5000]


def extract_xlsx_chunks(
    file_id: str,
    drive_service: Any,
    mime_type: str = "",
    max_rows_per_chunk: int = 30,
    max_chunk_body_chars: int = 8000,
) -> list[dict]:
    """
    Structure-aware chunking for ``.xlsx`` and legacy ``.xls`` files.

    Strategy: **one chunk per sheet region** (a contiguous block of rows
    not separated by blank rows). Each chunk holds the full table — the
    dense embedding stays compact (file name + sheet + columns +
    preamble, *no values*) so it does not blow past the embedding-model
    token budget regardless of table size. The full body, including
    every row and value, lives in ``content`` and is what the LLM reads
    and what BM25 indexes for exact-term matching.

    The two-channel design lets us:
    - keep ONE embedding per table no matter how many rows it has,
    - retain exact-term precision (years, references, row labels) via
      the sparse pipeline,
    - inject the entire matched table into the LLM prompt so it can
      find any cell value the query asks about.

    Args:
        file_id:        Google Drive file ID.
        drive_service:  Authenticated Drive service.
        mime_type:      MIME hint (``application/vnd.ms-excel`` for legacy ``.xls``).

    Returns:
        List of dicts compatible with the ingestion pipeline:
        ``id``, ``content``, ``embedding_text``, ``chunk_index``,
        ``chunk_total``, ``chunk_strategy = "xlsx_structured"``,
        ``parent_chunk_id = None``.
    """
    import uuid

    request = drive_service.files().get_media(fileId=file_id, supportsAllDrives=True)
    buf = io.BytesIO(request.execute())

    sheets: list[tuple[str, list[list[Any]]]] = []
    is_legacy_xls = mime_type == "application/vnd.ms-excel"

    if not is_legacy_xls:
        try:
            import openpyxl
            wb = openpyxl.load_workbook(buf, read_only=True, data_only=True)
            for sh in wb.worksheets:
                rows = [list(r) for r in sh.iter_rows(values_only=True)]
                sheets.append((sh.title, rows))
            wb.close()
        except Exception:
            buf.seek(0)
            sheets = []  # fall through to pandas

    if not sheets:
        # Legacy .xls or openpyxl failure → pandas (xlrd for binary, openpyxl for OOXML)
        import pandas as pd
        all_sheets = pd.read_excel(buf, sheet_name=None, dtype=object, header=None)
        for name, df in all_sheets.items():
            df = df.where(df.notna(), None)
            sheets.append((str(name), df.values.tolist()))

    pending: list[dict] = []

    for sheet_name, rows in sheets:
        if not rows:
            continue
        for region_offset, region_rows in _split_regions(rows):
            if not region_rows:
                continue
            header_idx = _detect_header_row(region_rows)
            header = _forward_fill([_clean_cell(c) for c in region_rows[header_idx]])
            data_rows = region_rows[header_idx + 1 :]

            # Drop trailing all-blank rows — openpyxl can yield ghost rows
            # (formatting-only cells) up to the sheet's nominal row limit.
            while data_rows and all(
                c is None or str(c).strip() == "" for c in data_rows[-1]
            ):
                data_rows = data_rows[:-1]

            # Free-text preamble = rows above the detected header
            preamble_lines: list[str] = []
            for r in region_rows[: header_idx]:
                cells = [_clean_cell(c) for c in r if c is not None]
                cells = [c for c in cells if c.strip()]
                if cells:
                    preamble_lines.append(" ".join(cells))
            preamble = " — ".join(preamble_lines)

            data_offset_excel = region_offset + header_idx + 1 + 1  # first data row, 1-indexed

            if not data_rows:
                pending.append({
                    "_sheet":    sheet_name,
                    "_header":   header,
                    "_data":     [],
                    "_preamble": preamble,
                    "_start":    region_offset + header_idx + 1,
                    "_end":      region_offset + header_idx + 1,
                })
                continue

            # Adaptive packing: pack rows up to BOTH max_rows_per_chunk AND
            # max_chunk_body_chars (greedy). Wide rows (dashboards with 30+
            # cols) trigger smaller blocks; narrow rows fill up to the row cap.
            # No data is lost — over-cap rows go into the next sub-chunk
            # rather than being truncated.
            i = 0
            n = len(data_rows)
            while i < n:
                block: list[list[Any]] = []
                block_chars = _estimate_overhead_chars(
                    sheet_name, header, preamble
                )
                while i < n and len(block) < max_rows_per_chunk:
                    rc = _estimate_record_chars(header, data_rows[i])
                    if block and block_chars + rc > max_chunk_body_chars:
                        break
                    block.append(data_rows[i])
                    block_chars += rc
                    i += 1
                if not block:
                    # Single row exceeds the cap on its own — accept it whole;
                    # the row count will be 1 but we never lose data.
                    block.append(data_rows[i])
                    i += 1
                pending.append({
                    "_sheet":    sheet_name,
                    "_header":   header,
                    "_data":     block,
                    "_preamble": preamble,
                    "_start":    data_offset_excel + (i - len(block)),
                    "_end":      data_offset_excel + i - 1,
                })

    total = len(pending)
    final_chunks: list[dict] = []
    for idx, c in enumerate(pending):
        body = _format_chunk_md(
            c["_sheet"], c["_header"], c["_data"], c["_start"], c["_end"]
        )
        if c["_preamble"]:
            body = f"Préambule : {c['_preamble']}\n\n{body}"
        # Adaptive packing already keeps the body within max_chunk_body_chars
        # whenever possible. The only case where it can still go over is a
        # single row larger than the cap on its own; we keep it untruncated
        # so no data is lost.
        # The dense embedding only uses the column-level summary (no row
        # values) — the body is what BM25 indexes and what the LLM reads.
        embedding_text = _format_table_summary(
            c["_sheet"], c["_header"], c["_preamble"], file_name="",
        )
        final_chunks.append({
            "id":              str(uuid.uuid4()),
            "content":         body,
            "embedding_text":  embedding_text,
            "chunk_index":     idx,
            "chunk_total":     total,
            "chunk_strategy":  "xlsx_structured",
            "parent_chunk_id": None,
        })
    return final_chunks


def _estimate_record_chars(header: list[str], row: list[Any]) -> int:
    """Approximate length of a 'ligne X — h: v | h: v ...' line."""
    total = 12  # 'ligne XXXX — '
    for h, c in zip(header, row):
        cv = "" if c is None else str(c)
        total += len(h) + len(cv) + 5  # 'h: cv | '
    return total


def _estimate_overhead_chars(
    sheet_name: str,
    header: list[str],
    preamble: str,
) -> int:
    """Approximate fixed overhead per chunk (filename injected later in extract_chunks)."""
    overhead = 30  # '=== sheet (lignes X-Y) ==='
    overhead += len(sheet_name)
    if preamble:
        overhead += len(preamble) + 15  # 'Préambule : ...\n\n'
    if any(h.strip() for h in header):
        overhead += sum(len(h) for h in header) + len(header) * 2 + 15
    return overhead


def _format_table_summary(
    sheet_name: str,
    header: list[str],
    preamble: str,
    file_name: str,
) -> str:
    """
    Compact column-level summary used **only** for the dense embedding.

    Drops every cell value (including textual labels) — the row data
    lives in ``chunk['content']`` and is matched by BM25 / served to
    the LLM. The dense vector therefore captures *what the table is
    about*, not *what numbers it contains*.
    """
    parts: list[str] = []
    if file_name:
        parts.append(f"Fichier : {file_name}")
    if preamble:
        parts.append(f"Préambule : {preamble}")
    parts.append(f"Feuille : {sheet_name}")
    safe_header = [h for h in (_clean_cell(c) for c in header) if h]
    if safe_header:
        parts.append("Colonnes : " + ", ".join(safe_header))
    return "\n".join(parts)


def extract_from_xlsx(file_id: str, drive_service: Any, mime_type: str = "") -> str:
    """
    Download and extract text from a .xlsx or legacy .xls file.

    Each sheet is exported as tab-separated rows.

    Strategy:
    - .xlsx (Office Open XML) → openpyxl
    - .xls  (legacy binary)   → pandas + xlrd fallback

    Args:
        file_id:       Google Drive file ID.
        drive_service: Authenticated Google Drive API service object.
        mime_type:     Optional MIME hint to pick the right reader. If empty,
                       openpyxl is tried first and we fall back on failure.

    Returns:
        Plain text with one row per line, sheets separated by headers.
    """
    request = drive_service.files().get_media(fileId=file_id, supportsAllDrives=True)
    buf = io.BytesIO(request.execute())

    is_legacy_xls = mime_type == "application/vnd.ms-excel"

    if not is_legacy_xls:
        try:
            import openpyxl
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
        except Exception:
            buf.seek(0)  # reset stream for pandas fallback

    # Legacy .xls (or openpyxl failure) — use pandas which dispatches
    # to xlrd / olefile for the legacy binary format.
    import pandas as pd
    sheets = pd.read_excel(buf, sheet_name=None, dtype=str)  # dict {name: df}
    lines = []
    for name, df in sheets.items():
        lines.append(f"=== {name} ===")
        df = df.fillna("")
        # Header row
        lines.append("\t".join(str(c) for c in df.columns))
        for _, row in df.iterrows():
            cells = [str(c) for c in row.values if str(c).strip()]
            if cells:
                lines.append("\t".join(cells))
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
        return extract_from_xlsx(file_id, drive_service, mime_type=mime)
    elif mime in (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.ms-powerpoint",
    ):
        return extract_from_pptx(file_id, drive_service)
    else:
        raise ValueError(f"Unsupported MIME type '{mime}' for file {file_id}")


def list_drive_tree(drive_service: Any) -> dict[str, dict[str, list[dict]]]:
    """
    Scan the Drive root folder and return a two-level tree: L1 → L2 → files.

    Supports two structural patterns under DRIVE_ROOT_FOLDER:

    1. **Standard two-level** — L1 folder contains only L2 subfolders::

        Rag_indica/
          └── Finances/                → tree["finances"]["budget"] = [FileInfo, ...]
              └── Budget/

    2. **Flat L1** — L1 folder contains files directly (no L2 subfolder)::

        Rag_indica/
          └── Patrimoine/              → tree["patrimoine"][""] = [FileInfo, ...]
              ├── inventaire.xlsx
              └── plan.pdf

    Mixed L1 (both files at L1 + L2 subfolders) is also supported — the
    files at L1 are collected under the empty notion key ``""`` while
    each L2 subfolder gets its own key.

    The empty notion key is later interpreted by ``hybrid_add_data_auto``
    as a request for a single-segment index name (no ``__`` separator).

    Returns:
        Dict like ``{"finances": {"budget": [...], "": [files_at_L1]}, ...}``.
        Keys are lowercased; the special ``""`` key holds files at L1.
    """
    from hybrid.config import DRIVE_ROOT_FOLDER

    root_id = _find_folder_by_name(DRIVE_ROOT_FOLDER, drive_service)
    if not root_id:
        raise ValueError(f"Drive root folder '{DRIVE_ROOT_FOLDER}' not found.")

    tree: dict[str, dict[str, list[dict]]] = {}

    # Level 1 — top-level folders
    l1_folders = _list_subfolders(root_id, drive_service)
    for l1_name, l1_id in l1_folders:
        l1_key = l1_name.strip().lower()
        tree[l1_key] = {}

        # Files DIRECTLY at L1 (non-recursive — recursion is reserved for L2)
        l1_direct_files = _list_folder_files_only(l1_id, drive_service)
        if l1_direct_files:
            tree[l1_key][""] = l1_direct_files

        # Level 2 — sub-folders, each gets its own (recursive) file list
        l2_folders = _list_subfolders(l1_id, drive_service)
        for l2_name, l2_id in l2_folders:
            l2_key = l2_name.strip().lower()
            files = _list_folder_by_id(l2_id, drive_service, recursive=True)
            tree[l1_key][l2_key] = files

    return tree


def _list_folder_files_only(folder_id: str, drive_service: Any) -> list[dict]:
    """
    Like ``_list_folder_by_id`` but **non-recursive** and skips subfolders.

    Used to collect files sitting directly at L1 (without an L2 wrapper)
    so they can be ingested into a single-segment index.
    """
    q = f"'{folder_id}' in parents and trashed = false"
    result: list[dict] = []
    page_token = None

    while True:
        kwargs: dict = dict(
            q=q,
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
            if mime == "application/vnd.google-apps.folder":
                continue  # skip subfolders — only direct files
            if mime not in _MIME_TO_TYPE:
                continue
            owners = f.get("owners", [])
            author = owners[0].get("displayName", "") if owners else ""
            result.append({
                "file_id":    f["id"],
                "file_name":  f.get("name", ""),
                "file_type":  _MIME_TO_TYPE.get(mime, "Other"),
                "mime_type":  mime,
                "source_url": f.get("webViewLink", ""),
                "created_at": f.get("createdTime", ""),
                "updated_at": f.get("modifiedTime", ""),
                "author":     author,
            })

        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return result


def _list_subfolders(parent_id: str, drive_service: Any) -> list[tuple[str, str]]:
    """
    List direct child folders of *parent_id*.

    Returns:
        List of (folder_name, folder_id) tuples.
    """
    q = (
        f"'{parent_id}' in parents "
        "and mimeType = 'application/vnd.google-apps.folder' "
        "and trashed = false"
    )
    results: list[tuple[str, str]] = []
    page_token = None
    while True:
        kwargs: dict = dict(
            q=q,
            fields="nextPageToken, files(id, name)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            pageSize=1000,
        )
        if page_token:
            kwargs["pageToken"] = page_token
        resp = drive_service.files().list(**kwargs).execute()
        for f in resp.get("files", []):
            results.append((f["name"], f["id"]))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return results


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
