"""
ADK tool: hybrid_add_data

Ingests Google Drive folders into a hybrid index:
  1. Connects to Drive via service-account credentials.
  2. Lists all supported files in each named folder.
  3. Skips files already indexed (deduplication by file_name + updated_at — safe to re-run).
  4. If a file with the same name exists but with a newer updated_at, deletes the stale
     chunks and re-ingests the updated version.
  5. Extracts text content from each file.
  6. Chunks the text with the configured strategy.
  7. Detects metadata (language, domain, tags).
  8. Embeds each chunk in batches.
  9. Inserts everything into the store.

For large Drive folders use max_files to process in multiple passes:
  Pass 1 → hybrid_add_data(..., max_files=20)
  Pass 2 → hybrid_add_data(..., max_files=20)   ← already-indexed files skipped
  ...
"""

from hybrid.config import (
    DEFAULT_EMBEDDING_MODEL,
    SERVICE_ACCOUNT_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SEMANTIC_BREAKPOINT_THRESHOLD,
    PARENT_CHUNK_SIZE,
    CHILD_CHUNK_SIZE,
)
from hybrid.stores import get_store
from hybrid.embeddings import get_embedding_model
from hybrid.ingestion.extractor import list_drive_folder, extract_from_url
from hybrid.ingestion.chunker import chunk_text
from hybrid.ingestion.metadata import build_metadata

# Max characters extracted per file to protect against huge documents
_MAX_FILE_CHARS = 500_000  # ~100 pages


def hybrid_add_data(
    index_name: str,
    folder_names: list[str],
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    chunk_strategy: str = "fixed",
    max_files: int = 0,
) -> dict:
    """
    Ingest Google Drive folders into a hybrid RAG index.

    For each folder, lists all supported files (PDF, Doc, Sheet, Slide),
    skips files already present in the index, extracts text, chunks it,
    and stores the embedded chunks in the configured store backend.

    **Resumable** — already-indexed files are detected by ``file_name`` +
    ``updated_at`` and skipped, so the tool can be called repeatedly on the
    same folder without creating duplicate chunks.  If a file has been updated
    in Drive (same name, newer ``updated_at``), the stale chunks are deleted
    and the file is re-ingested cleanly.

    **For large folders** use ``max_files`` to process in batches and avoid
    ADK request timeouts::

        # First call — processes up to 20 files
        hybrid_add_data(index_name="idx", folder_names=["Big Folder"], max_files=20)
        # Second call — skips the 20 already indexed, processes the next 20
        hybrid_add_data(index_name="idx", folder_names=["Big Folder"], max_files=20)

    Args:
        index_name:      Target index to insert chunks into.
        folder_names:    List of Google Drive folder names to ingest.
        embedding_model: Embedding model key (default from config).
        chunk_strategy:  ``"fixed"``, ``"semantic"``, or ``"hierarchical"``.
        max_files:       Maximum number of **new** files to process in this
                         call.  0 (default) means no limit — process all.
                         Use 10–30 for large folders to avoid timeouts.

    Returns:
        Dict with keys:
        - ``status``            : ``"success"`` or ``"error"``
        - ``message``           : Human-readable summary
        - ``index_name``        : Echo of the target index
        - ``files_processed``   : Files ingested in this call
        - ``files_already_indexed``: Files skipped (already in index, unchanged)
        - ``files_updated``     : Files whose stale chunks were replaced
        - ``files_remaining``   : Estimated files still to process (if max_files set)
        - ``chunks_created``    : Total chunks inserted in this call
        - ``files_skipped``     : Files that failed with error reasons
        - ``folders_not_found`` : Folder names not found in Drive
        - ``embedding_model``   : Model used
        - ``chunk_strategy``    : Strategy used
    """
    index_name = index_name.strip().lower()
    if not index_name:
        return {"status": "error", "message": "index_name is required."}
    if not folder_names:
        return {"status": "error", "message": "folder_names must not be empty."}

    # -- Build Drive service -------------------------------------------------
    try:
        from googleapiclient.discovery import build
        from google.oauth2 import service_account

        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_PATH,
            scopes=["https://www.googleapis.com/auth/drive.readonly"],
        )
        drive_service = build("drive", "v3", credentials=creds)
    except Exception as exc:
        return {
            "status":  "error",
            "message": f"Failed to connect to Google Drive: {exc}",
        }

    # -- Load embedding model ------------------------------------------------
    try:
        embedder = get_embedding_model(embedding_model)
        emb_dim = embedder.get_dimension()
    except Exception as exc:
        return {
            "status":  "error",
            "message": f"Failed to load embedding model '{embedding_model}': {exc}",
        }

    # -- Open store + load already-indexed file state ------------------------
    store = get_store()
    already_indexed_urls       = _get_indexed_urls(store, index_name)
    # maps file_name → updated_at (YYYY-MM-DD) for all chunks in the index
    indexed_file_dates         = _get_indexed_file_names_with_dates(store, index_name)

    # -- Chunk strategy parameters -------------------------------------------
    chunk_params: dict = {}
    if chunk_strategy == "fixed":
        chunk_params = {"size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP}
    elif chunk_strategy == "semantic":
        chunk_params = {"threshold": SEMANTIC_BREAKPOINT_THRESHOLD}
    elif chunk_strategy == "hierarchical":
        chunk_params = {
            "parent_size": PARENT_CHUNK_SIZE,
            "child_size":  CHILD_CHUNK_SIZE,
        }

    # -- Process folders -----------------------------------------------------
    files_processed        = 0
    files_already_indexed  = 0
    files_updated          = 0
    chunks_created         = 0
    files_remaining        = 0
    files_skipped: list[dict] = []
    folders_not_found: list[str] = []

    for folder_name in folder_names:
        try:
            file_infos = list_drive_folder(folder_name, drive_service)
        except ValueError:
            folders_not_found.append(folder_name)
            continue
        except Exception as exc:
            folders_not_found.append(f"{folder_name} (error: {exc})")
            continue

        for file_info in file_infos:
            url = file_info.get("source_url", "")
            file_name = file_info.get("file_name", "")
            # Parse to YYYY-MM-DD — same format stored in the DB
            file_updated_at = _parse_date_str(file_info.get("updated_at", ""))

            # -- Deduplication logic -----------------------------------------
            # 1. URL already indexed → same exact file, skip unconditionally
            if url and url in already_indexed_urls:
                files_already_indexed += 1
                continue

            # 2. Same file_name already indexed
            if file_name and file_name in indexed_file_dates:
                stored_date = indexed_file_dates[file_name]
                if stored_date == file_updated_at:
                    # Same version — nothing to do
                    files_already_indexed += 1
                    continue
                # Different updated_at → file was modified in Drive:
                # purge stale chunks so we ingest a clean version
                store.delete_chunks_by_file_name(file_name, index_name)
                files_updated += 1
                # Remove from local cache so URL check stays coherent
                already_indexed_urls = {
                    u for u in already_indexed_urls
                    if u != url
                }

            # -- Respect max_files limit -------------------------------------
            if max_files and files_processed >= max_files:
                files_remaining += 1
                continue

            try:
                # 1. Extract text (truncate to protect against huge files)
                text = extract_from_url(url, drive_service)
                if not text.strip():
                    files_skipped.append(
                        {"file": file_info["file_name"], "reason": "empty content"}
                    )
                    continue

                if len(text) > _MAX_FILE_CHARS:
                    text = text[:_MAX_FILE_CHARS]

                # 2. Chunk
                chunks = chunk_text(text, strategy=chunk_strategy, **chunk_params)
                if not chunks:
                    files_skipped.append(
                        {"file": file_info["file_name"], "reason": "no chunks produced"}
                    )
                    continue

                # 3. Embed in batches of 64 to limit memory pressure
                texts = [c["content"] for c in chunks]
                embeddings = _embed_in_batches(embedder, texts, batch_size=64)

                # 4. Assemble metadata + attach embeddings
                records: list[dict] = []
                for chunk, embedding in zip(chunks, embeddings):
                    meta = build_metadata(
                        file_info=file_info,
                        chunk=chunk,
                        embedding_model=embedding_model,
                        index_name=index_name,
                        embedding_dim=emb_dim,
                    )
                    meta["embedding"] = embedding
                    records.append(meta)

                # 5. Insert into store
                store.insert_chunks(records)

                files_processed += 1
                chunks_created  += len(records)

            except Exception as exc:
                files_skipped.append(
                    {"file": file_info.get("file_name", "unknown"), "reason": str(exc)}
                )

    # -- Build response ------------------------------------------------------
    done = max_files == 0 or files_remaining == 0
    status = "success" if (files_processed > 0 or files_already_indexed > 0) else "error"

    resume_hint = (
        f" {files_remaining} file(s) still to process — call again with the same "
        f"parameters to continue (already-indexed files will be skipped)."
        if files_remaining > 0 else ""
    )

    return {
        "status":               status,
        "message": (
            f"Ingestion {'complete' if done else 'partial'}. "
            f"{files_processed} file(s) processed, "
            f"{chunks_created} chunk(s) created into index '{index_name}'. "
            f"{files_already_indexed} file(s) already indexed (skipped). "
            f"{files_updated} file(s) updated (stale chunks replaced)."
            f"{resume_hint}"
        ),
        "index_name":           index_name,
        "files_processed":      files_processed,
        "files_already_indexed": files_already_indexed,
        "files_updated":        files_updated,
        "files_remaining":      files_remaining,
        "chunks_created":       chunks_created,
        "files_skipped":        files_skipped,
        "folders_not_found":    folders_not_found,
        "embedding_model":      embedding_model,
        "chunk_strategy":       chunk_strategy,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_indexed_urls(store, index_name: str) -> set[str]:
    """
    Return the set of source_urls already present in the index.

    Uses a raw DuckDB query for efficiency; falls back to empty set on error.

    Args:
        store:      Store instance (DuckDBStore or AlloyDBStore).
        index_name: Index to check.

    Returns:
        Set of source_url strings already indexed.
    """
    try:
        from hybrid.stores.duckdb_store import DuckDBStore
        if isinstance(store, DuckDBStore):
            conn = store._get_conn()
            rows = conn.execute(
                "SELECT DISTINCT source_url FROM chunks WHERE index_name = ?",
                [index_name],
            ).fetchall()
            return {row[0] for row in rows if row[0]}
    except Exception:
        pass
    return set()


def _get_indexed_file_names_with_dates(store, index_name: str) -> dict[str, str]:
    """
    Return a mapping of file_name → updated_at (YYYY-MM-DD) for all files
    already present in the index.

    When the same file_name appears in multiple chunks, the most recent
    updated_at is used (so a partially-ingested update doesn't get skipped).

    Args:
        store:      Store instance (DuckDBStore or AlloyDBStore).
        index_name: Index to check.

    Returns:
        Dict mapping file_name → updated_at string (YYYY-MM-DD or empty).
    """
    try:
        from hybrid.stores.duckdb_store import DuckDBStore
        if isinstance(store, DuckDBStore):
            conn = store._get_conn()
            rows = conn.execute(
                """
                SELECT file_name, MAX(CAST(updated_at AS VARCHAR))
                FROM chunks
                WHERE index_name = ?
                GROUP BY file_name
                """,
                [index_name],
            ).fetchall()
            return {row[0]: (row[1] or "") for row in rows if row[0]}
    except Exception:
        pass
    return {}


def _parse_date_str(dt_str: str) -> str:
    """
    Extract the YYYY-MM-DD prefix from an ISO 8601 datetime string.

    Returns an empty string if parsing fails.
    """
    import re
    if not dt_str:
        return ""
    match = re.match(r"(\d{4}-\d{2}-\d{2})", dt_str)
    return match.group(1) if match else ""


def _embed_in_batches(
    embedder, texts: list[str], batch_size: int = 64
) -> list[list[float]]:
    """
    Embed texts in fixed-size batches to limit peak memory usage.

    Args:
        embedder:   Loaded BaseEmbedding instance.
        texts:      All texts to embed.
        batch_size: Number of texts per batch.

    Returns:
        Flat list of embedding vectors in the same order as texts.
    """
    results: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        results.extend(embedder.embed_documents(batch))
    return results
