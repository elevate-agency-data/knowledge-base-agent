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
from hybrid.ingestion.extractor import (
    list_drive_folder,
    list_drive_tree,
    extract_from_url,
)
from hybrid.ingestion.chunker import chunk_text
from hybrid.ingestion.metadata import build_metadata

# Separator used to build hierarchical index names: company__notion
INDEX_SEP = "__"

# Max characters extracted per file to protect against huge documents
_MAX_FILE_CHARS = 500_000  # ~100 pages


def hybrid_add_data(
    index_name: str,
    folder_names: list[str],
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
    embedding_model = DEFAULT_EMBEDDING_MODEL  # always use configured default

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

    print(f"[hybrid_add_data] Starting ingestion -> index='{index_name}' | folders={folder_names} | model={embedding_model} | strategy={chunk_strategy}")

    for folder_name in folder_names:
        print(f"[hybrid_add_data] Listing folder: '{folder_name}'")
        try:
            file_infos = list_drive_folder(folder_name, drive_service)
        except ValueError:
            print(f"[hybrid_add_data]   ERROR Folder not found: '{folder_name}'")
            folders_not_found.append(folder_name)
            continue
        except Exception as exc:
            print(f"[hybrid_add_data]   ERROR listing '{folder_name}': {exc}")
            folders_not_found.append(f"{folder_name} (error: {exc})")
            continue

        print(f"[hybrid_add_data]   Found {len(file_infos)} file(s) in '{folder_name}'")

        for file_info in file_infos:
            url = file_info.get("source_url", "")
            file_name = file_info.get("file_name", "")
            # Parse to YYYY-MM-DD — same format stored in the DB
            file_updated_at = _parse_date_str(file_info.get("updated_at", ""))

            # -- Deduplication logic -----------------------------------------
            # 1. URL already indexed → same exact file, skip unconditionally
            if url and url in already_indexed_urls:
                print(f"[hybrid_add_data]   Skip (URL already indexed): {file_name}")
                files_already_indexed += 1
                continue

            # 2. Same file_name already indexed
            if file_name and file_name in indexed_file_dates:
                stored_date = indexed_file_dates[file_name]
                if stored_date == file_updated_at:
                    # Same version — nothing to do
                    print(f"[hybrid_add_data]   Skip (unchanged): {file_name}")
                    files_already_indexed += 1
                    continue
                # Different updated_at → file was modified in Drive:
                # purge stale chunks so we ingest a clean version
                print(f"[hybrid_add_data]   Update detected ({stored_date} -> {file_updated_at}): {file_name}")
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

            print(f"[hybrid_add_data]   Processing ({files_processed + 1}): {file_name}")
            try:
                # 1. Extract text (truncate to protect against huge files)
                print(f"[hybrid_add_data]      Extracting text…")
                text = extract_from_url(url, drive_service)
                if not text.strip():
                    print(f"[hybrid_add_data]      WARN Empty content - skipped")
                    files_skipped.append(
                        {"file": file_info["file_name"], "reason": "empty content"}
                    )
                    continue

                if len(text) > _MAX_FILE_CHARS:
                    text = text[:_MAX_FILE_CHARS]

                # 2. Chunk
                chunks = chunk_text(text, strategy=chunk_strategy, **chunk_params)
                if not chunks:
                    print(f"[hybrid_add_data]      WARN No chunks produced - skipped")
                    files_skipped.append(
                        {"file": file_info["file_name"], "reason": "no chunks produced"}
                    )
                    continue

                # Detect language and domain ONCE on the full document text
                from hybrid.ingestion.metadata import detect_language, detect_domaine
                _sample = text[:5000]
                doc_language = detect_language(_sample)
                doc_domaine  = detect_domaine(_sample, file_info.get("file_name", ""))

                print(f"[hybrid_add_data]      Chunked → {len(chunks)} chunks | {doc_domaine}/{doc_language} | Embedding…")

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
                        doc_language=doc_language,
                        doc_domaine=doc_domaine,
                    )
                    meta["embedding"] = embedding
                    records.append(meta)

                # 5. Insert into store
                store.insert_chunks(records)

                files_processed += 1
                chunks_created  += len(records)
                print(f"[hybrid_add_data]      Done -> {len(records)} chunks inserted (total: {chunks_created})")

            except Exception as exc:
                print(f"[hybrid_add_data]      ERROR: {exc}")
                files_skipped.append(
                    {"file": file_info.get("file_name", "unknown"), "reason": str(exc)}
                )

    # -- Build response ------------------------------------------------------
    print(
        f"[hybrid_add_data] Ingestion done -> index='{index_name}' | "
        f"processed={files_processed} | skipped={files_already_indexed} | "
        f"updated={files_updated} | chunks={chunks_created} | errors={len(files_skipped)}"
    )
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


# ---------------------------------------------------------------------------
# Auto-ingest from Drive tree (company → notion → files)
# ---------------------------------------------------------------------------


def hybrid_add_data_auto(
    company_filter: list[str] | None = None,
    chunk_strategy: str = "fixed",
    max_files_per_index: int = 0,
) -> dict:
    """
    Auto-ingest all companies and notions from the Drive tree.

    Scans DRIVE_ROOT_FOLDER for the two-level structure::

        RAG (Test & Co)/
          ├── Celio/
          │    ├── RH/         → index "celio__rh"
          │    └── Commercial/  → index "celio__commercial"
          └── ClientB/
               └── Juridique/  → index "clientb__juridique"

    For each company/notion pair:
      1. Creates the index ``company__notion`` if it does not exist.
      2. Ingests all files from the notion folder into that index.
      3. The ``domaine`` metadata field is set to the notion folder name.

    Args:
        company_filter:      Optional list of company names to process.
                             If empty/None, all companies are processed.
        chunk_strategy:      ``"fixed"``, ``"semantic"``, or ``"hierarchical"``.
        max_files_per_index: Max new files per index per call (0 = no limit).

    Returns:
        Dict with per-index results and overall summary.
    """
    embedding_model = DEFAULT_EMBEDDING_MODEL

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
        return {"status": "error", "message": f"Failed to connect to Google Drive: {exc}"}

    # -- Scan Drive tree -----------------------------------------------------
    try:
        tree = list_drive_tree(drive_service)
    except Exception as exc:
        return {"status": "error", "message": f"Failed to scan Drive tree: {exc}"}

    if not tree:
        return {"status": "error", "message": "No company folders found in Drive root."}

    # -- Optional filter on companies ----------------------------------------
    if company_filter:
        allowed = {c.strip().lower() for c in company_filter}
        tree = {k: v for k, v in tree.items() if k in allowed}
        if not tree:
            return {
                "status": "error",
                "message": f"None of the requested companies found. Available: {list(tree.keys())}",
            }

    # -- Load embedding model ------------------------------------------------
    try:
        embedder = get_embedding_model(embedding_model)
        emb_dim = embedder.get_dimension()
    except Exception as exc:
        return {"status": "error", "message": f"Failed to load embedding model: {exc}"}

    # -- Chunk strategy parameters -------------------------------------------
    chunk_params: dict = {}
    if chunk_strategy == "fixed":
        chunk_params = {"size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP}
    elif chunk_strategy == "semantic":
        chunk_params = {"threshold": SEMANTIC_BREAKPOINT_THRESHOLD}
    elif chunk_strategy == "hierarchical":
        chunk_params = {"parent_size": PARENT_CHUNK_SIZE, "child_size": CHILD_CHUNK_SIZE}

    store = get_store()
    per_index_results: list[dict] = []
    total_files = 0
    total_chunks = 0

    for company, notions in tree.items():
        for notion, file_infos in notions.items():
            # Empty notion key = files sitting directly at L1 (no L2 subfolder).
            # Use a single-segment index name (no `__` separator) so the
            # registry / role filter / UI all treat it as a flat L1 index.
            index_name = (
                f"{company}{INDEX_SEP}{notion}" if notion else company
            )
            print(f"[auto] === {index_name} ({len(file_infos)} file(s)) ===")

            # Ensure index exists
            try:
                store.initialize(index_name, embedding_model=embedding_model, chunk_strategy=chunk_strategy)
            except Exception as exc:
                per_index_results.append({
                    "index_name": index_name, "status": "error",
                    "message": f"Failed to create index: {exc}",
                })
                continue

            # Dedup state
            already_indexed_urls = _get_indexed_urls(store, index_name)
            indexed_file_dates = _get_indexed_file_names_with_dates(store, index_name)

            files_processed = 0
            files_skipped_count = 0
            files_updated = 0
            chunks_created = 0

            for file_info in file_infos:
                url = file_info.get("source_url", "")
                file_name = file_info.get("file_name", "")
                file_updated_at = _parse_date_str(file_info.get("updated_at", ""))

                # Dedup
                if url and url in already_indexed_urls:
                    files_skipped_count += 1
                    continue
                if file_name and file_name in indexed_file_dates:
                    stored_date = indexed_file_dates[file_name]
                    if stored_date == file_updated_at:
                        files_skipped_count += 1
                        continue
                    store.delete_chunks_by_file_name(file_name, index_name)
                    files_updated += 1
                    already_indexed_urls = {u for u in already_indexed_urls if u != url}

                if max_files_per_index and files_processed >= max_files_per_index:
                    break

                print(f"[auto]   Processing: {file_name}")
                try:
                    text = extract_from_url(url, drive_service)
                    if not text.strip():
                        continue
                    if len(text) > _MAX_FILE_CHARS:
                        text = text[:_MAX_FILE_CHARS]

                    chunks = chunk_text(text, strategy=chunk_strategy, **chunk_params)
                    if not chunks:
                        continue

                    from hybrid.ingestion.metadata import detect_language
                    _sample = text[:5000]
                    doc_language = detect_language(_sample)
                    # Use the notion folder name as domaine instead of keyword detection.
                    # When notion is empty (files at L1), fall back to the company name.
                    doc_domaine = (notion or company).upper()

                    texts_to_embed = [c["content"] for c in chunks]
                    embeddings = _embed_in_batches(embedder, texts_to_embed, batch_size=64)

                    records: list[dict] = []
                    for chunk, embedding in zip(chunks, embeddings):
                        meta = build_metadata(
                            file_info=file_info,
                            chunk=chunk,
                            embedding_model=embedding_model,
                            index_name=index_name,
                            embedding_dim=emb_dim,
                            doc_language=doc_language,
                            doc_domaine=doc_domaine,
                        )
                        meta["embedding"] = embedding
                        records.append(meta)

                    store.insert_chunks(records)
                    files_processed += 1
                    chunks_created += len(records)

                except Exception as exc:
                    print(f"[auto]   ERROR: {file_name} → {exc}")

            total_files += files_processed
            total_chunks += chunks_created
            per_index_results.append({
                "index_name": index_name,
                "status": "success",
                "files_processed": files_processed,
                "files_already_indexed": files_skipped_count,
                "files_updated": files_updated,
                "chunks_created": chunks_created,
            })
            print(f"[auto]   Done: {files_processed} files, {chunks_created} chunks")

    return {
        "status": "success",
        "message": (
            f"Auto-ingestion complete. {len(per_index_results)} index(es) processed, "
            f"{total_files} file(s), {total_chunks} chunk(s) created."
        ),
        "indexes": per_index_results,
        "total_files": total_files,
        "total_chunks": total_chunks,
    }
