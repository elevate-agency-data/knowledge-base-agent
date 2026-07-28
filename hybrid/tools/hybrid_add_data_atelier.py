"""
ADK / CLI tool: hybrid_add_data_atelier

Ingest a Google Drive tree using the ATELIER model:

    <root>/<LigneProduit>/<Domaine>/[<Zone>/]<file>
    _meta.yaml (audience_role, matiere) inherited nearest-wins

Unlike ``hybrid_add_data_auto`` (fixed 2-level company__notion), this scans the
tree at ANY depth, projects each file onto ONE domain index, and carries
ligne_produit / zone / audience_role / matiere as filterable metadata. The
folder→dimension mapping is shared with the local ingester via atelier_map, so
Drive and local produce identical indexes.

Resumable: files whose source_url is already indexed are skipped.
"""

from __future__ import annotations

from typing import Any

from hybrid.config import (
    DEFAULT_EMBEDDING_MODEL, SERVICE_ACCOUNT_PATH, DRIVE_ROOT_FOLDER,
    CHUNK_SIZE, CHUNK_OVERLAP,
)
from hybrid.stores import get_store
from hybrid.embeddings import get_embedding_model
from hybrid.ingestion.extractor import (
    _MIME_TO_TYPE, _find_folder_by_name, extract_from_url,
)
from hybrid.ingestion.chunker import chunk_text
from hybrid.ingestion.metadata import build_metadata, detect_language
from hybrid.ingestion.atelier_map import (
    META_FILENAME, map_folders, merge_meta, as_list,
)
from hybrid.tools.hybrid_add_data import _get_indexed_urls, _parse_date_str

_FOLDER_MIME = "application/vnd.google-apps.folder"
_MAX_FILE_CHARS = 500_000


# ── Drive helpers ─────────────────────────────────────────────────────────────

def _list_children(folder_id: str, drive_service: Any) -> list[dict]:
    """All direct children of a folder (files + subfolders), paginated."""
    q = f"'{folder_id}' in parents and trashed = false"
    out: list[dict] = []
    page_token = None
    while True:
        kwargs: dict = dict(
            q=q,
            fields=("nextPageToken, files(id, name, mimeType, webViewLink, "
                    "createdTime, modifiedTime, owners)"),
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            pageSize=1000,
        )
        if page_token:
            kwargs["pageToken"] = page_token
        resp = drive_service.files().list(**kwargs).execute()
        out.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return out


def _download_yaml(file_id: str, drive_service: Any) -> dict:
    """Download and parse a _meta.yaml file from Drive."""
    import yaml
    try:
        data = drive_service.files().get_media(
            fileId=file_id, supportsAllDrives=True
        ).execute()
        parsed = yaml.safe_load(data.decode("utf-8")) or {}
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _walk(folder_id: str, folders: list[str], inherited_meta: dict,
          drive_service: Any):
    """Yield (file_info, folders, meta) for every supported file, recursively.

    ``folders`` is the root-relative chain of folder names (excludes the root
    and the filename). ``meta`` is the merged _meta.yaml chain (nearest wins).
    """
    children = _list_children(folder_id, drive_service)

    # 1. absorb this folder's _meta.yaml (overrides inherited)
    local_meta = dict(inherited_meta)
    for c in children:
        if c.get("name") == META_FILENAME and c.get("mimeType") != _FOLDER_MIME:
            local_meta = merge_meta(local_meta, _download_yaml(c["id"], drive_service))

    # 2. recurse into subfolders, emit supported files
    for c in children:
        mime = c.get("mimeType", "")
        name = c.get("name", "")
        if mime == _FOLDER_MIME:
            yield from _walk(c["id"], folders + [name], local_meta, drive_service)
        elif mime in _MIME_TO_TYPE:
            owners = c.get("owners", [])
            file_info = {
                "file_id":    c["id"],
                "file_name":  name,
                "file_type":  _MIME_TO_TYPE.get(mime, "Other"),
                "mime_type":  mime,
                "source_url": c.get("webViewLink", ""),
                "created_at": c.get("createdTime", ""),
                "updated_at": c.get("modifiedTime", ""),
                "author":     owners[0].get("displayName", "") if owners else "",
            }
            yield file_info, folders, local_meta


# ── main tool ─────────────────────────────────────────────────────────────────

def hybrid_add_data_atelier(
    root_folder: str = "",
    chunk_strategy: str = "fixed",
    max_files: int = 0,
) -> dict:
    """
    Ingest a Drive atelier tree into the active store.

    Args:
        root_folder:    Drive root folder name. Defaults to DRIVE_ROOT_FOLDER
                        (the active brand's ``drive_root_folder``).
        chunk_strategy: ``"fixed"`` | ``"semantic"`` | ``"hierarchical"``.
        max_files:      Max NEW files to process this call (0 = all). Skipped
                        (already-indexed) files don't count.

    Returns:
        Summary dict: indexes, chunks_per_index, files_ingested, skipped, etc.
    """
    root_folder = root_folder or DRIVE_ROOT_FOLDER

    # -- Drive service -------------------------------------------------------
    try:
        from googleapiclient.discovery import build
        from google.oauth2 import service_account
        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_PATH,
            scopes=["https://www.googleapis.com/auth/drive.readonly"],
        )
        drive_service = build("drive", "v3", credentials=creds)
    except Exception as exc:
        return {"status": "error", "message": f"Drive connection failed: {exc}"}

    root_id = _find_folder_by_name(root_folder, drive_service)
    if not root_id:
        return {"status": "error", "message": f"Root folder not found: {root_folder}"}

    # -- Model + store -------------------------------------------------------
    try:
        embedder = get_embedding_model(DEFAULT_EMBEDDING_MODEL)
        emb_dim = embedder.get_dimension()
    except Exception as exc:
        return {"status": "error", "message": f"Embedding model failed: {exc}"}

    store = get_store()
    chunk_params = ({"size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP}
                    if chunk_strategy == "fixed" else {})

    per_index: dict[str, int] = {}
    files_done = 0
    files_remaining = 0
    files_skipped: list[dict] = []
    seen_indexes: set[str] = set()
    indexed_urls_cache: dict[str, set[str]] = {}

    print(f"[atelier] Scanning Drive root '{root_folder}' (id={root_id})")

    for file_info, folders, meta in _walk(root_id, [], {}, drive_service):
        fname = file_info["file_name"]
        mp = map_folders(folders)
        if not mp:
            files_skipped.append({"file": fname, "reason": "too shallow"})
            continue

        index_name = mp["index"]

        # ensure index + load its already-indexed URLs once
        if index_name not in seen_indexes:
            store.initialize(index_name, embedding_model=DEFAULT_EMBEDDING_MODEL,
                             chunk_strategy=chunk_strategy)
            indexed_urls_cache[index_name] = _get_indexed_urls(store, index_name)
            seen_indexes.add(index_name)

        # dedup by source_url
        url = file_info.get("source_url", "")
        if url and url in indexed_urls_cache.get(index_name, set()):
            files_skipped.append({"file": fname, "reason": "already indexed"})
            continue

        if max_files and files_done >= max_files:
            files_remaining += 1
            continue

        # extract → chunk → embed → metadata → insert
        try:
            text = extract_from_url(url, drive_service)
            if not text.strip():
                files_skipped.append({"file": fname, "reason": "empty"})
                continue
            if len(text) > _MAX_FILE_CHARS:
                text = text[:_MAX_FILE_CHARS]

            chunks = chunk_text(text, strategy=chunk_strategy, **chunk_params)
            if not chunks:
                files_skipped.append({"file": fname, "reason": "no chunks"})
                continue

            audience_role = as_list(meta.get("audience_role"))
            matiere = as_list(meta.get("matiere"))
            langue = detect_language(text[:5000])

            embeddings = embedder.embed_documents([c["content"] for c in chunks])
            records = []
            for chunk, emb in zip(chunks, embeddings):
                rec = build_metadata(
                    file_info=file_info, chunk=chunk,
                    embedding_model=DEFAULT_EMBEDDING_MODEL,
                    index_name=index_name, embedding_dim=emb_dim,
                    doc_language=langue, doc_domaine=mp["domaine"].upper(),
                    ligne_produit=mp["ligne_produit"], zone=mp["zone"],
                    audience_role=audience_role, matiere=matiere,
                )
                rec["embedding"] = emb
                records.append(rec)

            store.insert_chunks(records)
            per_index[index_name] = per_index.get(index_name, 0) + len(records)
            files_done += 1
            print(f"[atelier]   [{index_name:16s}] {fname}  "
                  f"(ligne={mp['ligne_produit'] or '-'} zone={mp['zone'] or '-'} "
                  f"role={audience_role or '-'} +{len(records)} chunks)")

        except Exception as exc:
            files_skipped.append({"file": fname, "reason": str(exc)})

    done = max_files == 0 or files_remaining == 0
    return {
        "status": "success" if (files_done or not files_skipped) else "error",
        "message": (
            f"Atelier ingestion {'complete' if done else 'partial'}. "
            f"{files_done} file(s), {sum(per_index.values())} chunk(s) into "
            f"{len(per_index)} index(es)."
            + (f" {files_remaining} remaining — call again to continue."
               if files_remaining else "")
        ),
        "root_folder": root_folder,
        "indexes": sorted(per_index),
        "chunks_per_index": per_index,
        "files_ingested": files_done,
        "files_remaining": files_remaining,
        "total_chunks": sum(per_index.values()),
        "files_skipped": files_skipped,
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "chunk_strategy": chunk_strategy,
    }
