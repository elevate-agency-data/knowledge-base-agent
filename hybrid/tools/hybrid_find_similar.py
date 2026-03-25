"""
ADK tool: hybrid_find_similar

Find documents similar to a given Google Drive document URL.

Instead of a text query, this tool:
  1. Extracts the text content of the source document from Drive.
  2. Chunks the content and embeds each chunk.
  3. Averages the chunk embeddings → a single representative document vector.
  4. Runs a direct vector cosine similarity search against one or more indexes.

This is pure document-to-document similarity — no text query, no LLM rewriting.
A capability unique to Hybrid RAG (embeddings stored locally in DuckDB).
"""

from hybrid.config import DEFAULT_EMBEDDING_MODEL, SERVICE_ACCOUNT_PATH, TOP_K
from hybrid.stores import get_store
from hybrid.embeddings import get_embedding_model
from hybrid.ingestion.extractor import extract_from_url
from hybrid.ingestion.chunker import chunk_text
from hybrid.retrieval.dense import _normalise_scores


_MAX_CHARS = 200_000  # ~40 pages


def hybrid_find_similar(
    document_url: str,
    index_names: list[str],
    top_k: int = TOP_K,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> dict:
    """
    Find documents similar to a given Google Drive document.

    Extracts the source document content, builds a representative embedding
    by averaging its chunk vectors, then runs a cosine similarity search
    across the specified indexes — no text query involved.

    Args:
        document_url:    Full URL of the Google Drive source document.
                         Supported: Google Docs, Slides, Sheets, PDF.
                         Example: "https://docs.google.com/presentation/d/..."
        index_names:     List of hybrid indexes to search in.
                         Example: ["rh", "marketing"]
        top_k:           Total number of similar documents to return (default 10).
        embedding_model: Embedding model key (default from config).
                         Should match the model used when indexes were built.

    Returns:
        Dict with keys:
        - ``status``          : ``"success"`` or ``"error"``
        - ``source_document`` : URL of the reference document
        - ``indexes_searched``: Indexes that were queried
        - ``results``         : List of similar documents, each with:
                                ``file_name``, ``source_url``, ``index_name``,
                                ``score``, ``file_type``, ``domaine``, ``langue``
        - ``total_results``   : Number of results returned
    """
    if not document_url or not document_url.strip():
        return {"status": "error", "message": "document_url is required."}

    index_names = [n.strip().lower() for n in (index_names or []) if n.strip()]
    if not index_names:
        from hybrid.tools.hybrid_list_indexes import hybrid_list_indexes
        list_result = hybrid_list_indexes()
        index_names = [idx["index_name"] for idx in list_result.get("indexes", [])]
    if not index_names:
        return {"status": "error", "message": "No hybrid indexes available."}

    # -- Connect to Drive ----------------------------------------------------
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

    # -- Extract source document content ------------------------------------
    try:
        text = extract_from_url(document_url, drive_service)
        if not text or not text.strip():
            return {"status": "error", "message": "Source document is empty or unreadable."}
        if len(text) > _MAX_CHARS:
            text = text[:_MAX_CHARS]
    except Exception as exc:
        return {"status": "error", "message": f"Failed to extract document content: {exc}"}

    # -- Build document vector (average of chunk embeddings) ----------------
    try:
        embedder = get_embedding_model(embedding_model)
        chunks = chunk_text(text, strategy="fixed")
        if not chunks:
            return {"status": "error", "message": "Could not chunk source document."}

        texts = [c["content"] for c in chunks]
        all_embeddings = []
        for i in range(0, len(texts), 64):
            all_embeddings.extend(embedder.embed_documents(texts[i:i + 64]))

        # Average all chunk vectors → single document-level vector
        dim = len(all_embeddings[0])
        doc_vector = [
            sum(emb[d] for emb in all_embeddings) / len(all_embeddings)
            for d in range(dim)
        ]
    except Exception as exc:
        return {"status": "error", "message": f"Embedding failed: {exc}"}

    # -- Cosine similarity search across indexes ----------------------------
    try:
        store = get_store()
        top_k_per_index = max(1, top_k // len(index_names))

        all_results: list[dict] = []
        for index_name in index_names:
            raw = store.dense_search(
                index_name=index_name,
                embedding=doc_vector,
                top_k=top_k_per_index,
                filters={},
            )
            for chunk in raw:
                chunk["index_name"] = index_name
            all_results.extend(raw)

        all_results = _normalise_scores(all_results)

        # Deduplicate by source_url, keep highest score, exclude source doc
        seen: dict[str, dict] = {}
        for chunk in sorted(all_results, key=lambda x: x.get("score", 0), reverse=True):
            url = chunk.get("source_url", "")
            if not url or url == document_url:
                continue
            if url not in seen:
                seen[url] = {
                    "file_name":  chunk.get("file_name", ""),
                    "source_url": url,
                    "index_name": chunk.get("index_name", ""),
                    "score":      round(chunk.get("score", 0), 4),
                    "file_type":  chunk.get("file_type", ""),
                    "domaine":    chunk.get("domaine", ""),
                    "langue":     chunk.get("langue", ""),
                }

        results = list(seen.values())[:top_k]

    except Exception as exc:
        return {"status": "error", "message": f"Similarity search failed: {exc}"}

    return {
        "status":           "success",
        "source_document":  document_url,
        "indexes_searched": index_names,
        "results":          results,
        "total_results":    len(results),
    }
