"""
ADK tool: hybrid_list_drive

Lists the complete contents of a Google Drive folder (recursive).
Useful for verifying the exact folder/file names before ingestion.
"""

from hybrid.config import SERVICE_ACCOUNT_PATH


_SUPPORTED: dict[str, str] = {
    # Google native
    "application/vnd.google-apps.document":     "Doc",
    "application/vnd.google-apps.spreadsheet":  "Sheet",
    "application/vnd.google-apps.presentation": "Slide",
    # PDF
    "application/pdf":                          "PDF",
    # Office Open XML
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document":   "Docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":         "Xlsx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "Pptx",
    # Legacy Office
    "application/msword":            "Docx",
    "application/vnd.ms-excel":      "Xlsx",
    "application/vnd.ms-powerpoint": "Pptx",
}


def _list_folder_tree(
    folder_id: str,
    drive_service,
    folder_name: str = "",
    depth: int = 0,
) -> dict:
    """
    Recursively list all contents of a folder.

    Returns a dict with:
      - name       : folder display name
      - subfolders : list of nested folder dicts (same schema, recursive)
      - files      : list of file dicts {name, type, url}
    """
    all_query = f"'{folder_id}' in parents and trashed = false"
    items = []
    page_token = None

    while True:
        kwargs = dict(
            q=all_query,
            fields="nextPageToken, files(id, name, mimeType, webViewLink)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            pageSize=1000,
        )
        if page_token:
            kwargs["pageToken"] = page_token
        resp = drive_service.files().list(**kwargs).execute()
        items.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    subfolders = []
    files = []

    for item in items:
        mime = item.get("mimeType", "")
        if mime == "application/vnd.google-apps.folder":
            sub = _list_folder_tree(
                item["id"], drive_service, folder_name=item["name"], depth=depth + 1
            )
            subfolders.append(sub)
        elif mime in _SUPPORTED:
            files.append({
                "name": item["name"],
                "type": _SUPPORTED[mime],
                "url":  item.get("webViewLink", ""),
            })

    return {
        "name":       folder_name,
        "subfolders": sorted(subfolders, key=lambda x: x["name"]),
        "files":      sorted(files, key=lambda x: x["name"]),
    }


def _count_tree(node: dict) -> tuple[int, int]:
    """Return (total_files, total_folders) for the whole tree."""
    files   = len(node["files"])
    folders = len(node["subfolders"])
    for sub in node["subfolders"]:
        sf, ss = _count_tree(sub)
        files   += sf
        folders += ss
    return files, folders


def hybrid_list_drive(folder_name: str) -> dict:
    """
    List the complete contents of a Google Drive folder (recursive).

    Searches inside the configured DRIVE_ROOT_FOLDER to avoid picking
    up a same-named folder from another location in Drive.
    Shows all subfolders and supported files (PDF, Doc, Sheet, Slide,
    Docx, Xlsx, Pptx) at every depth level.

    Args:
        folder_name: Name of the Drive folder to inspect (e.g. "CELIO").

    Returns:
        Dict with keys:
        - ``status``      : ``"success"`` or ``"error"``
        - ``folder_name`` : Echo of the folder searched
        - ``tree``        : Recursive folder tree (subfolders + files)
        - ``total_files`` : Total supported files found across all levels
        - ``total_folders``: Total subfolders found across all levels
    """
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

    try:
        from hybrid.ingestion.extractor import _resolve_folder_id, _find_folder_by_name
        from hybrid.config import DRIVE_ROOT_FOLDER

        # Résolution avec logs pour vérifier quel dossier est trouvé
        root_id = _find_folder_by_name(DRIVE_ROOT_FOLDER, drive_service)
        print(f"[hybrid_list_drive] Root '{DRIVE_ROOT_FOLDER}' → id={root_id}")

        folder_id = _resolve_folder_id(folder_name, drive_service)
        print(f"[hybrid_list_drive] '{folder_name}' → id={folder_id}")

    except ValueError as exc:
        return {"status": "error", "message": str(exc)}
    except Exception as exc:
        return {"status": "error", "message": f"Folder lookup failed: {exc}"}

    tree = _list_folder_tree(folder_id, drive_service, folder_name=folder_name)
    total_files, total_folders = _count_tree(tree)

    return {
        "status":        "success",
        "folder_name":   folder_name,
        "folder_id":     folder_id,           # pour déboguer
        "root_id":       root_id or "non trouvé",
        "tree":          tree,
        "total_files":   total_files,
        "total_folders": total_folders,
    }
