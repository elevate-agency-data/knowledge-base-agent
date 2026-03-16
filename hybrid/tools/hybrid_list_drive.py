"""
ADK tool: hybrid_list_drive

Lists subfolders and files inside a Google Drive folder.
Useful for finding the exact folder/file names before ingestion.
"""

from hybrid.config import SERVICE_ACCOUNT_PATH


def hybrid_list_drive(
    folder_name: str,
) -> dict:
    """
    List the contents of a Google Drive folder.

    Shows all subfolders and supported files (PDF, Doc, Sheet, Slide)
    inside the specified folder. Use this to find exact folder names
    before calling hybrid_add_data.

    Args:
        folder_name: Exact name of the Drive folder to inspect.

    Returns:
        Dict with keys:
        - ``status``     : ``"success"`` or ``"error"``
        - ``folder_name``: Echo of the folder searched
        - ``subfolders`` : List of subfolder names found inside
        - ``files``      : List of file dicts (name, type, url)
        - ``total``      : Total items found
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

    # Find the folder
    escaped = folder_name.replace("'", "\\'")
    resp = drive_service.files().list(
        q=(
            f"name = '{escaped}' "
            "and mimeType = 'application/vnd.google-apps.folder' "
            "and trashed = false"
        ),
        fields="files(id, name)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()

    folders = resp.get("files", [])
    if not folders:
        return {
            "status":  "error",
            "message": f"Folder '{folder_name}' not found. Check the exact name and sharing permissions.",
        }

    folder_id = folders[0]["id"]

    # List everything inside
    items_resp = drive_service.files().list(
        q=f"'{folder_id}' in parents and trashed = false",
        fields="files(id, name, mimeType, webViewLink)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
        pageSize=200,
    ).execute()

    _SUPPORTED = {
        "application/vnd.google-apps.document":     "Doc",
        "application/vnd.google-apps.spreadsheet":  "Sheet",
        "application/vnd.google-apps.presentation": "Slide",
        "application/pdf":                          "PDF",
    }

    subfolders = []
    files = []

    for item in items_resp.get("files", []):
        mime = item.get("mimeType", "")
        if mime == "application/vnd.google-apps.folder":
            subfolders.append(item["name"])
        elif mime in _SUPPORTED:
            files.append({
                "name": item["name"],
                "type": _SUPPORTED[mime],
                "url":  item.get("webViewLink", ""),
            })

    return {
        "status":      "success",
        "folder_name": folder_name,
        "subfolders":  sorted(subfolders),
        "files":       files,
        "total":       len(subfolders) + len(files),
    }
