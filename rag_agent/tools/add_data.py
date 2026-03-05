"""
Tool for adding new data sources to a Vertex AI RAG corpus.
"""
import re
from googleapiclient.discovery import build
from google.oauth2 import service_account
from vertexai import rag
from .utils import get_corpus_resource_name
from google.adk.tools.tool_context import ToolContext
from ..config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_REQUESTS_PER_MIN,
    SERVICE_ACCOUNT_PATH 
)

def add_data(
    folder_names: list[str], # Changement en liste
    corpus_name: str,
    tool_context: ToolContext,
) -> dict:
    try:
        
        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_PATH, 
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        drive_service = build('drive', 'v3', credentials=creds)

        all_paths = []
        not_found = []

        # 1. Boucle pour récupérer tous les IDs
        for name in folder_names:
            escaped_name = name.replace("'", "\\'")
            query = f"name = '{escaped_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
            # On ajoute supportsAllDrives pour être sûr de couvrir les dossiers partagés
            response = drive_service.files().list(
                q=query, 
                fields="files(id, name)",
                supportsAllDrives=True,
                includeItemsFromAllDrives=True
            ).execute()
            
            folders = response.get('files', [])
            if folders:
                folder_id=folders[0]['id']
                path = f"https://drive.google.com/drive/u/0/folders/{folder_id}"
                all_paths.append(path)
            else:
                not_found.append(name)

        if not all_paths:
            return {
                "status": "error", 
                "message": f"Aucun des dossiers suivants n'a été trouvé : {', '.join(not_found)}"
            }
        
         # Get the corpus resource name
        corpus_resource_name = get_corpus_resource_name(corpus_name)

        # Config
        transformation_config = rag.TransformationConfig(
            chunking_config = rag.ChunkingConfig(
                            chunk_size=DEFAULT_CHUNK_SIZE,        # Nombre de tokens par morceau
                            chunk_overlap=DEFAULT_CHUNK_OVERLAP       # Chevauchement entre les morceaux
                        )
        )

        # 2. Importation groupée via Vertex AI RAG        
        import_result = rag.import_files(
            corpus_resource_name,
            all_paths,
            transformation_config=transformation_config,
            max_embedding_requests_per_min=DEFAULT_EMBEDDING_REQUESTS_PER_MIN,
        )

        # Set this as the current corpus if not already set
        if not tool_context.state.get("current_corpus"):
            tool_context.state["current_corpus"] = corpus_name

        # 3. Préparation du message de retour
        msg = f"Importation réussie de {len(all_paths)} dossier(s)."
        if not_found:
            msg += f" Attention, dossiers introuvables : {', '.join(not_found)}"

        return {
            "status": "success",
            "message": msg,
            "corpus_name": corpus_name,
            "paths": all_paths,
        }

    except Exception as e:
        return {"status": "error", "message": f"Erreur lors de l'ajout : {str(e)}"}