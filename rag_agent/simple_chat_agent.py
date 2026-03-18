"""
Simple Chat Agent — chatbot conversationnel sans pipeline RAG vectoriel.

Dispose uniquement d'outils Drive/index pour lire des documents et explorer
les dossiers. Ne fait PAS de recherche vectorielle (rag_query, hybrid_query).

Utilisé par la page Simple Chat (mode « Sans RAG »).
"""

from rag_agent.config import MODEL

from google.adk.agents import Agent

from hybrid.tools.hybrid_list_drive   import hybrid_list_drive
from hybrid.tools.hybrid_list_indexes import hybrid_list_indexes
from hybrid.tools.hybrid_index_info   import hybrid_index_info
from rag_agent.tools.get_document_content import get_document_content


simple_chat_agent = Agent(
    name="SimpleChatAgent",
    model=MODEL,
    description="Chatbot conversationnel avec accès aux outils Drive, sans RAG vectoriel.",
    tools=[
        hybrid_list_drive,
        hybrid_list_indexes,
        hybrid_index_info,
        get_document_content,
    ],
    instruction="""
    Tu es un assistant conversationnel intelligent.

    ## Ce que tu peux faire

    - Répondre à n'importe quelle question depuis ta connaissance générale.
    - Lister le contenu d'un dossier Google Drive : `hybrid_list_drive(folder_name="...")`
    - Lister les index de la base de connaissances : `hybrid_list_indexes()`
    - Voir le détail d'un index : `hybrid_index_info(index_name="...")`
    - Lire un document Drive depuis son URL : `get_document_content(url="...")`

    ## Ce que tu ne fais PAS

    Tu ne fais **pas** de recherche vectorielle RAG.
    Si l'utilisateur veut interroger la base de connaissances (recherche sémantique),
    dis-lui de changer de mode dans la barre latérale (Vertex AI ou Hybrid RAG).

    ## Comportement

    - Réponds toujours en français sauf si l'utilisateur écrit dans une autre langue.
    - Utilise les outils Drive uniquement si la demande le justifie explicitement.
    - Sois direct et concis.
    """,
)
