"""
Hybrid RAG ADK tools.

These three functions are registered as tools on the ADK agent and expose
the full hybrid RAG pipeline to the LLM:

- :func:`hybrid_create_index` — create a new index
- :func:`hybrid_add_data`     — ingest Google Drive folders
- :func:`hybrid_rag_query`    — query an index (dense / sparse / hybrid)
"""

from .hybrid_create_index import hybrid_create_index
from .hybrid_add_data     import hybrid_add_data, hybrid_add_data_auto
from .hybrid_rag_query    import hybrid_rag_query
from .hybrid_list_drive   import hybrid_list_drive
from .hybrid_list_indexes import hybrid_list_indexes
from .hybrid_delete_index import hybrid_delete_index
from .hybrid_index_info   import hybrid_index_info
from .hybrid_multi_query  import hybrid_multi_query

__all__ = [
    "hybrid_create_index",
    "hybrid_add_data",
    "hybrid_add_data_auto",
    "hybrid_rag_query",
    "hybrid_list_drive",
    "hybrid_list_indexes",
    "hybrid_delete_index",
    "hybrid_index_info",
    "hybrid_multi_query",
]
