"""
Tool for querying Vertex AI RAG corpora and retrieving relevant information.
"""

import logging

from google.adk.tools.tool_context import ToolContext
from vertexai.preview.generative_models import GenerativeModel, Tool
from vertexai import rag

from ..config import (
    DEFAULT_DISTANCE_THRESHOLD,
    DEFAULT_TOP_K,
    MODEL
)
from .utils import check_corpus_exists, get_corpus_resource_name
from shared.query_rewriter import rewrite_query


def rag_query(
    corpus_name: str,
    query: str,
    tool_context: ToolContext,
    context: str = "",
) -> dict:
    """
    Query a Vertex AI RAG corpus with a user question and return relevant information.

    Args:
        corpus_name (str): The name of the corpus to query. If empty, the current corpus will be used.
                          Preferably use the resource_name from list_corpora results.
        query (str): The text query to search for in the corpus
        tool_context (ToolContext): The tool context

    Returns:
        dict: The query results and status
    """
    try:

        # Check if the corpus exists
        if not check_corpus_exists(corpus_name, tool_context):
            return {
                "status": "error",
                "message": f"Corpus '{corpus_name}' does not exist. Please create it first using the create_corpus tool.",
                "query": query,
                "corpus_name": corpus_name,
            }

        # Get the corpus resource name
        corpus_resource_name = get_corpus_resource_name(corpus_name)


        rag_store = rag.VertexRagStore(
        rag_resources=[
            rag.RagResource(
                rag_corpus=corpus_resource_name
                )
            ],
        rag_retrieval_config= rag.RagRetrievalConfig(
                    top_k=DEFAULT_TOP_K,
                    filter=rag.utils.resources.Filter(vector_distance_threshold=DEFAULT_DISTANCE_THRESHOLD),
                )
        )

        # Rewrite query for better retrieval recall
        retrieval_query = rewrite_query(query, context=context)

        # Perform the query
        print(f"Performing retrieval query... (rewritten: '{retrieval_query}')")
        rag_retrieval_tool = Tool.from_retrieval(
            retrieval=rag.Retrieval(source=rag_store)
        )

        rag_model = GenerativeModel(
        model_name=MODEL, tools=[rag_retrieval_tool]
        )

        response = rag_model.generate_content(retrieval_query)

        # Process the response into a more usable format
        results = []
        if hasattr(response, "text") and response.text:
            result = response.text
            results.append(result)

        # If we didn't find any results
        if not results:
            return {
                "status": "warning",
                "message": f"No results found in corpus '{corpus_name}' for query: '{query}'",
                "query": query,
                "corpus_name": corpus_name,
                "results": [],
            }
        
        final_results = {
            "answer": results,
            "sources": []
        }

        sources_info = []

        if response.candidates[0].grounding_metadata:
            metadata = response.candidates[0].grounding_metadata
            seen_uris = set()
            
            for chunk in metadata.grounding_chunks:
                # Récupération sécurisée du contexte de retrieval
                ctx = getattr(chunk, 'retrieved_context', None)
                uri = getattr(ctx, 'uri', None) if ctx else None
                title = getattr(ctx, 'title', None) if ctx else "Document sans titre"
                
                if uri and uri not in seen_uris:
                    seen_uris.add(uri)
                    sources_info.append({
                        "title": title,
                        "uri": uri
                    })
                    

        final_results["sources"] = sources_info

        return {
            "status": "success",
            "message": f"Successfully queried corpus '{corpus_name}'",
            "query": query,
            "retrieval_query": retrieval_query,
            "corpus_name": corpus_name,
            "results": final_results,
        }

    except Exception as e:
        error_msg = f"Error querying corpus: {str(e)}"
        logging.error(error_msg)
        return {
            "status": "error",
            "message": error_msg,
            "query": query,
            "corpus_name": corpus_name,
        }