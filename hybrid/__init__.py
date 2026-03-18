"""
Hybrid RAG pipeline — combines dense and sparse retrieval for more powerful,
production-quality document search than the Vertex AI RAG baseline.

Architecture:
    - Stores  : DuckDB (local dev) | AlloyDB + pgvector (GCP prod)
    - Embeddings : MiniLM-384, MPNet-768, E5-Large-1024, BGE-M3, Vertex-005
    - Retrieval  : Dense, Sparse (BM25), Hybrid (Reciprocal Rank Fusion)
    - Ingestion  : PDF, Google Docs/Sheets/Slides via Drive API
"""
