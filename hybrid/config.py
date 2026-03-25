"""
Central configuration for the Hybrid RAG pipeline.

Switch between local (DuckDB) and GCP (AlloyDB) by setting ENV.
All other modules import from here — nothing is hardcoded elsewhere.
"""

# ---------------------------------------------------------------------------
# GCP settings (shared with rag_agent/)
# ---------------------------------------------------------------------------
PROJECT_ID: str = "knowledge-base-agent-485813"
LOCATION: str = "europe-west1"
SERVICE_ACCOUNT_PATH: str = "rag_agent/key.json"

# ---------------------------------------------------------------------------
# Drive — dossier racine contenant les sous-dossiers clients
# Modifier cette valeur pour pointer vers un autre répertoire Drive.
# ---------------------------------------------------------------------------
DRIVE_ROOT_FOLDER: str = "RAG (Test & Co)"

# ---------------------------------------------------------------------------
# Environment: "local" → DuckDB  |  "gcp" → AlloyDB
# ---------------------------------------------------------------------------
ENV: str = "local"

# ---------------------------------------------------------------------------
# DuckDB (local dev)
# ---------------------------------------------------------------------------
DUCKDB_PATH: str = "hybrid/data/hybrid.duckdb"

# ---------------------------------------------------------------------------
# AlloyDB (GCP prod) — fill in the real DSN for production
# ---------------------------------------------------------------------------
ALLOYDB_CONNECTION_STRING: str = ""  # e.g. "postgresql://user:pass@host:5432/db"

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE: int = 0          # 0 = auto from model max_seq_length (e5-base-768: 512 tokens)
CHUNK_OVERLAP: int = 32     # in tokens
SEMANTIC_BREAKPOINT_THRESHOLD: float = 0.85
PARENT_CHUNK_SIZE: int = 1024
CHILD_CHUNK_SIZE: int = 256

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
# Available models (key → dim / max_seq_length / notes):
#   "minilm-384"     384 dim   128 tokens   lightweight, fast
#   "mpnet-768"      768 dim   128 tokens   multilingual, small context
#   "e5-base-768"    768 dim   512 tokens   multilingual FR/EN, recommended
#   "e5-large-1024" 1024 dim   512 tokens   best quality, heavy (~1.3 GB)
#   "bge-m3"        1024 dim  8192 tokens   dense+sparse, heaviest (~2 GB)
#   "vertex"         768 dim  2048 tokens   cloud-only (GCP)
DEFAULT_EMBEDDING_MODEL: str = "e5-base-768"
VERTEX_EMBEDDING_MODEL: str = "publishers/google/models/text-embedding-005"
EMBEDDING_REQUESTS_PER_MIN: int = 1000

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K: int = 10
DISTANCE_THRESHOLD: float = 0.5
DENSE_WEIGHT: float = 0.7
SPARSE_WEIGHT: float = 0.3
RRF_K: int = 60

# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
BENCHMARK_EMBEDDING_MODELS: list[str] = [
    "minilm-384",
    "mpnet-768",
    "e5-large-1024",
    "bge-m3",
    "vertex",
]
BENCHMARK_CHUNK_SIZES: list[int] = [128, 256, 512, 1024]   # tokens, fixed strategy only
BENCHMARK_CHUNK_STRATEGIES: list[str] = ["fixed", "semantic", "hierarchical"]
BENCHMARK_RETRIEVAL_MODES: list[str] = ["dense", "sparse", "hybrid"]

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
MODEL: str = "gemini-2.5-pro"
