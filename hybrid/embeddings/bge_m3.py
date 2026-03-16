"""
BGE-M3 embedding model.

Uses BAAI/bge-m3 via FlagEmbedding.  Uniquely produces BOTH dense (1024-dim)
and sparse (lexical weight dict) vectors in a single forward pass, enabling
true hybrid retrieval without separate models.
"""

from .base import BaseEmbedding

_MODEL_ID = "BAAI/bge-m3"
_DIMENSION = 1024


class BGEM3Embedding(BaseEmbedding):
    """
    BGE-M3 hybrid embedding model (dense: 1024 dim + sparse lexical weights).

    This is the recommended default model for the hybrid pipeline because it
    provides both dense and sparse representations simultaneously, cutting
    embedding cost roughly in half compared to running two separate models.

    Sparse output format (per text)::

        {token_id: weight, ...}   # dict[int, float]

    Use ``encode_with_sparse()`` to get both representations at once.
    """

    def __init__(self) -> None:
        """Initialise without loading the model yet."""
        self._model = None

    def _load(self) -> None:
        """Load the BGEM3FlagModel if not already loaded."""
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel
            self._model = BGEM3FlagModel(_MODEL_ID, use_fp16=True)

    # ------------------------------------------------------------------
    # BaseEmbedding interface
    # ------------------------------------------------------------------

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of document passages (dense vectors only).

        For both dense + sparse use ``encode_with_sparse()`` instead.

        Args:
            texts: Passages to embed.

        Returns:
            List of 1024-dimensional float vectors.
        """
        self._load()
        output = self._model.encode(
            texts,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return [vec.tolist() for vec in output["dense_vecs"]]

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query string (dense vector only).

        Args:
            text: Query string.

        Returns:
            1024-dimensional float vector.
        """
        self._load()
        output = self._model.encode(
            [text],
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return output["dense_vecs"][0].tolist()

    def get_dimension(self) -> int:
        """Return vector dimension: 1024."""
        return _DIMENSION

    def get_model_name(self) -> str:
        """Return the HuggingFace model identifier."""
        return _MODEL_ID

    # ------------------------------------------------------------------
    # BGE-M3 specific — dense + sparse in one call
    # ------------------------------------------------------------------

    def encode_with_sparse(
        self, texts: list[str]
    ) -> list[dict]:
        """
        Encode texts and return both dense vectors and sparse weights.

        Args:
            texts: List of text strings to encode.

        Returns:
            List of dicts, one per input text::

                {
                    "dense":  list[float],           # 1024-dim dense vector
                    "sparse": dict[int, float],       # token_id → weight
                }
        """
        self._load()
        output = self._model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        results = []
        for dense_vec, sparse_weights in zip(
            output["dense_vecs"], output["lexical_weights"]
        ):
            results.append(
                {
                    "dense": dense_vec.tolist(),
                    "sparse": sparse_weights,  # dict[int, float]
                }
            )
        return results

    def encode_query_with_sparse(self, text: str) -> dict:
        """
        Encode a single query and return both dense vector and sparse weights.

        Args:
            text: Query string.

        Returns:
            Dict with keys ``"dense"`` (list[float]) and
            ``"sparse"`` (dict[int, float]).
        """
        results = self.encode_with_sparse([text])
        return results[0]

    @staticmethod
    def sparse_to_storage(sparse_weights: dict[int, float]) -> str:
        """
        Serialise a sparse weight dict to a compact string for storage.

        Format: ``"token_id1:weight1,token_id2:weight2,…"``

        Args:
            sparse_weights: Dict mapping token IDs to lexical weights.

        Returns:
            Comma-separated ``token_id:weight`` string.
        """
        return ",".join(f"{tid}:{w:.6f}" for tid, w in sparse_weights.items())

    @staticmethod
    def sparse_from_storage(stored: str) -> dict[int, float]:
        """
        Deserialise a sparse weight string produced by ``sparse_to_storage``.

        Args:
            stored: String in ``"token_id:weight,…"`` format.

        Returns:
            Dict mapping integer token IDs to float weights.
        """
        if not stored:
            return {}
        result = {}
        for pair in stored.split(","):
            if ":" in pair:
                tid, w = pair.split(":", 1)
                result[int(tid)] = float(w)
        return result

    @staticmethod
    def sparse_dot_product(
        query_sparse: dict[int, float],
        doc_sparse: dict[int, float],
    ) -> float:
        """
        Compute dot-product similarity between two sparse weight dicts.

        Args:
            query_sparse: Sparse weights for the query.
            doc_sparse:   Sparse weights for a document chunk.

        Returns:
            Scalar similarity score (higher is more relevant).
        """
        score = 0.0
        for token_id, weight in query_sparse.items():
            if token_id in doc_sparse:
                score += weight * doc_sparse[token_id]
        return score
