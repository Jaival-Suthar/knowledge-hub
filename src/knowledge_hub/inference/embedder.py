"""Local, M1-compatible embedding capability for dense retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

DEFAULT_EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_EMBEDDING_DIMENSION = 384
DEFAULT_EMBEDDING_DEVICE = "cpu"
DEFAULT_EMBEDDING_NORMALIZE = True


class Embedder(Protocol):
    """Capability required by dense retrieval and vector indexing."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""


class SentenceTransformerEmbedder:
    """Lazy local Sentence Transformers embedder matching the M1 baseline."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
        dimension: int = DEFAULT_EMBEDDING_DIMENSION,
        device: str = DEFAULT_EMBEDDING_DEVICE,
        normalize_embeddings: bool = DEFAULT_EMBEDDING_NORMALIZE,
    ) -> None:
        if dimension <= 0:
            raise ValueError("embedding dimension must be positive")
        self.model_name = model_name
        self.dimension = dimension
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
            )
        return self._model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        values = list(texts)
        if not values:
            return []

        encoded = self.model.encode(
            values,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        vectors = [list(map(float, vector)) for vector in encoded]

        if len(vectors) != len(values):
            raise ValueError(
                "embedding model returned "
                f"{len(vectors)} vectors for {len(values)} texts"
            )
        invalid = [
            index
            for index, vector in enumerate(vectors)
            if len(vector) != self.dimension
        ]
        if invalid:
            raise ValueError(
                "embedding model returned unexpected dimension "
                f"at vector indices {invalid}; expected {self.dimension}"
            )

        return vectors
