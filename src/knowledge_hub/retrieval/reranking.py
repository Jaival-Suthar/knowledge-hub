from __future__ import annotations

from knowledge_hub.retrieval.types import RankedChunk


class CrossEncoderReranker:
    """M1-derived BGE-style cross-encoder boundary, lazy-loading the model."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self, query: str, candidates: list[RankedChunk], top_k: int
    ) -> list[RankedChunk]:
        if not candidates or top_k <= 0:
            return []
        scores = self.model.predict(
            [(query, item.chunk.content) for item in candidates]
        )
        ordered = sorted(
            zip(candidates, scores, strict=True),
            key=lambda pair: float(pair[1]),
            reverse=True,
        )[:top_k]
        return [
            RankedChunk(
                chunk=item.chunk, score=float(score), rank=rank, channel="reranker"
            )
            for rank, (item, score) in enumerate(ordered, start=1)
        ]
