from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from knowledge_hub.models import Chunk
from knowledge_hub.retrieval.types import RankedChunk


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_.$:/@-]+", text.lower())


class BM25Retriever:
    """In-memory sparse retrieval for M2 v1; suitable for small personal corpora."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        self._index = BM25Okapi([_tokens(chunk.content) for chunk in self._chunks])

    def search(self, query: str, top_k: int) -> list[RankedChunk]:
        if not self._chunks or top_k <= 0:
            return []
        scores = self._index.get_scores(_tokens(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :top_k
        ]
        return [
            RankedChunk(
                chunk=self._chunks[i], score=float(scores[i]), rank=rank, channel="bm25"
            )
            for rank, i in enumerate(order, start=1)
        ]
