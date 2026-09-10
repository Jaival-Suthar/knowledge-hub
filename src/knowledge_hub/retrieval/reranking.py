from abc import ABC, abstractmethod

from knowledge_hub.models import Chunk


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
        raise NotImplementedError
