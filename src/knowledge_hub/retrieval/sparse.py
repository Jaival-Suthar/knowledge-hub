from abc import ABC, abstractmethod

from knowledge_hub.models import Chunk


class SparseRetriever(ABC):
    @abstractmethod
    def search(self, query: str, top_k: int) -> list[Chunk]:
        raise NotImplementedError
