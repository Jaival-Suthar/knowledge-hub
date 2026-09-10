from abc import ABC, abstractmethod

from knowledge_hub.models import Chunk


class Index(ABC):
    @abstractmethod
    def upsert(self, chunks: list[Chunk]) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete(self, document_id: str) -> None:
        raise NotImplementedError
