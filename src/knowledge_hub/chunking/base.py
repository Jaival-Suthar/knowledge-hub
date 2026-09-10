from abc import ABC, abstractmethod

from knowledge_hub.models import Chunk, Document


class Chunker(ABC):
    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        raise NotImplementedError
