from abc import ABC, abstractmethod

from knowledge_hub.models import Chunk


class EligibilityFilter(ABC):
    @abstractmethod
    def apply(self, chunks: list[Chunk]) -> list[Chunk]:
        raise NotImplementedError
