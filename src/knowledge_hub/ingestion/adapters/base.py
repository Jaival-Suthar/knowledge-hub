from abc import ABC, abstractmethod
from pathlib import Path

from knowledge_hub.models import Document


class SourceAdapter(ABC):
    @abstractmethod
    def supports(self, source: str | Path) -> bool:
        raise NotImplementedError

    @abstractmethod
    def extract(self, source: str | Path) -> Document:
        raise NotImplementedError
