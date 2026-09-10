from abc import ABC, abstractmethod
from typing import Any


class RankFusion(ABC):
    @abstractmethod
    def fuse(self, ranked_lists: list[list[Any]]) -> list[Any]:
        raise NotImplementedError
