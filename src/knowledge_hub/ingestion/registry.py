from knowledge_hub.ingestion.adapters.base import SourceAdapter


class AdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, SourceAdapter] = {}

    def register(self, source_type: str, adapter: SourceAdapter) -> None:
        self._adapters[source_type] = adapter

    def get(self, source_type: str) -> SourceAdapter:
        return self._adapters[source_type]
