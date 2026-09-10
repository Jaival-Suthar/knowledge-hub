from collections.abc import Iterable

from knowledge_hub.models import Chunk, Document


class IngestionPipeline:
    def ingest(self, document: Document) -> Iterable[Chunk]:
        raise NotImplementedError
