import pytest

from knowledge_hub.indexing.embedding import embed_and_upsert
from knowledge_hub.models import Chunk, SourceType


class FakeEmbedder:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.inputs = None

    def embed(self, texts):
        self.inputs = list(texts)
        return self.embeddings


class FakeIndex:
    vector_size = 2

    def __init__(self):
        self.received = None

    def upsert(self, chunks, embeddings):
        self.received = (chunks, embeddings)


def make_chunk(content):
    return Chunk(
        document_id="document",
        chunk_id=content,
        content=content,
        source_type=SourceType.MARKDOWN,
    )


def test_embed_and_upsert_supports_generators_and_preserves_order():
    chunks = [make_chunk("one"), make_chunk("two")]
    embedder = FakeEmbedder([[1.0, 2.0], [3.0, 4.0]])
    index = FakeIndex()

    embed_and_upsert(index, (chunk for chunk in chunks), embedder)

    assert embedder.inputs == ["one", "two"]
    assert index.received == (chunks, [[1.0, 2.0], [3.0, 4.0]])


def test_embed_and_upsert_rejects_count_mismatch():
    with pytest.raises(ValueError, match="vectors for 2 chunks"):
        embed_and_upsert(
            FakeIndex(),
            [make_chunk("one"), make_chunk("two")],
            FakeEmbedder([[1.0, 2.0]]),
        )


def test_embed_and_upsert_rejects_dimension_mismatch():
    with pytest.raises(ValueError, match="expected 2"):
        embed_and_upsert(
            FakeIndex(),
            [make_chunk("one")],
            FakeEmbedder([[1.0, 2.0, 3.0]]),
        )
