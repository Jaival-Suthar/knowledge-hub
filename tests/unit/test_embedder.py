import sys
from types import SimpleNamespace
from typing import ClassVar

import pytest

from knowledge_hub.inference.embedder import (
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL_NAME,
    SentenceTransformerEmbedder,
)


class FakeSentenceTransformer:
    instances: ClassVar[list["FakeSentenceTransformer"]] = []
    vectors: ClassVar[list[list[float]]] = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    def __init__(self, model_name, *, device):
        self.model_name = model_name
        self.device = device
        self.encode_calls = []
        self.instances.append(self)

    def encode(self, texts, *, normalize_embeddings, convert_to_numpy):
        self.encode_calls.append((list(texts), normalize_embeddings, convert_to_numpy))
        return self.vectors[: len(texts)]


def install_fake_sentence_transformers(monkeypatch):
    FakeSentenceTransformer.instances.clear()
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )


def test_embedder_is_lazy_and_uses_m1_configuration(monkeypatch):
    install_fake_sentence_transformers(monkeypatch)
    embedder = SentenceTransformerEmbedder(
        model_name="test/model",
        dimension=3,
        device="cpu",
        normalize_embeddings=True,
    )

    assert FakeSentenceTransformer.instances == []
    assert embedder.embed(text for text in ("first", "second")) == [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]
    model = FakeSentenceTransformer.instances[0]
    assert (model.model_name, model.device) == ("test/model", "cpu")
    assert model.encode_calls == [
        (["first", "second"], True, True),
    ]


def test_empty_input_does_not_initialize_model(monkeypatch):
    install_fake_sentence_transformers(monkeypatch)
    assert SentenceTransformerEmbedder().embed([]) == []
    assert FakeSentenceTransformer.instances == []


def test_default_model_and_dimension_match_m1():
    embedder = SentenceTransformerEmbedder()
    assert embedder.model_name == DEFAULT_EMBEDDING_MODEL_NAME
    assert embedder.dimension == DEFAULT_EMBEDDING_DIMENSION


def test_embedder_rejects_wrong_count(monkeypatch):
    install_fake_sentence_transformers(monkeypatch)
    FakeSentenceTransformer.vectors = [[0.1, 0.2, 0.3]]
    with pytest.raises(ValueError, match="vectors for 2 texts"):
        SentenceTransformerEmbedder(dimension=3).embed(["a", "b"])


def test_embedder_rejects_wrong_dimension(monkeypatch):
    install_fake_sentence_transformers(monkeypatch)
    FakeSentenceTransformer.vectors = [[0.1, 0.2]]
    with pytest.raises(ValueError, match="expected 3"):
        SentenceTransformerEmbedder(dimension=3).embed(["a"])
    FakeSentenceTransformer.vectors = [[0.1, 0.2], [0.3, 0.4]]
