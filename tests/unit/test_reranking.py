from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import ClassVar

import pytest

from knowledge_hub.models import Chunk, ContentRole, Provenance, SourceType
from knowledge_hub.retrieval.metadata import MetadataFilters
from knowledge_hub.retrieval.pipeline import RetrievalPipeline
from knowledge_hub.retrieval.reranking import CrossEncoderReranker
from knowledge_hub.retrieval.types import RankedChunk


class FakeCrossEncoder:
    instances: ClassVar[list[FakeCrossEncoder]] = []
    scores: ClassVar[list[float]] = []

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.calls = []
        self.instances.append(self)

    def predict(self, pairs):
        self.calls.append(list(pairs))
        return self.scores


def install_fake_model(monkeypatch, scores: list[float]) -> None:
    FakeCrossEncoder.instances.clear()
    FakeCrossEncoder.scores = scores
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(CrossEncoder=FakeCrossEncoder),
    )


def make_result(chunk_id: str, rank: int, score: float = 0.1) -> RankedChunk:
    chunk = Chunk(
        document_id="doc",
        chunk_id=chunk_id,
        content=f"content {chunk_id}",
        source_type=SourceType.CODE,
        source_uri="file:///src/example.py",
        project="knowledge_hub",
        language="python",
        content_role=ContentRole.IMPLEMENTATION,
        provenance=Provenance(
            repository="org/repo",
            path="src/example.py",
            symbol=chunk_id,
        ),
    )
    return RankedChunk(chunk, score, rank, "rrf")


def test_reranker_is_lazy_and_orders_by_cross_encoder_score(monkeypatch) -> None:
    install_fake_model(monkeypatch, [0.2, 0.9, 0.5])
    candidates = [make_result("A", 1), make_result("B", 2), make_result("C", 3)]
    reranker = CrossEncoderReranker("test/bge")

    assert FakeCrossEncoder.instances == []
    result = reranker.rerank("query", candidates, top_k=3)

    assert [item.chunk.chunk_id for item in result] == ["B", "C", "A"]
    assert [item.rank for item in result] == [1, 2, 3]
    assert [item.score for item in result] == pytest.approx([0.9, 0.5, 0.2])
    model = FakeCrossEncoder.instances[0]
    assert model.model_name == "test/bge"
    assert model.calls == [
        [("query", "content A"), ("query", "content B"), ("query", "content C")]
    ]


def test_reranker_top_k_and_fewer_candidates() -> None:
    reranker = CrossEncoderReranker()
    reranker._model = SimpleNamespace(
        predict=lambda pairs: [0.1, 0.9, 0.5][: len(pairs)]
    )
    candidates = [make_result("A", 1), make_result("B", 2), make_result("C", 3)]

    assert len(reranker.rerank("q", candidates, top_k=2)) == 2
    assert len(reranker.rerank("q", candidates[:2], top_k=5)) == 2


def test_empty_candidates_return_without_model_inference() -> None:
    reranker = CrossEncoderReranker()

    assert reranker.rerank("q", [], top_k=5) == []
    assert reranker._model is None


def test_equal_scores_use_original_retrieval_rank() -> None:
    reranker = CrossEncoderReranker()
    reranker._model = SimpleNamespace(predict=lambda pairs: [0.8, 0.8])
    candidates = [make_result("later", 5), make_result("earlier", 2)]

    result = reranker.rerank("q", candidates, top_k=5)

    assert [item.chunk.chunk_id for item in result] == ["earlier", "later"]
    assert [item.rank for item in result] == [1, 2]


def test_reranking_preserves_chunks_metadata_and_provenance() -> None:
    source = make_result("A", 1, score=0.99)
    reranker = CrossEncoderReranker()
    reranker._model = SimpleNamespace(predict=lambda pairs: [0.4])

    result = reranker.rerank("q", [source], top_k=1)[0]

    assert result.chunk is source.chunk
    assert result.chunk.chunk_id == "A"
    assert result.chunk.source_type is SourceType.CODE
    assert result.chunk.project == "knowledge_hub"
    assert result.chunk.language == "python"
    assert result.chunk.provenance.repository == "org/repo"
    assert result.chunk.provenance.path == "src/example.py"
    assert result.chunk.content_role == ContentRole.IMPLEMENTATION
    assert result.score == 0.4
    assert result.channel == "reranker"


def test_cross_encoder_score_controls_order_over_rrf_score() -> None:
    reranker = CrossEncoderReranker()
    reranker._model = SimpleNamespace(predict=lambda pairs: [0.2, 0.9])
    candidates = [make_result("A", 1, score=0.99), make_result("B", 2, score=0.01)]

    result = reranker.rerank("q", candidates, top_k=2)

    assert [item.chunk.chunk_id for item in result] == ["B", "A"]


def test_pipeline_reranks_after_structural_and_metadata_filtering() -> None:
    navigation = make_result("navigation", 1)
    navigation.chunk.content_role = ContentRole.NAVIGATION
    documentation = make_result("documentation", 2)
    documentation.chunk.language = "typescript"
    implementation = make_result("implementation", 1)
    reranker = CrossEncoderReranker()
    reranker._model = SimpleNamespace(predict=lambda pairs: [0.9])

    class FakeRetriever:
        def __init__(self, results):
            self.results = results

        def search(self, query, top_k):
            return self.results

    trace = RetrievalPipeline(
        FakeRetriever([navigation, documentation]),
        FakeRetriever([implementation]),
        reranker=reranker,
    ).search(
        "q",
        dense_k=3,
        sparse_k=3,
        rerank_k=5,
        metadata_filters=MetadataFilters(language="python"),
    )

    assert [item.chunk.chunk_id for item in trace.fusion_results] == [
        "implementation",
        "navigation",
        "documentation",
    ]
    assert [item.chunk.chunk_id for item in trace.filtered_results] == [
        "implementation",
        "documentation",
    ]
    assert [item.chunk.chunk_id for item in trace.metadata_filtered_results] == [
        "implementation"
    ]
    assert [item.chunk.chunk_id for item in trace.reranked_results] == [
        "implementation"
    ]
    assert trace.final_evidence == trace.reranked_results
