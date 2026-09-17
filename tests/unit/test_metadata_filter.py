from __future__ import annotations

import pytest

from knowledge_hub.models import Chunk, ContentRole, Provenance, SourceType
from knowledge_hub.retrieval.metadata import MetadataFilter, MetadataFilters
from knowledge_hub.retrieval.pipeline import RetrievalPipeline
from knowledge_hub.retrieval.types import RankedChunk


def make_result(
    chunk_id: str,
    *,
    source_type=SourceType.GITHUB,
    project="knowledge_hub",
    language="Python",
    repository="Jaival-Suthar/Knowledge-Hub",
    path="src/knowledge_hub/retrieval/structural.py",
    content_role=ContentRole.IMPLEMENTATION,
    rank=1,
    score=0.8,
) -> RankedChunk:
    chunk = Chunk(
        document_id="doc",
        chunk_id=chunk_id,
        content=chunk_id,
        source_type=source_type,
        project=project,
        language=language,
        content_role=content_role,
        provenance=Provenance(repository=repository, path=path),
    )
    return RankedChunk(chunk, score, rank, "rrf")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_type", "github"),
        ("project", "knowledge_hub"),
        ("language", "python"),
        ("repository", "jaival-suthar/knowledge-hub"),
        ("path", "src/knowledge_hub/retrieval/structural.py"),
        ("content_role", "implementation"),
    ],
)
def test_each_metadata_dimension_filters_independently(field, value) -> None:
    matching = make_result("match")
    other_values = {
        "source_type": SourceType.MARKDOWN,
        "project": "other_project",
        "language": "typescript",
        "repository": "other/repository",
        "path": "other/path.py",
        "content_role": ContentRole.REFERENCE,
    }
    other = make_result("other", **{field: other_values[field]})

    result = MetadataFilter().filter(
        [matching, other], MetadataFilters(**{field: value})
    )

    assert result == [matching]


def test_filters_use_and_semantics_and_multi_values_use_or() -> None:
    matching = make_result("match")
    wrong_role = make_result("wrong-role", content_role=ContentRole.REFERENCE)
    wrong_language = make_result("wrong-language", language="rust")

    result = MetadataFilter().filter(
        [matching, wrong_role, wrong_language],
        MetadataFilters(
            source_type="github",
            language=["python", "typescript"],
            content_role="implementation",
        ),
    )

    assert result == [matching]


def test_missing_metadata_does_not_match_but_survives_unfiltered() -> None:
    missing = make_result("missing", language=None, repository=None, path=None)
    metadata_filter = MetadataFilter()

    assert metadata_filter.filter([missing], MetadataFilters(language="python")) == []
    assert metadata_filter.filter([missing], None) == [missing]


def test_unknown_filter_value_is_a_deterministic_non_match() -> None:
    result = make_result("chunk")

    assert MetadataFilter().filter([result], MetadataFilters(language="go")) == []


def test_empty_input_and_no_constraints_are_safe_and_preserve_identity() -> None:
    result = make_result("chunk")
    metadata_filter = MetadataFilter()

    assert metadata_filter.filter([], MetadataFilters()) == []
    assert metadata_filter.filter([result], MetadataFilters()) == [result]
    assert metadata_filter.filter([result], MetadataFilters())[0] is result
    assert result.rank == 1
    assert result.score == 0.8
    assert result.chunk.provenance.repository == "Jaival-Suthar/Knowledge-Hub"


def test_reference_role_remains_filterable_and_eligible() -> None:
    result = make_result("reference", content_role=ContentRole.REFERENCE)

    assert MetadataFilter().filter(
        [result], MetadataFilters(content_role="reference")
    ) == [result]


def test_pipeline_applies_metadata_after_structural_filtering() -> None:
    navigation = make_result("navigation", content_role=ContentRole.NAVIGATION, rank=1)
    documentation = make_result(
        "documentation",
        content_role=ContentRole.DOCUMENTATION,
        language="typescript",
        rank=2,
    )
    metadata = make_result("metadata", content_role=ContentRole.METADATA, rank=1)
    implementation = make_result(
        "implementation",
        content_role=ContentRole.IMPLEMENTATION,
        language="python",
        rank=2,
    )

    class FakeRetriever:
        def __init__(self, results):
            self.results = results

        def search(self, query, top_k):
            return self.results

    trace = RetrievalPipeline(
        FakeRetriever([navigation, documentation]),
        FakeRetriever([metadata, implementation]),
    ).search(
        "query",
        dense_k=4,
        sparse_k=4,
        rerank_k=10,
        metadata_filters=MetadataFilters(language="python"),
    )

    assert {item.chunk.chunk_id for item in trace.fusion_results} == {
        "navigation",
        "documentation",
        "metadata",
        "implementation",
    }
    assert {item.chunk.chunk_id for item in trace.filtered_results} == {
        "documentation",
        "implementation",
    }
    assert [item.chunk.chunk_id for item in trace.metadata_filtered_results] == [
        "implementation"
    ]
    structural_item = next(
        item
        for item in trace.filtered_results
        if item.chunk.chunk_id == "implementation"
    )
    assert trace.metadata_filtered_results[0] is structural_item
    assert trace.metadata_filtered_results[0].chunk is implementation.chunk
    assert trace.final_evidence == trace.metadata_filtered_results
