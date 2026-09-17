import pytest

from knowledge_hub.models import Chunk, ContentRole, Provenance, SourceType
from knowledge_hub.retrieval.structural import (
    EligibilityStatus,
    StructuralEligibility,
    structural_eligible,
)
from knowledge_hub.retrieval.types import RankedChunk


def make_chunk(
    content: str,
    *,
    parent_structure: str | None = None,
    structural_type: str | None = None,
    content_role: str | None = None,
    chunk_id: str = "chunk",
) -> Chunk:
    return Chunk(
        document_id="doc",
        chunk_id=chunk_id,
        content=content,
        source_type=SourceType.PDF,
        parent_structure=parent_structure,
        structural_type=structural_type,
        content_role=content_role,
    )


@pytest.mark.parametrize(
    "chunk",
    [
        make_chunk("Contents\nIntroduction\nChapter 1", parent_structure="Contents"),
        make_chunk("Table of Contents\nIntroduction\nChapter 1"),
        make_chunk(
            "Copyright © 2011 by Example. All rights reserved.\nPublished by Example Press.",
        ),
        make_chunk(
            "Requests for permission should be sent to the publisher.\n"
            "Reproduction in any form is prohibited.\n"
            "Copyright Clearance Center, Inc.",
        ),
        make_chunk(
            "capability, actions, and mind-set of\nDaily actions\nDanger, embracing\n"
            "Deal, closing\nFailure avoiding\nFamily\nGoal achievement\nGrowth, continued\n"
            "Habits\nLife situations\n",
        ),
        make_chunk(
            "Glossary\nA term — its definition\nAnother term — its definition",
            parent_structure="Glossary",
        ),
    ],
)
def test_structural_material_is_ineligible(chunk: Chunk) -> None:
    assert not structural_eligible(chunk)


@pytest.mark.parametrize(
    "content",
    [
        "The system stores documentation in a searchable index for later retrieval.",
        "This documentation discusses copyright law and licensing choices.",
        "See the glossary for definitions, then continue with the implementation.",
    ],
)
def test_ordinary_documentation_remains_eligible(content: str) -> None:
    assert structural_eligible(make_chunk(content))


def test_explicit_documentation_role_is_authoritative() -> None:
    chunk = make_chunk("Contents\nChapter One", content_role="documentation")
    assert StructuralEligibility().classify(chunk) is EligibilityStatus.ELIGIBLE
    assert structural_eligible(chunk)


def test_explicit_reference_role_is_eligible() -> None:
    chunk = make_chunk("References\nUseful source", content_role="reference")

    assert StructuralEligibility().classify(chunk) is EligibilityStatus.ELIGIBLE
    assert structural_eligible(chunk)


@pytest.mark.parametrize("role", list(ContentRole))
def test_every_content_role_is_representable(role: ContentRole) -> None:
    chunk = make_chunk("content", content_role=role.value)

    assert StructuralEligibility().classify(chunk) in {
        EligibilityStatus.ELIGIBLE,
        EligibilityStatus.INELIGIBLE,
    }


@pytest.mark.parametrize(
    ("role", "expected"),
    [
        ("navigation", EligibilityStatus.INELIGIBLE),
        ("metadata", EligibilityStatus.INELIGIBLE),
        ("documentation", EligibilityStatus.ELIGIBLE),
        ("implementation", EligibilityStatus.ELIGIBLE),
        ("evidence", EligibilityStatus.ELIGIBLE),
        ("reference", EligibilityStatus.ELIGIBLE),
        ("test", EligibilityStatus.ELIGIBLE),
        ("configuration", EligibilityStatus.ELIGIBLE),
        ("future-role", EligibilityStatus.UNKNOWN),
        (None, EligibilityStatus.UNKNOWN),
    ],
)
def test_explicit_policy_and_unknown_roles(role, expected) -> None:
    assert (
        StructuralEligibility().classify(make_chunk("content", content_role=role))
        == expected
    )
    assert structural_eligible(make_chunk("content", content_role=role)) is not (
        expected is EligibilityStatus.INELIGIBLE
    )


def test_filter_preserves_ranked_chunks_and_original_ranks() -> None:
    eligible = make_chunk("docs", content_role=ContentRole.DOCUMENTATION.value)
    excluded = make_chunk("nav", content_role=ContentRole.NAVIGATION.value)
    result = StructuralEligibility().filter(
        [
            RankedChunk(excluded, 1.0, 1, "rrf"),
            RankedChunk(eligible, 0.8, 2, "rrf"),
        ]
    )

    assert result[0].chunk is eligible
    assert (result[0].chunk.chunk_id, result[0].rank, result[0].score) == (
        "chunk",
        2,
        0.8,
    )
    assert result[0].chunk.source_type is SourceType.PDF
    assert result[0].chunk.content_role == ContentRole.DOCUMENTATION


def test_filter_mixed_explicit_roles_preserves_objects_metadata_and_ranks() -> None:
    roles = [
        ContentRole.NAVIGATION,
        ContentRole.METADATA,
        ContentRole.DOCUMENTATION,
        ContentRole.IMPLEMENTATION,
        ContentRole.EVIDENCE,
        ContentRole.REFERENCE,
        ContentRole.TEST,
        ContentRole.CONFIGURATION,
        "future-role",
    ]
    ranked = []
    for rank, role in enumerate(roles, start=1):
        chunk = make_chunk(
            role.value if isinstance(role, ContentRole) else role,
            content_role=role.value if isinstance(role, ContentRole) else role,
            chunk_id=f"chunk-{rank}",
        ).model_copy(
            update={
                "source_uri": f"file:///chunk-{rank}",
                "provenance": Provenance(source_uri=f"file:///chunk-{rank}"),
            }
        )
        ranked.append(RankedChunk(chunk, 1.0 / rank, rank, "rrf"))

    result = StructuralEligibility().filter(ranked)

    assert [item.chunk.chunk_id for item in result] == [
        "chunk-3",
        "chunk-4",
        "chunk-5",
        "chunk-6",
        "chunk-7",
        "chunk-8",
        "chunk-9",
    ]
    assert [item.rank for item in result] == [3, 4, 5, 6, 7, 8, 9]
    assert [item.score for item in result] == [
        1 / 3,
        1 / 4,
        1 / 5,
        1 / 6,
        1 / 7,
        1 / 8,
        1 / 9,
    ]
    assert all(item.chunk.source_type is SourceType.PDF for item in result)
    assert all(
        item.chunk.source_uri == item.chunk.provenance.source_uri for item in result
    )
    assert all(item.chunk is ranked[item.rank - 1].chunk for item in result)


def test_unknown_role_is_retained_with_metadata_and_provenance() -> None:
    chunk = make_chunk("unknown", content_role=None)
    chunk = chunk.model_copy(
        update={
            "source_uri": "file:///unknown",
            "provenance": Provenance(source_uri="file:///unknown"),
        }
    )

    result = StructuralEligibility().filter([RankedChunk(chunk, 1.0, 1, "rrf")])

    assert result[0].chunk is chunk
    assert result[0].chunk.source_uri == "file:///unknown"
    assert result[0].chunk.provenance.source_uri == "file:///unknown"
