import pytest

from knowledge_hub.models import Chunk, SourceType
from knowledge_hub.retrieval.structural import structural_eligible


def make_chunk(
    content: str,
    *,
    parent_structure: str | None = None,
    structural_type: str | None = None,
    content_role: str | None = "documentation",
) -> Chunk:
    return Chunk(
        document_id="doc",
        chunk_id="chunk",
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


def test_content_role_is_only_a_fallback() -> None:
    chunk = make_chunk("Contents\nChapter One", content_role="documentation")
    assert not structural_eligible(chunk)
