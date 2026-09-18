from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from knowledge_hub.api.app import create_app
from knowledge_hub.api.contracts import (
    DocumentResponse,
    IngestRequest,
    RankedChunkResponse,
    RetrieveRequest,
    SearchRequest,
    SourceResponse,
)
from knowledge_hub.config.settings import Settings
from knowledge_hub.models import Chunk, Document, Provenance, SourceType


def test_create_app_uses_existing_settings_configuration() -> None:
    app = create_app(Settings(app_name="Test Knowledge Hub"))

    assert app.title == "Test Knowledge Hub"
    assert app.version == "0.1.0"


def test_health_endpoint_returns_deterministic_status() -> None:
    response = TestClient(create_app()).get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_search_request_validates_and_serializes() -> None:
    request = SearchRequest(
        query="  find authentication  ",
        top_k=5,
        metadata_filters={"source_type": "github"},
    )

    assert request.model_dump() == {
        "query": "  find authentication  ",
        "top_k": 5,
        "metadata_filters": {
            "source_type": "github",
            "project": None,
            "language": None,
            "repository": None,
            "path": None,
            "content_role": None,
        },
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": "query", "top_k": 0},
        {"query": "query", "top_k": -1},
    ],
)
def test_search_request_rejects_invalid_payload(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        SearchRequest.model_validate(payload)


def test_other_request_contracts_reject_invalid_payloads() -> None:
    with pytest.raises(ValidationError):
        RetrieveRequest.model_validate({"chunk_ids": []})
    with pytest.raises(ValidationError):
        IngestRequest.model_validate({"source": ""})


def test_canonical_domain_objects_serialize_through_api_responses() -> None:
    provenance = Provenance(
        source_uri="https://example.test/source",
        repository="org/repo",
        path="src/auth.py",
        symbol="Auth.login",
    )
    document = Document(
        document_id="doc-1",
        source_type=SourceType.GITHUB,
        title="Auth",
        source_uri="https://example.test/source",
        provenance=provenance,
    )
    chunk = Chunk(
        document_id="doc-1",
        chunk_id="chunk-1",
        content="def login(): pass",
        source_type=SourceType.GITHUB,
        provenance=provenance,
    )

    document_response = DocumentResponse(document=document)
    ranked_response = RankedChunkResponse(
        chunk=chunk,
        score=0.9,
        rank=1,
        channel="rrf",
    )
    source_response = SourceResponse(
        source_id="source-1",
        source_type=SourceType.GITHUB,
        source_uri=document.source_uri,
        provenance=provenance,
    )

    assert (
        document_response.model_dump(mode="json")["document"]["source_type"] == "github"
    )
    assert ranked_response.model_dump(mode="json")["chunk"]["chunk_id"] == "chunk-1"
    assert (
        source_response.model_dump(mode="json")["provenance"]["repository"]
        == "org/repo"
    )


def test_business_endpoints_are_not_implemented_in_foundation() -> None:
    client = TestClient(create_app())

    assert client.post("/search", json={"query": "anything"}).status_code == 404
