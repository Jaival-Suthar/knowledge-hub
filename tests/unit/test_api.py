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
from knowledge_hub.api.service import KnowledgeRetrievalService
from knowledge_hub.config.settings import Settings
from knowledge_hub.models import Chunk, Document, Provenance, SourceType
from knowledge_hub.retrieval.metadata import MetadataFilters
from knowledge_hub.retrieval.types import RankedChunk, RetrievalTrace


def _chunk(chunk_id: str = "chunk-1") -> Chunk:
    provenance = Provenance(
        source_uri="https://example.test/source",
        repository="org/repo",
        path="src/auth.py",
        symbol="Auth.login",
    )
    return Chunk(
        document_id="doc-1",
        chunk_id=chunk_id,
        content="def login(): pass",
        source_type=SourceType.GITHUB,
        provenance=provenance,
    )


class _StubPipeline:
    def __init__(self, result: RetrievalTrace | None = None) -> None:
        self.result = result or RetrievalTrace(query="")
        self.calls: list[dict[str, object]] = []

    def search(self, **kwargs: object) -> RetrievalTrace:
        self.calls.append(kwargs)
        return self.result


class _FailingService:
    def search(self, request: SearchRequest) -> list[RankedChunk]:
        raise RuntimeError("backend unavailable")

    def retrieve(self, chunk_ids: list[str]) -> list[Chunk]:
        raise RuntimeError("backend unavailable")


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

    assert client.post("/ingest", json={"source": "anything"}).status_code == 404


def test_search_delegates_request_to_existing_pipeline_and_preserves_provenance() -> (
    None
):
    chunk = _chunk()
    trace = RetrievalTrace(
        query="authentication",
        final_evidence=[
            RankedChunk(chunk=chunk, score=0.9, rank=1, channel="rrf"),
        ],
    )
    pipeline = _StubPipeline(trace)
    service = KnowledgeRetrievalService(pipeline, [chunk])
    client = TestClient(create_app(retrieval_service=service))

    response = client.post(
        "/search",
        json={
            "query": "authentication",
            "top_k": 7,
            "metadata_filters": {
                "source_type": "github",
                "language": ["python", "typescript"],
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["results"][0]["chunk"]["provenance"]["path"] == "src/auth.py"
    assert response.json()["results"][0]["channel"] == "rrf"
    assert pipeline.calls == [
        {
            "query": "authentication",
            "dense_k": 7,
            "sparse_k": 7,
            "rerank_k": 7,
            "metadata_filters": MetadataFilters(
                source_type="github",
                language=("python", "typescript"),
            ),
        }
    ]


def test_search_returns_service_failure_as_500() -> None:
    client = TestClient(create_app(retrieval_service=_FailingService()))

    response = client.post("/search", json={"query": "anything"})

    assert response.status_code == 500
    assert response.json()["detail"] == "retrieval service failed"


def test_search_without_service_returns_503() -> None:
    response = TestClient(create_app()).post("/search", json={"query": "anything"})

    assert response.status_code == 503
    assert response.json()["detail"] == "retrieval service is not configured"


def test_retrieve_returns_requested_canonical_chunks_in_order() -> None:
    first = _chunk("chunk-1")
    second = _chunk("chunk-2")
    service = KnowledgeRetrievalService(_StubPipeline(), [first, second])

    response = TestClient(create_app(retrieval_service=service)).post(
        "/retrieve",
        json={"chunk_ids": ["chunk-2", "chunk-1"]},
    )

    assert response.status_code == 200
    assert [chunk["chunk_id"] for chunk in response.json()["chunks"]] == [
        "chunk-2",
        "chunk-1",
    ]
    assert response.json()["chunks"][0]["provenance"]["source_uri"] == (
        "https://example.test/source"
    )


def test_retrieve_returns_404_for_missing_chunks() -> None:
    service = KnowledgeRetrievalService(_StubPipeline(), [_chunk()])

    response = TestClient(create_app(retrieval_service=service)).post(
        "/retrieve",
        json={"chunk_ids": ["missing", "also-missing"]},
    )

    assert response.status_code == 404
    assert response.json()["detail"]["missing_chunk_ids"] == [
        "also-missing",
        "missing",
    ]


def test_retrieve_rejects_empty_chunk_id_request() -> None:
    response = TestClient(create_app()).post("/retrieve", json={"chunk_ids": []})

    assert response.status_code == 422
