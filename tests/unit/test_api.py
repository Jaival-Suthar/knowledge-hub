from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import knowledge_hub.api.service as retrieval_service_module
from knowledge_hub.api.app import create_app
from knowledge_hub.api.contracts import (
    DocumentResponse,
    IngestRequest,
    RankedChunkResponse,
    RetrieveRequest,
    SearchRequest,
    SourceResponse,
)
from knowledge_hub.api.corpus import CanonicalCorpus
from knowledge_hub.api.ingestion import IngestionResult, KnowledgeIngestionService
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
        self.sparse = _StubSparse()

    def search(self, **kwargs: object) -> RetrievalTrace:
        self.calls.append(kwargs)
        return self.result


class _StubSparse:
    def __init__(self) -> None:
        self.updated: list[tuple[Chunk, ...]] = []

    def update(self, chunks: object) -> None:
        self.updated.append(tuple(chunks))


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

    assert client.get("/not-an-endpoint").status_code == 404


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
    app = create_app()
    app.state.retrieval_service = None

    response = TestClient(app).post("/search", json={"query": "anything"})

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


def test_retrieve_uses_canonical_corpus_when_pipeline_is_not_configured() -> None:
    chunk = _chunk()
    client = TestClient(create_app(corpus=CanonicalCorpus(chunks=[chunk])))

    response = client.post("/retrieve", json={"chunk_ids": ["chunk-1"]})

    assert response.status_code == 200
    assert response.json()["chunks"][0]["chunk_id"] == "chunk-1"


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


class _StubIngestionService:
    def __init__(self, result: IngestionResult) -> None:
        self.result = result
        self.calls: list[tuple[object, object]] = []

    def ingest(self, source: object, source_type: object = None) -> IngestionResult:
        self.calls.append((source, source_type))
        return self.result


def test_ingest_delegates_and_returns_canonical_results() -> None:
    document = Document(
        document_id="doc-1",
        source_type=SourceType.MARKDOWN,
        title="Guide",
        source_uri="file:///guide.md",
    )
    chunk = _chunk()
    result = IngestionResult(SourceType.MARKDOWN, [document], [chunk])
    ingestion = _StubIngestionService(result)

    retrieval = KnowledgeRetrievalService(_StubPipeline(), [])
    response = TestClient(
        create_app(ingestion_service=ingestion, retrieval_service=retrieval)
    ).post(
        "/ingest",
        json={"source": "guide.md", "source_type": "markdown"},
    )

    assert response.status_code == 200
    assert response.json()["source_type"] == "markdown"
    assert response.json()["documents"][0]["document_id"] == "doc-1"
    assert response.json()["chunks"][0]["chunk_id"] == "chunk-1"
    assert ingestion.calls == [("guide.md", SourceType.MARKDOWN)]


def test_ingest_upload_delegates_file_and_preserves_canonical_source() -> None:
    document = Document(
        document_id="doc-1",
        source_type=SourceType.MARKDOWN,
        source_uri="file:///guide.md",
    )
    ingestion = _StubIngestionService(
        IngestionResult(SourceType.MARKDOWN, [document], [_chunk()])
    )

    retrieval = KnowledgeRetrievalService(_StubPipeline(), [])
    response = TestClient(
        create_app(ingestion_service=ingestion, retrieval_service=retrieval)
    ).post(
        "/ingest",
        files={"file": ("guide.md", b"# Guide\n\nContent", "text/markdown")},
    )

    assert response.status_code == 200
    assert Path(ingestion.calls[0][0]).suffix == ".md"


def test_add_chunks_updates_bm25_and_persists_only_new_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = _chunk("existing")
    added = _chunk("added")
    pipeline = _StubPipeline()
    calls: list[tuple[object, tuple[Chunk, ...], object]] = []

    def fake_embed_and_upsert(index: object, chunks: object, embedder: object) -> None:
        calls.append((index, tuple(chunks), embedder))

    monkeypatch.setattr(
        retrieval_service_module,
        "embed_and_upsert",
        fake_embed_and_upsert,
    )
    index = object()
    embedder = object()
    service = KnowledgeRetrievalService(
        pipeline,
        [existing],
        index=index,
        embedder=embedder,
    )

    service.add_chunks([added])
    service.add_chunks([added])

    assert service.chunks["added"] == added
    assert calls == [(index, (added,), embedder)]
    assert pipeline.sparse.updated == [(existing, added), (existing, added)]


def test_fresh_qdrant_corpus_hydrates_new_persisted_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk = _chunk("persisted")
    fake_client = SimpleNamespace(
        scroll=lambda **_kwargs: (
            [SimpleNamespace(payload=chunk.model_dump(mode="json"))],
            None,
        )
    )
    monkeypatch.setattr(
        "knowledge_hub.api.corpus.QdrantIndex",
        lambda **_kwargs: SimpleNamespace(client=fake_client, collection="knowledge"),
    )

    hydrated = CanonicalCorpus.from_qdrant(Settings())

    assert hydrated.chunks_by_ids(["persisted"])[0] == chunk
    assert hydrated.documents(100, 0)[0].document_id == chunk.document_id


def test_ingest_rejects_unsupported_content_type() -> None:
    response = TestClient(create_app()).post(
        "/ingest",
        content=b"unsupported",
        headers={"content-type": "text/plain"},
    )

    assert response.status_code == 415


def test_ingest_rejects_invalid_zip_as_bad_input(tmp_path: Path) -> None:
    archive = tmp_path / "broken.zip"
    archive.write_bytes(b"not a zip")

    response = TestClient(create_app()).post(
        "/ingest",
        files={"file": (archive.name, archive.read_bytes(), "application/zip")},
    )

    assert response.status_code == 400


def test_ingest_openapi_declares_json_and_multipart_file_input() -> None:
    schema = TestClient(create_app()).get("/openapi.json").json()
    content = schema["paths"]["/ingest"]["post"]["requestBody"]["content"]

    assert "application/json" in content
    assert "multipart/form-data" in content
    multipart_schema = content["multipart/form-data"]["schema"]
    file_schema = multipart_schema["properties"]["file"]
    assert any(
        option.get("type") == "string" and option.get("format") == "binary"
        for option in file_schema["anyOf"]
    )
    assert any(
        option.get("type") == "string"
        for option in multipart_schema["properties"]["source"]["anyOf"]
    )
    assert "source_type" in multipart_schema["properties"]


def test_documents_and_sources_use_canonical_corpus() -> None:
    document = Document(
        document_id="doc-1",
        source_type=SourceType.GITHUB,
        source_uri="https://github.com/org/repo",
        provenance=Provenance(
            source_uri="https://github.com/org/repo",
            repository="org/repo",
            path="src/auth.py",
        ),
    )
    app = create_app(corpus=CanonicalCorpus([document], [_chunk()]))
    client = TestClient(app)

    documents = client.get("/documents", params={"limit": 1, "offset": 0})
    sources = client.get("/sources")

    assert documents.status_code == 200
    assert documents.json()["documents"][0]["document_id"] == "doc-1"
    assert sources.status_code == 200
    assert sources.json()["sources"][0]["provenance"]["repository"] == "org/repo"


def test_documents_reject_invalid_pagination() -> None:
    response = TestClient(create_app()).get("/documents", params={"limit": 0})

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("source", "method"),
    [
        ("https://author.hashnode.dev/article", "_ingest_hashnode"),
        ("https://github.com/org/repo", "_ingest_github"),
    ],
)
def test_url_ingestion_dispatches_to_existing_acquisition_flows(
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    method: str,
) -> None:
    result = IngestionResult(SourceType.ARTICLE, [], [])
    service = KnowledgeIngestionService()
    calls: list[str] = []

    def fake_ingest(value: str) -> IngestionResult:
        calls.append(value)
        return result

    monkeypatch.setattr(service, method, fake_ingest)

    assert service.ingest(source) is result
    assert calls == [source]


@pytest.mark.parametrize("suffix", [".zip", ".md", ".markdown", ".pdf"])
def test_file_ingestion_dispatches_by_supported_extension(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    suffix: str,
) -> None:
    path = tmp_path / f"source{suffix}"
    path.write_bytes(b"placeholder")
    result = IngestionResult(SourceType.CODE, [], [])
    service = KnowledgeIngestionService()

    if suffix == ".zip":
        monkeypatch.setattr(service, "_ingest_zip", lambda value: result)
    elif suffix in {".md", ".markdown"}:
        monkeypatch.setattr(
            "knowledge_hub.api.ingestion.MarkdownAdapter.extract",
            lambda _adapter, _value: Document(
                document_id="doc", source_type=SourceType.MARKDOWN
            ),
        )
        monkeypatch.setattr(
            "knowledge_hub.api.ingestion.MarkdownChunker.chunk",
            lambda _chunker, _document: [],
        )
    else:
        monkeypatch.setattr(
            "knowledge_hub.api.ingestion.PdfAdapter.extract",
            lambda _adapter, _value: Document(
                document_id="doc", source_type=SourceType.PDF
            ),
        )
        monkeypatch.setattr(
            "knowledge_hub.api.ingestion.PdfChunker.chunk",
            lambda _chunker, _document: [],
        )

    output = service.ingest(path)

    expected_type = {
        ".zip": SourceType.CODE,
        ".md": SourceType.MARKDOWN,
        ".markdown": SourceType.MARKDOWN,
        ".pdf": SourceType.PDF,
    }[suffix]
    assert output.source_type is expected_type
