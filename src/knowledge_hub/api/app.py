from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import ValidationError

from knowledge_hub import __version__
from knowledge_hub.config.settings import Settings, settings
from knowledge_hub.models import SourceType

from .contracts import (
    DocumentListResponse,
    IngestMultipartRequest,
    IngestRequest,
    IngestResponse,
    RankedChunkResponse,
    RetrieveRequest,
    RetrieveResponse,
    SearchRequest,
    SearchResponse,
    SourcesResponse,
)
from .corpus import CanonicalChunksNotFoundError, CanonicalCorpus
from .ingestion import IngestionServiceError, KnowledgeIngestionService
from .service import (
    ChunksNotFoundError,
    KnowledgeRetrievalService,
    build_retrieval_service,
)


def health() -> dict[str, str]:
    return {"status": "ok", "service": "knowledge-hub", "version": __version__}


def _configured_service(request: Request) -> KnowledgeRetrievalService:
    service = getattr(request.app.state, "retrieval_service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="retrieval service is not configured",
        )
    return service


def search(payload: SearchRequest, request: Request) -> SearchResponse:
    """Run the existing retrieval pipeline for an HTTP search request."""
    service = _configured_service(request)
    try:
        results = service.search(payload)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="retrieval service failed") from exc
    return SearchResponse(
        query=payload.query,
        results=[
            RankedChunkResponse(
                chunk=result.chunk,
                score=result.score,
                rank=result.rank,
                channel=result.channel,
            )
            for result in results
        ],
    )


def retrieve(payload: RetrieveRequest, request: Request) -> RetrieveResponse:
    """Look up canonical chunks from the configured service collection."""
    service = getattr(request.app.state, "retrieval_service", None)
    try:
        if service is not None:
            chunks = service.retrieve(payload.chunk_ids)
        else:
            chunks = request.app.state.corpus.chunks_by_ids(payload.chunk_ids)
    except (ChunksNotFoundError, CanonicalChunksNotFoundError) as exc:
        raise HTTPException(
            status_code=404,
            detail={"missing_chunk_ids": list(exc.chunk_ids)},
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="retrieval service failed") from exc
    return RetrieveResponse(chunks=chunks)


async def ingest(request: Request) -> IngestResponse:
    """Dispatch JSON URLs or multipart files to existing ingestion services."""
    content_type = request.headers.get("content-type", "").lower()
    source: str | Path | None = None
    source_type = None
    temporary_path: Path | None = None

    try:
        if content_type.startswith("application/json"):
            try:
                payload = IngestRequest.model_validate(await request.json())
            except (ValidationError, ValueError) as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            source = payload.source
            source_type = payload.source_type
        elif content_type.startswith("multipart/form-data"):
            form = await request.form()
            form_source = form.get("source")
            form_type = form.get("source_type")
            upload = form.get("file")
            if upload is not None and hasattr(upload, "filename"):
                filename = getattr(upload, "filename", None)
                if not filename:
                    raise HTTPException(status_code=422, detail="file name is required")
                suffix = Path(filename).suffix.lower()
                with NamedTemporaryFile(
                    prefix="knowledge-hub-api-",
                    suffix=suffix,
                    delete=False,
                ) as temporary:
                    temporary_path = Path(temporary.name)
                    temporary.write(await upload.read())
                source = temporary_path
            elif isinstance(form_source, str) and form_source.strip():
                source = form_source
            else:
                raise HTTPException(
                    status_code=422,
                    detail="provide a source URL/path or an uploaded file",
                )
            if form_type is not None:
                try:
                    source_type = SourceType(str(form_type))
                except ValueError as exc:
                    raise HTTPException(
                        status_code=422, detail="invalid source_type"
                    ) from exc
        else:
            raise HTTPException(
                status_code=415,
                detail="Content-Type must be application/json or multipart/form-data",
            )

        service = getattr(request.app.state, "ingestion_service", None)
        if service is None:
            raise HTTPException(
                status_code=503, detail="ingestion service is not configured"
            )
        result = service.ingest(source, source_type)
        request.app.state.corpus.add(result.documents, result.chunks)
        retrieval_service = getattr(request.app.state, "retrieval_service", None)
        if retrieval_service is not None:
            retrieval_service.add_chunks(result.chunks)
        return IngestResponse(
            source_type=result.source_type,
            documents=list(result.documents),
            chunks=list(result.chunks),
        )
    except HTTPException:
        raise
    except IngestionServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="ingestion service failed") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def documents(
    request: Request,
    limit: int = Query(default=100, gt=0, le=1000),
    offset: int = Query(default=0, ge=0),
    source_type: str | None = None,
) -> DocumentListResponse:
    """Return canonical documents from the API's canonical corpus view."""
    try:
        parsed_source_type = None
        if source_type is not None:
            parsed_source_type = SourceType(source_type)
        values = request.app.state.corpus.documents(limit, offset, parsed_source_type)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="invalid source_type") from exc
    return DocumentListResponse(documents=list(values), limit=limit, offset=offset)


def sources(request: Request) -> SourcesResponse:
    """Return deterministic deduplicated provenance-derived sources."""
    return SourcesResponse(sources=list(request.app.state.corpus.sources()))


def _ingest_openapi() -> dict[str, object]:
    """Describe both supported ingestion request encodings to Swagger."""
    return {
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {"schema": IngestRequest.model_json_schema()},
                "multipart/form-data": {
                    "schema": IngestMultipartRequest.model_json_schema()
                },
            },
        }
    }


def create_app(
    config: Settings = settings,
    retrieval_service: KnowledgeRetrievalService | None = None,
    ingestion_service: KnowledgeIngestionService | None = None,
    corpus: CanonicalCorpus | None = None,
) -> FastAPI:
    """Create the API application using the existing settings object."""
    application = FastAPI(title=config.app_name, version=__version__)
    configured_corpus = (
        corpus if corpus is not None else CanonicalCorpus.from_qdrant(config)
    )
    configured_retrieval = (
        retrieval_service
        if retrieval_service is not None
        else build_retrieval_service(config, configured_corpus.chunks())
    )
    application.add_api_route("/health", health, methods=["GET"])
    application.add_api_route(
        "/search",
        search,
        methods=["POST"],
        response_model=SearchResponse,
    )
    application.add_api_route(
        "/retrieve",
        retrieve,
        methods=["POST"],
        response_model=RetrieveResponse,
    )
    application.add_api_route(
        "/ingest",
        ingest,
        methods=["POST"],
        response_model=IngestResponse,
        openapi_extra=_ingest_openapi(),
    )
    application.add_api_route(
        "/documents",
        documents,
        methods=["GET"],
        response_model=DocumentListResponse,
    )
    application.add_api_route(
        "/sources",
        sources,
        methods=["GET"],
        response_model=SourcesResponse,
    )
    application.state.retrieval_service = configured_retrieval
    application.state.ingestion_service = (
        ingestion_service or KnowledgeIngestionService()
    )
    application.state.corpus = configured_corpus
    return application


app = create_app()
