from fastapi import FastAPI, HTTPException, Request

from knowledge_hub import __version__
from knowledge_hub.config.settings import Settings, settings

from .contracts import (
    RankedChunkResponse,
    RetrieveRequest,
    RetrieveResponse,
    SearchRequest,
    SearchResponse,
)
from .service import ChunksNotFoundError, KnowledgeRetrievalService


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
    service = _configured_service(request)
    try:
        chunks = service.retrieve(payload.chunk_ids)
    except ChunksNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={"missing_chunk_ids": list(exc.chunk_ids)},
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="retrieval service failed") from exc
    return RetrieveResponse(chunks=chunks)


def create_app(
    config: Settings = settings,
    retrieval_service: KnowledgeRetrievalService | None = None,
) -> FastAPI:
    """Create the API application using the existing settings object."""
    application = FastAPI(title=config.app_name, version=__version__)
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
    application.state.retrieval_service = retrieval_service
    return application


app = create_app()
