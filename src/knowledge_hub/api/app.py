from fastapi import FastAPI

from knowledge_hub import __version__
from knowledge_hub.config.settings import Settings, settings


def health() -> dict[str, str]:
    return {"status": "ok", "service": "knowledge-hub", "version": __version__}


def create_app(config: Settings = settings) -> FastAPI:
    """Create the API application using the existing settings object."""
    application = FastAPI(title=config.app_name, version=__version__)
    application.add_api_route("/health", health, methods=["GET"])
    return application


app = create_app()
