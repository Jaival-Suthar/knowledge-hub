from fastapi import FastAPI

from knowledge_hub import __version__

app = FastAPI(title="Knowledge Hub", version=__version__)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "knowledge-hub", "version": __version__}
