from pydantic import BaseModel, Field


class Provenance(BaseModel):
    source_uri: str | None = None
    repository: str | None = None
    branch: str | None = None
    commit_sha: str | None = None
    path: str | None = None
    page: int | None = None
    section: str | None = None
    symbol: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    extra: dict[str, str] = Field(default_factory=dict)
