from enum import StrEnum


class SourceType(StrEnum):
    PDF = "pdf"
    MARKDOWN = "markdown"
    CODE = "code"
    GITHUB = "github"
    ARTICLE = "article"
    NOTE = "note"
