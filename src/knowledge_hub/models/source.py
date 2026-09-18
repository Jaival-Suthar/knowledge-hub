from enum import StrEnum


class SourceType(StrEnum):
    PDF = "pdf"
    MARKDOWN = "markdown"
    CODE = "code"
    GITHUB = "github"
    ARTICLE = "article"
    NOTE = "note"


class ContentRole(StrEnum):
    """Canonical role describing the primary purpose of chunk content."""

    NAVIGATION = "navigation"
    METADATA = "metadata"
    DOCUMENTATION = "documentation"
    IMPLEMENTATION = "implementation"
    EVIDENCE = "evidence"
    REFERENCE = "reference"
    TEST = "test"
    CONFIGURATION = "configuration"
