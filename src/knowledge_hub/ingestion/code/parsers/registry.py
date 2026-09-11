from __future__ import annotations

from collections.abc import Mapping

from ..contracts import Language, LanguageParser, UnsupportedLanguageError
from .cpp import CParser, CppParser
from .javascript import JavaScriptParser
from .python import PythonParser
from .typescript import TypeScriptParser


class ParserRegistry:
    """Deterministic mapping from canonical languages to parser instances."""

    def __init__(self, parsers: Mapping[Language, LanguageParser]) -> None:
        self._parsers = dict(parsers)

    @classmethod
    def default(cls) -> ParserRegistry:
        return cls(
            {
                Language.TYPESCRIPT: TypeScriptParser(),
                Language.JAVASCRIPT: JavaScriptParser(),
                Language.PYTHON: PythonParser(),
                Language.C: CParser(),
                Language.CPP: CppParser(),
            }
        )

    def get(self, language: Language) -> LanguageParser:
        try:
            return self._parsers[language]
        except KeyError as error:
            raise UnsupportedLanguageError(
                f"no parser registered for language: {language}"
            ) from error

    def languages(self) -> tuple[Language, ...]:
        return tuple(self._parsers)
