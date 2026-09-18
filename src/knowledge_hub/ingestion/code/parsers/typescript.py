from __future__ import annotations

from typing import ClassVar

import tree_sitter_typescript
from tree_sitter import Language as TreeSitterLanguage
from tree_sitter import Parser

from ..contracts import CodeFile, CodeParseResult, CodeSymbolType, Language
from .base import TreeSitterParser


class TypeScriptParser(TreeSitterParser):
    symbol_types: ClassVar[dict[str, CodeSymbolType | str]] = {
        "class_declaration": CodeSymbolType.CLASS,
        "function_declaration": CodeSymbolType.FUNCTION,
        "method_definition": CodeSymbolType.METHOD,
        "interface_declaration": CodeSymbolType.INTERFACE,
        "type_alias_declaration": CodeSymbolType.TYPE,
        "enum_declaration": CodeSymbolType.ENUM,
    }

    def __init__(self, tsx: bool = False) -> None:
        super().__init__(
            Language.TYPESCRIPT,
            tree_sitter_typescript.language_tsx
            if tsx
            else tree_sitter_typescript.language_typescript,
        )
        self._tsx = tsx
        self._tsx_parser = (
            None
            if tsx
            else Parser(TreeSitterLanguage(tree_sitter_typescript.language_tsx()))
        )

    def parse(self, code_file: CodeFile) -> CodeParseResult:
        source = code_file.content.encode("utf-8")
        if self._tsx or code_file.relative_path.lower().endswith(".tsx"):
            parser = self._parser if self._tsx else self._tsx_parser
            assert parser is not None
            return self._parse_with_parser(parser, code_file, source)
        return self._parse_with_parser(self._parser, code_file, source)
