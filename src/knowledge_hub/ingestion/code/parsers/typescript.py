from __future__ import annotations

from typing import ClassVar

import tree_sitter_typescript

from ..contracts import CodeSymbolType, Language
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
