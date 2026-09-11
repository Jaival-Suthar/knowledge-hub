from __future__ import annotations

from typing import ClassVar

import tree_sitter_javascript

from ..contracts import CodeSymbolType, Language
from .base import TreeSitterParser


class JavaScriptParser(TreeSitterParser):
    symbol_types: ClassVar[dict[str, CodeSymbolType | str]] = {
        "class_declaration": CodeSymbolType.CLASS,
        "function_declaration": CodeSymbolType.FUNCTION,
        "method_definition": CodeSymbolType.METHOD,
    }

    def __init__(self) -> None:
        super().__init__(Language.JAVASCRIPT, tree_sitter_javascript.language)
