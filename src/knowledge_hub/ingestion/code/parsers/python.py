from __future__ import annotations

from typing import ClassVar

import tree_sitter_python

from ..contracts import CodeSymbolType, Language
from .base import TreeSitterParser


class PythonParser(TreeSitterParser):
    symbol_types: ClassVar[dict[str, CodeSymbolType | str]] = {
        "class_definition": CodeSymbolType.CLASS,
        "function_definition": CodeSymbolType.FUNCTION,
    }

    def __init__(self) -> None:
        super().__init__(Language.PYTHON, tree_sitter_python.language)

    def _symbol_type(self, node, parent):
        if node.type == "function_definition" and node.text:
            if parent is not None and parent.symbol_type == CodeSymbolType.CLASS:
                return CodeSymbolType.METHOD
            if node.text.startswith(b"async "):
                return CodeSymbolType.ASYNC_FUNCTION
        return super()._symbol_type(node, parent)
