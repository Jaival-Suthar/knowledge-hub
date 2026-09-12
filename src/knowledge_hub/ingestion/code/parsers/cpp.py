from __future__ import annotations

from typing import ClassVar

import tree_sitter_c
import tree_sitter_cpp

from ..contracts import CodeSymbolType, Language
from .base import TreeSitterParser


class CParser(TreeSitterParser):
    symbol_types: ClassVar[dict[str, CodeSymbolType | str]] = {
        "struct_specifier": CodeSymbolType.STRUCT,
        "function_definition": CodeSymbolType.FUNCTION,
        "enum_specifier": CodeSymbolType.ENUM,
    }

    def __init__(self) -> None:
        super().__init__(Language.C, tree_sitter_c.language)


class CppParser(TreeSitterParser):
    symbol_types: ClassVar[dict[str, CodeSymbolType | str]] = {
        "class_specifier": CodeSymbolType.CLASS,
        "struct_specifier": CodeSymbolType.STRUCT,
        "function_definition": CodeSymbolType.FUNCTION,
        "namespace_definition": CodeSymbolType.NAMESPACE,
        "enum_specifier": CodeSymbolType.ENUM,
    }

    def __init__(self) -> None:
        super().__init__(Language.CPP, tree_sitter_cpp.language)

    def _symbol_type(self, node, parent):
        if (
            node.type == "function_definition"
            and parent is not None
            and parent.symbol_type in {CodeSymbolType.CLASS, CodeSymbolType.STRUCT}
        ):
            return CodeSymbolType.METHOD
        return super()._symbol_type(node, parent)
