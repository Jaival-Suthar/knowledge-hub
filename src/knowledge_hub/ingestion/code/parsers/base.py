from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import ClassVar

from tree_sitter import Language as TreeSitterLanguage
from tree_sitter import Node, Parser

from ..contracts import (
    CodeFile,
    CodeParseError,
    CodeParseResult,
    CodeSymbol,
    CodeSymbolType,
    Language,
    LanguageParser,
)

GrammarFactory = Callable[[], object]


class TreeSitterParser(LanguageParser):
    """Shared Tree-sitter parsing and source-location handling."""

    language: Language
    symbol_types: ClassVar[dict[str, CodeSymbolType | str]] = {}

    def __init__(self, language: Language, grammar_factory: GrammarFactory) -> None:
        self.language = language
        self._parser = Parser(TreeSitterLanguage(grammar_factory()))

    def parse(self, code_file: CodeFile) -> CodeParseResult:
        source = code_file.content.encode("utf-8")
        tree = self._parser.parse(source)
        errors = tuple(self._errors(tree.root_node, code_file))
        symbols = tuple(self._symbols(tree.root_node, source, code_file))
        return CodeParseResult(symbols=symbols, errors=errors)

    def _symbols(
        self, root: Node, source: bytes, code_file: CodeFile
    ) -> Iterable[CodeSymbol]:
        yield from self._walk(root, source, code_file, None)

    def _walk(
        self,
        node: Node,
        source: bytes,
        code_file: CodeFile,
        parent: CodeSymbol | None,
    ) -> Iterable[CodeSymbol]:
        symbol_type = self._symbol_type(node, parent)
        current = parent
        if symbol_type is not None:
            name = self._node_name(node)
            if name:
                qualified_name = f"{parent.qualified_name}.{name}" if parent else name
                current = CodeSymbol(
                    name=name,
                    qualified_name=qualified_name,
                    symbol_type=symbol_type,
                    language=code_file.language,
                    path=code_file.relative_path,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    content=source[node.start_byte : node.end_byte].decode(
                        "utf-8", errors="replace"
                    ),
                    parent=parent.qualified_name if parent else None,
                )
                yield current
        for child in node.named_children:
            yield from self._walk(child, source, code_file, current)

    def _errors(self, root: Node, code_file: CodeFile) -> Iterable[CodeParseError]:
        for node in self._error_nodes(root):
            yield CodeParseError(
                message=f"Tree-sitter syntax error: {node.type}",
                path=code_file.relative_path,
                line=node.start_point[0] + 1,
            )

    def _error_nodes(self, node: Node) -> Iterable[Node]:
        if node.type in {"ERROR", "MISSING"}:
            yield node
        for child in node.named_children:
            yield from self._error_nodes(child)

    def _node_name(self, node: Node) -> str | None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            name_node = node.child_by_field_name("declarator")
        if name_node is not None:
            return self._first_identifier(name_node)
        return None

    def _symbol_type(
        self, node: Node, parent: CodeSymbol | None
    ) -> CodeSymbolType | str | None:
        return self.symbol_types.get(node.type)

    def _first_identifier(self, node: Node) -> str | None:
        if node.type in {
            "identifier",
            "type_identifier",
            "field_identifier",
            "namespace_identifier",
            "property_identifier",
        }:
            return node.text.decode("utf-8", errors="replace") if node.text else None
        for child in node.named_children:
            value = self._first_identifier(child)
            if value:
                return value
        return None
