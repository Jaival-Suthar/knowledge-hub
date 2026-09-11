from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from .contracts import CodeFile, CodeParseResult, CodeSymbol

DEFAULT_MAX_CHUNK_CHARS = 4_000


@dataclass(frozen=True)
class SemanticCodeChunk:
    """A semantic chunk backed by one parser-produced symbol."""

    symbol: CodeSymbol
    content: str
    chunk_id: str
    start_line: int
    end_line: int
    part_index: int = 0

    @property
    def name(self) -> str:
        return self.symbol.name

    @property
    def qualified_name(self) -> str:
        return self.symbol.qualified_name

    @property
    def symbol_type(self):
        return self.symbol.symbol_type

    @property
    def parent(self) -> str | None:
        return self.symbol.parent

    @property
    def language(self):
        return self.symbol.language

    @property
    def path(self) -> str:
        return self.symbol.path


class CodeChunker:
    """Create deterministic semantic chunks from parser-produced symbols.

    Symbols are kept whole whenever they fit within the configured source
    character limit.

    When a symbol is oversized:
    - symbols with semantic children are subdivided into those children;
    - oversized leaf symbols use deterministic exact source slicing.

    Size is measured in source characters, including whitespace. This keeps
    the C4 contract tokenizer-independent until canonical tokenization is
    introduced.
    """

    def __init__(self, max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS) -> None:
        if max_chunk_chars <= 0:
            raise ValueError("max_chunk_chars must be positive")

        self.max_chunk_chars = max_chunk_chars

    def chunk(
        self,
        code_file: CodeFile,
        parse_result: CodeParseResult,
    ) -> tuple[SemanticCodeChunk, ...]:
        """Create deterministic semantic chunks from parser output."""
        del code_file

        children: dict[str | None, list[CodeSymbol]] = {}

        for symbol in parse_result.symbols:
            children.setdefault(symbol.parent, []).append(symbol)

        output: list[SemanticCodeChunk] = []
        emitted: set[str] = set()

        for symbol in parse_result.symbols:
            if symbol.parent is None:
                self._emit(symbol, children, output, emitted)

        return tuple(output)

    def _emit(
        self,
        symbol: CodeSymbol,
        children: dict[str | None, list[CodeSymbol]],
        output: list[SemanticCodeChunk],
        emitted: set[str],
    ) -> None:
        if symbol.qualified_name in emitted:
            return

        nested = children.get(symbol.qualified_name, [])

        if len(symbol.content) <= self.max_chunk_chars:
            output.append(
                self._chunk(
                    symbol=symbol,
                    content=symbol.content,
                    part_index=0,
                    start_line=symbol.start_line,
                    end_line=symbol.end_line,
                )
            )
            emitted.add(symbol.qualified_name)

            for child in nested:
                self._emit(child, children, output, emitted)

            return

        if nested:
            # The semantic parent is too large, so subdivide it into its
            # parser-produced semantic children. This intentionally avoids
            # emitting a duplicate oversized parent chunk.
            emitted.add(symbol.qualified_name)

            for child in nested:
                self._emit(child, children, output, emitted)

            return

        # Oversized leaf: there is no semantic child boundary available,
        # therefore exact deterministic slicing is the final fallback.
        for part_index, (
            content,
            start_line,
            end_line,
        ) in enumerate(
            self._split_exact(
                content=symbol.content,
                start_line=symbol.start_line,
            )
        ):
            output.append(
                self._chunk(
                    symbol=symbol,
                    content=content,
                    part_index=part_index,
                    start_line=start_line,
                    end_line=end_line,
                )
            )

        emitted.add(symbol.qualified_name)

    def _chunk(
        self,
        symbol: CodeSymbol,
        content: str,
        part_index: int,
        start_line: int,
        end_line: int,
    ) -> SemanticCodeChunk:
        chunk_id = sha256(
            f"{symbol.path}:{symbol.qualified_name}:{part_index}:{content}".encode()
        ).hexdigest()

        return SemanticCodeChunk(
            symbol=symbol,
            content=content,
            chunk_id=chunk_id,
            start_line=start_line,
            end_line=end_line,
            part_index=part_index,
        )

    def _split_exact(
        self,
        content: str,
        start_line: int,
    ) -> list[tuple[str, int, int]]:
        """Split source deterministically without dropping any characters."""
        limit = self.max_chunk_chars
        parts: list[tuple[str, int, int]] = []

        offset = 0
        current_line = start_line

        while offset < len(content):
            remaining_length = len(content) - offset

            if remaining_length <= limit:
                cut = remaining_length
            else:
                window_end = offset + limit

                # Prefer the last newline within the size limit.
                newline = content.rfind("\n", offset + 1, window_end + 1)

                if newline > offset:
                    cut = newline - offset + 1
                else:
                    # Otherwise prefer a space.
                    space = content.rfind(" ", offset + 1, window_end + 1)

                    if space > offset:
                        cut = space - offset + 1
                    else:
                        cut = limit

            piece = content[offset : offset + cut]

            if not piece:
                break

            end_line = current_line + piece.count("\n")

            parts.append(
                (
                    piece,
                    current_line,
                    end_line,
                )
            )

            offset += cut
            current_line = end_line

        return parts
