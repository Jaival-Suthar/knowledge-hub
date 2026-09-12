# Changelog

## 0.2.2 — Code ingestion

### Added

- Typed code-ingestion contracts for sources, files, languages, symbols, and parser results/errors.

- Safe recursive codebase discovery with language classification, UTF-8 validation, SHA-256 hashing, size limits, and generated/vendor/cache exclusions.

- Tree-sitter AST parsing for TypeScript, JavaScript, Python, C, and C++.

- AST-aware semantic chunking for classes, methods, functions, async functions, interfaces, types, enums, structs, and namespaces.

- Secure ZIP codebase ingestion with path traversal protection, symlink and encrypted ZIP rejection, extraction limits, pre-extraction validation, and temporary cleanup.

- Canonical Document and Chunk integration with deterministic identities and code provenance.

- Developer inspection CLI for tracing code from source through discovery, parsing, AST symbols, semantic chunks, and canonical representations.

### Validation

- Validated against the real PerfEngine codebase: 252 supported files, 730 AST symbols, 730 semantic chunks, 252 canonical documents, and 730 canonical chunks.

- 95 tests passed with Ruff, compileall, and diff checks passing.

## 0.2.1 — Markdown ingestion

### Added

- Markdown adapter for `.md` and `.markdown` files using canonical Documents.

- Heading-aware Markdown chunking with preserved block types and provenance.

- Markdown corpus and PDF/Markdown retrieval coexistence smoke tests.

## 0.2.0 — M2 Retrieval Foundation

### Added

- Generalized proven M1 PDF ingestion into M2 canonical models.

- Source-agnostic recursive chunking.

- Qdrant indexing and dense retrieval boundaries.

- BM25, RRF, structural eligibility, and reranking components.

- Inspectable retrieval trace.

- M0 inference/embedding HTTP client.

### Validation

- Preserved the 35-question M1 regression dataset.

## 0.1.0 — Repository Foundation

### Added

- Canonical document/chunk/source/provenance contracts.

- Ingestion/chunking/indexing/retrieval interfaces.

- Retrieval trace structure.

- M0 inference boundary.

- Evaluation metric foundation.

- FastAPI health endpoint.

- Test scaffolding.