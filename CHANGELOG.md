# Changelog

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
