# Knowledge Hub

M2 — Source-Agnostic Knowledge & Retrieval Substrate.

**Repository foundation.** Establish architecture,
canonical contracts, and test scaffolding. Feature work belongs in later PRs.

```text
PDF / Markdown / Code / GitHub / Articles / Notes
                    ↓
             Source Adapters
                    ↓
          Unified Document Model
                    ↓
           Source-Aware Chunking
                ↙       ↘
             Dense       BM25
            Qdrant    Sparse Store
                ↘       ↙
          Reciprocal Rank Fusion
                    ↓
        Structural Eligibility
                    ↓
                Reranker
                    ↓
            Evidence Selection
                    ↓
           Context Construction
                    ↓
              M0 Inference
```

M1 (`Knowledge Vault`) is a separate repository and is **not** a runtime
dependency. Commit 2 will selectively generalize proven M1 components.

Planned phases: canonical model → PDF/M1 regression → Markdown → AST code →
Git repositories → BM25 → hybrid/RRF → structural filtering → reranking →
metadata filtering → incremental indexing → evaluation/retrieval forensics.


## M2 source ingestion

Markdown files (`.md` and `.markdown`) enter through `MarkdownAdapter`, which
produces the canonical `Document`. `MarkdownChunker` preserves heading paths,
code blocks, lists, tables, and blockquotes while producing canonical `Chunk`
objects. Those chunks use the existing source-agnostic indexing and retrieval
pipeline alongside PDF chunks.

```text
Markdown → MarkdownAdapter → Document → MarkdownChunker → Chunk → retrieval
```

## M1-derived retrieval foundation

This work selectively absorbs proven M1 architecture into M2-owned abstractions:

- text-layer `PdfAdapter` producing the canonical `Document`
- M1-style recursive chunking
- Qdrant index boundary
- dense retrieval boundary
- lightweight BM25 retrieval
- Reciprocal Rank Fusion
- structural eligibility filtering
- lazy cross-encoder reranking
- inspectable `RetrievalTrace`
- M0 `/v1/generate` and `/v1/embed` client boundary
- preserved 35-question M1 regression dataset

M1 remains a separate repository and is not imported at runtime. No M1 source package,
virtual environment, model weights, caches, or generated artifacts are included here.

## Code ingestion

Code files enter through safe codebase discovery and Tree-sitter parsing, producing AST symbols that are converted into source-aware semantic chunks and canonical `Document` / `Chunk` objects. Code ingestion supports TypeScript, JavaScript, Python, C, and C++ and can ingest both directories and ZIP codebases.

The ZIP ingestion path validates archive members before extraction and rejects unsafe paths, symlinks, encrypted entries, and archives exceeding configured file/count/size limits.

```
Code / ZIP → CodeDiscovery → Tree-sitter → AST Symbols → Semantic Chunks → Canonical Documents / Chunks
```

Code chunks preserve symbol identity, language, source location, parent structure, content hashes, and provenance. Oversized symbols are handled deterministically.

A developer inspection CLI provides an inspectable path from source acquisition through canonicalization for full codebases or individual files.

The implementation was validated against the real PerfEngine codebase:

- 1,928 ZIP members
- 252 supported source files
- 247 TypeScript files
- 5 JavaScript files
- 730 AST symbols
- 730 semantic chunks
- 252 canonical documents
- 730 canonical chunks
- 70 recoverable parser errors

Code ingestion is intentionally limited to acquisition, parsing, semantic chunking, and canonicalization. Git/GitHub ingestion, embeddings, indexing changes, BM25, hybrid retrieval, reranking, metadata retrieval, UI, OCR, and agents remain separate work.