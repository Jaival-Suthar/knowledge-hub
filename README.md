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
