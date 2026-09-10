# Architecture

Commit 1 defines the boundaries only.

Source-specific extraction is isolated behind adapters. Every source converges
on canonical Document/Chunk models. Retrieval consumes those canonical units.

M0 owns model inference. M2 owns ingestion, indexing, retrieval, evidence and
retrieval diagnostics.

M1 remains an independent repository. Commit 2 selectively generalizes M1
implementations without adding an M1 runtime dependency.
