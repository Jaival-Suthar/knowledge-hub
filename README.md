# Knowledge Hub

**M2 — Source-Agnostic Knowledge & Retrieval Substrate**

Knowledge Hub is a multi-source knowledge ingestion and retrieval system designed to turn heterogeneous technical knowledge into **canonical, provenance-preserving evidence** that can be searched through hybrid retrieval and served through a FastAPI interface.

It is built around a simple idea:

> **Ingest different kinds of knowledge once, normalize them into a common representation, and evaluate retrieval as an engineering system rather than treating RAG as a black box.**

The system currently supports:

* PDF documents
* Markdown documents
* Source code
* GitHub repositories
* Hashnode articles

It provides:

* canonical document and chunk contracts
* source-aware ingestion
* AST-aware code chunking
* dense retrieval
* BM25 lexical retrieval
* Reciprocal Rank Fusion
* structural eligibility filtering
* metadata-aware filtering
* optional cross-encoder reranking
* provenance preservation
* reproducible retrieval evaluation
* FastAPI-based ingestion and retrieval
* Qdrant-backed persistence
* startup corpus hydration

---

## Architecture

```text
                         Knowledge Sources
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
         PDF               Markdown              Code
          │                    │                    │
     PdfAdapter        MarkdownAdapter        CodeDiscovery
          │                    │                    │
          │              MarkdownChunker       Tree-sitter
          │                    │                    │
          │                    │               AST Symbols
          │                    │                    │
          │                    │              Semantic Chunks
          │                    │                    │
          └──────────────┬─────┴────────────────────┘
                         │
                  Canonical Model
              Document / Chunk / Provenance
                         │
          ┌──────────────┼───────────────┐
          │              │               │
       GitHub         Articles       Local Sources
          │              │
     Safe Git       Hashnode
     Acquisition    Acquisition
          │              │
          └──────────────┼───────────────┘
                         │
                   Canonical Chunks
                         │
              ┌──────────┴──────────┐
              │                     │
         Dense Retrieval          BM25
              │                     │
           Qdrant               Sparse Index
              │                     │
              └──────────┬──────────┘
                         │
                Reciprocal Rank Fusion
                         │
              Structural Eligibility
                         │
                Metadata Filtering
                         │
                 Cross-Encoder
                  Reranking
                         │
                  Ranked Evidence
                         │
                  Provenance / Trace
                         │
                     FastAPI
              ┌──────────┼──────────┐
              │          │          │
           /search   /retrieve   /ingest
              │
       /documents /sources
```

The API is intentionally thin. It reuses the same canonical ingestion and retrieval infrastructure rather than introducing a separate API-specific processing path.

---

# Why Knowledge Hub Exists

A conventional RAG implementation often looks like:

```text
documents
   ↓
embeddings
   ↓
vector database
   ↓
top-k
   ↓
LLM
```

That architecture hides several important engineering questions:

* What happens when the source is code rather than prose?
* How should GitHub repositories be represented?
* How do exact identifiers interact with semantic search?
* Does BM25 actually contribute useful evidence?
* Does hybrid retrieval improve retrieval quality?
* Does reranking actually improve the measured metrics?
* Should structural content such as navigation and metadata be eligible evidence?
* How does retrieval behave across heterogeneous source types?
* Can the system expose provenance rather than returning anonymous text?
* Can the same canonical representation support ingestion, retrieval, evaluation, and API serving?

Knowledge Hub treats these as **measurable system-design questions**.

The project therefore evolved through controlled engineering branches rather than implementing every retrieval technique at once.

---

# Source Ingestion

## Supported sources

| Source   | Acquisition      | Parsing / Chunking                        |
| -------- | ---------------- | ----------------------------------------- |
| PDF      | Local file       | PyMuPDF + PDF chunking                    |
| Markdown | Local file       | Markdown parser + section-aware chunking  |
| Code     | Directory / ZIP  | Tree-sitter + AST-aware semantic chunking |
| GitHub   | Git repository   | Git acquisition + existing code pipeline  |
| Hashnode | HTTP acquisition | Markdown parsing + article-aware chunking |

All sources eventually converge on the same canonical representation:

```text
Document
   ↓
Chunk
   ↓
Provenance
```

This allows retrieval and evaluation components to remain source-agnostic.

---

# Canonical Knowledge Model

The core abstraction is a canonical `Document` / `Chunk` model with provenance.

Chunks preserve information such as:

* `document_id`
* `chunk_id`
* content
* source type
* source URI
* content role
* project
* language
* repository
* path
* content hash
* location
* provenance
* parent structure

This allows retrieval results to remain traceable back to their original source.

For example:

```text
Query
  ↓
Ranked Chunk
  ↓
source_type = github
repository = Jaival-Suthar/RippleTalk
path = rippletalk/src/context/AuthContext.ts
symbol = AuthContextType
lines = 4–12
```

The retrieval layer therefore returns **evidence with identity**, not just text.

---

# Markdown Ingestion

Markdown became a first-class M2 source through a dedicated adapter and chunker.

```text
Markdown
   ↓
MarkdownAdapter
   ↓
Canonical Document
   ↓
MarkdownChunker
   ↓
Canonical Chunks
```

The Markdown pipeline preserves:

* heading hierarchy
* heading paths
* paragraphs
* fenced code blocks
* lists
* tables
* blockquotes
* chunk text fidelity

Chunking is both section-aware and token-aware.

Parser edge cases were explicitly tested, including:

* headings containing trailing `#` characters such as `C#`
* fenced-code boundaries
* chunk text fidelity
* Markdown/PDF coexistence

The Markdown PR was validated against the real corpus and the full test suite.

---

# AST-Aware Code Ingestion

Code is not treated as ordinary text.

The code ingestion pipeline uses Tree-sitter to identify semantic program structures before chunking.

```text
Code / ZIP
    ↓
CodeDiscovery
    ↓
Tree-sitter
    ↓
AST Symbols
    ↓
Semantic Chunks
    ↓
Canonical Documents / Chunks
```

Supported languages:

* TypeScript
* JavaScript
* Python
* C
* C++

Supported semantic structures include:

* classes
* methods
* functions
* interfaces
* types
* enums
* structs
* namespaces

The ingestion layer also provides:

* recursive discovery
* language classification
* UTF-8 validation
* SHA-256 content hashing
* file-size limits
* generated/vendor/cache exclusions
* deterministic oversized-symbol handling

### Secure ZIP ingestion

ZIP archives are validated before extraction.

The implementation rejects:

* path traversal
* absolute paths
* Windows absolute paths
* symlinks
* encrypted entries
* archives exceeding configured file/count/size limits

Temporary extraction directories are cleaned deterministically.

### Real-project validation

The pipeline was validated against the real **PerfEngine** codebase:

| Metric                    | Result |
| ------------------------- | -----: |
| ZIP members               |  1,928 |
| Supported files           |    252 |
| TypeScript                |    247 |
| JavaScript                |      5 |
| Parsed files              |    252 |
| AST symbols               |    730 |
| Semantic chunks           |    730 |
| Canonical documents       |    252 |
| Canonical chunks          |    730 |
| Recoverable parser errors |     70 |

A developer inspection path was also verified against the complete archive and individual real source files.

---

# GitHub Repository Ingestion

GitHub repositories reuse the existing code ingestion pipeline rather than creating a second code parser.

```text
GitHub URL
    ↓
Safe Git Acquisition
    ↓
CodeDiscovery
    ↓
Tree-sitter
    ↓
AST-aware Semantic Chunking
    ↓
Canonical Documents / Chunks
    ↓
GitHub Provenance
```

Repository acquisition provides:

* shallow cloning
* explicit ref handling
* commit SHA validation
* bounded clone timeouts
* isolated temporary workspaces
* deterministic cleanup
* failure-path cleanup

GitHub provenance preserves:

* repository URL
* owner
* repository name
* ref
* commit SHA
* file path
* source symbol information where available

Document identity remains content/path based rather than being tied to a particular commit.

### Real repository validation

Validated against:

```text
Repository: Jaival-Suthar/RippleTalk
Ref: main
Commit: c6245f7c3e0514bc1bd96b4a3fa7479bef7c7ad0
```

Results:

| Metric              | Result |
| ------------------- | -----: |
| Files discovered    |     38 |
| Files parsed        |     38 |
| Semantic chunks     |     25 |
| Canonical documents |     38 |
| Canonical chunks    |     25 |

GitHub acquisition, failure handling, cleanup, deterministic identities, provenance propagation, and multi-file output were tested.

---

# Article Ingestion

Hashnode articles are treated as a first-class source.

```text
Hashnode Article
      ↓
Acquisition
      ↓
Markdown Parsing
      ↓
Article-aware Chunking
      ↓
Canonical Document / Chunk
      ↓
Provenance
```

The article pipeline includes:

* bounded HTTP acquisition
* article-specific source contracts
* structured Markdown parsing
* heading hierarchy preservation
* deterministic token-aware chunking
* stable chunk identity
* exact content preservation
* canonicalization
* provenance propagation

An end-to-end confidence gate was run against three real Hashnode articles.

Validation included:

* determinism
* isolation
* network-disabled behavior
* real article ingestion

The article ingestion branch completed with the retrieval layer intentionally kept separate.

---

# Retrieval Architecture

Knowledge Hub supports two independent retrieval channels.

## Dense retrieval

Dense retrieval uses:

* Sentence Transformers
* `BAAI/bge-small-en-v1.5`
* Qdrant
* canonical M2 chunks

The embedding model produces 384-dimensional normalized vectors.

## BM25 retrieval

BM25 operates directly over canonical chunk content and metadata.

It:

* builds a sparse lexical index
* returns the shared `RankedChunk` contract
* preserves chunk identity and provenance
* supports index rebuild/update
* remains independent from Qdrant

The two channels intentionally remain independently measurable.

---

# Reciprocal Rank Fusion

Hybrid retrieval combines Dense and BM25 through Reciprocal Rank Fusion.

```text
Dense Retrieval ──┐
                   ├── RRF ──> Hybrid Results
BM25 Retrieval ───┘
```

RRF uses:

```text
score = 1 / (k + rank)
```

Important characteristics:

* rank-based rather than score-based
* independent of the original retriever score scales
* merges by canonical `chunk_id`
* preserves metadata and provenance
* deterministic tie-breaking
* configurable `k`
* configurable `top_k`

The hybrid retriever executes both retrieval channels with the same candidate depth before fusion.

---

# Retrieval Experiments

A major goal of M2 was to avoid assuming that every additional retrieval technique improves the system.

The retrieval stack was therefore evaluated progressively.

---

## Experiment 1 — Dense vs BM25

The first controlled benchmark used:

* 35 questions
* 30 answerable questions
* 5 deliberately unanswerable questions
* 70 explicit gold/acceptable evidence records

The retrieval-quality aggregates use the 30 answerable queries.

### Results

| Metric    |     Dense |     BM25 |
| --------- | --------: | -------: |
| Recall@1  |    40.00% |   10.00% |
| Recall@5  |    66.67% |   46.67% |
| Recall@10 |    80.00% |   53.33% |
| Recall@20 |    80.00% |   70.00% |
| MRR       |    0.5322 |   0.2438 |
| nDCG@5    |    0.4768 |   0.2196 |
| Latency   | ~23.06 ms | ~0.73 ms |

The result is intentionally presented as a benchmark observation rather than a universal statement about lexical versus semantic retrieval.

On this evaluation corpus, Dense produced stronger ranking metrics while BM25 was substantially faster and still recovered relevant evidence at deeper retrieval depths.

The BM25 branch also added per-query failure analysis for exact terminology, identifiers, function/class names, filenames, repository names, and other lexical cases.

---

# Experiment 2 — Dense + BM25 + RRF

The next experiment evaluated:

```text
Dense
BM25
Hybrid Dense + BM25 + RRF
```

on the same controlled evaluation set.

### Results

| Metric    |      Dense |     BM25 |     Hybrid |
| --------- | ---------: | -------: | ---------: |
| Recall@1  |     40.00% |   10.00% |     36.67% |
| Recall@5  |     66.67% |   46.67% |     60.00% |
| Recall@10 |     80.00% |   53.33% |     76.67% |
| Recall@20 |     80.00% |   70.00% | **86.67%** |
| MRR       | **0.5322** |   0.2438 |     0.4615 |
| nDCG@5    | **0.4768** |   0.2196 |     0.3930 |
| Latency   |  ~23.06 ms | ~0.73 ms |  ~32.64 ms |

### Finding

The hybrid result is **not simply "better."**

It increased deep retrieval coverage:

```text
Dense     80.00%
BM25      70.00%
Hybrid    86.67%
```

while early-ranking metrics were lower than Dense:

```text
MRR

Dense     0.5322
Hybrid    0.4615
```

and:

```text
nDCG@5

Dense     0.4768
Hybrid    0.3930
```

Hybrid latency also increased because both retrieval channels are executed.

The experiment therefore demonstrated a more useful result:

> **Dense and BM25 contribute partially different candidate sets, but naïve equal RRF fusion does not automatically improve early ranking quality.**

---

# RRF Sensitivity

RRF was evaluated across multiple values of `k`:

```text
10
20
40
60
100
200
```

Across the evaluation set:

* Recall@1 remained unchanged
* Recall@10 remained unchanged
* Recall@20 remained unchanged
* Recall@5 showed modest variation
* MRR and nDCG@5 showed corresponding modest changes

This indicates that the observed deep-recall behavior was relatively robust across a broad range of RRF constants.

The experiments were intended to understand sensitivity rather than to declare a dataset-specific "optimal" constant.

---

# Retriever Complementarity

A separate analysis examined whether Dense and BM25 were actually retrieving the same candidates.

Across the 30 answerable queries:

| Outcome            | Queries |
| ------------------ | ------: |
| Dense only         |       5 |
| BM25 only          |       2 |
| Both               |      19 |
| Neither            |       4 |
| BM25 rescued Dense |       2 |
| Dense rescued BM25 |       5 |

Additional measurements:

```text
Mean candidate intersection: 4.77 / 20
Mean Jaccard overlap:        0.1393
```

This provides evidence that the two retrieval channels produce meaningfully different candidate sets.

That complementarity explains why BM25 can contribute useful candidates even when its standalone ranking metrics are weaker.

---

# Source-Aware Retrieval

After the controlled PDF retrieval experiments, the retrieval architecture was evaluated against a heterogeneous Knowledge Hub corpus.

The source-aware layer adds:

* structural eligibility
* metadata-aware filtering
* configurable cross-encoder reranking
* multi-source gold evidence
* retrieval ablation reports

---

## Structural Eligibility

Chunks receive content roles such as:

```text
navigation
metadata
documentation
implementation
evidence
reference
test
configuration
```

Structural eligibility allows content such as navigation and metadata to be excluded from final evidence while preserving unknown roles rather than aggressively discarding them.

This operates after RRF.

---

# Metadata Filtering

Retrieval can optionally filter on:

* `source_type`
* `project`
* `language`
* `repository`
* `path`
* `content_role`

Semantics:

```text
AND across dimensions
OR within a dimension
```

This provides explicit source/provenance control without making metadata filtering mandatory for the generic retrieval pipeline.

---

# Cross-Encoder Reranking

Knowledge Hub integrates a configurable BGE cross-encoder reranker.

The pipeline ordering is:

```text
Dense
   +
BM25
   ↓
RRF
   ↓
Structural Eligibility
   ↓
Metadata Filtering
   ↓
Cross-Encoder Reranking
   ↓
Final Evidence
```

Reranking preserves canonical chunks and provenance.

It is optional rather than mandatory because the evaluation results showed that adding a more sophisticated ranking stage does not automatically mean better measured retrieval quality.

---

# Multi-Source Evaluation

The source-aware evaluation introduced a multi-source gold-evidence benchmark covering:

* articles
* GitHub repositories
* code/project sources

The controlled ablation sequence was:

```text
E1 — Dense retrieval

E2 — BM25 retrieval

E3 — Hybrid Dense + BM25 + RRF

E4 — Hybrid + structural eligibility

E5 — Hybrid + structural eligibility + cross-encoder reranking

E6 — Hybrid + structural eligibility + metadata filtering + reranking
```

The benchmark measures:

* Recall@1
* Recall@5
* Recall@10
* Recall@20
* MRR
* nDCG@5
* retrieval latency

---

## Multi-Source Results

The multi-source benchmark produced several useful observations:

* Hybrid Dense + BM25 + RRF reached **100% Recall@5** on the 21-query benchmark.
* Structural eligibility preserved the measured ranking metrics with minimal additional latency.
* The evaluated BGE reranker configuration **reduced the measured ranking metrics while adding substantial latency**.
* Source-aware metadata filtering produced only marginal changes to measured ranking metrics while providing explicit provenance-based retrieval control.

These results are deliberately reported as **benchmark-specific observations**, not universal claims about reranking or metadata filtering.

---

# Evaluation Philosophy

Knowledge Hub treats retrieval evaluation as a first-class engineering concern.

The benchmark infrastructure evaluates explicit gold evidence rather than answer-string heuristics.

Metrics include:

```text
Recall@1
Recall@5
Recall@10
Recall@20
MRR
nDCG@5
Mean retrieval latency
```

Per-query records preserve:

* query ID
* question
* relevant chunk IDs
* graded relevance
* retrieved chunk IDs
* retrieval scores
* latency
* answerability

Unanswerable questions remain visible in reports while being excluded from strict retrieval-quality aggregates.

This makes it possible to inspect **why** a retrieval system succeeded or failed rather than relying only on a single aggregate score.

---

# FastAPI Knowledge API

The final M2 layer exposes ingestion, retrieval, and corpus inspection through FastAPI.

## Endpoints

| Method | Endpoint     | Purpose                               |
| ------ | ------------ | ------------------------------------- |
| GET    | `/health`    | Service health                        |
| POST   | `/search`    | Hybrid knowledge retrieval            |
| POST   | `/retrieve`  | Direct canonical chunk lookup         |
| POST   | `/ingest`    | Ingest new knowledge                  |
| GET    | `/documents` | Inspect canonical documents           |
| GET    | `/sources`   | Inspect source/provenance information |

Swagger/OpenAPI is available through FastAPI's standard documentation interface.

---

## `/search`

The search endpoint uses the existing retrieval pipeline.

It preserves:

* chunk identity
* source information
* provenance
* rank
* score
* retrieval channel

Example flow:

```text
POST /search
      ↓
RetrievalPipeline
      ↓
Dense + BM25
      ↓
RRF
      ↓
Source-aware stages
      ↓
Ranked evidence
```

---

## `/retrieve`

Directly retrieves canonical chunks by identity.

This is useful for:

* evidence inspection
* deterministic lookup
* provenance verification
* API consumers that already know the desired chunk ID

---

## `/ingest`

The ingestion endpoint supports:

* PDF uploads
* Markdown uploads/paths
* Hashnode article URLs
* GitHub repository URLs
* ZIP codebase uploads

The API delegates to the existing ingestion pipelines rather than duplicating their logic.

---

## Corpus Inspection

The API exposes:

```text
GET /documents
GET /sources
```

These provide visibility into the canonical corpus and its provenance.

---

# Persistence and Startup Hydration

Qdrant stores the canonical retrieval representation.

When a fresh API process starts, the application can hydrate its in-memory corpus view from the configured Qdrant collection.

```text
Qdrant
  ↓
Persisted canonical chunks
  ↓
Startup hydration
  ↓
Canonical corpus view
  ↓
/documents
/sources
/retrieve
```

This avoids introducing a second persistent corpus store.

The API was verified in a fresh Uvicorn process and with real ingestion/retrieval flows.

---

# End-to-End Verification

The API was tested against a newly ingested copy of **Inference Engineering**.

The query:

```text
What are the four factors that affect cold start times?
```

returned the relevant `Inference Engineering` chunk as the **top-ranked RRF result**.

The retrieved evidence identified:

1. GPU procurement
2. Image loading
3. Model loading
4. Engine startup

The response also preserved:

* document ID
* chunk ID
* PDF source type
* source URI
* page provenance
* retrieval rank
* retrieval channel

This demonstrates the complete path:

```text
PDF
 ↓
/ingest
 ↓
Canonical Document / Chunks
 ↓
Qdrant + Retrieval Indexes
 ↓
Dense + BM25
 ↓
RRF
 ↓
/search
 ↓
Correct evidence + provenance
```

---

# M2 Engineering Progression

The implementation was developed as a sequence of focused branches.

```text
feat/markdown-adapter
        ↓
feat/code-ingestion
        ↓
feat/git-repository-ingestion
        ↓
feat/article-ingestion
        ↓
feat/bm25-retrieval
        ↓
feat/hybrid-retrieval
        ↓
feat/source-aware-ranking
        ↓
feat/knowledge-api
        ↓
M2 COMPLETE
```

Each stage answered a specific engineering question.

| Stage                | Engineering question                                                       |
| -------------------- | -------------------------------------------------------------------------- |
| Markdown ingestion   | Can non-PDF documents enter the canonical model cleanly?                   |
| Code ingestion       | Can source code be represented semantically rather than as arbitrary text? |
| GitHub ingestion     | Can complete repositories be safely acquired and normalized?               |
| Article ingestion    | Can remote technical articles become deterministic canonical knowledge?    |
| BM25                 | Where does lexical retrieval provide useful coverage?                      |
| Hybrid/RRF           | Do Dense and BM25 contribute complementary evidence?                       |
| Source-aware ranking | Do structural and provenance signals improve retrieval control?            |
| FastAPI              | Can the complete system be operated through a stable API boundary?         |

This progression deliberately separates **implementation from experimental conclusions**.

---

# What the Experiments Actually Showed

The project does not claim that every additional retrieval component improves quality.

Instead, the experiments produced several concrete observations.

### Dense vs BM25

On the controlled PDF benchmark:

* Dense produced stronger ranking metrics.
* BM25 was substantially faster.
* BM25 still recovered meaningful relevant evidence at deeper retrieval depths.

### Hybrid RRF

On the same benchmark:

* Hybrid increased Recall@20 from **80.00% to 86.67%**.
* Dense remained stronger on MRR and nDCG@5.
* Hybrid introduced additional latency.
* Dense and BM25 produced substantially different candidate sets.

### Structural eligibility

On the multi-source benchmark:

* It preserved measured retrieval metrics.
* It added minimal latency.
* It provided an explicit mechanism for excluding structurally inappropriate evidence.

### Cross-encoder reranking

On the evaluated configuration:

* It reduced the measured ranking metrics.
* It added substantial latency.

Therefore it is **not enabled as an unquestioned mandatory improvement**.

### Metadata filtering

On the evaluated multi-source benchmark:

* It produced marginal metric changes.
* It provided explicit provenance-based retrieval control.

The resulting system is therefore based on **measured behavior rather than the assumption that more sophisticated retrieval always means better retrieval**.

---

# Source Synchronization

Knowledge Hub currently treats ingestion as an **explicit operation** rather than providing continuous synchronization with external sources.

If an external source changes, it must currently be re-ingested explicitly.

Automatic:

* change detection
* incremental indexing
* GitHub commit synchronization
* article update synchronization
* deletion propagation

are intentionally **outside the completed M2 scope**.

This is a deliberate boundary rather than an implicit capability.

Future productionization could use:

```text
content_hash
document identity
source version
      ↓
NEW / MODIFIED / UNCHANGED / DELETED
      ↓
incremental document/chunk updates
      ↓
Qdrant + BM25 synchronization
```

---

# Retrieval Inspector

A dedicated retrieval-inspector UI is also outside the completed M2 scope.

The underlying retrieval architecture already preserves the information needed for deeper inspection, including:

* chunk identity
* provenance
* retrieval channel
* rank
* scores
* source metadata

A future inspector could expose the full pipeline:

```text
QUERY
  ↓
Dense Results
  ↓
BM25 Results
  ↓
RRF
  ↓
Structural Eligibility
  ↓
Metadata Filtering
  ↓
Reranking
  ↓
Final Evidence
```

The important distinction is that the **retrieval infrastructure exists**, while a dedicated visual inspection interface remains future work.

---

# M1 Relationship

Knowledge Vault is the separate M1 repository from which selected architectural ideas were generalized.

M1 remains independent.

Knowledge Hub does **not** depend on the M1 repository at runtime.

The M2 repository does not require:

* the M1 source package
* M1 virtual environments
* M1 model weights
* M1 caches
* M1 generated artifacts

The M2 evaluation foundation preserves the original 35-question regression semantics while adapting the evidence representation to the M2 canonical chunk model.

---

# Technology Stack

| Component          | Technology                             |
| ------------------ | -------------------------------------- |
| Language           | Python 3.11+                           |
| API                | FastAPI                                |
| Validation         | Pydantic                               |
| PDF parsing        | PyMuPDF                                |
| Dense embeddings   | Sentence Transformers                  |
| Embedding model    | `BAAI/bge-small-en-v1.5`               |
| Vector database    | Qdrant                                 |
| Sparse retrieval   | BM25                                   |
| Fusion             | Reciprocal Rank Fusion                 |
| Code parsing       | Tree-sitter                            |
| Supported code     | TypeScript, JavaScript, Python, C, C++ |
| Reranking          | BGE Cross-Encoder                      |
| Package/runtime    | uv                                     |
| Testing            | pytest                                 |
| Linting/formatting | Ruff                                   |

---

# Running the API

Install the project with its development dependencies using `uv`.

Then start the API:

```bash
uv run uvicorn knowledge_hub.api.app:app --host 127.0.0.1 --port 8000
```

API health:

```text
http://127.0.0.1:8000/health
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

OpenAPI:

```text
http://127.0.0.1:8000/openapi.json
```

---

# Testing

The final API branch was validated with the complete test suite.

Current final suite status:

```text
324 passed
6 skipped
```

Additional validation performed throughout M2 included:

* pytest
* Ruff checks
* Ruff formatting
* compileall
* `git diff --check`
* real corpus ingestion
* real GitHub repository ingestion
* real Hashnode article ingestion
* real PerfEngine codebase ingestion
* real Uvicorn API verification
* Swagger/OpenAPI verification
* end-to-end PDF ingestion and retrieval

The final API branch also required a small pytest path configuration so repository-level developer scripts and the `src` package resolve correctly during test collection.

---

# Project Structure

The repository is organized around source ingestion, canonical models, retrieval, evaluation, and API serving.

```text
knowledge-hub/
│
├── src/
│   └── knowledge_hub/
│       ├── api/
│       ├── ingestion/
│       ├── retrieval/
│       ├── evaluation/
│       └── ...
│
├── tests/
│
├── scripts/
│
├── data/
│   ├── raw/
│   └── ...
│
├── evaluation/
│   ├── datasets/
│   └── reports/
│
├── pyproject.toml
├── uv.lock
└── README.md
```

---

# Design Principles

### 1. Canonical contracts

Every source eventually becomes the same `Document` / `Chunk` abstraction.

### 2. Provenance first

Retrieval results should remain traceable to their original source.

### 3. Source-specific ingestion, source-agnostic retrieval

PDFs, code, GitHub repositories, Markdown, and articles need different ingestion strategies, but retrieval should operate over canonical evidence.

### 4. Controlled experiments

Retrieval components are evaluated independently before being combined.

### 5. Evidence over intuition

A component is not considered an improvement merely because it is more sophisticated.

### 6. Explicit scope

Capabilities such as automatic synchronization and a visual retrieval inspector are documented as future work rather than being implied as completed functionality.

### 7. Persistence without unnecessary duplication

Qdrant provides persistent canonical retrieval state, while startup hydration reconstructs the API's in-memory corpus view.

---

# Final Status

## M2 — COMPLETE

Knowledge Hub now provides a complete source-agnostic ingestion, retrieval, evaluation, and API substrate covering:

```text
PDF
Markdown
Code
GitHub
Articles
   ↓
Canonical Knowledge
   ↓
Dense + BM25
   ↓
RRF Hybrid Retrieval
   ↓
Structural Eligibility
   ↓
Metadata Filtering
   ↓
Optional Reranking
   ↓
Evidence + Provenance
   ↓
FastAPI
```

The project has moved beyond a basic RAG prototype.

It provides a system where:

* heterogeneous technical sources can be normalized
* source structure is preserved
* code is chunked semantically
* retrieval channels can be compared independently
* hybrid retrieval can be experimentally evaluated
* ranking stages can be ablated
* provenance remains attached to evidence
* retrieval can be accessed through an API
* persistence survives API process restarts
* retrieval behavior can be measured rather than assumed

The completed M2 result is therefore not simply a vector-search application.

It is a **source-agnostic knowledge ingestion and retrieval substrate built around canonical evidence, measurable retrieval behavior, and explicit engineering boundaries.**

---

# Future Work

The following are intentionally deferred from the completed M2 milestone:

* incremental source synchronization
* content-hash change detection
* GitHub commit-aware synchronization
* article update synchronization
* deletion propagation
* incremental Qdrant/BM25 updates
* dedicated retrieval inspector UI
* richer retrieval tracing
* additional source types
* larger-scale evaluation
* production deployment hardening

These can be developed as a separate productionization phase without changing the completed M2 architecture.

---

## M2 in One Sentence

> **Knowledge Hub is a multi-source, provenance-preserving knowledge retrieval system that turns PDFs, Markdown, code, GitHub repositories, and technical articles into canonical evidence, evaluates Dense/BM25/hybrid retrieval empirically, and exposes the resulting system through a FastAPI interface.**
