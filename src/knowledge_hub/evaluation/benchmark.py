"""Command-line runner for comparable dense and BM25 experiments."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from knowledge_hub.config.settings import settings
from knowledge_hub.indexing import QdrantIndex, embed_and_upsert
from knowledge_hub.inference.embedder import SentenceTransformerEmbedder
from knowledge_hub.models import SourceType
from knowledge_hub.retrieval.bm25 import BM25Retriever
from knowledge_hub.retrieval.dense import QdrantDenseRetriever
from knowledge_hub.retrieval.hybrid import HybridRetriever
from knowledge_hub.retrieval.metadata import MetadataFilters
from knowledge_hub.retrieval.pipeline import RetrievalPipeline
from knowledge_hub.retrieval.reranking import CrossEncoderReranker
from knowledge_hub.retrieval.types import RankedChunk
from scripts.ingest_corpus import RAW_DIR, ingest_corpus

from .retrieval import EvaluationQuestion, evaluate_retriever

DEFAULT_DATASET = Path("evaluation/datasets/m2_multisource_gold-evidence-v1.json")
DEFAULT_REPORT_DIR = Path("evaluation/reports")
DEFAULT_TOP_K = 20
DEFAULT_HYBRID_CANDIDATE_K = 20
DEFAULT_RRF_K = 60

EVALUATION_SOURCE_TYPES = {
    "article": SourceType.ARTICLE,
    "github": SourceType.GITHUB,
    "code": SourceType.CODE,
}


@dataclass(frozen=True)
class PipelineExperimentConfig:
    """Stages enabled for a pipeline-backed retrieval experiment."""

    name: str
    enable_structural_filter: bool
    enable_metadata_filter: bool
    enable_reranking: bool


def load_questions(
    path: Path = DEFAULT_DATASET,
) -> tuple[EvaluationQuestion, ...]:
    """Load M2 questions with explicit M2 gold evidence."""
    payload = json.loads(path.read_text(encoding="utf-8"))

    questions: list[EvaluationQuestion] = []

    if isinstance(payload, dict):
        items = payload.get("questions", [])
    elif isinstance(payload, list):
        items = payload
    else:
        raise TypeError(
            f"Unsupported evaluation dataset format: {type(payload).__name__}"
        )

    for item in items:
        gold_evidence = item.get("m2_gold_evidence", [])

        gold_chunk_ids = frozenset(
            evidence["m2_chunk_id"]
            for evidence in gold_evidence
            if evidence.get("m2_chunk_id")
        )

        graded_relevance: dict[str, int] = {}

        for evidence in gold_evidence:
            chunk_id = evidence.get("m2_chunk_id")
            grade = evidence.get("grade")

            if chunk_id and grade is not None:
                graded_relevance[chunk_id] = max(
                    graded_relevance.get(chunk_id, 0),
                    int(grade),
                )

        questions.append(
            EvaluationQuestion(
                id=str(item["id"]),
                category=("unanswerable" if not item["answerable"] else "answerable"),
                question=str(item["question"]),
                answerable=bool(item["answerable"]),
                gold_chunk_ids=gold_chunk_ids,
                graded_relevance=graded_relevance,
                source_type=_source_type(item.get("source_type")),
                source=item.get("source"),
            )
        )

    return tuple(questions)


def _source_type(value: object) -> SourceType | None:
    if value is None:
        return None
    try:
        return EVALUATION_SOURCE_TYPES[str(value).strip().casefold()]
    except KeyError as error:
        raise ValueError(f"unsupported evaluation source_type: {value!r}") from error


def load_corpus():
    """Build the canonical M2 multi-source evaluation corpus."""
    result = ingest_corpus(RAW_DIR)

    print(f"Evaluation corpus: {len(result.chunks)} chunks")
    print(f"Evaluation documents: {len(result.documents)}")

    return result


def run_bm25_experiment(
    chunks,
    questions: tuple[EvaluationQuestion, ...],
):
    return evaluate_retriever(
        BM25Retriever(chunks),
        chunks,
        questions,
        name="bm25",
        top_k=DEFAULT_TOP_K,
    )


def run_dense_experiment(
    chunks,
    questions: tuple[EvaluationQuestion, ...],
    *,
    qdrant_url: str,
    collection: str,
):
    retriever = _build_dense_retriever(
        qdrant_url=qdrant_url,
        collection=collection,
        chunks=chunks,
    )

    return evaluate_retriever(
        retriever,
        chunks,
        questions,
        name="dense",
        top_k=DEFAULT_TOP_K,
    )


def _build_dense_retriever(
    *,
    qdrant_url: str,
    collection: str,
    chunks,
):
    embedder = SentenceTransformerEmbedder(
        model_name=settings.embedding_model_name,
        dimension=settings.embedding_dimension,
        device=settings.embedding_device,
        normalize_embeddings=settings.embedding_normalize,
    )

    index = QdrantIndex(
        url=qdrant_url,
        collection=collection,
        vector_size=settings.embedding_dimension,
    )

    embed_and_upsert(
        index,
        chunks,
        embedder,
    )

    return QdrantDenseRetriever(
        client=index.client,
        collection=collection,
        embedder=embedder,
    )


class _HybridEvaluationAdapter:
    """Adapt HybridRetriever to the evaluation Searcher call shape."""

    def __init__(
        self,
        retriever: HybridRetriever,
        candidate_k: int,
    ) -> None:
        self.retriever = retriever
        self.candidate_k = candidate_k

    @property
    def last_timing(self):
        return self.retriever.last_timing

    def search(
        self,
        query: str,
        top_k: int,
    ) -> list[RankedChunk]:
        return self.retriever.search(
            query,
            top_k=top_k,
            candidate_k=self.candidate_k,
        )


class _PipelineEvaluationAdapter:
    """Adapt RetrievalPipeline traces to the evaluator's Searcher contract."""

    def __init__(
        self,
        pipeline: RetrievalPipeline,
        candidate_k: int,
        questions: tuple[EvaluationQuestion, ...],
        enable_metadata_filter: bool,
    ) -> None:
        self.pipeline = pipeline
        self.candidate_k = candidate_k
        self.enable_metadata_filter = enable_metadata_filter
        self._questions_by_text = defaultdict(deque)
        for question in questions:
            self._questions_by_text[question.question].append(question)

    @property
    def last_timing(self):
        return getattr(self.pipeline.dense, "last_timing", None)

    def search(self, query: str, top_k: int) -> list[RankedChunk]:
        question = self._questions_by_text[query].popleft()
        metadata_filters = (
            _source_metadata_filters(question) if self.enable_metadata_filter else None
        )
        trace = self.pipeline.search(
            query,
            dense_k=self.candidate_k,
            sparse_k=self.candidate_k,
            rerank_k=top_k,
            metadata_filters=metadata_filters,
        )
        return trace.final_evidence


def run_hybrid_experiment(
    chunks,
    questions: tuple[EvaluationQuestion, ...],
    *,
    qdrant_url: str,
    collection: str,
    candidate_k: int = DEFAULT_HYBRID_CANDIDATE_K,
    rrf_k: int = DEFAULT_RRF_K,
):
    dense_retriever = _build_dense_retriever(
        qdrant_url=qdrant_url,
        collection=collection,
        chunks=chunks,
    )

    hybrid_retriever = HybridRetriever(
        dense_retriever,
        BM25Retriever(chunks),
        rrf_k=rrf_k,
    )

    report = evaluate_retriever(
        _HybridEvaluationAdapter(
            hybrid_retriever,
            candidate_k,
        ),
        chunks,
        questions,
        name="hybrid",
        top_k=DEFAULT_TOP_K,
    )

    return replace(
        report,
        rrf_k=rrf_k,
    )


def run_pipeline_experiment(
    chunks,
    questions: tuple[EvaluationQuestion, ...],
    *,
    qdrant_url: str,
    collection: str,
    config: PipelineExperimentConfig,
    candidate_k: int = DEFAULT_HYBRID_CANDIDATE_K,
    rrf_k: int = DEFAULT_RRF_K,
):
    """Evaluate one explicitly configured RetrievalPipeline experiment."""
    dense_retriever = _build_dense_retriever(
        qdrant_url=qdrant_url,
        collection=collection,
        chunks=chunks,
    )
    reranker = (
        CrossEncoderReranker(settings.reranker_model_name)
        if config.enable_reranking
        else None
    )
    pipeline = RetrievalPipeline(
        dense_retriever,
        BM25Retriever(chunks),
        reranker=reranker,
        rrf_k=rrf_k,
        enable_structural_filter=config.enable_structural_filter,
    )
    report = evaluate_retriever(
        _PipelineEvaluationAdapter(
            pipeline,
            candidate_k,
            questions,
            config.enable_metadata_filter,
        ),
        chunks,
        questions,
        name=config.name,
        top_k=DEFAULT_TOP_K,
    )
    return replace(report, rrf_k=rrf_k)


def run_e4_experiment(chunks, questions, **kwargs):
    return run_pipeline_experiment(
        chunks,
        questions,
        config=PipelineExperimentConfig("e4", True, False, False),
        **kwargs,
    )


def run_e5_experiment(chunks, questions, **kwargs):
    return run_pipeline_experiment(
        chunks,
        questions,
        config=PipelineExperimentConfig("e5", True, False, True),
        **kwargs,
    )


def run_e6_experiment(chunks, questions, **kwargs):
    return run_pipeline_experiment(
        chunks,
        questions,
        config=PipelineExperimentConfig("e6", True, True, True),
        **kwargs,
    )


def _source_metadata_filters(question: EvaluationQuestion) -> MetadataFilters:
    if question.source_type is None:
        raise ValueError(
            f"E6 requires source_type metadata for question {question.id!r}"
        )
    return MetadataFilters(source_type=question.source_type)


def write_report(
    report: Any,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            report.as_dict(),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _positive_int(value: str) -> int:
    parsed = int(value)

    if parsed <= 0:
        raise argparse.ArgumentTypeError("rrf-k must be a positive integer")

    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--mode",
        choices=("bm25", "dense", "hybrid", "e4", "e5", "e6"),
        required=True,
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
    )

    parser.add_argument(
        "--output",
        type=Path,
    )

    parser.add_argument(
        "--qdrant-url",
        default="http://localhost:6333",
    )

    parser.add_argument(
        "--collection",
        default="knowledge_hub",
    )

    parser.add_argument(
        "--rrf-k",
        type=_positive_int,
        default=DEFAULT_RRF_K,
    )

    args = parser.parse_args()

    corpus = load_corpus()
    chunks = corpus.chunks

    questions = load_questions(args.dataset)

    print(f"Evaluation questions: {len(questions)}")
    print(f"Evaluation top-k: {DEFAULT_TOP_K}")

    if args.mode == "bm25":
        report = run_bm25_experiment(
            chunks,
            questions,
        )

    elif args.mode == "dense":
        report = run_dense_experiment(
            chunks,
            questions,
            qdrant_url=args.qdrant_url,
            collection=args.collection,
        )

    elif args.mode == "hybrid":
        report = run_hybrid_experiment(
            chunks,
            questions,
            qdrant_url=args.qdrant_url,
            collection=args.collection,
            candidate_k=DEFAULT_HYBRID_CANDIDATE_K,
            rrf_k=args.rrf_k,
        )

    else:
        experiment = {
            "e4": run_e4_experiment,
            "e5": run_e5_experiment,
            "e6": run_e6_experiment,
        }[args.mode]
        report = experiment(
            chunks,
            questions,
            qdrant_url=args.qdrant_url,
            collection=args.collection,
            candidate_k=DEFAULT_HYBRID_CANDIDATE_K,
            rrf_k=args.rrf_k,
        )

    output = args.output or DEFAULT_REPORT_DIR / f"{args.mode}.json"

    write_report(
        report,
        output,
    )

    print(
        json.dumps(
            report.as_dict(),
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
