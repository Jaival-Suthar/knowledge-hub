"""Command-line runner for comparable dense and BM25 experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from knowledge_hub.chunking.pdf import PdfChunker
from knowledge_hub.config.settings import settings
from knowledge_hub.indexing import QdrantIndex, embed_and_upsert
from knowledge_hub.inference.embedder import SentenceTransformerEmbedder
from knowledge_hub.ingestion.adapters.pdf import PdfAdapter
from knowledge_hub.retrieval.bm25 import BM25Retriever
from knowledge_hub.retrieval.dense import QdrantDenseRetriever

from .retrieval import EvaluationQuestion, evaluate_retriever

DEFAULT_DATASET = Path("evaluation/datasets/m2_gold-evidence-v1.json")
DEFAULT_PDF = Path("data/raw/The 10X Rule.pdf")
DEFAULT_REPORT_DIR = Path("evaluation/reports")


def load_questions(
    path: Path = DEFAULT_DATASET,
) -> tuple[EvaluationQuestion, ...]:
    """Load M2 questions with explicit M2 gold evidence."""
    payload = json.loads(path.read_text(encoding="utf-8"))

    questions: list[EvaluationQuestion] = []

    for item in payload:
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
                category="unanswerable" if not item["answerable"] else "answerable",
                question=str(item["question"]),
                answerable=bool(item["answerable"]),
                gold_chunk_ids=gold_chunk_ids,
                graded_relevance=graded_relevance,
            )
        )

    return tuple(questions)


def load_pdf_chunks(path: Path = DEFAULT_PDF):
    """Build the shared PDF corpus used by both retrieval experiments."""
    document = PdfAdapter().extract(path)
    return tuple(PdfChunker().chunk(document))


def run_bm25_experiment(
    chunks,
    questions: tuple[EvaluationQuestion, ...],
):
    return evaluate_retriever(
        BM25Retriever(chunks),
        chunks,
        questions,
        name="bm25",
    )


def run_dense_experiment(
    chunks,
    questions: tuple[EvaluationQuestion, ...],
    *,
    qdrant_url: str,
    collection: str,
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
    embed_and_upsert(index, chunks, embedder)

    retriever = QdrantDenseRetriever(
        client=index.client,
        collection=collection,
        embedder=embedder,
    )

    return evaluate_retriever(
        retriever,
        chunks,
        questions,
        name="dense",
    )


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--mode",
        choices=("bm25", "dense"),
        required=True,
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
    )

    parser.add_argument(
        "--pdf",
        type=Path,
        default=DEFAULT_PDF,
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

    args = parser.parse_args()

    chunks = load_pdf_chunks(args.pdf)
    questions = load_questions(args.dataset)

    if args.mode == "bm25":
        report = run_bm25_experiment(
            chunks,
            questions,
        )
    else:
        report = run_dense_experiment(
            chunks,
            questions,
            qdrant_url=args.qdrant_url,
            collection=args.collection,
        )

    output = args.output or DEFAULT_REPORT_DIR / f"{args.mode}.json"

    write_report(report, output)

    print(
        json.dumps(
            report.as_dict(),
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
