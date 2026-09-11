from pathlib import Path

from knowledge_hub.chunking.markdown import MarkdownChunker
from knowledge_hub.ingestion.adapters.markdown import MarkdownAdapter
from knowledge_hub.models import SourceType
from knowledge_hub.retrieval.pipeline import RetrievalPipeline
from knowledge_hub.retrieval.types import RankedChunk


def test_real_markdown_corpus_enters_canonical_flow() -> None:
    corpus = Path("data/raw/markdown")
    adapter = MarkdownAdapter()
    documents = [adapter.extract(path) for path in sorted(corpus.iterdir())]
    chunks = [chunk for document in documents for chunk in MarkdownChunker().chunk(document)]

    assert len(documents) == 4
    assert sum(len(document.structure["blocks"]) for document in documents) >= 12
    assert chunks
    assert all(chunk.source_type is SourceType.MARKDOWN for chunk in chunks)
    assert all(chunk.provenance.path for chunk in chunks)
    assert {chunk.parent_structure for chunk in chunks} >= {
        "Architecture > Sources",
        "Retrieval > Ranking",
    }


class _Retriever:
    def __init__(self, result: RankedChunk) -> None:
        self.result = result

    def search(self, query: str, limit: int) -> list[RankedChunk]:
        return [self.result]


def test_pdf_and_markdown_chunks_use_same_retrieval_path(tmp_path: Path) -> None:
    markdown_path = tmp_path / "guide.md"
    markdown_path.write_text("# Guide\n\nShared canonical chunk.", encoding="utf-8")
    markdown_chunk = MarkdownChunker().chunk(MarkdownAdapter().extract(markdown_path))[0]
    pdf_like = markdown_chunk.model_copy(update={"source_type": SourceType.PDF})

    markdown_result = RankedChunk(markdown_chunk, 1.0, 1, "dense")
    pdf_result = RankedChunk(pdf_like, 1.0, 1, "dense")
    markdown_trace = RetrievalPipeline(
        _Retriever(markdown_result), _Retriever(markdown_result)
    ).search("shared")
    pdf_trace = RetrievalPipeline(_Retriever(pdf_result), _Retriever(pdf_result)).search("shared")

    assert markdown_trace.final_evidence[0].chunk.source_type is SourceType.MARKDOWN
    assert pdf_trace.final_evidence[0].chunk.source_type is SourceType.PDF
