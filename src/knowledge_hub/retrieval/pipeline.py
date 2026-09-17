from __future__ import annotations

from knowledge_hub.retrieval.fusion import ReciprocalRankFusion
from knowledge_hub.retrieval.metadata import MetadataFilter, MetadataFilters
from knowledge_hub.retrieval.reranking import CrossEncoderReranker
from knowledge_hub.retrieval.structural import StructuralEligibility
from knowledge_hub.retrieval.types import RetrievalTrace


class RetrievalPipeline:
    """Inspectable M2 retrieval pipeline: dense + BM25 -> RRF -> filter -> rerank."""

    def __init__(
        self, dense, sparse, reranker: CrossEncoderReranker | None = None
    ) -> None:
        self.dense = dense
        self.sparse = sparse
        self.reranker = reranker
        self.fusion = ReciprocalRankFusion()
        self.structural_eligibility = StructuralEligibility()
        self.metadata_filter = MetadataFilter()

    def search(
        self,
        query: str,
        dense_k: int = 20,
        sparse_k: int = 20,
        rerank_k: int = 5,
        metadata_filters: MetadataFilters | None = None,
    ) -> RetrievalTrace:
        trace = RetrievalTrace(query=query)
        trace.dense_results = self.dense.search(query, dense_k)
        trace.bm25_results = self.sparse.search(query, sparse_k)
        trace.fusion_results = self.fusion.fuse(
            [trace.dense_results, trace.bm25_results],
            top_k=max(dense_k, sparse_k),
        )
        trace.filtered_results = self.structural_eligibility.filter(
            trace.fusion_results
        )
        trace.metadata_filtered_results = self.metadata_filter.filter(
            trace.filtered_results, metadata_filters
        )
        if self.reranker is not None:
            trace.reranked_results = self.reranker.rerank(
                query, trace.metadata_filtered_results, rerank_k
            )
            trace.final_evidence = trace.reranked_results
        else:
            trace.final_evidence = trace.metadata_filtered_results[:rerank_k]
        return trace
