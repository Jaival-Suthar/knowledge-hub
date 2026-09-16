"""Pure Reciprocal Rank Fusion for independently produced ranked results."""

from __future__ import annotations

from collections.abc import Iterable

from knowledge_hub.retrieval.types import RankedChunk


def reciprocal_rank_fusion(
    result_lists: list[list[RankedChunk]],
    *,
    k: int = 60,
    top_k: int | None = None,
) -> list[RankedChunk]:
    """Combine ranked chunks using rank-based Reciprocal Rank Fusion.

    Each list contributes ``1 / (k + rank)`` using one-based ranks; raw
    retrieval scores are intentionally ignored. ``k`` controls contribution
    decay, and ``top_k`` optionally limits the deterministically ordered output.
    """
    if k <= 0:
        raise ValueError("RRF k must be positive")
    if top_k is not None and top_k <= 0:
        return []

    scores: dict[str, float] = {}
    representatives: dict[str, RankedChunk] = {}
    for results in result_lists:
        _add_ranked_results(results, k, scores, representatives)

    ordered_ids = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    if top_k is not None:
        ordered_ids = ordered_ids[:top_k]

    return [
        RankedChunk(
            chunk=representatives[chunk_id].chunk,
            score=scores[chunk_id],
            rank=rank,
            channel="rrf",
        )
        for rank, chunk_id in enumerate(ordered_ids, start=1)
    ]


def _add_ranked_results(
    results: Iterable[RankedChunk],
    k: int,
    scores: dict[str, float],
    representatives: dict[str, RankedChunk],
) -> None:
    for item in results:
        chunk_id = item.chunk.chunk_id
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + item.rank)
        representatives.setdefault(chunk_id, item)
