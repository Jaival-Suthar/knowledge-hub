def recall_at_k(relevant_ids: set[str], ranked_ids: list[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    return float(bool(relevant_ids.intersection(ranked_ids[:k])))


def reciprocal_rank(relevant_ids: set[str], ranked_ids: list[str]) -> float:
    for rank, item_id in enumerate(ranked_ids, start=1):
        if item_id in relevant_ids:
            return 1.0 / rank
    return 0.0
