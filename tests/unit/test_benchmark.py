from __future__ import annotations

import sys

from knowledge_hub.evaluation import benchmark


def test_benchmark_cli_selects_each_retrieval_mode(monkeypatch, tmp_path) -> None:
    selected: list[str] = []

    monkeypatch.setattr(benchmark, "load_pdf_chunks", lambda path: ())
    monkeypatch.setattr(benchmark, "load_questions", lambda path: ())

    def fake_report(name: str):
        selected.append(name)
        return type("Report", (), {"as_dict": lambda self: {"retriever": name}})()

    monkeypatch.setattr(
        benchmark,
        "run_bm25_experiment",
        lambda chunks, questions: fake_report("bm25"),
    )
    monkeypatch.setattr(
        benchmark,
        "run_dense_experiment",
        lambda chunks, questions, **kwargs: fake_report("dense"),
    )
    monkeypatch.setattr(
        benchmark,
        "run_hybrid_experiment",
        lambda chunks, questions, **kwargs: fake_report("hybrid"),
    )

    for mode in ("dense", "bm25", "hybrid"):
        monkeypatch.setattr(
            sys,
            "argv",
            ["benchmark", "--mode", mode, "--output", str(tmp_path / f"{mode}.json")],
        )
        assert benchmark.main() == 0

    assert selected == ["dense", "bm25", "hybrid"]


def test_hybrid_evaluation_adapter_forwards_top_and_candidate_k() -> None:
    class FakeHybrid:
        def __init__(self):
            self.calls = []

        def search(self, query, *, top_k, candidate_k):
            self.calls.append((query, top_k, candidate_k))
            return []

    retriever = FakeHybrid()
    adapter = benchmark._HybridEvaluationAdapter(retriever, candidate_k=20)

    assert adapter.search("query", 5) == []
    assert retriever.calls == [("query", 5, 20)]
