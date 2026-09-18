from __future__ import annotations

import sys

import pytest

from knowledge_hub.evaluation import benchmark
from knowledge_hub.evaluation.retrieval import (
    EvaluationQuestion,
    EvaluationReport,
    RetrievalMetrics,
)
from knowledge_hub.models import SourceType


def test_benchmark_cli_selects_each_retrieval_mode(monkeypatch, tmp_path) -> None:
    selected: list[tuple[str, int | None]] = []

    monkeypatch.setattr(
        benchmark,
        "load_corpus",
        lambda: type("Corpus", (), {"chunks": (), "documents": ()})(),
    )
    monkeypatch.setattr(benchmark, "load_questions", lambda path: ())

    def fake_report(name: str, rrf_k: int | None = None):
        selected.append((name, rrf_k))
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
        lambda chunks, questions, **kwargs: fake_report("hybrid", kwargs["rrf_k"]),
    )

    for mode in ("dense", "bm25", "hybrid"):
        monkeypatch.setattr(
            sys,
            "argv",
            ["benchmark", "--mode", mode, "--output", str(tmp_path / f"{mode}.json")],
        )
        assert benchmark.main() == 0

    assert selected == [("dense", None), ("bm25", None), ("hybrid", 60)]


def test_load_questions_preserves_and_maps_declared_source_metadata(tmp_path) -> None:
    dataset = tmp_path / "questions.json"
    dataset.write_text(
        '{"questions": ['
        '{"id":"article","source_type":"article",'
        '"source":"https://example.test/article","question":"a",'
        '"answerable":true,"m2_gold_evidence":[]},'
        '{"id":"github","source_type":"github",'
        '"source":"https://github.com/example/repo","question":"b",'
        '"answerable":true,"m2_gold_evidence":[]},'
        '{"id":"code","source_type":"code",'
        '"source":"perfengine/file.ts","question":"c",'
        '"answerable":true,"m2_gold_evidence":[]}'
        "]}",
        encoding="utf-8",
    )

    questions = benchmark.load_questions(dataset)

    assert [(item.source_type, item.source) for item in questions] == [
        (SourceType.ARTICLE, "https://example.test/article"),
        (SourceType.GITHUB, "https://github.com/example/repo"),
        (SourceType.CODE, "perfengine/file.ts"),
    ]


def test_benchmark_cli_forwards_explicit_rrf_k_to_hybrid(monkeypatch, tmp_path) -> None:
    captured: dict[str, int] = {}
    monkeypatch.setattr(
        benchmark,
        "load_corpus",
        lambda: type("Corpus", (), {"chunks": (), "documents": ()})(),
    )
    monkeypatch.setattr(benchmark, "load_questions", lambda path: ())
    monkeypatch.setattr(
        benchmark,
        "run_hybrid_experiment",
        lambda chunks, questions, **kwargs: (
            captured.update(rrf_k=kwargs["rrf_k"]),
            type("Report", (), {"as_dict": lambda self: {}})(),
        )[1],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--mode",
            "hybrid",
            "--rrf-k",
            "10",
            "--output",
            str(tmp_path / "hybrid.json"),
        ],
    )

    assert benchmark.main() == 0
    assert captured == {"rrf_k": 10}


@pytest.mark.parametrize("value", ["0", "-1"])
def test_benchmark_cli_rejects_non_positive_rrf_k(monkeypatch, value: str) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmark", "--mode", "hybrid", "--rrf-k", value],
    )

    with pytest.raises(SystemExit) as error:
        benchmark.main()

    assert error.value.code == 2


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


@pytest.mark.parametrize("rrf_k", [60, 10])
def test_hybrid_report_records_the_value_forwarded_to_hybrid(
    monkeypatch, rrf_k: int
) -> None:
    forwarded: dict[str, int] = {}
    base_report = EvaluationReport(
        retriever="hybrid",
        metrics=RetrievalMetrics(0, 0, 0, 0, 0, 0, 0, 0),
        by_category={},
        queries=(),
    )

    class FakeHybrid:
        def __init__(self, dense, bm25, *, rrf_k):
            forwarded["rrf_k"] = rrf_k

    monkeypatch.setattr(benchmark, "_build_dense_retriever", lambda **kwargs: object())
    monkeypatch.setattr(benchmark, "BM25Retriever", lambda chunks: object())
    monkeypatch.setattr(benchmark, "HybridRetriever", FakeHybrid)
    monkeypatch.setattr(
        benchmark, "evaluate_retriever", lambda *args, **kwargs: base_report
    )

    if rrf_k == 60:
        report = benchmark.run_hybrid_experiment((), (), qdrant_url="q", collection="c")
    else:
        report = benchmark.run_hybrid_experiment(
            (), (), qdrant_url="q", collection="c", rrf_k=rrf_k
        )

    assert report.rrf_k == rrf_k
    assert forwarded == {"rrf_k": rrf_k}
    assert report.as_dict()["rrf_k"] == rrf_k


def test_pipeline_evaluation_adapter_preserves_pipeline_trace_results() -> None:
    question = EvaluationQuestion(
        id="q1",
        category="answerable",
        question="query",
        answerable=True,
        gold_chunk_ids=frozenset(),
        graded_relevance={},
        source_type=SourceType.ARTICLE,
        source="https://example.test/article",
    )

    class FakePipeline:
        dense = type("Dense", (), {"last_timing": None})()

        def __init__(self):
            self.calls = []

        def search(self, query, **kwargs):
            self.calls.append((query, kwargs))
            return type("Trace", (), {"final_evidence": []})()

    pipeline = FakePipeline()
    adapter = benchmark._PipelineEvaluationAdapter(
        pipeline,
        candidate_k=20,
        questions=(question,),
        enable_metadata_filter=True,
    )

    assert adapter.search("query", 5) == []
    assert pipeline.calls == [
        (
            "query",
            {
                "dense_k": 20,
                "sparse_k": 20,
                "rerank_k": 5,
                "metadata_filters": benchmark.MetadataFilters(
                    source_type=SourceType.ARTICLE
                ),
            },
        )
    ]


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (
            benchmark.PipelineExperimentConfig("e4", True, False, False),
            (True, False, False),
        ),
        (
            benchmark.PipelineExperimentConfig("e5", True, False, True),
            (True, False, True),
        ),
        (
            benchmark.PipelineExperimentConfig("e6", True, True, True),
            (True, True, True),
        ),
    ],
)
def test_pipeline_experiment_passes_stage_configuration(
    monkeypatch, config, expected
) -> None:
    captured = {}
    base_report = EvaluationReport(
        retriever=config.name,
        metrics=RetrievalMetrics(0, 0, 0, 0, 0, 0, 0, 0),
        by_category={},
        queries=(),
    )

    class FakePipeline:
        def __init__(self, dense, sparse, **kwargs):
            captured["pipeline"] = kwargs
            captured["has_reranker"] = kwargs["reranker"] is not None
            self.dense = type("Dense", (), {"last_timing": None})()

    monkeypatch.setattr(benchmark, "_build_dense_retriever", lambda **kwargs: object())
    monkeypatch.setattr(benchmark, "BM25Retriever", lambda chunks: object())
    monkeypatch.setattr(benchmark, "RetrievalPipeline", FakePipeline)
    monkeypatch.setattr(benchmark, "CrossEncoderReranker", lambda model_name: object())
    monkeypatch.setattr(
        benchmark, "evaluate_retriever", lambda *args, **kwargs: base_report
    )

    benchmark.run_pipeline_experiment(
        (),
        (),
        qdrant_url="q",
        collection="c",
        config=config,
    )

    assert captured["pipeline"]["enable_structural_filter"] is expected[0]
    assert captured["has_reranker"] is expected[2]
