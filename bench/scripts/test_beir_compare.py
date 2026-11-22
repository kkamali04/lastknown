"""Regression checks for denominators and like-for-like scoring, without corpus downloads."""
import beir_compare


def test_missing_failed_and_unjudged_queries_stay_in_denominator():
    qrels = {"a": {"doc-a": 1}, "b": {"doc-b": 1}, "c": {"doc-c": 1}}
    run = {"fts5": {"a": {"status": "ok", "ids": ["doc-a"]},
                    "b": {"status": "error", "ids": ["doc-b"]},
                    "outside-qrels": {"status": "ok", "ids": ["doc-c"]}}}
    values, failures = beir_compare.scores(run, qrels, sorted(qrels), "fts5")
    assert values == [1.0, 0.0, 0.0]
    assert failures == 2
    assert beir_compare.mean(values) == 1 / 3


def test_identical_id_and_repeated_hits_are_removed_before_scoring():
    qrels = {"a": {"doc-a": 1}}
    run = {"fts5": {"a": {"status": "ok", "ids": ["a", "doc-a", "doc-a"]}}}
    assert beir_compare.scores(run, qrels, ["a"], "fts5") == ([1.0], 0)
