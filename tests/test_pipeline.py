"""End to end: task in, run on disk, report and charts out."""

from __future__ import annotations

import json

import pytest

from jevcal import analysis, report
from jevcal.cli import main
from jevcal.datasets import from_rows, save_task, synthetic_task
from jevcal.providers import build_provider
from jevcal.runner import align, completed_ids, read_predictions, run_task
from jevcal.types import load_task


def test_a_task_survives_a_round_trip_through_disk(tmp_path):
    task = synthetic_task(n=25, kind="score", n_labels=4, seed=1)
    spec = save_task(task, tmp_path / "t.json", tmp_path / "t.jsonl")
    loaded = load_task(spec)
    assert loaded.labels == task.labels
    assert loaded.kind == "score"
    assert [e.text for e in loaded.examples] == [e.text for e in task.examples]


def test_a_task_can_carry_its_examples_inline(tmp_path):
    task = synthetic_task(n=5, seed=2)
    spec = save_task(task, tmp_path / "inline.json")
    assert "examples" in json.loads(spec.read_text())
    assert len(load_task(spec).examples) == 5


def test_rows_with_boolean_labels_become_a_noul_task():
    task = from_rows(
        [{"text": "a", "label": True}, {"text": "b", "label": False}],
        name="n",
        question="q?",
        kind="noul",
    )
    assert task.labels == ("true", "false")
    assert [e.label for e in task.examples] == ["true", "false"]


def test_a_full_audit_produces_a_report_and_charts(tmp_path):
    task = synthetic_task(n=400, kind="choice", n_labels=3, seed=5)
    provider = build_provider("simulated", seed=5)
    results = tmp_path / "run.jsonl"
    meta = run_task(task, provider, results, concurrency=4, progress=False)
    assert meta.n_ok == 400
    assert meta.n_failed == 0
    assert meta.simulated

    examples, predictions = align(task, list(read_predictions(results)))
    result = analysis.analyse(task, examples, predictions, bootstrap_resamples=50, simulated=True)
    assert result.n == 400
    assert 0 <= result.accuracy <= 1
    assert result.ece_ci[0] <= result.ece <= result.ece_ci[1]

    from jevcal.plots import render_all

    charts = render_all(result, tmp_path / "report", confidences=[p.confidence for p in predictions])
    assert len(charts) == 6
    assert all(c.exists() and c.stat().st_size > 5000 for c in charts)

    path = report.write_report(result, tmp_path / "report", charts)
    text = path.read_text()
    assert "SIMULATED" in text.upper()
    assert "## Verdict" in text
    assert "error budget" in text
    payload = json.loads((tmp_path / "report" / "results.json").read_text())
    assert payload["n"] == 400
    assert "extra" not in payload
    assert len(payload["coverage"]) <= report.MAX_COVERAGE_POINTS


def test_a_resumed_run_neither_repeats_nor_skips_work(tmp_path):
    task = synthetic_task(n=60, seed=6)
    provider = build_provider("simulated", seed=6)
    results = tmp_path / "run.jsonl"

    run_task(task.subset(20), provider, results, concurrency=2, progress=False)
    assert len(completed_ids(results)) == 20

    meta = run_task(task, provider, results, concurrency=2, resume=True, progress=False)
    ids = [p.example_id for p in read_predictions(results)]
    assert len(ids) == 60
    assert len(set(ids)) == 60
    assert meta.n_ok == 60


def test_starting_over_discards_the_previous_run(tmp_path):
    task = synthetic_task(n=30, seed=7)
    provider = build_provider("simulated", seed=7)
    results = tmp_path / "run.jsonl"
    run_task(task, provider, results, progress=False)
    run_task(task.subset(10), provider, results, resume=False, progress=False)
    assert len(list(read_predictions(results))) == 10


def test_a_duplicated_row_is_counted_once(tmp_path):
    task = synthetic_task(n=10, seed=8)
    provider = build_provider("simulated", seed=8)
    results = tmp_path / "run.jsonl"
    run_task(task, provider, results, progress=False)
    with results.open("a") as fh:
        fh.write(results.read_text().splitlines()[0] + "\n")
    examples, predictions = align(task, list(read_predictions(results)))
    assert len(examples) == 10 == len(predictions)


def test_a_truncated_final_line_does_not_break_a_resume(tmp_path):
    task = synthetic_task(n=10, seed=9)
    provider = build_provider("simulated", seed=9)
    results = tmp_path / "run.jsonl"
    run_task(task.subset(5), provider, results, progress=False)
    with results.open("a") as fh:
        fh.write('{"example_id": "syn-000005", "pred')  # killed mid-write
    meta = run_task(task, provider, results, resume=True, progress=False)
    assert meta.n_ok == 10


def test_failed_rows_are_recorded_and_then_excluded(tmp_path):
    task = synthetic_task(n=10, seed=10)

    class HalfBroken:
        name = "half-broken"

        def __init__(self):
            self.inner = build_provider("simulated", seed=10)

        def predict(self, task, example):
            p = self.inner.predict(task, example)
            if example.id.endswith(("0", "1")):
                return type(p)(example_id=example.id, predicted=None, confidence=None, error="boom")
            return p

    results = tmp_path / "run.jsonl"
    meta = run_task(task, HalfBroken(), results, progress=False)
    assert meta.n_failed == 2
    examples, predictions = align(task, list(read_predictions(results)))
    assert len(examples) == 8


def test_analysis_refuses_to_score_nothing():
    task = synthetic_task(n=5, seed=11)
    with pytest.raises(ValueError):
        analysis.analyse(task, [], [])


def test_the_base_rate_is_reported_so_accuracy_can_be_judged(tmp_path):
    rows = [{"text": f"t{i}", "label": "yes" if i < 90 else "no"} for i in range(100)]
    task = from_rows(rows, name="skewed", question="q?", labels=("yes", "no"))
    provider = build_provider("simulated", seed=12)
    results = tmp_path / "run.jsonl"
    run_task(task, provider, results, progress=False)
    examples, predictions = align(task, list(read_predictions(results)))
    result = analysis.analyse(task, examples, predictions, bootstrap_resamples=20)
    assert result.base_rate == pytest.approx(0.9)


def test_the_cli_runs_an_audit_end_to_end(tmp_path, capsys):
    task = synthetic_task(n=120, seed=13)
    spec = save_task(task, tmp_path / "t.json", tmp_path / "t.jsonl")
    code = main([
        "audit", str(spec), "--provider", "simulated",
        "--out", str(tmp_path / "run.jsonl"),
        "--report-dir", str(tmp_path / "rep"),
        "--bootstrap", "50", "--no-charts",
    ])
    assert code == 0
    assert (tmp_path / "rep" / "report.md").exists()
    assert "calibrated" in capsys.readouterr().out


def test_the_cli_reports_a_missing_task_file_without_a_traceback(tmp_path, capsys):
    assert main(["analyze", str(tmp_path / "nope.json"), str(tmp_path / "nope.jsonl")]) == 2
    assert "error:" in capsys.readouterr().err


def test_limit_and_sample_seed_select_different_subsets(tmp_path):
    task = synthetic_task(n=100, seed=14)
    first = task.subset(10)
    sampled = task.subset(10, seed=1)
    assert len(first.examples) == len(sampled.examples) == 10
    assert [e.id for e in first.examples] != [e.id for e in sampled.examples]


def test_a_task_cannot_carry_labels_its_examples_contradict():
    with pytest.raises(ValueError, match="outside the task's label set"):
        from_rows(
            [{"text": "a", "label": "purple"}],
            name="bad",
            question="q?",
            labels=("red", "blue"),
        )


def test_make_task_converts_a_csv_and_renames_its_labels(tmp_path):
    csv_path = tmp_path / "raw.csv"
    csv_path.write_text("text,label\ngreat kettle,1\nawful lamp,0\n")
    spec = tmp_path / "task.json"
    code = main([
        "make-task", str(csv_path), "--out", str(spec),
        "--question", "Is this positive?",
        "--label-map", '{"0": "negative", "1": "positive"}',
    ])
    assert code == 0
    task = load_task(spec)
    assert set(task.labels) == {"negative", "positive"}
    assert [e.label for e in task.examples] == ["positive", "negative"]
