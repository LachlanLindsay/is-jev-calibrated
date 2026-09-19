"""Does the audit recover a miscalibration it was given on purpose?

These are the tests that make the rest of the repo mean anything. The simulator
distorts confidence by a known amount in logit space; if the audit cannot read
that distortion back off the data -- and cannot tell a calibrated model from an
overconfident one -- then no verdict it prints about a real model is worth
anything either.
"""

from __future__ import annotations

import pytest

from jevcal.analysis import analyse
from jevcal.datasets import synthetic_task
from jevcal.providers import build_provider
from jevcal.runner import align, read_predictions, run_task


def audit(tmp_path, *, temperature: float, bias: float = 0.0, n: int = 6000, seed: int = 0):
    task = synthetic_task(n=n, kind="choice", n_labels=3, seed=seed)
    provider = build_provider("simulated", seed=seed, temperature=temperature, bias=bias)
    results = tmp_path / f"t{temperature}-b{bias}.jsonl"
    run_task(task, provider, results, concurrency=4, progress=False)
    examples, predictions = align(task, list(read_predictions(results)))
    return analyse(task, examples, predictions, bootstrap_resamples=200, simulated=True)


def test_a_calibrated_model_is_not_accused_of_miscalibration(tmp_path):
    result = audit(tmp_path, temperature=1.0)
    assert result.ece < 0.03
    assert abs(result.overconfidence) < 0.02
    assert result.fit["slope"] == pytest.approx(1.0, abs=0.15)
    assert result.brier_parts["reliability"] < 0.005


def test_an_overconfident_model_is_caught(tmp_path):
    result = audit(tmp_path, temperature=0.5)
    assert result.ece > 0.05
    assert result.overconfidence > 0.05
    # Temperature 0.5 squashes the logit by half, and the fit should say so.
    assert result.fit["slope"] == pytest.approx(0.5, abs=0.12)


def test_an_underconfident_model_is_caught_and_named_correctly(tmp_path):
    result = audit(tmp_path, temperature=2.0)
    assert result.overconfidence < -0.02
    assert result.fit["slope"] == pytest.approx(2.0, abs=0.6)


def test_the_ece_interval_covers_the_truth_for_a_calibrated_model(tmp_path):
    # A perfectly calibrated model has a true ECE of 0; the finite-sample
    # estimate is positive but small, and its interval should sit near zero.
    result = audit(tmp_path, temperature=1.0)
    assert result.ece_ci[0] < result.ece < result.ece_ci[1]
    assert result.ece_ci[0] < 0.03


def test_miscalibration_grows_monotonically_with_the_distortion(tmp_path):
    eces = [audit(tmp_path, temperature=t).ece for t in (1.0, 0.7, 0.45)]
    assert eces[0] < eces[1] < eces[2]


def test_calibration_and_discrimination_are_measured_separately(tmp_path):
    # Squashing the confidence is a monotone transform, so it wrecks calibration
    # while leaving the *ranking* of predictions untouched. If AUROC moved with
    # temperature, the two would be confounded and gating advice would be wrong.
    calibrated = audit(tmp_path, temperature=1.0)
    squashed = audit(tmp_path, temperature=0.5)
    assert squashed.ece > calibrated.ece + 0.04
    assert squashed.auroc == pytest.approx(calibrated.auroc, abs=0.02)


def test_a_recommended_gate_holds_up_on_data_the_threshold_was_not_chosen_on(tmp_path):
    """The gate is the thing people will actually ship, so hold it out.

    Choose the threshold on one half of the run and measure it on the other. If
    the recommendation only works on the sample it was fitted to, it is a
    cherry-pick and the audit is worse than useless.
    """
    from jevcal.selective import coverage_at_threshold, recommend_threshold

    task = synthetic_task(n=12000, kind="choice", n_labels=3, seed=21)
    provider = build_provider("simulated", seed=21, temperature=0.6)
    results = tmp_path / "split.jsonl"
    run_task(task, provider, results, concurrency=8, progress=False)
    examples, predictions = align(task, list(read_predictions(results)))

    conf = [p.confidence for p in predictions]
    correct = [p.predicted == e.label for e, p in zip(examples, predictions)]
    half = len(conf) // 2

    rec = recommend_threshold(conf[:half], correct[:half], target_error=0.05)
    assert rec.threshold is not None and rec.guaranteed

    held_out = coverage_at_threshold(conf[half:], correct[half:], rec.threshold)
    assert held_out.n > 100
    # The held-out error should land near the budget, not wildly above it.
    assert held_out.error < 0.08
