"""Metric correctness, checked against cases whose answers are known by hand."""

from __future__ import annotations

import math

import numpy as np
import pytest

from jevcal import metrics


def test_wilson_interval_matches_published_values():
    # 10/20 successes, 95%: the textbook Wilson interval.
    lo, hi = metrics.wilson_interval(10, 20)
    assert lo == pytest.approx(0.2993, abs=1e-3)
    assert hi == pytest.approx(0.7007, abs=1e-3)


def test_wilson_interval_stays_inside_unit_interval_at_the_extremes():
    lo, hi = metrics.wilson_interval(0, 5)
    assert lo == 0.0
    assert 0 < hi < 1
    lo, hi = metrics.wilson_interval(5, 5)
    assert hi == 1.0
    assert 0 < lo < 1


def test_wilson_interval_with_no_data_is_uninformative():
    assert metrics.wilson_interval(0, 0) == (0.0, 1.0)


def test_ece_is_zero_when_every_bin_delivers_what_it_claims():
    # 100 predictions at 0.8 of which exactly 80 are right, and so on.
    conf, correct = [], []
    for p, n in ((0.2, 100), (0.5, 100), (0.8, 100), (0.95, 100)):
        hits = round(p * n)
        conf.extend([p] * n)
        correct.extend([True] * hits + [False] * (n - hits))
    assert metrics.expected_calibration_error(conf, correct, n_bins=10) == pytest.approx(0.0, abs=1e-9)


def test_ece_equals_the_offset_for_a_uniform_offset():
    # Everything claimed at 0.9, only 70% right: the gap is 0.20 everywhere.
    conf = [0.9] * 1000
    correct = [True] * 700 + [False] * 300
    assert metrics.expected_calibration_error(conf, correct) == pytest.approx(0.2, abs=1e-9)
    assert metrics.maximum_calibration_error(conf, correct) == pytest.approx(0.2, abs=1e-9)
    assert metrics.overconfidence(conf, correct) == pytest.approx(0.2, abs=1e-9)


def test_underconfidence_is_reported_as_a_negative_gap():
    conf = [0.6] * 100
    correct = [True] * 90 + [False] * 10
    assert metrics.overconfidence(conf, correct) == pytest.approx(-0.3, abs=1e-9)


def test_mce_ignores_bins_below_the_minimum_count():
    # One tiny, wildly wrong bin plus one large, well behaved one.
    conf = [0.99] * 5 + [0.8] * 500
    correct = [False] * 5 + [True] * 400 + [False] * 100
    assert metrics.maximum_calibration_error(conf, correct, min_count=1) > 0.9
    assert metrics.maximum_calibration_error(conf, correct, min_count=50) == pytest.approx(0.0, abs=1e-9)


def test_equal_mass_bins_hold_roughly_equal_counts():
    rng = np.random.default_rng(0)
    conf = rng.beta(8, 2, size=1000)
    correct = rng.random(1000) < conf
    bins = metrics.reliability_bins(conf, correct, n_bins=10, scheme="equal_mass")
    counts = [b.count for b in bins if b.count]
    assert max(counts) - min(counts) <= 2


def test_every_prediction_lands_in_exactly_one_bin():
    conf = [0.0, 0.0001, 0.5, 0.9999, 1.0]
    correct = [False, False, True, True, True]
    bins = metrics.reliability_bins(conf, correct, n_bins=10)
    assert sum(b.count for b in bins) == len(conf)


def test_brier_score_matches_the_definition():
    conf = [0.9, 0.2]
    correct = [True, False]
    expected = ((0.9 - 1) ** 2 + (0.2 - 0) ** 2) / 2
    assert metrics.brier_score(conf, correct) == pytest.approx(expected)


def test_brier_decomposition_reconstructs_the_brier_score():
    rng = np.random.default_rng(1)
    conf = rng.beta(6, 2, size=4000)
    correct = rng.random(4000) < conf
    parts = metrics.brier_decomposition(conf, correct, n_bins=40, scheme="equal_mass")
    # The decomposition is bin-dependent, so it reconstructs Brier only
    # approximately; with fine bins the residual is small.
    assert parts["brier"] == pytest.approx(metrics.brier_score(conf, correct), abs=0.01)
    assert parts["reliability"] >= 0
    assert parts["uncertainty"] == pytest.approx(correct.mean() * (1 - correct.mean()))


def test_auroc_is_one_when_confidence_separates_errors_perfectly():
    conf = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
    correct = [True, True, True, False, False, False]
    assert metrics.auroc(conf, correct) == pytest.approx(1.0)


def test_auroc_is_zero_when_confidence_is_exactly_backwards():
    conf = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    correct = [True, True, True, False, False, False]
    assert metrics.auroc(conf, correct) == pytest.approx(0.0)


def test_auroc_is_one_half_when_confidence_carries_no_signal():
    conf = [0.5] * 10
    correct = [True] * 5 + [False] * 5
    assert metrics.auroc(conf, correct) == pytest.approx(0.5)


def test_auroc_is_undefined_without_both_outcomes():
    assert math.isnan(metrics.auroc([0.9, 0.8], [True, True]))


def test_calibration_fit_recovers_a_known_distortion():
    # Build a model whose reported confidence is a known squash of the truth,
    # then check the fit reads that squash back off the data.
    rng = np.random.default_rng(3)
    true_p = rng.uniform(0.35, 0.99, size=40000)
    correct = rng.random(40000) < true_p
    logit = np.log(true_p / (1 - true_p))
    reported = 1 / (1 + np.exp(-logit / 0.5))  # temperature 0.5 => slope 0.5
    fit = metrics.calibration_curve_fit(reported, correct)
    assert fit["slope"] == pytest.approx(0.5, abs=0.05)
    assert fit["intercept"] == pytest.approx(0.0, abs=0.08)


def test_calibration_fit_is_identity_for_a_calibrated_model():
    rng = np.random.default_rng(4)
    true_p = rng.uniform(0.3, 0.99, size=40000)
    correct = rng.random(40000) < true_p
    fit = metrics.calibration_curve_fit(true_p, correct)
    assert fit["slope"] == pytest.approx(1.0, abs=0.06)
    assert fit["intercept"] == pytest.approx(0.0, abs=0.06)


def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(5)
    conf = rng.beta(7, 2, size=800)
    correct = rng.random(800) < conf * 0.9
    point = metrics.expected_calibration_error(conf, correct)
    lo, hi = metrics.bootstrap_ci(metrics.expected_calibration_error, conf, correct, n_resamples=300)
    assert lo <= point <= hi


def test_negative_log_likelihood_punishes_confident_mistakes():
    assert metrics.negative_log_likelihood([0.9, 0.9]) < metrics.negative_log_likelihood([0.9, 0.01])


def test_multiclass_brier_counts_the_whole_distribution():
    dists = [{"a": 0.7, "b": 0.2, "c": 0.1}]
    expected = (0.7 - 1) ** 2 + 0.2**2 + 0.1**2
    assert metrics.multiclass_brier(dists, ["a"], ["a", "b", "c"]) == pytest.approx(expected)


@pytest.mark.parametrize(
    "conf, correct",
    [
        ([1.4], [True]),
        ([-0.1], [True]),
        ([float("nan")], [True]),
        ([0.5, 0.5], [True]),
        ([], []),
    ],
)
def test_bad_input_is_rejected_rather_than_silently_scored(conf, correct):
    with pytest.raises(ValueError):
        metrics.expected_calibration_error(conf, correct)
