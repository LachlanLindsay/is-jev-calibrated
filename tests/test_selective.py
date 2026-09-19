"""Confidence gating: the numbers an engineer would actually deploy against."""

from __future__ import annotations

import numpy as np
import pytest

from jevcal import selective


def test_full_coverage_point_matches_overall_accuracy():
    conf = [0.9, 0.8, 0.7, 0.6]
    correct = [True, True, False, True]
    points = selective.risk_coverage_curve(conf, correct)
    full = max(points, key=lambda p: p.coverage)
    assert full.coverage == 1.0
    assert full.error == pytest.approx(0.25)


def test_error_falls_as_coverage_falls_for_a_useful_confidence_signal():
    rng = np.random.default_rng(0)
    conf = rng.beta(6, 2, size=3000)
    correct = rng.random(3000) < conf
    points = sorted(selective.risk_coverage_curve(conf, correct), key=lambda p: p.coverage)
    assert points[0].error < points[-1].error


def test_thresholds_never_split_a_tie():
    # With every prediction at the same confidence, there is exactly one
    # deployable gate: take everything.
    points = selective.risk_coverage_curve([0.8] * 50, [True] * 40 + [False] * 10)
    assert len(points) == 1
    assert points[0].coverage == 1.0


def test_aurc_is_lower_for_a_confidence_that_ranks_its_errors_well():
    good_conf = [0.99, 0.98, 0.97, 0.10, 0.09, 0.08]
    outcomes = [True, True, True, False, False, False]
    bad_conf = list(reversed(good_conf))
    good = selective.area_under_risk_coverage(selective.risk_coverage_curve(good_conf, outcomes))
    bad = selective.area_under_risk_coverage(selective.risk_coverage_curve(bad_conf, outcomes))
    assert good < bad


def test_recommended_gate_actually_meets_its_budget():
    rng = np.random.default_rng(1)
    conf = rng.beta(9, 2, size=5000)
    correct = rng.random(5000) < conf
    rec = selective.recommend_threshold(conf, correct, target_error=0.05)
    assert rec.threshold is not None
    assert rec.guaranteed
    assert rec.error_upper <= 0.05
    measured = selective.coverage_at_threshold(conf, correct, rec.threshold)
    assert measured.error == pytest.approx(rec.observed_error)
    assert measured.n == rec.n_accepted


def test_a_looser_budget_buys_more_coverage():
    rng = np.random.default_rng(2)
    conf = rng.beta(9, 2, size=5000)
    correct = rng.random(5000) < conf
    tight = selective.recommend_threshold(conf, correct, 0.03)
    loose = selective.recommend_threshold(conf, correct, 0.10)
    assert loose.coverage >= tight.coverage


def test_an_unreachable_budget_is_refused_rather_than_fudged():
    rng = np.random.default_rng(3)
    # Confidence carries no signal at all, and the model is right half the time.
    conf = rng.uniform(0.4, 0.6, size=2000)
    correct = rng.random(2000) < 0.5
    rec = selective.recommend_threshold(conf, correct, target_error=0.01)
    assert rec.threshold is None
    assert not rec.guaranteed
    assert "no threshold reaches" in rec.note


def test_a_slice_too_thin_to_measure_is_not_recommended():
    # Twenty flawless predictions at the top do not justify a 1% claim.
    conf = list(np.linspace(0.99, 1.0, 20)) + list(np.linspace(0.1, 0.5, 500))
    correct = [True] * 20 + [False] * 500
    rec = selective.recommend_threshold(conf, correct, target_error=0.01, min_accepted=30)
    assert rec.threshold is None


def test_provisional_gates_are_flagged_not_hidden():
    # Enough rows to pass min_accepted, too few for the upper bound to clear 1%.
    conf = [0.99] * 40 + [0.2] * 200
    correct = [True] * 40 + [False] * 200
    rec = selective.recommend_threshold(conf, correct, target_error=0.01, min_accepted=30)
    assert rec.threshold is not None
    assert not rec.guaranteed
    assert rec.observed_error == 0.0
    assert rec.error_upper > 0.01


def test_a_threshold_above_every_prediction_accepts_nothing():
    point = selective.coverage_at_threshold([0.5, 0.6], [True, True], threshold=0.99)
    assert point.n == 0
    assert point.coverage == 0.0


@pytest.mark.parametrize("target", [0.0, 1.0, -0.1, 2.0])
def test_a_nonsensical_budget_is_rejected(target):
    with pytest.raises(ValueError):
        selective.recommend_threshold([0.9], [True], target)
