"""Selective prediction: where is it actually safe to automate?

Calibration is the scientific question; this module answers the engineering one.
If you only act on decisions above a confidence threshold and escalate the rest
to a human or a frontier model, what error rate do you get, and what fraction of
traffic do you still handle automatically?
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .metrics import Z95, as_scored_arrays, wilson_interval


@dataclass(frozen=True)
class CoveragePoint:
    threshold: float
    coverage: float
    accuracy: float
    error: float
    n: int
    #: Upper Wilson bound on the error rate -- the number to plan against.
    error_upper: float


def risk_coverage_curve(
    confidence: Sequence[float],
    correct: Sequence[bool],
    z: float = Z95,
) -> list[CoveragePoint]:
    """Error rate as a function of how much traffic you keep.

    Sweeps the threshold over every distinct confidence value, keeping all
    predictions at or above it. The first point is full coverage (accept
    everything); the last is the most selective non-empty slice.
    """
    conf, corr = as_scored_arrays(confidence, correct)
    order = np.argsort(-conf, kind="mergesort")
    conf_sorted = conf[order]
    corr_sorted = corr[order].astype(int)
    cum_hits = np.cumsum(corr_sorted)
    n_total = conf.size

    points: list[CoveragePoint] = []
    # Only cut at the boundary between distinct confidence values: a threshold
    # that splits a tie is not a threshold you could actually deploy.
    boundaries = np.flatnonzero(np.diff(conf_sorted) != 0)
    cut_indices = np.append(boundaries, n_total - 1)
    for i in cut_indices:
        n = int(i + 1)
        hits = int(cum_hits[i])
        acc = hits / n
        _, err_hi = wilson_interval(n - hits, n, z)
        points.append(
            CoveragePoint(
                threshold=float(conf_sorted[i]),
                coverage=n / n_total,
                accuracy=acc,
                error=1.0 - acc,
                n=n,
                error_upper=err_hi,
            )
        )
    points.reverse()  # low coverage first, so the curve reads left to right
    return points


def area_under_risk_coverage(points: Sequence[CoveragePoint]) -> float:
    """AURC: mean error over all coverage levels. Lower is better.

    A single number for "how well does this confidence signal let me trade
    coverage for accuracy", comparable across models on the same task.
    """
    if not points:
        raise ValueError("no coverage points")
    ordered = sorted(points, key=lambda p: p.coverage)
    cov = np.array([p.coverage for p in ordered])
    err = np.array([p.error for p in ordered])
    if cov.size == 1:
        return float(err[0])
    # ``trapezoid`` is the numpy >= 2 spelling of ``trapz``.
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return float(integrate(err, cov) / (cov[-1] - cov[0]))


@dataclass(frozen=True)
class GateRecommendation:
    """The answer to "where do I set the threshold?"."""

    target_error: float
    threshold: float | None
    coverage: float
    observed_error: float
    error_upper: float
    n_accepted: int
    guaranteed: bool
    note: str


def recommend_threshold(
    confidence: Sequence[float],
    correct: Sequence[bool],
    target_error: float,
    z: float = Z95,
    min_accepted: int = 30,
) -> GateRecommendation:
    """Lowest threshold whose error rate stays under ``target_error``.

    Two things make this honest rather than a cherry-pick:

    * We require the *upper* confidence bound on the observed error to clear the
      target, not the point estimate. Picking the threshold that happened to look
      best on this sample is exactly how a gate that "tested at 1% error" ships
      at 4%.
    * We refuse to recommend a slice thinner than ``min_accepted`` predictions,
      because an error rate measured on a dozen rows is not a measurement.

    Lowering the threshold only ever adds traffic, so we sweep from the most
    selective slice downward and keep the last point that still passes.
    """
    if not 0.0 < target_error < 1.0:
        raise ValueError("target_error must lie in (0, 1)")

    points = risk_coverage_curve(confidence, correct, z)
    eligible = [p for p in points if p.n >= min_accepted]
    if not eligible:
        return GateRecommendation(
            target_error=target_error,
            threshold=None,
            coverage=0.0,
            observed_error=float("nan"),
            error_upper=1.0,
            n_accepted=0,
            guaranteed=False,
            note=f"no slice holds at least {min_accepted} predictions; run more examples",
        )

    passing = [p for p in eligible if p.error_upper <= target_error]
    if passing:
        best = max(passing, key=lambda p: p.coverage)
        return GateRecommendation(
            target_error=target_error,
            threshold=best.threshold,
            coverage=best.coverage,
            observed_error=best.error,
            error_upper=best.error_upper,
            n_accepted=best.n,
            guaranteed=True,
            note="upper 95% bound on error clears the target at this threshold",
        )

    # Nothing clears the bar with statistical room to spare. Report the best
    # point estimate instead, clearly flagged as not guaranteed.
    by_point = [p for p in eligible if p.error <= target_error]
    if by_point:
        best = max(by_point, key=lambda p: p.coverage)
        return GateRecommendation(
            target_error=target_error,
            threshold=best.threshold,
            coverage=best.coverage,
            observed_error=best.error,
            error_upper=best.error_upper,
            n_accepted=best.n,
            guaranteed=False,
            note=(
                "observed error clears the target but the 95% upper bound "
                f"({best.error_upper:.1%}) does not; treat as provisional"
            ),
        )

    strictest = min(eligible, key=lambda p: p.coverage)
    return GateRecommendation(
        target_error=target_error,
        threshold=None,
        coverage=0.0,
        observed_error=float("nan"),
        error_upper=1.0,
        n_accepted=0,
        guaranteed=False,
        note=(
            f"no threshold reaches {target_error:.1%} error; the most selective "
            f"usable slice still errs at {strictest.error:.1%}"
        ),
    )


def coverage_at_threshold(
    confidence: Sequence[float],
    correct: Sequence[bool],
    threshold: float,
    z: float = Z95,
) -> CoveragePoint:
    """Measure one specific gate, e.g. the round number you plan to ship."""
    conf, corr = as_scored_arrays(confidence, correct)
    mask = conf >= threshold
    n = int(mask.sum())
    if n == 0:
        return CoveragePoint(threshold, 0.0, float("nan"), float("nan"), 0, 1.0)
    hits = int(corr[mask].sum())
    _, err_hi = wilson_interval(n - hits, n, z)
    return CoveragePoint(
        threshold=threshold,
        coverage=n / conf.size,
        accuracy=hits / n,
        error=1.0 - hits / n,
        n=n,
        error_upper=err_hi,
    )
