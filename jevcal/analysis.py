"""Turn a results file into the numbers the report and the charts need."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from . import metrics, selective
from .types import Example, Prediction, Task

#: Error budgets an engineer actually has to choose between when deciding what
#: to automate outright and what to escalate.
DEFAULT_TARGETS = (0.01, 0.05, 0.10)


@dataclass
class AuditResult:
    task: str
    kind: str
    labels: list[str]
    model: str
    simulated: bool
    n: int
    accuracy: float
    base_rate: float
    mean_confidence: float
    overconfidence: float
    ece: float
    ece_ci: tuple[float, float]
    ece_adaptive: float
    mce: float
    brier: float
    brier_parts: dict[str, float]
    auroc: float
    fit: dict[str, float]
    bins: list[metrics.Bin]
    bins_adaptive: list[metrics.Bin]
    coverage: list[selective.CoveragePoint]
    aurc: float
    gates: list[selective.GateRecommendation]
    nll: float | None = None
    multiclass_brier: float | None = None
    latency_ms: dict[str, float] = field(default_factory=dict)
    cost_usd: float | None = None
    ordinal: dict[str, float] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def analyse(
    task: Task,
    examples: Sequence[Example],
    predictions: Sequence[Prediction],
    n_bins: int = 15,
    targets: Sequence[float] = DEFAULT_TARGETS,
    bootstrap_resamples: int = 1000,
    seed: int = 0,
    simulated: bool = False,
) -> AuditResult:
    """Compute every statistic the audit publishes.

    ``examples`` and ``predictions`` must already be aligned row for row -- see
    :func:`jevcal.runner.align`.
    """
    if len(examples) != len(predictions):
        raise ValueError("examples and predictions must align")
    if not examples:
        raise ValueError("nothing to analyse: no successful predictions")

    confidence = np.array([p.confidence for p in predictions], dtype=float)
    correct = np.array([p.predicted == e.label for e, p in zip(examples, predictions)], dtype=bool)

    ece = metrics.expected_calibration_error(confidence, correct, n_bins, "equal_width")
    ece_ci = metrics.bootstrap_ci(
        lambda c, k: metrics.expected_calibration_error(c, k, n_bins, "equal_width"),
        confidence,
        correct,
        n_resamples=bootstrap_resamples,
        seed=seed,
    )

    coverage = selective.risk_coverage_curve(confidence, correct)
    gates = [selective.recommend_threshold(confidence, correct, t) for t in targets]

    result = AuditResult(
        task=task.name,
        kind=task.kind,
        labels=list(task.labels),
        model=next((p.model for p in predictions if p.model), ""),
        simulated=simulated,
        n=len(examples),
        accuracy=float(correct.mean()),
        base_rate=_majority_base_rate(examples, task),
        mean_confidence=float(confidence.mean()),
        overconfidence=metrics.overconfidence(confidence, correct),
        ece=ece,
        ece_ci=ece_ci,
        ece_adaptive=metrics.expected_calibration_error(confidence, correct, n_bins, "equal_mass"),
        mce=metrics.maximum_calibration_error(confidence, correct, n_bins, "equal_width", min_count=30),
        brier=metrics.brier_score(confidence, correct),
        brier_parts=metrics.brier_decomposition(confidence, correct, n_bins, "equal_mass"),
        auroc=metrics.auroc(confidence, correct),
        fit=metrics.calibration_curve_fit(confidence, correct),
        bins=metrics.reliability_bins(confidence, correct, n_bins, "equal_width"),
        bins_adaptive=metrics.reliability_bins(confidence, correct, n_bins, "equal_mass"),
        coverage=coverage,
        aurc=selective.area_under_risk_coverage(coverage),
        gates=gates,
        latency_ms=_latency_summary(predictions),
        cost_usd=_total_cost(predictions),
    )

    distributions = [p.distribution for p in predictions if p.distribution]
    if len(distributions) == len(predictions) and distributions:
        true_probs = [p.distribution.get(e.label, 0.0) for e, p in zip(examples, predictions)]
        result.nll = metrics.negative_log_likelihood(true_probs)
        result.multiclass_brier = metrics.multiclass_brier(
            distributions, [e.label for e in examples], task.labels
        )

    if task.kind == "score":
        result.ordinal = _ordinal_summary(task, examples, predictions)

    return result


def _majority_base_rate(examples: Sequence[Example], task: Task) -> float:
    """Accuracy of always answering the most common label.

    The floor any real result has to clear. Quoting 91% accuracy on a task whose
    majority class is 90% is not a result.
    """
    counts = {label: 0 for label in task.labels}
    for e in examples:
        counts[e.label] += 1
    return max(counts.values()) / len(examples)


def _latency_summary(predictions: Sequence[Prediction]) -> dict[str, float]:
    values = np.array([p.latency_ms for p in predictions if p.latency_ms is not None], dtype=float)
    if values.size == 0:
        return {}
    return {
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def _total_cost(predictions: Sequence[Prediction]) -> float | None:
    costs = [p.cost_usd for p in predictions if p.cost_usd is not None]
    return float(sum(costs)) if costs else None


def _ordinal_summary(task: Task, examples: Sequence[Example], predictions: Sequence[Prediction]) -> dict[str, float]:
    """Score tasks are ordered, so "wrong" has a magnitude.

    Exact-match accuracy alone under-sells a rubric model that is usually one
    rung off, and over-sells one that is occasionally at the far end.
    """
    true_idx = np.array([task.label_index(e.label) for e in examples], dtype=float)
    pred_idx = np.array([task.label_index(p.predicted) for p in predictions], dtype=float)
    diff = pred_idx - true_idx
    return {
        "mae_rungs": float(np.abs(diff).mean()),
        "within_one_rung": float((np.abs(diff) <= 1).mean()),
        "mean_signed_error": float(diff.mean()),
    }


def calibrated_bins_table(bins: Sequence[metrics.Bin]) -> list[dict[str, Any]]:
    """The reliability diagram as rows, so every plotted value is also readable."""
    return [
        {
            "bin": f"{b.lower:.2f}-{b.upper:.2f}",
            "n": b.count,
            "mean_confidence": round(b.mean_confidence, 4) if b.count else None,
            "observed_accuracy": round(b.accuracy, 4) if b.count else None,
            "ci_low": round(b.ci_low, 4) if b.count else None,
            "ci_high": round(b.ci_high, 4) if b.count else None,
            "gap": round(b.gap, 4) if b.count else None,
        }
        for b in bins
    ]
