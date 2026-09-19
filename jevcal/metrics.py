"""Calibration metrics.

A model is *calibrated* if, among the decisions it reports at confidence p, a
fraction p are right. Everything here is a different way of asking how far from
that a model is, and how sure we are of the answer given a finite sample.

All functions take plain numpy arrays so they can be reused on any model's
output, not just Jev's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import numpy as np

BinScheme = Literal["equal_width", "equal_mass"]

#: 1.959964 -- the two-sided normal quantile for a 95% interval.
Z95 = 1.959963984540054


def as_scored_arrays(confidence: Sequence[float], correct: Sequence[bool]) -> tuple[np.ndarray, np.ndarray]:
    """Validate and coerce a (confidence, correctness) pair.

    Shared by every metric so that bad input fails once, here, rather than
    producing a plausible-looking number somewhere downstream.
    """
    conf = np.asarray(confidence, dtype=float)
    corr = np.asarray(correct, dtype=bool)
    if conf.shape != corr.shape:
        raise ValueError(f"confidence and correct must align: {conf.shape} vs {corr.shape}")
    if conf.size == 0:
        raise ValueError("no predictions to score")
    if np.any(~np.isfinite(conf)):
        raise ValueError("confidence contains non-finite values")
    if np.any((conf < 0) | (conf > 1)):
        raise ValueError("confidence must lie in [0, 1]")
    return conf, corr


def wilson_interval(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation here because calibration bins are
    often small and often near 0 or 1, exactly where the normal interval breaks
    (it can leave [0, 1] and its coverage collapses).
    """
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


@dataclass(frozen=True)
class Bin:
    """One bucket of a reliability diagram."""

    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float
    ci_low: float
    ci_high: float

    @property
    def gap(self) -> float:
        """Signed miscalibration: positive means overconfident in this bin."""
        return self.mean_confidence - self.accuracy


def bin_edges(confidence: Sequence[float], n_bins: int, scheme: BinScheme) -> np.ndarray:
    """Edges for ``n_bins`` buckets over [0, 1].

    ``equal_width`` is the textbook reliability diagram: readable, but bins in
    the sparse middle can hold a handful of points and dominate nothing.
    ``equal_mass`` (adaptive) puts the same number of predictions in each bin,
    which gives every bin a comparable error bar -- the honest default for a
    model like Jev whose confidence piles up near 1.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    if scheme == "equal_width":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if scheme == "equal_mass":
        conf = np.asarray(confidence, dtype=float)
        quantiles = np.quantile(conf, np.linspace(0.0, 1.0, n_bins + 1))
        quantiles[0], quantiles[-1] = 0.0, 1.0
        # Collapse duplicate edges (heavy ties at 1.0 are common) so we never
        # emit an empty zero-width bin.
        return np.unique(quantiles)
    raise ValueError(f"unknown binning scheme: {scheme!r}")


def reliability_bins(
    confidence: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = 15,
    scheme: BinScheme = "equal_width",
    z: float = Z95,
) -> list[Bin]:
    """Bucket predictions and measure observed accuracy in each bucket."""
    conf, corr = as_scored_arrays(confidence, correct)
    edges = bin_edges(conf, n_bins, scheme)
    # Right-closed bins so that a confidence of exactly 1.0 lands in the top bin.
    idx = np.clip(np.searchsorted(edges, conf, side="left") - 1, 0, len(edges) - 2)
    idx[conf <= edges[0]] = 0

    bins: list[Bin] = []
    for b in range(len(edges) - 1):
        mask = idx == b
        count = int(mask.sum())
        if count == 0:
            bins.append(Bin(float(edges[b]), float(edges[b + 1]), 0, float("nan"), float("nan"), 0.0, 1.0))
            continue
        hits = int(corr[mask].sum())
        lo, hi = wilson_interval(hits, count, z)
        bins.append(
            Bin(
                lower=float(edges[b]),
                upper=float(edges[b + 1]),
                count=count,
                mean_confidence=float(conf[mask].mean()),
                accuracy=hits / count,
                ci_low=lo,
                ci_high=hi,
            )
        )
    return bins


def expected_calibration_error(
    confidence: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = 15,
    scheme: BinScheme = "equal_width",
) -> float:
    """ECE: the count-weighted mean absolute gap between confidence and accuracy.

    Read it as "on average, the reported probability is off by this much."
    """
    bins = reliability_bins(confidence, correct, n_bins, scheme)
    total = sum(b.count for b in bins)
    return float(sum(b.count * abs(b.gap) for b in bins if b.count) / total)


def maximum_calibration_error(
    confidence: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = 15,
    scheme: BinScheme = "equal_width",
    min_count: int = 1,
) -> float:
    """MCE: the worst gap in any bin holding at least ``min_count`` predictions.

    This is the number that matters for gating, because a single badly
    miscalibrated band is what a confidence threshold will trip over.
    """
    bins = [b for b in reliability_bins(confidence, correct, n_bins, scheme) if b.count >= max(1, min_count)]
    return float(max((abs(b.gap) for b in bins), default=0.0))


def brier_score(confidence: Sequence[float], correct: Sequence[bool]) -> float:
    """Mean squared error of the top-label confidence. Lower is better.

    Brier is a *proper* score: it rewards being calibrated and being accurate at
    once, so a model cannot game it by hedging everything to its base rate.
    """
    conf, corr = as_scored_arrays(confidence, correct)
    return float(np.mean((conf - corr.astype(float)) ** 2))


def brier_decomposition(
    confidence: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = 15,
    scheme: BinScheme = "equal_mass",
) -> dict[str, float]:
    """Murphy's decomposition: Brier = reliability - resolution + uncertainty.

    * reliability -- miscalibration (lower is better; 0 means perfectly calibrated)
    * resolution  -- how far bins separate from the base rate (higher is better)
    * uncertainty -- the irreducible difficulty of the task itself
    """
    conf, corr = as_scored_arrays(confidence, correct)
    base = float(corr.mean())
    bins = reliability_bins(conf, corr, n_bins, scheme)
    n = conf.size
    reliability = sum(b.count * (b.mean_confidence - b.accuracy) ** 2 for b in bins if b.count) / n
    resolution = sum(b.count * (b.accuracy - base) ** 2 for b in bins if b.count) / n
    uncertainty = base * (1 - base)
    return {
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
        "brier": float(reliability - resolution + uncertainty),
    }


def negative_log_likelihood(probabilities: Sequence[float], eps: float = 1e-12) -> float:
    """Mean ``-log p`` of the probability assigned to the *true* label.

    Needs the full distribution, so it is only reported when a provider returns
    one. It punishes confident mistakes far harder than Brier does.
    """
    p = np.clip(np.asarray(probabilities, dtype=float), eps, 1.0)
    if p.size == 0:
        raise ValueError("no probabilities to score")
    return float(-np.mean(np.log(p)))


def multiclass_brier(distributions: Sequence[dict[str, float]], labels: Sequence[str], classes: Sequence[str]) -> float:
    """Brier over the whole probability vector, not just the winning label."""
    classes = list(classes)
    if len(distributions) != len(labels):
        raise ValueError("distributions and labels must align")
    if not distributions:
        raise ValueError("no distributions to score")
    total = 0.0
    for dist, label in zip(distributions, labels):
        for cls in classes:
            target = 1.0 if cls == label else 0.0
            total += (dist.get(cls, 0.0) - target) ** 2
    return float(total / len(distributions))


def auroc(confidence: Sequence[float], correct: Sequence[bool]) -> float:
    """AUROC of confidence as a detector of its own errors.

    Calibration and discrimination are different questions: a model can be
    badly calibrated yet still rank its mistakes below its hits, which is all a
    confidence *gate* actually needs. 0.5 means the confidence carries no signal
    about correctness; 1.0 means every error sits below every hit.
    """
    conf, corr = as_scored_arrays(confidence, correct)
    pos = conf[corr]
    neg = conf[~corr]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    # Rank-sum (Mann-Whitney U) form, which handles ties correctly.
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(order.size, dtype=float)
    sorted_vals = np.concatenate([pos, neg])[order]
    i = 0
    while i < sorted_vals.size:
        j = i
        while j + 1 < sorted_vals.size and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    rank_sum_pos = ranks[: pos.size].sum()
    return float((rank_sum_pos - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def overconfidence(confidence: Sequence[float], correct: Sequence[bool]) -> float:
    """Signed headline gap: mean confidence minus accuracy.

    Positive means the model claims more than it delivers. Unlike ECE this can
    cancel out across bins, so report it *alongside* ECE, never instead of it.
    """
    conf, corr = as_scored_arrays(confidence, correct)
    return float(conf.mean() - corr.mean())


def calibration_curve_fit(confidence: Sequence[float], correct: Sequence[bool], eps: float = 1e-6) -> dict[str, float]:
    """Fit ``P(correct) = sigmoid(a * logit(conf) + b)`` by Newton's method.

    A compact, bin-free summary of the miscalibration's *shape*:

    * ``slope`` < 1 -- confidences are too extreme (the classic overconfidence);
      > 1 -- too timid.
    * ``intercept`` != 0 -- a constant bias toward or away from "correct".

    Perfect calibration is slope 1, intercept 0.
    """
    conf, corr = as_scored_arrays(confidence, correct)
    x = np.log(np.clip(conf, eps, 1 - eps) / (1 - np.clip(conf, eps, 1 - eps)))
    y = corr.astype(float)
    X = np.column_stack([x, np.ones_like(x)])
    w = np.zeros(2)
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-X @ w))
        grad = X.T @ (y - p)
        s = np.clip(p * (1 - p), 1e-10, None)
        hess = X.T @ (X * s[:, None])
        try:
            step = np.linalg.solve(hess + 1e-9 * np.eye(2), grad)
        except np.linalg.LinAlgError:  # pragma: no cover - degenerate input
            break
        w = w + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return {"slope": float(w[0]), "intercept": float(w[1])}


def bootstrap_ci(
    statistic: Callable[[np.ndarray, np.ndarray], float],
    confidence: Sequence[float],
    correct: Sequence[bool],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for any of the statistics above.

    ECE is a biased, noisy estimator on small samples; publishing it without an
    interval is how audits end up arguing about noise.
    """
    conf, corr = as_scored_arrays(confidence, correct)
    rng = np.random.default_rng(seed)
    n = conf.size
    values = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        values[i] = statistic(conf[idx], corr[idx])
    lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))
