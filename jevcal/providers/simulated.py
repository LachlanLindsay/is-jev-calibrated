"""An offline provider that fakes a miscalibrated model.

This exists for two reasons, and neither of them is producing results worth
publishing:

1. The whole pipeline -- runner, metrics, plots, report -- can be exercised end
   to end in CI with no API key and no network.
2. You can point the analysis at a model whose true miscalibration you *chose*,
   and check that the audit recovers it. A calibration audit whose own
   machinery is unvalidated is not evidence of anything.

Every artefact generated from this provider is stamped ``simulated`` so it can
never be mistaken for a measurement of a real model.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from ..types import Example, Prediction, Task


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return float(np.log(p / (1 - p)))


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


@dataclass
class SimulatedProvider:
    """A model whose confidence is a deliberately distorted view of its skill.

    For each example we draw a latent *true* probability of being correct, ``q``,
    from a Beta distribution. Correctness is then drawn as ``Bernoulli(q)``, so
    ``q`` is by construction the perfectly calibrated confidence. The reported
    confidence is a distortion of it in logit space::

        reported = sigmoid(logit(q) / temperature + bias)

    ``temperature < 1`` sharpens -- the familiar overconfident model that says 0.99
    when it means 0.85. ``temperature = 1, bias = 0`` reproduces a perfectly
    calibrated model, which is what the metric tests assert against.
    """

    labels: tuple[str, ...] = ()
    skill_a: float = 6.0
    skill_b: float = 2.0
    temperature: float = 0.55
    bias: float = 0.0
    seed: int = 0
    latency_ms: float = 92.0
    name: str = "simulated"

    def _rng(self, example: Example) -> np.random.Generator:
        # Seeded per example so a run is reproducible and resumable, and so
        # re-running a subset gives the same answers for the same rows.
        digest = hashlib.sha256(f"{self.seed}:{example.id}".encode()).digest()
        return np.random.default_rng(int.from_bytes(digest[:8], "big"))

    def predict(self, task: Task, example: Example) -> Prediction:
        rng = self._rng(example)
        n_labels = len(task.labels)

        # Latent skill on this example, floored at chance: a model cannot be
        # reliably worse than guessing on a balanced label set.
        chance = 1.0 / n_labels
        q = chance + (1 - chance) * float(rng.beta(self.skill_a, self.skill_b))
        is_correct = bool(rng.random() < q)

        reported = _sigmoid(_logit(q) / self.temperature + self.bias)
        reported = float(min(max(reported, chance), 0.999))

        if is_correct:
            predicted = example.label
        else:
            wrong = [l for l in task.labels if l != example.label]
            # Ordinal tasks fail locally: a score model that misses usually
            # misses by one rung, not by the whole rubric.
            if task.kind == "score":
                true_idx = task.label_index(example.label)
                weights = np.array([1.0 / (1 + abs(task.label_index(l) - true_idx)) ** 2 for l in wrong])
                predicted = str(rng.choice(wrong, p=weights / weights.sum()))
            else:
                predicted = str(rng.choice(wrong))

        distribution = self._spread(task, predicted, reported, rng)
        return Prediction(
            example_id=example.id,
            predicted=predicted,
            confidence=reported,
            distribution=distribution,
            latency_ms=float(rng.normal(self.latency_ms, self.latency_ms * 0.15)),
            cost_usd=0.0,
            model="simulated",
            raw={"latent_true_probability": q, "simulated": True},
        )

    def _spread(
        self, task: Task, predicted: str, confidence: float, rng: np.random.Generator
    ) -> dict[str, float]:
        """Spread the leftover mass over the other labels."""
        others = [l for l in task.labels if l != predicted]
        remaining = 1.0 - confidence
        if not others:
            return {predicted: 1.0}
        if task.kind == "score":
            pred_idx = task.label_index(predicted)
            weights = np.array([1.0 / (1 + abs(task.label_index(l) - pred_idx)) ** 2 for l in others])
        else:
            weights = rng.dirichlet(np.ones(len(others)))
        weights = weights / weights.sum()
        dist = {predicted: confidence}
        for label, w in zip(others, weights):
            dist[label] = remaining * float(w)
        return dist
