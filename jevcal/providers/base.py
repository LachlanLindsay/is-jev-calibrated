"""Provider interface.

A provider turns one :class:`~jevcal.types.Example` into one
:class:`~jevcal.types.Prediction`. That is the entire contract, which is what
makes the audit reusable: point it at a different provider and you are auditing
a different model's calibration with the same code and the same charts.
"""

from __future__ import annotations

from typing import Protocol

from ..types import Example, Prediction, Task


class Provider(Protocol):
    name: str

    def predict(self, task: Task, example: Example) -> Prediction:  # pragma: no cover - protocol
        ...


class ProviderError(RuntimeError):
    """Raised for configuration problems the runner should not retry."""


def normalise_distribution(dist: dict[str, float], labels: tuple[str, ...]) -> dict[str, float]:
    """Restrict a distribution to the task's labels and renormalise.

    A provider may return extra keys, drop near-zero mass, or return counts
    rather than probabilities. Downstream metrics assume a proper distribution
    over exactly ``labels``, so we enforce that in one place.
    """
    filtered = {label: float(dist.get(label, 0.0)) for label in labels}
    if any(v < 0 for v in filtered.values()):
        raise ValueError(f"negative probability in distribution: {dist}")
    total = sum(filtered.values())
    if total <= 0:
        raise ValueError(f"distribution has no mass over the task labels: {dist}")
    return {k: v / total for k, v in filtered.items()}
