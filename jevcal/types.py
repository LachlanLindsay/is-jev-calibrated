"""Core data types for the calibration audit.

Jev exposes exactly three decision primitives, and this module mirrors them:

* ``choice`` -- pick one of up to 255 options, with a probability distribution.
* ``score``  -- a position on an ordered rubric.
* ``noul``   -- a yes/no with the calibrated probability that it is true.

Everything downstream (runner, metrics, plots, report) speaks these types, so a
new provider only has to produce a :class:`Prediction`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

TaskKind = Literal["choice", "score", "noul"]

#: Label used for the two outcomes of a ``noul`` task, so that a noul can be
#: treated as a two-option choice wherever that is convenient.
NOUL_LABELS = ("true", "false")


@dataclass(frozen=True)
class Example:
    """One labelled input: the state Jev sees plus the ground-truth answer."""

    id: str
    text: str
    label: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Task:
    """A dataset plus the typed question asked about every row of it.

    ``labels`` is the ordered set of permitted answers. For ``choice`` the order
    is arbitrary; for ``score`` it is the rubric, lowest rung first; for ``noul``
    it is always :data:`NOUL_LABELS`.
    """

    name: str
    kind: TaskKind
    question: str
    labels: tuple[str, ...]
    examples: tuple[Example, ...]
    instructions: str = ""
    source: str = ""
    #: Optional per-label descriptions, sent to the model alongside the option
    #: names. Jev's Choice/Score questions take a description per option and the
    #: docs are explicit that they carry real signal, so whatever is sent here is
    #: part of the experiment and belongs in the committed task spec. When a
    #: label has no entry the provider sends the label itself.
    criteria: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in ("choice", "score", "noul"):
            raise ValueError(f"unknown task kind: {self.kind!r}")
        if self.kind == "noul" and tuple(self.labels) != NOUL_LABELS:
            raise ValueError(f"noul tasks must use labels {NOUL_LABELS}, got {self.labels}")
        if len(self.labels) < 2:
            raise ValueError("a task needs at least two labels")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("task labels must be unique")
        if self.kind == "choice" and len(self.labels) > 255:
            raise ValueError("Jev Choice supports at most 255 options")
        unknown = {e.label for e in self.examples} - set(self.labels)
        if unknown:
            raise ValueError(f"examples carry labels outside the task's label set: {sorted(unknown)}")
        stray = set(self.criteria) - set(self.labels)
        if stray:
            raise ValueError(f"criteria describe labels not in the task: {sorted(stray)}")

    def label_index(self, label: str) -> int:
        return self.labels.index(label)

    def subset(self, n: int, *, seed: int | None = None) -> "Task":
        """Return a copy holding at most ``n`` examples.

        With ``seed`` the subset is a deterministic random sample; without it the
        first ``n`` rows are kept, which is what you want when resuming a run.
        """
        examples = list(self.examples)
        if seed is not None:
            import random

            random.Random(seed).shuffle(examples)
        return Task(
            name=self.name,
            kind=self.kind,
            question=self.question,
            labels=self.labels,
            examples=tuple(examples[:n]),
            instructions=self.instructions,
            source=self.source,
            criteria=self.criteria,
        )


@dataclass
class Prediction:
    """One typed decision returned by a provider.

    ``confidence`` is the probability the provider assigns to ``predicted`` -- the
    number whose honesty this whole repo exists to check. ``distribution`` holds
    the full probability vector when the provider returns one, which lets us
    score the tail of the distribution and not just the top label.
    """

    example_id: str
    predicted: str | None
    confidence: float | None
    distribution: dict[str, float] | None = None
    latency_ms: float | None = None
    cost_usd: float | None = None
    #: Jev's own ``confidence`` statistic, when the surface returns one. It is
    #: *not* the probability of ``predicted``: it is 1 - the normalised entropy
    #: of the whole distribution, so it answers "how peaked is this?" rather
    #: than "how likely is this to be right?". Recorded because TypeSafe's docs
    #: tell users to gate on it, which makes it worth auditing separately -- but
    #: it must never be plotted against the reliability diagonal.
    reported_confidence: float | None = None
    model: str = ""
    error: str | None = None
    raw: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.predicted is not None and self.confidence is not None

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        # ``raw`` is kept for auditability but can be large; callers strip it.
        return d


def load_task(path: str | Path) -> Task:
    """Load a task spec.

    The spec is JSON::

        {
          "name": "sst2-sentiment",
          "kind": "choice",
          "question": "What is the sentiment of this review?",
          "labels": ["positive", "negative"],
          "instructions": "optional extra framing shown to the model",
          "data": "sst2.jsonl"            # or "examples": [...]
        }

    ``data`` is a JSONL file, resolved relative to the spec, with one
    ``{"id": ..., "text": ..., "label": ...}`` object per line. Any other keys on
    a row are kept in ``Example.meta`` so you can slice the results later.
    """
    path = Path(path)
    spec = json.loads(path.read_text())

    kind: TaskKind = spec.get("kind", "choice")
    labels = tuple(spec["labels"]) if "labels" in spec else NOUL_LABELS

    rows: Iterable[dict[str, Any]]
    if "examples" in spec:
        rows = spec["examples"]
    else:
        data_path = (path.parent / spec["data"]).resolve()
        rows = _read_jsonl(data_path)

    examples = tuple(_row_to_example(row, i) for i, row in enumerate(rows))
    return Task(
        name=spec.get("name", path.stem),
        kind=kind,
        question=spec["question"],
        labels=labels,
        examples=examples,
        instructions=spec.get("instructions", ""),
        source=spec.get("source", str(path)),
        criteria=dict(spec.get("criteria", {}) or {}),
    )


def _row_to_example(row: dict[str, Any], index: int) -> Example:
    meta = {k: v for k, v in row.items() if k not in ("id", "text", "label")}
    label = row["label"]
    if isinstance(label, bool):
        label = "true" if label else "false"
    return Example(
        id=str(row.get("id", index)),
        text=row["text"],
        label=str(label),
        meta=meta,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
