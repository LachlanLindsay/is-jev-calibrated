"""Getting labelled data into a :class:`~jevcal.types.Task`.

Deliberately unopinionated: the audit is only as credible as the dataset behind
it, so this module converts *your* labelled data rather than shipping a
favourite benchmark. Bring any CSV or JSONL with a text column and a label
column and you have a task.

:func:`synthetic_task` is for exercising the pipeline offline. It generates text
that no model should be asked to classify for real; it exists so that
``--provider simulated`` has something to run against at 10,000 rows.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

from .types import Example, NOUL_LABELS, Task, TaskKind, write_jsonl


def from_rows(
    rows: Iterable[dict[str, Any]],
    *,
    name: str,
    question: str,
    kind: TaskKind = "choice",
    labels: Sequence[str] | None = None,
    text_field: str = "text",
    label_field: str = "label",
    id_field: str | None = "id",
    instructions: str = "",
    source: str = "",
    label_map: dict[str, str] | None = None,
) -> Task:
    """Build a task from dictionaries.

    ``label_map`` renames raw labels to the wording you want the model to see
    (``{"0": "negative", "1": "positive"}``). For ``score`` tasks pass ``labels``
    explicitly -- the rubric's order is the whole point and cannot be inferred.
    """
    examples: list[Example] = []
    seen: list[str] = []
    for i, row in enumerate(rows):
        raw_label = row[label_field]
        if isinstance(raw_label, bool):
            raw_label = "true" if raw_label else "false"
        label = str(raw_label)
        if label_map:
            label = label_map.get(label, label)
        if label not in seen:
            seen.append(label)
        meta = {k: v for k, v in row.items() if k not in (text_field, label_field, id_field)}
        examples.append(
            Example(
                id=str(row.get(id_field, i)) if id_field else str(i),
                text=str(row[text_field]),
                label=label,
                meta=meta,
            )
        )

    if labels is not None:
        resolved = tuple(labels)
    elif kind == "noul":
        resolved = NOUL_LABELS
    else:
        resolved = tuple(sorted(seen))

    return Task(
        name=name,
        kind=kind,
        question=question,
        labels=resolved,
        examples=tuple(examples),
        instructions=instructions,
        source=source,
    )


def from_jsonl(path: str | Path, **kwargs: Any) -> Task:
    path = Path(path)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    kwargs.setdefault("source", str(path))
    return from_rows(rows, **kwargs)


def from_csv(path: str | Path, **kwargs: Any) -> Task:
    path = Path(path)
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    kwargs.setdefault("source", str(path))
    return from_rows(rows, **kwargs)


def save_task(task: Task, spec_path: str | Path, data_path: str | Path | None = None) -> Path:
    """Write a task out as a spec (+ JSONL) that ``load_task`` can read back."""
    spec_path = Path(spec_path)
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec: dict[str, Any] = {
        "name": task.name,
        "kind": task.kind,
        "question": task.question,
        "labels": list(task.labels),
    }
    if task.instructions:
        spec["instructions"] = task.instructions
    if task.source:
        spec["source"] = task.source

    rows = [
        {"id": e.id, "text": e.text, "label": e.label, **e.meta}
        for e in task.examples
    ]
    if data_path is None:
        spec["examples"] = rows
    else:
        data_path = Path(data_path)
        write_jsonl(data_path, rows)
        try:
            spec["data"] = str(data_path.relative_to(spec_path.parent))
        except ValueError:
            spec["data"] = str(data_path.resolve())

    spec_path.write_text(json.dumps(spec, indent=2) + "\n")
    return spec_path


_TEMPLATES = {
    "positive": ["genuinely great {noun}", "loved the {noun}", "{noun} exceeded what I expected"],
    "negative": ["the {noun} was a letdown", "would not buy this {noun} again", "{noun} broke in a week"],
    "neutral": ["the {noun} arrived on time", "it is a {noun}", "standard {noun}, nothing notable"],
}
_NOUNS = ["kettle", "monitor", "jacket", "bike light", "mattress", "router", "lamp", "backpack"]


def synthetic_task(
    n: int = 1000,
    kind: TaskKind = "choice",
    n_labels: int = 3,
    seed: int = 0,
    name: str = "synthetic-demo",
) -> Task:
    """A throwaway labelled task, for exercising the pipeline with no network.

    The text is template-generated nonsense. It is here so the plumbing and the
    charts can be checked end to end, never as a benchmark.
    """
    rng = random.Random(seed)
    if kind == "noul":
        labels: tuple[str, ...] = NOUL_LABELS
        question = "Is this review positive?"
    elif kind == "score":
        labels = tuple(f"{i}" for i in range(1, n_labels + 1))
        question = "Rate this review from 1 (worst) to {} (best).".format(n_labels)
    else:
        pool = ["positive", "negative", "neutral"]
        labels = tuple(pool[:n_labels]) if n_labels <= 3 else tuple(pool + [f"other-{i}" for i in range(n_labels - 3)])
        question = "What is the sentiment of this review?"

    examples: list[Example] = []
    for i in range(n):
        label = rng.choice(labels)
        template_key = label if label in _TEMPLATES else rng.choice(list(_TEMPLATES))
        text = rng.choice(_TEMPLATES[template_key]).format(noun=rng.choice(_NOUNS))
        examples.append(Example(id=f"syn-{i:06d}", text=text, label=label, meta={"synthetic": True}))

    return Task(
        name=name,
        kind=kind,
        question=question,
        labels=labels,
        examples=tuple(examples),
        instructions="",
        source="jevcal.datasets.synthetic_task (generated, not a benchmark)",
    )
