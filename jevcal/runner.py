"""Batch runner: many typed decisions, resumably.

An audit is only as good as its sample size, and 10,000+ requests is long enough
that something will go wrong in the middle of it. So: results stream to disk one
row at a time, a rerun skips what is already on disk, and a failed row is
recorded as a failure rather than dropped.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

from .providers.base import Provider
from .types import Example, Prediction, Task


@dataclass
class RunMetadata:
    """Everything needed to say what a results file actually is."""

    task: str
    task_source: str
    kind: str
    labels: list[str]
    provider: str
    #: The URL the decisions were actually fetched from. ``provider`` alone is
    #: ambiguous -- "gateway" is this repo's client class, which can speak either
    #: the Vercel AI Gateway or TypeSafe's own API -- and "which endpoint served
    #: this?" is a question a published number has to be able to answer.
    endpoint: str
    model: str
    n_examples: int
    n_ok: int
    n_failed: int
    started_at: float
    finished_at: float
    wall_seconds: float
    total_cost_usd: float | None
    simulated: bool

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")


def _close_last_line(path: Path) -> None:
    """Terminate a half-written final line before appending to it.

    A run killed mid-flush can leave a JSON fragment with no newline. Appending
    straight onto it would splice the next result into the fragment and lose a
    second row as well, so we close the line first.
    """
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb") as fh:
        fh.seek(-1, 2)
        if fh.read(1) != b"\n":
            with path.open("a") as out:
                out.write("\n")


def completed_ids(path: Path) -> set[str]:
    """Example ids already present in a results file."""
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(str(json.loads(line)["example_id"]))
            except (json.JSONDecodeError, KeyError):
                # A partial final line from a killed run: ignore it, the example
                # simply gets redone.
                continue
    return done


def run_task(
    task: Task,
    provider: Provider,
    out_path: str | Path,
    concurrency: int = 8,
    resume: bool = True,
    progress: bool = True,
    keep_raw: bool = False,
) -> RunMetadata:
    """Run every example in ``task`` through ``provider``, appending to ``out_path``."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    already = completed_ids(out_path) if resume else set()
    if not resume and out_path.exists():
        out_path.unlink()
    else:
        _close_last_line(out_path)
    pending = [e for e in task.examples if e.id not in already]

    if progress and already:
        print(f"resuming: {len(already)} done, {len(pending)} to go", file=sys.stderr)

    write_lock = threading.Lock()
    counter = {"done": 0, "failed": 0, "cost": 0.0}
    started = time.time()

    with out_path.open("a") as fh:
        def work(example: Example) -> None:
            prediction = provider.predict(task, example)
            row = prediction.to_json()
            if not keep_raw:
                row.pop("raw", None)
            with write_lock:
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                counter["done"] += 1
                if not prediction.ok:
                    counter["failed"] += 1
                if prediction.cost_usd:
                    counter["cost"] += prediction.cost_usd
                if progress and counter["done"] % 50 == 0:
                    _print_progress(counter["done"], len(pending), counter["failed"], started)

        if concurrency <= 1:
            for example in pending:
                work(example)
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                list(pool.map(work, pending))

    if progress and pending:
        _print_progress(counter["done"], len(pending), counter["failed"], started, final=True)

    finished = time.time()
    rows = list(read_predictions(out_path))
    n_ok = sum(1 for p in rows if p.ok)
    # Sum what is on disk, not what this invocation happened to fetch. Runs are
    # resumable, so a run finished in two passes would otherwise report only the
    # second pass's spend and understate the cost of the result.
    costed = [p.cost_usd for p in rows if p.cost_usd is not None]
    model = next((p.model for p in rows if p.model), "")
    return RunMetadata(
        task=task.name,
        task_source=task.source,
        kind=task.kind,
        labels=list(task.labels),
        provider=getattr(provider, "name", provider.__class__.__name__),
        endpoint=_endpoint_of(provider),
        model=model,
        n_examples=len(rows),
        n_ok=n_ok,
        n_failed=len(rows) - n_ok,
        started_at=started,
        finished_at=finished,
        wall_seconds=finished - started,
        total_cost_usd=sum(costed) if costed else None,
        simulated=getattr(provider, "name", "") == "simulated",
    )


def _endpoint_of(provider: Any) -> str:
    """The URL a provider posts to, when it posts to one at all."""
    return str(getattr(provider, "endpoint", ""))


def _print_progress(done: int, total: int, failed: int, started: float, final: bool = False) -> None:
    elapsed = max(time.time() - started, 1e-6)
    rate = done / elapsed
    eta = (total - done) / rate if rate > 0 and not final else 0.0
    tail = "" if final else f" eta {eta:6.0f}s"
    end = "\n" if final else "\r"
    print(
        f"  {done}/{total} decisions  {rate:6.1f}/s  {failed} failed{tail}",
        file=sys.stderr,
        end=end,
        flush=True,
    )


def read_predictions(path: str | Path) -> Iterator[Prediction]:
    """Stream predictions back off disk."""
    path = Path(path)
    if not path.exists():
        return
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield Prediction(
                example_id=str(row["example_id"]),
                predicted=row.get("predicted"),
                confidence=row.get("confidence"),
                distribution=row.get("distribution"),
                latency_ms=row.get("latency_ms"),
                cost_usd=row.get("cost_usd"),
                reported_confidence=row.get("reported_confidence"),
                model=row.get("model", ""),
                error=row.get("error"),
                raw=row.get("raw"),
            )


def align(task: Task, predictions: list[Prediction]) -> tuple[list[Example], list[Prediction]]:
    """Pair predictions with their examples, dropping failures.

    Deduplicates by example id (a resumed run can write a row twice if it was
    killed mid-flush) and keeps the task's ordering so slices stay comparable.
    """
    by_id: dict[str, Prediction] = {}
    for p in predictions:
        if p.ok:
            by_id.setdefault(p.example_id, p)
    examples: list[Example] = []
    aligned: list[Prediction] = []
    for example in task.examples:
        prediction = by_id.get(example.id)
        if prediction is not None:
            examples.append(example)
            aligned.append(prediction)
    return examples, aligned
