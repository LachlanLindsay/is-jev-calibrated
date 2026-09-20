#!/usr/bin/env python3
"""Flatten every run into one CSV you can open in a spreadsheet.

The raw `runs/*.jsonl` carry a full 150-way distribution per row, which makes
them 75MB and unreadable by eye. This keeps one row per decision with the fields
you actually look at, plus the model's top three options, which is usually enough
to see why an error happened.

Where a decision was also run in the described-criteria variant, that result is
joined onto the same row as `described_*`, so a before/after sits side by side.

    python scripts/export_decisions.py            # -> results/decisions.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jevcal.runner import align, read_predictions
from jevcal.types import load_task

RUNS = [
    ("E1-inscope", "tasks/clinc150-inscope.json", "runs/clinc150-inscope.jsonl"),
    ("E2-oos", "tasks/clinc150-oos.json", "runs/clinc150-oos.jsonl"),
    ("E3-gate", "tasks/clinc150-gate.json", "runs/clinc150-gate.jsonl"),
]
DESCRIBED = ("tasks/clinc150-described.json", "runs/clinc150-described.jsonl")

FIELDS = [
    "experiment", "id", "split", "utterance", "gold", "predicted", "correct",
    "confidence", "jev_confidence", "top1", "p1", "top2", "p2", "top3", "p3",
    "n_nonzero_options", "latency_ms", "cost_usd",
    "described_predicted", "described_correct", "described_confidence",
]


def top_n(dist: dict[str, float] | None, n: int = 3) -> list[tuple[str, float]]:
    """The n most likely options, skipping any the model gave no mass at all.

    With 150 options most answers are one-hot after the API's 0.01 rounding, so
    filling top2/top3 with alphabetically-first zeroes would just be noise.
    """
    if not dist:
        return []
    ranked = sorted(dist.items(), key=lambda kv: -kv[1])
    return [(k, v) for k, v in ranked[:n] if v > 0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results/decisions.csv")
    args = ap.parse_args()

    described: dict[str, tuple[str, bool, float]] = {}
    if Path(DESCRIBED[1]).exists():
        dt = load_task(DESCRIBED[0])
        for e, p in zip(*align(dt, list(read_predictions(DESCRIBED[1])))):
            described[e.id] = (p.predicted, p.predicted == e.label, p.confidence)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for name, task_path, run_path in RUNS:
            if not Path(run_path).exists():
                continue
            task = load_task(task_path)
            ex, pr = align(task, list(read_predictions(run_path)))
            for e, p in zip(ex, pr):
                tops = top_n(p.distribution)
                row = {
                    "experiment": name,
                    "id": e.id,
                    "split": e.meta.get("split", e.id.split("-")[0]),
                    "utterance": e.text,
                    "gold": e.label,
                    "predicted": p.predicted,
                    "correct": int(p.predicted == e.label),
                    "confidence": round(p.confidence, 4),
                    "jev_confidence": (round(p.reported_confidence, 4)
                                       if p.reported_confidence is not None else ""),
                    "n_nonzero_options": sum(1 for v in (p.distribution or {}).values() if v > 0),
                    "latency_ms": round(p.latency_ms) if p.latency_ms else "",
                    "cost_usd": p.cost_usd or "",
                }
                for i, (lab, prob) in enumerate(tops, 1):
                    row[f"top{i}"] = lab
                    row[f"p{i}"] = round(prob, 4)
                if name == "E1-inscope" and e.id in described:
                    d = described[e.id]
                    row["described_predicted"] = d[0]
                    row["described_correct"] = int(d[1])
                    row["described_confidence"] = round(d[2], 4)
                w.writerow(row)
                n += 1
    size = out.stat().st_size / 1e6
    print(f"wrote {out} -- {n:,} decisions, {size:.1f}MB")
    print(f"  {len(described):,} rows carry a described_* before/after comparison")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
