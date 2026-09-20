#!/usr/bin/env python3
"""E2's P(oos) versus E3's noul, as out-of-scope detectors, on identical rows.

E2 and E3 ask the same underlying question -- "is this utterance something the
assistant covers?" -- through different primitives. E2 embeds it in a 151-way
Choice whose distribution includes an ``oos`` option; E3 asks it directly as a
yes/no. Comparing their headline numbers is loose, because they are different
metrics over different row mixes.

This is the tight version. Both runs scored the same 1,200 out-of-scope rows,
and E2 also scored the 1,200 in-scope rows E3 sampled. So for exactly the same
2,400 utterances we have two single-number out-of-scope scores from the same
model: E2's P(oos) and E3's P(false). Rank them as detectors and compare.

    python scripts/rejector_comparison.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jevcal.metrics import auroc
from jevcal.runner import align, read_predictions
from jevcal.types import load_task


def _pfalse(task_path: str, run_path: str) -> dict[str, tuple[str, float]]:
    task = load_task(task_path)
    ex, pr = align(task, list(read_predictions(run_path)))
    return {e.id: (e.label, p.distribution["false"]) for e, p in zip(ex, pr)}


def scores() -> tuple[np.ndarray, list[tuple[str, np.ndarray]]]:
    gate = _pfalse("tasks/clinc150-gate.json", "runs/clinc150-gate.jsonl")

    oos_task = load_task("tasks/clinc150-oos.json")
    oex, opr = align(oos_task, list(read_predictions("runs/clinc150-oos.jsonl")))
    e2 = {e.id: p.distribution.get("oos", 0.0) for e, p in zip(oex, opr)}

    ids = [i for i in gate if i in e2]
    truth = np.array([gate[i][0] == "false" for i in ids])
    detectors = [
        ("E3 noul P(false), one-sentence scope", np.array([gate[i][1] for i in ids])),
        ("E2 choice P(oos)", np.array([e2[i] for i in ids])),
    ]
    rich_path = Path("runs/clinc150-gate-rich.jsonl")
    if rich_path.exists():
        rich = _pfalse("tasks/clinc150-gate-rich.json", str(rich_path))
        detectors.insert(1, ("E3b noul P(false), scope enumerated",
                             np.array([rich[i][1] for i in ids])))
    return truth, detectors


def at_false_alarm(score: np.ndarray, truth: np.ndarray, budget: float) -> tuple[float, float, float]:
    """Recall and realised false-alarm rate at the threshold spending ``budget``."""
    threshold = float(np.quantile(score[~truth], 1.0 - budget))
    return float((score[truth] >= threshold).mean()), float((score[~truth] >= threshold).mean()), threshold


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", dest="as_json", default="results/rejector-comparison.json")
    args = ap.parse_args()

    truth, detectors = scores()
    n = truth.size
    print(f"{n:,} identical rows, {int(truth.sum()):,} out-of-scope\n")
    rows = []
    for name, score in detectors:
        a = auroc(score.tolist(), truth.tolist())
        entry = {"detector": name, "auroc": a, "operating_points": []}
        print(f"{name}:  AUROC {a:.4f}")
        for budget in (0.01, 0.05, 0.10):
            recall, fa, t = at_false_alarm(score, truth, budget)
            entry["operating_points"].append(
                {"false_alarm_budget": budget, "recall": recall, "false_alarm": fa, "threshold": t})
            print(f"   at {budget:>4.0%} false alarms: catches {recall:6.1%} of out-of-scope (t={t:.2f})")
        rows.append(entry)
        print()

    Path(args.as_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.as_json).write_text(json.dumps({"n": n, "detectors": rows}, indent=2) + "\n")
    print(f"wrote {args.as_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
