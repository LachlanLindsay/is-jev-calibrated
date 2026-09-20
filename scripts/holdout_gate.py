#!/usr/bin/env python3
"""Choose a confidence gate on half a run and measure it on the other half.

The gate is the thing anyone reading this audit would actually ship, and a
threshold picked on the same data it is reported on is a cherry-pick. The
harness already guards against that internally -- it requires the *upper* 95%
bound on the error to clear the budget, not the point estimate -- but the only
test that really counts is holding data out.

``tests/test_calibration_recovery.py`` does this against the simulator, where
the answer is known. This does the same thing against a real run.

    python scripts/holdout_gate.py tasks/clinc150-inscope.json runs/clinc150-inscope.jsonl

The split is a seeded shuffle rather than a cut down the middle, because rows
arrive grouped by dataset split and a positional halving would compare train
against test rather than like against like.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jevcal.runner import align, read_predictions
from jevcal.selective import coverage_at_threshold, recommend_threshold
from jevcal.types import load_task

DEFAULT_TARGETS = (0.01, 0.02, 0.05, 0.10)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("task")
    ap.add_argument("predictions")
    ap.add_argument("--targets", type=float, nargs="*", default=list(DEFAULT_TARGETS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", dest="as_json", help="also write the table here")
    args = ap.parse_args()

    task = load_task(args.task)
    examples, predictions = align(task, list(read_predictions(args.predictions)))
    conf = [p.confidence for p in predictions]
    correct = [p.predicted == e.label for e, p in zip(examples, predictions)]

    import random

    order = list(range(len(conf)))
    random.Random(args.seed).shuffle(order)
    half = len(order) // 2
    fit, held = order[:half], order[half:]

    print(f"{task.name}: {len(conf):,} scored decisions, split {len(fit):,} / {len(held):,} (seed {args.seed})\n")
    print(f"{'budget':>7}  {'threshold':>9}  {'status':>11}  "
          f"{'fit err':>8}  {'held err':>8}  {'held cov':>8}  {'held n':>7}  holds?")

    rows = []
    for target in args.targets:
        rec = recommend_threshold([conf[i] for i in fit], [correct[i] for i in fit], target_error=target)
        if rec.threshold is None:
            print(f"{target:>7.0%}  {'-':>9}  {'none':>11}  "
                  f"{'-':>8}  {'-':>8}  {'-':>8}  {'-':>7}  n/a")
            rows.append({"target": target, "threshold": None, "status": "none"})
            continue

        status = "guaranteed" if rec.guaranteed else "provisional"
        out = coverage_at_threshold([conf[i] for i in held], [correct[i] for i in held], rec.threshold)
        # The budget is a claim about the error rate, so the honest check is
        # whether the held-out error stays under it.
        holds = out.n > 0 and out.error <= target
        print(f"{target:>7.0%}  {rec.threshold:>9.4f}  {status:>11}  "
              f"{rec.point.error:>8.4f}  {out.error:>8.4f}  {out.coverage:>8.1%}  {out.n:>7,}  "
              f"{'yes' if holds else 'NO'}")
        rows.append({
            "target": target, "threshold": rec.threshold, "status": status,
            "fit_error": rec.point.error, "held_error": out.error,
            "held_coverage": out.coverage, "held_n": out.n, "holds": holds,
        })

    if args.as_json:
        Path(args.as_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.as_json).write_text(json.dumps(
            {"task": task.name, "n": len(conf), "seed": args.seed, "rows": rows}, indent=2) + "\n")
        print(f"\nwrote {args.as_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
