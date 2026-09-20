#!/usr/bin/env python3
"""What do class descriptions buy? Bare names versus described, same rows.

Two comparisons, one per run this repo contains:

- **targeted** (E1b): the 16 classes involved in the dominant confusions,
  described by hand after seeing the errors. An upper bound by construction.
- **clean** (E1c): all 150 classes described by one mechanical, error-blind rule
  (each class's first three training utterances, verbatim), scored on val+test
  only so the exemplar source and the evaluated rows are disjoint. The number
  that can be quoted.

In both cases the baseline is the bare-name E1 run restricted to exactly the
same rows, so the description is the only thing that differs.

    python scripts/description_effect.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jevcal.metrics import expected_calibration_error, overconfidence
from jevcal.runner import align, read_predictions
from jevcal.types import load_task

COMPARISONS = [
    ("targeted (16 classes, post-hoc: upper bound)",
     "tasks/clinc150-described.json", "runs/clinc150-described.jsonl"),
    ("clean (all 150, error-blind rule, val+test)",
     "tasks/clinc150-described-full.json", "runs/clinc150-described-full.jsonl"),
]


def stats(conf: list[float], corr: list[bool]) -> dict[str, float]:
    n = len(corr)
    at1 = [(c, k) for c, k in zip(conf, corr) if c == 1.0]
    return {
        "n": n,
        "accuracy": sum(corr) / n,
        "errors": n - sum(corr),
        "overconfidence": overconfidence(conf, corr),
        "ece": expected_calibration_error(conf, corr),
        "n_at_1": len(at1),
        "errors_at_1": sum(1 for _, k in at1 if not k),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", dest="as_json", default="results/description-effect.json")
    args = ap.parse_args()

    base_task = load_task("tasks/clinc150-inscope.json")
    bex, bpr = align(base_task, list(read_predictions("runs/clinc150-inscope.jsonl")))
    bare = {e.id: (p.confidence, p.predicted == e.label) for e, p in zip(bex, bpr)}

    out = {}
    for name, task_path, run_path in COMPARISONS:
        if not Path(run_path).exists():
            print(f"[skip] {name}: {run_path} not found")
            continue
        task = load_task(task_path)
        ex, pr = align(task, list(read_predictions(run_path)))
        ids = [e.id for e in ex if e.id in bare]
        described = {e.id: (p.confidence, p.predicted == e.label) for e, p in zip(ex, pr)}
        b = stats(*map(list, zip(*(bare[i] for i in ids))))
        d = stats(*map(list, zip(*(described[i] for i in ids))))
        out[name] = {"bare": b, "described": d}

        print(f"=== {name} ===  ({b['n']:,} identical rows)")
        print(f"{'':<22}{'bare names':>12}{'described':>12}")
        for key, fmt in (("accuracy", "{:.4f}"), ("errors", "{:,.0f}"),
                         ("overconfidence", "{:+.4f}"), ("ece", "{:.4f}"),
                         ("errors_at_1", "{:,.0f}")):
            print(f"{key:<22}{fmt.format(b[key]):>12}{fmt.format(d[key]):>12}")
        print(f"{'share at p=1.000':<22}{b['n_at_1']/b['n']:>12.1%}{d['n_at_1']/d['n']:>12.1%}")
        print()

    Path(args.as_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.as_json).write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.as_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
