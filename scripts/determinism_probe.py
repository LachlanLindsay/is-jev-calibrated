#!/usr/bin/env python3
"""Does the same request return the same distribution?

If not, every single-shot number in the audit measures the model's *average*
behaviour rather than a fixed function -- still the operationally relevant
quantity for a gate, but a different claim, and one worth establishing rather
than assuming.

Sends each probe utterance N times and compares the returned distributions at
the API's own reported precision (two decimals). Probes are drawn evenly from
high-confidence and low-confidence rows of the E1 run, since instability, if it
exists, should show up first where the model is torn.

    python scripts/determinism_probe.py --repeats 3 --per-bucket 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jevcal.providers import build_provider
from jevcal.runner import align, read_predictions
from jevcal.types import load_task


def signature(dist: dict[str, float]) -> tuple:
    """A distribution at the precision the API reports (0.01)."""
    return tuple(sorted((k, round(v, 2)) for k, v in dist.items() if v > 0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--per-bucket", type=int, default=10)
    ap.add_argument("--surface", default="typesafe")
    ap.add_argument("--json", dest="as_json", default="results/determinism.json")
    args = ap.parse_args()

    task = load_task("tasks/clinc150-inscope.json")
    ex, pr = align(task, list(read_predictions("runs/clinc150-inscope.jsonl")))
    certain = [e for e, p in zip(ex, pr) if p.confidence == 1.0][: args.per_bucket]
    torn = [e for e, p in zip(ex, pr) if p.confidence < 0.6][: args.per_bucket]
    provider = build_provider("gateway", surface=args.surface)

    rows = []
    stable = 0
    for example in certain + torn:
        signatures = []
        for _ in range(args.repeats):
            pred = provider.predict(task, example)
            if pred.error:
                raise SystemExit(f"request failed: {pred.error[:200]}")
            signatures.append(signature(pred.distribution))
        identical = len(set(signatures)) == 1
        stable += identical
        rows.append({"id": example.id, "text": example.text, "identical": identical,
                     "distinct_signatures": len(set(signatures))})
        if not identical:
            print(f'VARIES ({len(set(signatures))} variants): "{example.text[:70]}"')
            for s in set(signatures):
                print(f"    {dict(s)}")

    n = len(rows)
    print(f"\n{stable}/{n} utterances identical across {args.repeats} repeats "
          f"({args.per_bucket} high-confidence, {args.per_bucket} low-confidence)")
    Path(args.as_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.as_json).write_text(json.dumps(
        {"repeats": args.repeats, "n": n, "identical": stable, "rows": rows}, indent=2) + "\n")
    print(f"wrote {args.as_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
