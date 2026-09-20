#!/usr/bin/env python3
"""What does Jev get wrong when it reports a probability of exactly 1.000?

The answer is not "a random 1.6% of everything". The confident errors are
concentrated, and they concentrate on a specific kind of distinction.

CLINC150 deliberately splits several intents by *pragmatics* rather than topic:

    reminder             read my reminders        vs  reminder_update      create one
    last_maintenance     when did I last ...      vs  oil_change_when      when should I next ...
    credit_score         what is my score         vs  improve_credit_score what affects it
    accept_reservations  does this place book     vs  restaurant_reservation book me a table

Same subject matter on both sides; the difference is tense, speech act, or
whether a value is being queried or acted on. Jev collapses all four, always in
the same direction -- toward the broader or more literal reading -- and it does
so while reporting certainty. 74% of its errors at p=1.000 are one of these four.

That is a model finding, not a dataset one. An earlier version of this analysis
claimed the gold labels were wrong; checking what CLINC150 labels with the *other*
intent showed they are not. The two pairs below that genuinely are questionable
move ECE by 0.0007, which is nothing.

    python scripts/confident_errors.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jevcal.metrics import expected_calibration_error, overconfidence
from jevcal.runner import align, read_predictions
from jevcal.types import load_task

#: Pairs CLINC150 separates on purpose, with the distinction it is drawing and
#: what the dataset's own rows for each side look like. Verified by sampling the
#: other intent's utterances, which is the check the first version of this
#: analysis skipped.
COLLAPSED: list[tuple[str, str, str]] = [
    ("reminder_update", "reminder",
     "speech act: 'reminder' rows read the list back ('what reminders did i have'); "
     "'reminder_update' rows create one ('remind me to call bob')."),
    ("accept_reservations", "restaurant_reservation",
     "possibility vs execution: 'restaurant_reservation' rows all carry a party "
     "size and a time ('reserve a table for 3 at 7'); these ask only whether "
     "booking is possible ('does olive garden take reservations')."),
    ("improve_credit_score", "credit_score",
     "value vs mechanism: 'credit_score' rows query the number ('show me my credit "
     "score'); 'improve_credit_score' rows are about what raises or harms it."),
    ("last_maintenance", "oil_change_when",
     "tense: 'oil_change_when' rows are all forward-looking ('when will i need my "
     "next oil change'); these ask about the past."),
]

#: Pairs where CLINC150's label really does look wrong. Kept separate, and small.
QUESTIONABLE: list[tuple[str, str, str]] = [
    ("pto_used", "pto_balance",
     "'how much pto do i have left' is labelled pto_used, but 'pto_balance' rows "
     "say exactly that ('how many pto days do i have left')."),
    ("ingredients_list", "recipe",
     "'show me the recipe for broccoli cheddar soup' is labelled ingredients_list "
     "while 'recipe' rows include 'find a recipe for baked ziti'. Mixed."),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", default="tasks/clinc150-inscope.json")
    ap.add_argument("--predictions", default="runs/clinc150-inscope.jsonl")
    ap.add_argument("--json", dest="as_json", default="results/confident-errors.json")
    args = ap.parse_args()

    task = load_task(args.task)
    ex, pr = align(task, list(read_predictions(args.predictions)))
    conf = [p.confidence for p in pr]
    strict = [p.predicted == e.label for e, p in zip(ex, pr)]
    n = len(strict)

    at1 = [(e, p) for e, p in zip(ex, pr) if p.confidence == 1.0]
    wrong1 = [(e, p) for e, p in at1 if p.predicted != e.label]
    print(f"{task.name}: {n:,} decisions, accuracy {sum(strict)/n:.4f}, "
          f"ECE {expected_calibration_error(conf, strict):.4f}\n")
    print(f"decisions at a reported probability of exactly 1.000: {len(at1):,} "
          f"({len(at1)/n:.1%} of the run)")
    print(f"   of those, wrong: {len(wrong1)} ({len(wrong1)/len(at1):.2%})\n")

    errs = Counter((e.label, p.predicted) for e, p in zip(ex, pr) if p.predicted != e.label)
    w1 = Counter((e.label, p.predicted) for e, p in wrong1)

    rows = []
    print(f"{'pair':<48}{'all':>6}{'@p=1':>6}  distinction being collapsed")
    covered = 0
    for gold, said, why in COLLAPSED:
        covered += w1[(gold, said)]
        print(f"{gold + ' / ' + said:<48}{errs[(gold, said)]:>6}{w1[(gold, said)]:>6}  {why.split(':')[0]}")
        rows.append({"gold": gold, "predicted": said, "errors": errs[(gold, said)],
                     "errors_at_one": w1[(gold, said)], "kind": "collapsed", "note": why})
    print(f"\n{covered} of {len(wrong1)} confident errors ({covered/len(wrong1):.0%}) "
          f"are one of these four collapses.")
    print(f"All run one way: {sum(errs[(a, b)] for a, b, _ in COLLAPSED)} forward against "
          f"{sum(errs[(b, a)] for a, b, _ in COLLAPSED)} reverse.\n")

    print("pairs where CLINC150's own label looks wrong:")
    q = {(a, b) for a, b, _ in QUESTIONABLE}
    for gold, said, why in QUESTIONABLE:
        print(f"   {gold} / {said}: {errs[(gold, said)]} errors -- {why}")
        rows.append({"gold": gold, "predicted": said, "errors": errs[(gold, said)],
                     "errors_at_one": w1[(gold, said)], "kind": "questionable", "note": why})
    lenient = [s or ((e.label, p.predicted) in q) for s, (e, p) in zip(strict, zip(ex, pr))]
    print(f"\nforgiving only those: ECE {expected_calibration_error(conf, strict):.4f} -> "
          f"{expected_calibration_error(conf, lenient):.4f}, "
          f"overconfidence {overconfidence(conf, strict):+.4f} -> {overconfidence(conf, lenient):+.4f}")
    print("Negligible. The strict numbers are the numbers.")

    Path(args.as_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.as_json).write_text(json.dumps({
        "task": task.name, "n": n,
        "accuracy": sum(strict) / n,
        "ece": expected_calibration_error(conf, strict),
        "at_probability_one": {"n": len(at1), "errors": len(wrong1),
                               "explained_by_collapse": covered},
        "pairs": rows,
    }, indent=2) + "\n")
    print(f"\nwrote {args.as_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
