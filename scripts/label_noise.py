#!/usr/bin/env python3
"""How much of the measured miscalibration is CLINC150's annotation noise?

Inspecting the decisions Jev got wrong *while reporting a probability of
exactly 1.000* turned up something that changes how the headline should be
read: 78% of them come from six pairs of near-synonymous intents, and on
several of those the gold label is contestable or simply wrong.

    "when was my last oil change"        gold last_maintenance  -> said oil_change_when
    "give me a recipe for tacos"         gold ingredients_list  -> said recipe
    "how much paid time off do i have left"  gold pto_used      -> said pto_balance
    "set a reminder to call my mom"      gold reminder_update   -> said reminder

None of those are model failures in any useful sense. So this script reports the
audit's numbers twice: once scored strictly against CLINC150, and once with the
six pairs treated as equivalent.

**The strict number stays the headline.** These pairs were chosen after seeing
which confusions were costly, and a post-hoc rule that flatters the subject is
its own bias -- so this is published as a sensitivity analysis with the pairs
named, not as a replacement. The gap between the two is the honest object: it is
a floor on what any calibration study of this dataset can resolve, because no
model can be scored better than its labels are.

Equivalence is symmetric, so the rule cannot be tuned by picking a direction.
In practice it barely matters: the confusion runs 395 one way against 3 the
other, which is itself a finding -- Jev consistently prefers the broader intent
over CLINC150's finer split.

    python scripts/label_noise.py
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

#: Intent pairs CLINC150 separates and this analysis treats as one, with why.
#: Written down rather than inferred, so the judgement is auditable.
CONTESTED_PAIRS: list[tuple[str, str, str]] = [
    ("reminder_update", "reminder",
     "'set a reminder to call my mom' is labelled reminder_update, but nothing is "
     "being updated -- these are plain reminder-creation utterances."),
    ("accept_reservations", "restaurant_reservation",
     "'can i make a reservation at chevy's' is genuinely ambiguous between asking "
     "whether reservations are possible and asking for one to be made."),
    ("improve_credit_score", "credit_score",
     "'will late payments hurt my credit' asks what affects a score, not how to "
     "improve one; credit_score is at least as defensible."),
    ("last_maintenance", "oil_change_when",
     "'when was my last oil change' is labelled last_maintenance even though an "
     "intent named oil_change_when exists. The prediction is the better answer."),
    ("ingredients_list", "recipe",
     "'give me a recipe for tacos' is labelled ingredients_list. It asks for a recipe."),
    ("pto_used", "pto_balance",
     "'how much pto do i have left' is labelled pto_used. 'Left' is a balance."),
]


def equivalent(gold: str, predicted: str) -> bool:
    return any({gold, predicted} == {a, b} for a, b, _ in CONTESTED_PAIRS)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", default="tasks/clinc150-inscope.json")
    ap.add_argument("--predictions", default="runs/clinc150-inscope.jsonl")
    ap.add_argument("--json", dest="as_json", default="results/label-noise.json")
    args = ap.parse_args()

    task = load_task(args.task)
    ex, pr = align(task, list(read_predictions(args.predictions)))
    conf = [p.confidence for p in pr]
    strict = [p.predicted == e.label for e, p in zip(ex, pr)]
    lenient = [s or equivalent(e.label, p.predicted)
               for s, (e, p) in zip(strict, zip(ex, pr))]
    n = len(strict)

    def block(correct: list[bool]) -> dict[str, float]:
        return {
            "accuracy": sum(correct) / n,
            "errors": n - sum(correct),
            "overconfidence": overconfidence(conf, correct),
            "ece": expected_calibration_error(conf, correct),
        }

    s, l = block(strict), block(lenient)
    at1 = [(a, b) for a, b, c in zip(strict, lenient, conf) if c == 1.0]
    s1 = sum(a for a, _ in at1)
    l1 = sum(b for _, b in at1)

    print(f"{task.name}: {n:,} decisions\n")
    print(f"{'':<22}{'strict':>12}{'pairs as ties':>16}{'difference':>13}")
    for key, fmt in (("accuracy", "{:.4f}"), ("errors", "{:,.0f}"),
                     ("overconfidence", "{:+.4f}"), ("ece", "{:.4f}")):
        d = l[key] - s[key]
        print(f"{key:<22}{fmt.format(s[key]):>12}{fmt.format(l[key]):>16}{fmt.format(d):>13}")
    print(f"\nat a reported probability of exactly 1.000 ({len(at1):,} decisions)")
    print(f"{'errors':<22}{len(at1)-s1:>12,}{len(at1)-l1:>16,}")
    print(f"{'error rate':<22}{1-s1/len(at1):>12.4f}{1-l1/len(at1):>16.4f}")

    affected = sum(1 for e, p in zip(ex, pr)
                   if p.predicted != e.label and equivalent(e.label, p.predicted))
    print(f"\ndecisions affected: {affected:,} ({affected/n:.2%} of the run)")

    err = Counter((e.label, p.predicted) for e, p in zip(ex, pr) if p.predicted != e.label)
    print(f"\n{'pair':<52}{'gold->said':>12}{'reverse':>9}")
    rows = []
    for a, b, why in CONTESTED_PAIRS:
        f, r = err[(a, b)], err[(b, a)]
        print(f"{a + ' / ' + b:<52}{f:>12}{r:>9}")
        rows.append({"gold": a, "predicted": b, "forward": f, "reverse": r, "rationale": why})

    out = {
        "task": task.name, "n": n, "strict": s, "lenient": l,
        "at_probability_one": {
            "n": len(at1), "strict_errors": len(at1) - s1, "lenient_errors": len(at1) - l1,
        },
        "affected_decisions": affected,
        "pairs": rows,
    }
    Path(args.as_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.as_json).write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {args.as_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
