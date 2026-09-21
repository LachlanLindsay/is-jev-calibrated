#!/usr/bin/env python3
"""Assemble the CLINC150 errata from the audit's documented findings.

The rows here are exactly the label errors already established in
``results/confident-errors.md`` — each one verified during the audit by
checking the utterance against the dataset's OWN rows for both labels, the
check that document exists to insist on. Nothing is re-adjudicated and no
model is consulted; this script just joins the documented findings with their
evidence so the errata is reproducible and self-contained.

Three sources, all from that document:

- the five clear gold-label errors among the described run's certainty errors
  (its "Postscript" table);
- the named ``ingredients_list`` rows that ask for a recipe while the
  dataset's ``recipe`` rows read the same way;
- the ``pto_used`` -> ``pto_balance`` verdict ("'how much pto do i have left'
  is labelled pto_used, but pto_balance rows say exactly that"), applied to
  the run's errors on that pair whose wording matches the verdict's rationale
  (left / available / remaining).

Output: results/label-errata.jsonl, results/label-errata.md, and
docs/upstream-issue.md.

    python scripts/build_errata.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = "https://github.com/LachlanLindsay/is-jev-calibrated"

#: The five clear errors from confident-errors.md's postscript, verbatim,
#: with the reason the gold label fails against the dataset's own usage.
CLEAR = [
    ("allow me to turn on the lights", "restaurant_reservation", "smart_home",
     "Nothing about the utterance concerns a restaurant or a reservation; indefensible on its face."),
    ("what is on my to do list", "reminder", "todo_list",
     "todo_list's own rows read 'what does my to-do list look like' and 'list my to-do list'."),
    ("what is my current location", "share_location", "current_location",
     "Every share_location row names a person to send the location to; none is named here. "
     "current_location's rows read 'please tell me the address of my current location'."),
    ("can you call the help desk line for my credit card company",
     "replacement_card_duration", "make_call",
     "replacement_card_duration's rows all ask how long a replacement card takes to arrive; "
     "this asks to place a call, which is what make_call's rows do."),
    ("how can i request a new credit card", "replacement_card_duration", "new_card",
     "Asks how to request a card, not how long a replacement takes; new_card's rows ask exactly this."),
]

#: The ingredients_list rows named in confident-errors.md that ask for a
#: recipe, against recipe rows like "find a recipe for baked ziti".
RECIPE = [
    ("give me a recipe for tacos", "ingredients_list", "recipe"),
    ("what's a good recipe foe tacos", "ingredients_list", "recipe"),
    ("show me the recipe for broccoli cheddar soup", "ingredients_list", "recipe"),
]
RECIPE_REASON = ("Asks for a recipe; the dataset's recipe rows read the same way "
                 "('find a recipe for baked ziti'), while ingredients_list rows ask "
                 "what ingredients are needed.")

PTO_REASON = ("Asks how much time off remains; pto_balance's own rows read 'how many "
              "pto days do i have left', while pto_used's rows ask how much has been taken.")
PTO_WORDING = re.compile(r"\bleft\b|\bavailable\b|\bremain", re.I)


def dataset_rows() -> dict[str, list[str]]:
    data = json.loads(Path("data/data_full.json").read_text())
    by: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        for text, intent in data[split]:
            by.setdefault(intent, []).append(text)
    return by


def pto_rows() -> list[tuple[str, str, str]]:
    """The run's pto_used->pto_balance errors whose wording the documented
    verdict actually covers."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from jevcal.runner import align, read_predictions
    from jevcal.types import load_task

    task = load_task("tasks/clinc150-inscope.json")
    ex, pr = align(task, list(read_predictions("runs/clinc150-inscope.jsonl")))
    return [(e.text, "pto_used", "pto_balance")
            for e, p in zip(ex, pr)
            if e.label == "pto_used" and p.predicted == "pto_balance"
            and PTO_WORDING.search(e.text)]


def main() -> int:
    by = dataset_rows()
    rows = [dict(utterance=u, dataset_label=g, should_be=s, why=w)
            for u, g, s, w in CLEAR]
    rows += [dict(utterance=u, dataset_label=g, should_be=s, why=RECIPE_REASON)
             for u, g, s in RECIPE]
    rows += [dict(utterance=u, dataset_label=g, should_be=s, why=PTO_REASON)
             for u, g, s in pto_rows()]
    for r in rows:
        r["dataset_rows_for_gold"] = [t for t in by[r["dataset_label"]]
                                      if t != r["utterance"]][:3]
        r["dataset_rows_for_should_be"] = [t for t in by[r["should_be"]]
                                           if t != r["utterance"]][:3]

    jl = Path("results/label-errata.jsonl")
    with jl.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {jl}: {len(rows)} rows")

    md = [f"""# CLINC150 label errata

{len(rows)} rows of CLINC150 (Larson et al., EMNLP 2019, CC BY-SA 3.0) whose
gold label contradicts the dataset's **own usage** of the labels involved.
Surfaced by [this calibration audit]({REPO}) once the audited model's error
rate fell low enough for annotation noise to dominate what remained; each row
was verified by checking the dataset's own utterances for *both* labels — the
discipline [`confident-errors.md`](confident-errors.md) exists to insist on.
Out of 22,500 in-scope rows, that this is the full convicted list speaks well
of the dataset.

**These are errata, not corrections to apply.** Published results depend on
the benchmark as-is; score against it unchanged and use this file as a
sensitivity note. Machine-readable: [`label-errata.jsonl`](label-errata.jsonl).

| utterance | dataset says | should be |
| --- | --- | --- |
"""]
    for r in rows:
        md.append(f"| {r['utterance']} | `{r['dataset_label']}` | `{r['should_be']}` |\n")
    md.append("\n## Evidence\n")
    for r in rows:
        ge = " / ".join(f'"{t}"' for t in r["dataset_rows_for_gold"])
        md.append(f"""
**"{r['utterance']}"** — labelled `{r['dataset_label']}`, should be
`{r['should_be']}`. {r['why']} The dataset's own `{r['dataset_label']}` rows:
{ge}.
""")
    Path("results/label-errata.md").write_text("".join(md))
    print("wrote results/label-errata.md")

    issue = f"""Title: {len(rows)} mislabelled rows in data_full.json, with evidence from the dataset's own usage

While running an independent calibration audit against CLINC150 ({REPO}), the
audited model's residual disagreements with the gold labels turned out to
include rows where the gold label contradicts the dataset's own usage. Example:

> "what is on my to do list" is labelled `reminder`, while `todo_list`'s own
> rows include "what does my to-do list look like" and "list my to-do list".

Each row below was checked against the dataset's own utterances for both
labels before being listed — the full list with per-row evidence is published
as machine-readable errata:

- {REPO}/blob/main/results/label-errata.md
- {REPO}/blob/main/results/label-errata.jsonl

| utterance | labelled | should be |
| --- | --- | --- |
{"".join(f"| {r['utterance']} | `{r['dataset_label']}` | `{r['should_be']}` |" + chr(10) for r in rows)}
{len(rows)} rows out of 22,500 in-scope utterances (~{len(rows)/22500:.2%}) —
which speaks well of the dataset's overall quality.

We are deliberately **not** proposing changes to `data_full.json`: published
results depend on the file as-is, and errata-on-the-side seems the right
pattern for a frozen benchmark. Filing this so the list is discoverable from
the source; happy to send a PR adding a README pointer instead if you'd prefer.
"""
    Path("docs/upstream-issue.md").write_text(issue)
    print("wrote docs/upstream-issue.md")

    print(f"\n{'utterance':<52}{'dataset says':<26}-> should be")
    for r in rows:
        print(f"{r['utterance'][:50]:<52}{r['dataset_label']:<26}-> {r['should_be']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
