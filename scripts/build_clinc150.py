#!/usr/bin/env python3
"""Turn CLINC150 into the three task specs this audit runs.

CLINC150 is Larson et al., EMNLP 2019, *An Evaluation Dataset for Intent
Classification and Out-of-Scope Prediction*, CC BY-SA 3.0. The source file is a
single JSON object of six keys, each a list of ``[utterance, intent]`` pairs:
``train`` (15,000), ``val`` (3,000), ``test`` (4,500), and the matching
``oos_*`` splits (100 / 100 / 1,000).

Three experiments come out of it:

E1 ``clinc150-inscope``  150-way Choice over the in-scope rows. The headline
                         reliability diagram. 150 options exercises Choice near
                         the interesting part of its range, and the classes are
                         exactly balanced, so the majority-class baseline is a
                         clean 1/150.
E2 ``clinc150-oos``      151-way Choice with ``oos`` added as an option, over
                         the in-scope rows plus all 1,200 out-of-scope ones.
                         Makes rejection a decision the model can take.
E3 ``clinc150-gate``     Noul: "is this in scope?". Balanced by sampling as many
                         in-scope rows as there are out-of-scope ones, so the
                         base rate does not flatter it.

Run ``python scripts/build_clinc150.py`` to write ``tasks/`` and the JSONL
beside it. Everything is deterministic: the only sampling is E3's balancing
draw, which is seeded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import urllib.request
from pathlib import Path

SOURCE_URL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"
ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "data_full.json"
TASKS = ROOT / "tasks"

IN_SCOPE_SPLITS = ("train", "val", "test")
OOS_SPLITS = ("oos_train", "oos_val", "oos_test")
OOS_LABEL = "oos"

#: The one description we write ourselves. Every in-scope option is named by its
#: own intent string, but ``oos`` is an internal dataset token that means nothing
#: to a model, and E2 is precisely the experiment about whether it gets chosen --
#: so it needs a description, and that description is part of the experiment.
OOS_DESCRIPTION = (
    "None of the other categories fit: the request is outside the set of things "
    "this assistant handles."
)

E3_QUESTION = "Is this user utterance something the assistant can handle?"
E3_INSTRUCTIONS = (
    "The assistant handles a fixed set of 150 everyday personal-assistant "
    "intents: banking, travel, vehicles, small talk, utilities, work and home "
    "automation. Answer true if the utterance is a request of that kind, and "
    "false if it falls outside what the assistant covers."
)


def fetch(force: bool = False) -> dict[str, list[list[str]]]:
    """Download the dataset once and cache it next to the repo."""
    if force or not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {SOURCE_URL}")
        with urllib.request.urlopen(SOURCE_URL, timeout=120) as resp:
            CACHE.write_bytes(resp.read())
    raw = CACHE.read_bytes()
    print(f"{CACHE.relative_to(ROOT)}: {len(raw):,} bytes, sha256 {hashlib.sha256(raw).hexdigest()[:16]}")
    return json.loads(raw)


def humanise(intent: str) -> str:
    return intent.replace("_", " ")


def rows(data: dict, splits: tuple[str, ...]) -> list[tuple[str, str, str]]:
    """``(id, utterance, intent)`` for every row of the named splits."""
    out: list[tuple[str, str, str]] = []
    for split in splits:
        for i, (text, intent) in enumerate(data[split]):
            out.append((f"{split}-{i}", text, intent))
    return out


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    print(f"  {path.relative_to(ROOT)}: {len(records):,} rows")


def write_spec(name: str, spec: dict) -> None:
    path = TASKS / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"  {path.relative_to(ROOT)}: {spec['kind']}, {len(spec.get('labels', []))} labels")


def build(seed: int = 0) -> None:
    data = fetch()

    in_scope = rows(data, IN_SCOPE_SPLITS)
    out_scope = rows(data, OOS_SPLITS)
    intents = sorted({intent for _, _, intent in in_scope})
    if len(intents) != 150:
        raise SystemExit(f"expected 150 intents, found {len(intents)}")

    source = f"CLINC150 (Larson et al., EMNLP 2019), CC BY-SA 3.0, {SOURCE_URL}"

    # ---- E1: 150-way Choice, in-scope only -----------------------------------
    print("\nE1 clinc150-inscope")
    write_jsonl(
        TASKS / "clinc150-inscope.jsonl",
        [{"id": i, "text": t, "label": y, "split": i.split("-")[0]} for i, t, y in in_scope],
    )
    write_spec("clinc150-inscope", {
        "name": "clinc150-inscope",
        "kind": "choice",
        "question": "Which intent does this user utterance express?",
        "labels": intents,
        "data": "clinc150-inscope.jsonl",
        "source": source,
        "notes": (
            "150 intents x 150 rows, exactly balanced; majority-class baseline 1/150 = 0.667%. "
            "Option descriptions are the intent names with underscores replaced by spaces, "
            "which the provider does by default -- no glosses are invented."
        ),
    })

    # ---- E2: 151-way Choice, oos as an option --------------------------------
    print("\nE2 clinc150-oos")
    write_jsonl(
        TASKS / "clinc150-oos.jsonl",
        [{"id": i, "text": t, "label": y, "split": i.split("-")[0]}
         for i, t, y in in_scope + out_scope],
    )
    write_spec("clinc150-oos", {
        "name": "clinc150-oos",
        "kind": "choice",
        "question": "Which intent does this user utterance express?",
        "labels": intents + [OOS_LABEL],
        "criteria": {OOS_LABEL: OOS_DESCRIPTION},
        "data": "clinc150-oos.jsonl",
        "source": source,
        "notes": (
            "The 150 in-scope intents plus 'oos'. Deliberately unbalanced: 22,500 in-scope "
            "rows against 1,200 out-of-scope, which is the realistic ratio and means the "
            "out-of-scope slice must be read on its own, not folded into the headline. "
            "'oos' is the one option given a written description, because the raw token "
            "means nothing to a model; that description is part of the experiment."
        ),
    })

    # ---- E3: Noul gate, balanced ---------------------------------------------
    print("\nE3 clinc150-gate")
    sampled = random.Random(seed).sample(in_scope, len(out_scope))
    gate = (
        [{"id": i, "text": t, "label": "true", "intent": y} for i, t, y in sampled]
        + [{"id": i, "text": t, "label": "false", "intent": y} for i, t, y in out_scope]
    )
    gate.sort(key=lambda r: r["id"])
    write_jsonl(TASKS / "clinc150-gate.jsonl", gate)
    write_spec("clinc150-gate", {
        "name": "clinc150-gate",
        "kind": "noul",
        "question": E3_QUESTION,
        "instructions": E3_INSTRUCTIONS,
        "labels": ["true", "false"],
        "data": "clinc150-gate.jsonl",
        "source": source,
        "notes": (
            f"Balanced 50/50: all {len(out_scope):,} out-of-scope rows against an equal "
            f"number of in-scope rows sampled with seed {seed}, so the majority-class "
            "baseline is 50%. The 150 intents are described to the model in the "
            "instructions rather than enumerated, because a Noul takes no option list."
        ),
    })

    print(f"\nin-scope {len(in_scope):,}  out-of-scope {len(out_scope):,}  intents {len(intents)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0, help="seed for E3's balancing sample")
    build(**vars(ap.parse_args()))
