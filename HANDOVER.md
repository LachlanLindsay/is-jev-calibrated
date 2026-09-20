# Handover: run the real audit

> **Status: done.** The run this brief asked for happened on 2026-09-19 —
> 51,000 decisions against `jev-1.13.0`, findings in [REPORT.md](REPORT.md).
> Step 1's warning proved out: the API surface assumed below was wrong
> ([docs/api-notes.md](docs/api-notes.md) is the verification trail), and the
> cost estimate was 8× low. This file is kept unedited below as the record of
> the brief.


**Status: this repo contains no measurements of Jev.** The harness is complete
and its 78 tests pass, but it has never been pointed at the model. It was built
in a cloud container whose egress was restricted to package registries —
`ai-gateway.vercel.sh` and `typesafe.ai` were both blocked at the proxy — so
every result committed so far comes from the built-in simulator and is
watermarked `SIMULATED DATA`.

Your job is to produce the real numbers. The dataset is chosen: **CLINC150**.

## Pick up where it left off

```bash
source .venv/bin/activate
git pull
python -m pytest -q                 # expect 78 passed
python -m jevcal audit examples/tasks/demo-choice.json --provider simulated
```

That last command needs no API key and writes `report/` in about ten seconds.
Read `README.md` before going further — it explains the design and the honesty
rules the harness enforces. Python 3.10+; rebuild `.venv` against 3.11 or 3.12
if it is older (macOS ships 3.9 as `/usr/bin/python3`).

## Step 1 — verify the API surface, before anything else

This is the single highest-risk assumption in the repo.
`jevcal/providers/gateway.py` sends an OpenAI-compatible request to the Vercel
AI Gateway against model `typesafe-ai/jev`, and reads the probability
distribution from one of three places, tried in order:

1. a native field on the response body — `probabilities` / `distribution` /
   `probs`, possibly nested under `decision`;
2. OpenAI-style `logprobs.content[*].top_logprobs`;
3. a JSON object in the message content carrying a `probabilities` map.

**That shape was inferred from the OpenAI-compatible surface, not verified
against live docs.** So:

```bash
export AI_GATEWAY_API_KEY=...        # or JEV_API_KEY
python -m jevcal probe examples/tasks/demo-choice.json
```

`probe` prints the request, the raw response, and which extraction paths succeed
against it.

- If exactly one path works, pin it for the real run with
  `--probability-source <body|logprobs|content>`.
- If none work, fix `extract_decision` in `jevcal/providers/gateway.py` and add
  a test to `tests/test_providers.py` (there are existing tests for each path to
  copy). **Do not loosen it into something that guesses** — the design rule is
  that a missing distribution fails loudly rather than inventing a confidence.
- Check the request side too: if the gateway rejects the `response_format`
  json_schema block or the `logprobs` / `top_logprobs` fields, adjust
  `build_payload`.

While you are in there, sanity-check `INPUT_USD_PER_MTOK`. It is set from
TypeSafe's published launch price ($0.042 per million input tokens, output free);
confirm it against what the gateway actually reports in `usage`.

Record which build `jev-latest` resolved to. The response's `model` field is
captured per row and surfaces in the report, but state it explicitly in the
writeup.

## Step 2 — CLINC150

Larson et al., EMNLP 2019, *An Evaluation Dataset for Intent Classification and
Out-of-Scope Prediction*. CC BY-SA 3.0.

Source (verified reachable and parsed, 2026-09-19):
`https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json`
— 2.5 MB, a single JSON object of six keys, each a list of `[utterance, intent]`
pairs:

| key | rows | note |
| --- | --- | --- |
| `train` | 15,000 | 150 intents × 100, exactly balanced |
| `val` | 3,000 | 150 intents × 20 |
| `test` | 4,500 | 150 intents × 30 |
| `oos_train` | 100 | label is the literal string `oos` |
| `oos_val` | 100 | |
| `oos_test` | 1,000 | |

22,500 in-scope rows over 150 intents, plus 1,200 out-of-scope. Utterances are
short — median 37 characters, max 125 — so context rot is not a confound and the
cost is trivial: a 150-name option list plus an utterance is roughly 300 input
tokens, so the whole in-scope set runs at well under a dollar at the published
price. Check the real per-request cost against `--limit 200` before assuming that.

### Why this dataset suits the audit

- **150 options exercises Choice near its interesting range** (Jev supports up to
  255). Everything published in the launch wave used binary or 3-class tasks.
- **The classes are exactly balanced**, so the majority-class baseline is 1/150 =
  0.67%. Accuracy means something without argument.
- **The out-of-scope split is a calibration question nobody has asked of Jev**:
  does confidence drop when the query isn't one of the things it was asked about?
  That is the confidence-gating use case in its purest form.

### Three experiments, all supported by the harness as it stands

Run **E1** at minimum. E2 and E3 are what make the result novel rather than
merely rigorous.

**E1 — 150-way Choice on in-scope data.** The headline reliability diagram.
Use `train + val + test` for 22,500 rows, or `test` alone (4,500) for a cheap
first pass. `--kind choice`, labels = the 150 intent names.

**E2 — 151-way Choice including `oos` as an option.** Adds the 1,200
out-of-scope rows and makes rejection a decision the model can take. Measures
whether it routes to `oos` and whether its confidence when it does so is
calibrated.

**E3 — Noul: "is this query in scope?"** Labels `true` / `false`, `true` for
in-scope rows and `false` for the 1,200 OOS ones. Build it balanced (sample
~1,200 in-scope rows to match) so the base rate doesn't flatter it. This is the
binary gate a real system would actually ship, and the primitive most people
will put in production.

Build each with the converter:

```bash
python -m jevcal make-task tasks/clinc150-inscope.jsonl \
  --out tasks/clinc150-inscope.json \
  --kind choice \
  --question "Which intent does this user utterance express?" \
  --labels accept_reservations account_blocked alarm ...   # all 150
```

You'll want a short script to turn `data_full.json` into the JSONL the converter
reads (`{"id": ..., "text": ..., "label": ...}` per line) and to emit the label
list. Commit it under `scripts/` and commit the resulting task specs, so the run
is reproducible by anyone.

### Caveats to carry into the writeup

- **Contamination.** CLINC150 has been public since 2019 and is very likely in
  pretraining corpora. This matters less for calibration than for accuracy —
  calibration is about the relationship between confidence and correctness, not
  the level of correctness — but say it rather than let a reader raise it.
- **Informative label names.** Intent names like `translate` and `balance` carry
  real signal. That is realistic for how Jev gets used, but it means this
  measures calibration on a task with helpful label strings, not on opaque
  classes.
- **One dataset licenses one claim.** Intent classification on short utterances
  is not "Jev is calibrated". Say what you measured.

## Step 3 — run it

```bash
python -m jevcal audit tasks/clinc150-inscope.json --provider gateway \
  --probability-source <whatever probe told you> \
  --concurrency 16
```

Start with `--limit 200` to confirm the plumbing and check cost per decision
before committing to the full run. Runs are resumable — results stream to
`runs/*.jsonl` a row at a time and a rerun skips what is already on disk — so an
interruption costs nothing.

Watch the failure count in the run metadata. Failed rows are recorded and
excluded rather than dropped silently, but a high failure rate means extraction
is wrong, not the model. Go back to step 1.

## Step 4 — read the result honestly

The report lands in `report/report.md` with `results.json` and six charts.
Before publishing:

- **Accuracy against the majority-class baseline.** Both are in the report.
- **ECE with its bootstrap interval, under both binning schemes.** If equal-width
  and equal-mass disagree much, the result is sensitive to binning and the honest
  headline is the range.
- **The confidence histogram.** If everything piles up above 0.99 there may be too
  few populated bins to draw a real curve — itself a finding worth reporting.
- **The recommended gate.** The harness only recommends a threshold when the
  *upper* 95% error bound clears the budget and the accepted slice holds 30+
  predictions; anything weaker comes back flagged `provisional`. Don't quote a
  provisional gate as if it were guaranteed.
- **Hold the gate out.** `tests/test_calibration_recovery.py` has a split-half
  pattern (`test_a_recommended_gate_holds_up_on_data_the_threshold_was_not_chosen_on`).
  Do the same on the real run: choose the threshold on one half, measure it on
  the other. If it doesn't hold, that is the story.

**Publish whichever way it comes out.** A negative result is the more useful
contribution here — nobody has published either. The README's "What a published
result should say" section lists what the writeup has to carry.

## Done looks like

- `probe` output confirmed, `--probability-source` pinned, and any `gateway.py`
  fix committed with a test.
- `scripts/` loader and `tasks/*.json` specs committed, so the run reproduces.
- E1 completed over 10,000+ decisions, with `runs/*.jsonl` and `runs/*.meta.json`
  on disk. E2 and E3 too, if you're going for the full result.
- `report/` regenerated from real data — no `SIMULATED DATA` watermark on the
  charts, no simulated banner in `report.md`.
- Real reports committed somewhere that does **not** overwrite
  `examples/sample-report/` (that one is deliberately synthetic and its README
  explains why) — e.g. `results/clinc150-inscope/`.
- README's "Status: no Jev numbers here yet" section replaced with the headline.
- `python -m pytest -q` still green.

## Don't

- Don't commit the API key, or paste it into the README or a commit message.
  `.env` is gitignored.
- Don't delete or "fix" the simulated provider. It keeps the pipeline testable in
  CI with no key, and `tests/test_calibration_recovery.py` is what validates that
  the audit's own machinery works — without it, no verdict this repo prints about
  a real model means anything.
- Don't quote TypeSafe's published speed, cost or accuracy figures as if this
  repo reproduced them. The harness measures what your run measured; the latency
  it records is end-to-end including network, which is not model time.
- Check which terms you're running under before publishing benchmark numbers. The
  harness defaults to the Vercel AI Gateway specifically because
  `typesafe-ai/jev` is generally available there, outside the waitlisted direct
  API's early-access agreement. If you switch to direct access via `JEV_BASE_URL`,
  confirm its terms allow publishing first.
