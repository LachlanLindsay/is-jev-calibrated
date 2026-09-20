# is-jev-calibrated

**An independent calibration audit for typed-decision models.** Built for
TypeSafe's [Jev](https://typesafe.ai), and useful for anything that returns a
probability alongside a decision.

Jev's speed and price are easy to check, and several people already have. Its
distinguishing claim is harder: that every decision arrives with a *calibrated*
probability. That claim is the reason you would put it in front of a frontier
model at all — a confidence you can gate on is worth more than a confidence you
can only read. This repo is the harness to settle it, on your data, with numbers
you ran yourself:

```
python -m jevcal audit my-task.json --provider gateway
```

Out comes a reliability diagram, an Expected Calibration Error with a bootstrap
interval, and the number an engineer actually needs: **the confidence threshold
above which it is safe to stop looking.**

## The results

We ran it: 60,900 decisions against `jev-1.13.0` on CLINC150, zero failed
requests, about $8. **[REPORT.md](REPORT.md)** is the write-up, written for
engineers with no stats background. The one-line version: **Jev is about as
calibrated as the question you send it.** With bare class names it is good but
overconfident (92.6% accuracy, ECE 0.021, and a wrong answer at probability
`1.000` two hundred times); with every class described by a mechanical
three-example rule it reaches 97.1% accuracy at **ECE 0.0046 with no systematic
lean**, on held-out rows. The boolean scope gate that looked broken (no usable
threshold at any budget) matched the 151-way Choice detector once its
instructions enumerated the scope — the failure was the specification, never
the primitive. What specification can't fix: probabilities quantised to 0.01,
a residual 0.26% error rate on certainty claims, ~4× worse calibration on
out-of-scope input, and non-determinism on contested rows.

Supporting material: [`results/`](results/) (charts, per-experiment reports,
every decision as [one CSV](results/decisions.csv)),
[`docs/api-notes.md`](docs/api-notes.md) (the wire format, verified against the
live service), and [`HANDOVER.md`](HANDOVER.md) (the original brief, kept for
history).

## What it measures

| Question | What answers it |
| --- | --- |
| When it says 90%, is it right 90% of the time? | reliability diagram, ECE, MCE |
| Does it lean one way? | mean confidence minus accuracy; logistic refit slope and intercept |
| Is the *shape* wrong, or just shifted? | slope < 1 means the probabilities are too extreme |
| Can confidence rank its own mistakes? | AUROC of confidence vs. correctness |
| Where do I set the gate? | risk-coverage curve, AURC, per-budget threshold recommendations |
| Is any of this above noise? | Wilson intervals per bin, bootstrap interval on ECE |

Calibration and discrimination are kept apart on purpose. A model can be badly
calibrated and still rank its errors below its hits, which is all a confidence
*gate* needs; and a model can be beautifully calibrated while carrying no
information at all. Conflating the two is the most common way a calibration
claim gets mis-sold in either direction.

## Quickstart (no API key, no network)

Python 3.10 or newer. On a Mac, note that `/usr/bin/python3` is 3.9 — build the
virtualenv against a newer interpreter (`python3.12 -m venv .venv`) rather than
the system one.

```bash
pip install -r requirements.txt
python -m jevcal audit examples/tasks/demo-choice.json --provider simulated
open report/report.md
```

That runs 2,000 synthetic decisions through a model whose miscalibration was
*chosen*, and writes `report/` with a Markdown report, a `results.json`, and six
charts. The output is committed at
[`examples/sample-report/`](examples/sample-report/) if you want to see the
shape of the result first.

The simulator is not a stand-in for results. It is there so the pipeline is
exercisable in CI, and so the audit can be checked against a known answer:
`tests/test_calibration_recovery.py` asserts that a perfectly calibrated
simulator is *not* accused of miscalibration, that a temperature-0.5 distortion
is read back as a fitted slope near 0.5, and that a recommended gate still holds
on data its threshold was not chosen on. An audit whose own machinery is
unvalidated is not evidence of anything.

## Reproducing the Jev audit

```bash
python scripts/build_clinc150.py          # downloads CLINC150, writes tasks/*.json
export JEV_API_KEY=...                    # or AI_GATEWAY_API_KEY for the gateway
python -m jevcal probe tasks/clinc150-inscope.json --surface typesafe
python -m jevcal run   tasks/clinc150-inscope.json --surface typesafe --concurrency 8
python -m jevcal run   tasks/clinc150-oos.json     --surface typesafe --concurrency 8
python -m jevcal run   tasks/clinc150-gate.json    --surface typesafe --concurrency 8
python -m jevcal analyze tasks/clinc150-inscope.json runs/clinc150-inscope.jsonl \
    --out results/clinc150-inscope        # and likewise for the other two
python scripts/holdout_gate.py tasks/clinc150-inscope.json runs/clinc150-inscope.jsonl
python scripts/rejector_comparison.py     # E2-vs-E3 head-to-head
python scripts/confident_errors.py        # what it gets wrong at certainty
python scripts/export_decisions.py        # everything as one CSV
```

The loader is deterministic (the only sampling is seeded), the task specs are
committed, and runs are resumable — results stream to `runs/*.jsonl` a row at a
time and a rerun skips what is already on disk. The full set costs about $5 and
half an hour. **Run `probe` before any long run**: it sends one request and
prints the payload, the raw response, and which extraction paths work, so a
mapping change surfaces on request one rather than request 10,000.

Keep `--concurrency` at 8; the public endpoint rate-limits above roughly that.

### Bringing your own dataset

Any CSV or JSONL with a text column and a label column:

```bash
python -m jevcal make-task reviews.csv \
  --out tasks/sentiment.json \
  --question "What is the sentiment of this review?" \
  --label-map '{"0": "negative", "1": "positive"}'
```

The audit is only as credible as the dataset behind it, so the repo ships
converters rather than a favourite benchmark. Pick something public, say which
split you used, and publish the task spec alongside the numbers. Supply
per-class `criteria` descriptions in the spec — the report's
[Experiment 1b](REPORT.md#experiment-1b--how-much-do-class-descriptions-matter)
measures what leaving them out costs.

### The two surfaces

Jev is an *evaluation*-modality model — it does not generate text and is not
reachable through OpenAI-compatible endpoints. `--surface` selects who serves
it:

| | `gateway` (default) | `typesafe` |
| --- | --- | --- |
| endpoint | `ai-gateway.vercel.sh/v1/evaluate` | `api.typesafe.ai/v1/systemone` |
| availability | generally available | waitlisted, early-access agreement |

Prefer the gateway where you can: its terms are the ones that let you publish
without checking an agreement first. [`docs/api-notes.md`](docs/api-notes.md)
has the verified request and response shapes for both, and the differences that
bit us.

One thing this client will never do is ask the model to *write* a confidence
number into its output. A self-reported number in generated text is a different
object from a calibrated distribution, and auditing it would answer a different
question than the one on the label.

## How the gate recommendation stays honest

Two rules, both of which cost coverage and are worth it:

1. A threshold is recommended only when the **upper** 95% bound on its error
   clears the budget, not the point estimate. Picking the threshold that
   happened to look best on your sample is how a gate that "tested at 1% error"
   ships at 4%. When only the point estimate clears, the recommendation comes
   back flagged `provisional`, not silently.
2. No recommendation is made on a slice thinner than 30 predictions. An error
   rate measured on a dozen rows is not a measurement.

And `scripts/holdout_gate.py` applies the only test that finally counts: choose
the threshold on half the run, measure it on the other half.

## Scope and suggested improvements

What one run on one dataset does and doesn't license, folded together with what
we would do next. Contributions welcome on any of these.

**Scope of the claim.** Calibration is per-distribution: these numbers cover
short-utterance intent classification with informative class names, and license
nothing beyond it. A model calibrated on support tickets can be badly calibrated
on medical text. Relatedly: *type-safe is not correct* — every error counted was
a valid in-schema value that was wrong — and *gating moves the problem rather
than removing it*: the escalated tail still needs somewhere to go, with its own
cost and error rate. Latency figures are end-to-end including network, and no
vendor figures are reproduced here.

**The Score primitive is unaudited.** The harness supports Jev's ordered-rubric
questions end-to-end (`--kind score`), but no experiment has exercised them
against the live model. An ordinal task (rating, severity, urgency) would
complete the coverage of Jev's three primitives.

**Why does full specification overshoot?** Well-specified runs flip from
overconfident to slightly *under*-confident (E1c bias −0.2%, E3b slope 1.23).
Characterising where the crossover sits would make the specification advice
quantitative.

**Description style.** Three verbatim exemplars worked; would one hand-written
sentence, or ten exemplars, work better or worse? A style sweep is cheap and
directly actionable.

**Option-order sensitivity.** `criteria` is an ordered map and we always sent
alphabetical order. Shuffle it and measure whether the distribution moves.

**A second dataset, different domain.** The single strongest upgrade to the
claim. Something with uninformative class names would also separate
label-string signal from state understanding.

**Long or noisy state.** Everything here fits comfortably in context; degraded
calibration under context pressure is a known failure mode of small models and
a separate experiment.

**Cross-surface check.** Rerun a slice through the Vercel gateway
(`/v1/evaluate`) when a gateway key is available, and confirm the two surfaces
return identical distributions for identical requests.

**Rate-limit handling.** The client retries with exponential backoff but
without jitter, and ignores any `Retry-After` header; under sustained 429s
throughput sawtooths (we saw 46/s collapse to ~10/s mid-run). Honouring
`Retry-After` with jitter would smooth long runs.

## Layout

```
jevcal/
  types.py       Task / Example / Prediction, and the task-spec format
  datasets.py    CSV and JSONL converters, plus a synthetic task for demos
  providers/
    gateway.py   Jev via the evaluation API, both surfaces, and the extraction
    simulated.py the offline miscalibration simulator
  runner.py      concurrent, resumable batch execution
  metrics.py     ECE, MCE, Brier + decomposition, NLL, AUROC, Wilson, bootstrap
  selective.py   risk-coverage curves and gate recommendations
  analysis.py    everything above, assembled into one AuditResult
  plots.py       the three charts, light and dark
  report.py      report.md and results.json
  cli.py         run / analyze / audit / probe / make-task
scripts/         dataset loader and the per-question analyses behind REPORT.md
tasks/           committed task specs for the CLINC150 experiments
results/         charts, per-experiment reports, decisions.csv
docs/            api-notes.md — the verified wire format
REPORT.md        the audit's findings, written for engineers
```

## Licence

MIT. See [LICENSE](LICENSE). CLINC150 is CC BY-SA 3.0 (Larson et al., EMNLP
2019).
