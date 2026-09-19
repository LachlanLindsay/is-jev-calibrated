# is-jev-calibrated

**An independent calibration audit for typed-decision models.** Built for
TypeSafe's [Jev](https://typesafe.ai), and useful for anything that returns a
probability alongside a decision.

Jev's speed and price are easy to check, and several people already have. Its
distinguishing claim is harder: that every decision arrives with a *calibrated*
probability. That claim is the reason you would put it in front of a frontier
model at all — a confidence you can gate on is worth more than a confidence you
can only read. As far as I can tell nobody has published a reliability diagram
for it, and the vendor hasn't either.

This repo is the harness to settle it, on your data, with numbers you ran
yourself.

```
python -m jevcal audit my-task.json --provider gateway
```

Out comes a reliability diagram, an Expected Calibration Error with a bootstrap
interval, and the number an engineer actually needs: **the confidence threshold
above which it is safe to stop looking.**

---

## Status: no Jev numbers here yet

This repository contains **no measurements of Jev**. The charts and reports you
can generate right now come from a built-in simulator and are stamped
`SIMULATED DATA` accordingly. Getting real numbers takes an API key and one
command — see [Running it for real](#running-it-for-real).

I'd rather ship an audit you can reproduce than a number you have to trust.

[`HANDOVER.md`](HANDOVER.md) is the brief for doing that run: which assumption to
verify first, the chosen dataset and how to build it, and what the result has to
carry to be worth publishing.

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
charts (three, in light and dark). The output is committed at
[`examples/sample-report/`](examples/sample-report/) if you want to see the shape
of the result first.

The simulator is not a stand-in for results. It is there so the pipeline is
exercisable in CI, and so the audit can be checked against a known answer:
`tests/test_calibration_recovery.py` asserts that a perfectly calibrated
simulator is *not* accused of miscalibration, that a temperature-0.5 distortion
is read back as a fitted slope near 0.5, and that a recommended gate still holds
on data its threshold was not chosen on. An audit whose own machinery is
unvalidated is not evidence of anything.

## Running it for real

### 1. Bring a labelled dataset

Any CSV or JSONL with a text column and a ground-truth label column:

```bash
python -m jevcal make-task reviews.csv \
  --out tasks/sentiment.json \
  --question "What is the sentiment of this review?" \
  --label-map '{"0": "negative", "1": "positive"}'
```

For an ordered rubric, pass the rungs in order — the order *is* the rubric:

```bash
python -m jevcal make-task tickets.jsonl \
  --out tasks/urgency.json --kind score \
  --labels low medium high critical \
  --question "How urgent is this support ticket?"
```

The audit is only as credible as the dataset behind it, so this repo ships
converters rather than a favourite benchmark. Pick something public, say which
split you used, and publish the task spec alongside the numbers.

### 2. Confirm where the probabilities live

```bash
export AI_GATEWAY_API_KEY=...
python -m jevcal probe tasks/sentiment.json
```

`probe` sends one request and prints the raw response plus every extraction path
that works against it. **Do this before a long run.** The client reads the
probability distribution from a native field on the body, from OpenAI-style
`top_logprobs`, or from JSON in the message content, and tries them in that
order; pin the right one with `--probability-source` rather than discovering
10,000 requests later that it picked a path you didn't intend. If none of them
match, the run fails loudly instead of inventing a confidence.

One thing this client will never do is ask the model to *write* a confidence
number into its output. A self-reported number in generated text is a different
object from a calibrated distribution, and auditing it would answer a different
question than the one on the label.

### 3. Run the audit

```bash
python -m jevcal audit tasks/sentiment.json --provider gateway --concurrency 16
```

Results stream to `runs/*.jsonl` one row at a time and a rerun skips what is
already on disk, so a run interrupted at example 7,431 costs you nothing to
resume. Failed requests are recorded as failures and excluded from the analysis
rather than quietly dropped; the report says how many there were.

Why the gateway and not the direct API: `typesafe-ai/jev` is generally available
through the [Vercel AI Gateway](https://vercel.com/docs/ai-gateway), while the
direct TypeSafe API is waitlisted behind an early-access agreement. Auditing the
generally-available endpoint keeps published numbers clear of any early-access
restriction, and is the endpoint most readers can reproduce against. Point
`JEV_BASE_URL` elsewhere if you have direct access and its terms allow it.

## The charts

**Reliability diagram** — predicted confidence against observed accuracy, with
95% Wilson intervals and the bin counts underneath. A calibrated model's bars sit
on the diagonal; bars below it are overconfidence. The count panel is there
because a dramatic gap in a bin holding nine predictions is noise, and that is
exactly the bin a screenshot will crop to.

**Risk-coverage curve** — error rate against the share of traffic you handle
automatically. The marked point is the recommended gate.

**Confidence histogram** — what it claims and how often, with the gap between
mean confidence and accuracy shaded.

Every plotted value also appears as a table row in `report.md`. Nothing here is
readable only as a picture.

## How the gate recommendation stays honest

Two rules, both of which cost coverage and are worth it:

1. A threshold is recommended only when the **upper** 95% bound on its error
   clears the budget, not the point estimate. Picking the threshold that happened
   to look best on your sample is how a gate that "tested at 1% error" ships at
   4%. When only the point estimate clears, the recommendation is returned
   flagged `provisional`, not returned silently.
2. No recommendation is made on a slice thinner than 30 predictions. An error
   rate measured on a dozen rows is not a measurement.

`tests/test_calibration_recovery.py` holds this to account the only way that
counts: choose the threshold on half the run, measure it on the other half.

## What a published result should say

If you write this up, the things that make it an audit rather than a vibe:

- the dataset, split, and number of examples, with the task spec committed;
- the model version (`jev-latest` resolves to a specific build — record it);
- the majority-class baseline next to the accuracy, so the accuracy means
  something;
- ECE **with** its interval, and the binning scheme, since ECE is a biased and
  noisy estimator and the bin count moves it;
- a negative result if you get one. An honest "it isn't calibrated on this data"
  is a more useful contribution than another 90%-accuracy screenshot, and it is
  the result the field is currently missing.

## What this does not show

- **Calibration is per-distribution.** A model calibrated on product reviews can
  be badly calibrated on medical text. One dataset licenses one claim.
- **Context length is a separate experiment.** Everything here fits comfortably
  in context; long or noisy states are known to degrade these models and are not
  tested by this harness.
- **Type-safe is not correct.** Every error counted here was a perfectly valid,
  in-schema value that happened to be wrong. "It cannot emit an out-of-schema
  value" and "it cannot be wrong" are different claims, and only the first one is
  structural.
- **Gating moves the problem, it doesn't remove it.** The escalated tail still
  has to go somewhere, and that somewhere has its own cost and its own error
  rate.
- **Vendor figures are not reproduced here.** Any speed, cost or accuracy numbers
  TypeSafe publishes are theirs; this repo measures what your own run measures
  and nothing else. Latency reported by the runner is end-to-end including
  network, which is not the same thing as model time.

## Layout

```
jevcal/
  types.py       Task / Example / Prediction, and the task-spec format
  datasets.py    CSV and JSONL converters, plus a synthetic task for demos
  providers/
    gateway.py   Jev via the Vercel AI Gateway, and the probability extraction
    simulated.py the offline miscalibration simulator
  runner.py      concurrent, resumable batch execution
  metrics.py     ECE, MCE, Brier + decomposition, NLL, AUROC, Wilson, bootstrap
  selective.py   risk-coverage curves and gate recommendations
  analysis.py    everything above, assembled into one AuditResult
  plots.py       the three charts, light and dark
  report.py      report.md and results.json
  cli.py         run / analyze / audit / probe / make-task
```

## Licence

MIT. See [LICENSE](LICENSE).
