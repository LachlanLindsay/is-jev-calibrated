# Jev on CLINC150: the first published reliability diagram

**Model:** `jev-1.13.0` (what `jev-latest` resolved to on 2026-09-19)
**Endpoint:** `api.typesafe.ai/v1/systemone`
**Dataset:** CLINC150 — Larson et al., EMNLP 2019, CC BY-SA 3.0
**Decisions:** 48,600 across three experiments, 0 failed
**Cost:** $4.90 total, ~$0.10 per 1,000 decisions

Reproduce with `python scripts/build_clinc150.py`, then `python -m jevcal run`
each of `tasks/clinc150-*.json`. The task specs are committed; the loader is
deterministic.

## The short version

Jev is **well calibrated in the aggregate and badly calibrated where it matters
most**. On 150-way intent classification it gets 92.6% right against a 0.67%
baseline, with an ECE of 2.1% — a genuinely good number, and the first one
published for this model. But:

- It reports **exactly 1.000** — literal certainty — on **62% of all decisions**,
  and is wrong 219 of those 13,977 times. A probability of 1.0 is a claim that
  cannot be correct, and it is the single most common thing this model says.
- Its probabilities are **far too extreme in shape**: a logistic refit gives a
  slope of 0.28, where 1.00 would be right.
- On the **binary in-scope/out-of-scope gate** — the confidence-gating use case
  in its purest form, and the thing most people would actually ship — reliability
  goes **non-monotone at the top**, and no confidence threshold meets an error
  budget of even 10%.

Calibration and discrimination come apart cleanly here, which is the honest way
to report it: an AUROC of 0.85 means confidence *ranks* its errors well on the
150-way task, even while the numbers themselves are too extreme.

## E1 — 150-way Choice, in-scope (22,500 decisions)

| metric | value |
| --- | --- |
| accuracy | 92.63% |
| majority-class baseline | 0.67% |
| mean confidence | 94.71% |
| overconfidence | +2.08% |
| ECE, 15 equal-width bins | 0.0209 (95% CI 0.0188–0.0245) |
| ECE, 15 equal-mass bins | 0.0208 |
| MCE (bins n ≥ 30) | 0.0714 |
| AUROC, confidence vs. correctness | 0.851 |
| logistic refit slope / intercept | 0.283 / +0.843 |
| latency p50 / p95 | 160 ms / 284 ms |

The two binning schemes agree to within 0.0001, so the headline is not an
artifact of the bin count.

### The top of the scale is where the story is

81% of decisions land in the top bin. At the resolution the API actually reports
(probabilities are quantised to 0.01):

| reported | n | share | accuracy | gap |
| --- | --- | --- | --- | --- |
| 1.00 | 13,977 | 62.1% | 98.43% | +1.57% |
| 0.99 | 1,710 | 7.6% | 96.08% | +2.92% |
| 0.98 | 885 | 3.9% | 94.92% | +3.08% |
| 0.97 | 577 | 2.6% | 94.80% | +2.20% |
| 0.96 | 443 | 2.0% | 93.23% | +2.77% |
| 0.95 | 356 | 1.6% | 91.29% | +3.71% |

The ordering is right — higher claims really are more accurate — and the gaps are
small. But the model spends most of its time asserting certainty it does not
have. If you gate on "act only when the model is certain", you accept 62% of
traffic at a 1.6% error rate, not 0%.

## E2 — 151-way Choice with `oos` as an option (23,700 decisions)

Accuracy 91.32%, ECE 0.0207, AUROC 0.851 — adding a 151st option costs almost
nothing on the in-scope rows. The interesting part is the slice:

| slice | n | accuracy | mean confidence | gap |
| --- | --- | --- | --- | --- |
| in-scope | 22,500 | 92.32% | 94.06% | +1.74% |
| out-of-scope | 1,200 | 72.67% | 79.70% | **+7.04%** |

It routes 72.7% of out-of-scope queries to `oos`, with a false-alarm rate of only
0.89% on in-scope traffic — a usable rejector. But **calibration is four times
worse on out-of-distribution input**, which is exactly where a confidence number
is load-bearing. When it misses an out-of-scope query it is 70.0% confident in a
concrete wrong intent; the commonest wrong destinations are `fun_fact` (4.5%),
`definition` (3.1%) and `date` (2.4%).

`oos` is the one option given a written description, because the bare token means
nothing to a model. That description is part of the experiment and is committed
in the task spec.

## E3 — Noul: "can the assistant handle this?" (2,400 decisions, balanced)

This is the binary gate a real system ships, and it is the weakest result.

| metric | value |
| --- | --- |
| accuracy | 71.92% |
| majority-class baseline | 50.00% |
| overconfidence | +4.68% |
| ECE | 0.0468 (95% CI 0.0369–0.0659) |
| MCE | 0.1417 |
| AUROC | 0.663 |
| logistic refit slope | 0.695 |

Reliability is **not monotone at the top**:

| confidence bin | n | claimed | delivered | gap |
| --- | --- | --- | --- | --- |
| 0.87–0.93 | 543 | 0.902 | 0.858 | +0.043 |
| 0.93–1.00 | 206 | 0.948 | **0.806** | **+0.142** |

The most confident band is *less* accurate than the band below it. Confidence is
close to anti-informative at the top of the scale, and an AUROC of 0.663 says it
ranks its own errors only a little better than chance. **No threshold meets an
error budget of 1%, 2%, 5% or 10%** on this task.

The same question asked as a 151-way Choice (E2) is answered far better than as a
direct yes/no (E3) — 72.7% recall with a calibrated-ish confidence, versus a gate
that cannot be set at all. If you need a scope check, make it a Choice with a
rejection option rather than a Noul.

## Holding the gate out

A threshold picked on the data it is then reported on is a cherry-pick. Gates
were chosen on half of each run and measured on the other half (seeded shuffle,
`scripts/holdout_gate.py`):

| task | budget | threshold | fit error | held-out error | held-out coverage | holds? |
| --- | --- | --- | --- | --- | --- | --- |
| E1 | 2% | ≥ 1.0000 | 1.58% | 1.55% | 61.8% | yes |
| E1 | 5% | ≥ 0.7000 | 4.53% | 4.40% | 93.2% | yes |
| E2 | 5% | ≥ 0.7576 | 4.52% | 4.14% | 89.2% | yes |
| E3 | any | — | — | — | — | no gate exists |

The E1 and E2 gates transfer to data their thresholds were not chosen on, which
is the claim that matters. Note what the 2% gate on E1 is: "act only when the
model reports 1.00", covering 62% of traffic. E2's 2% recommendation comes back
flagged `provisional` and should not be quoted as guaranteed.

## Jev's `confidence` field versus the probability

TypeSafe returns a `confidence` alongside the distribution and its docs tell you
to gate on it. It is **not** the probability of the chosen option — it is one
minus the normalised entropy of the distribution. On E1 the two happen to track
each other closely (they differ on 29.5% of decisions, never by more than 0.029)
because 150-option distributions are extremely peaked, so both sit near 1.

That closeness is a property of *this* task, not a general fact. On the 3-option
example in TypeSafe's own docs, probabilities of {0.84, 0.159, 0.001} carry a
`confidence` of 0.596 — the two numbers diverge by a quarter of the scale. Do not
assume they are interchangeable on tasks with fewer options or flatter answers.

## What this does not license

- **One dataset, one claim.** This is short-utterance intent classification with
  informative label names. It is not "Jev is calibrated".
- **Contamination.** CLINC150 has been public since 2019 and is very likely in
  pretraining data. This matters less for calibration than for accuracy — the
  relationship between confidence and correctness is the object of study, not the
  level of correctness — but the 92.6% should be read with it in mind.
- **Informative label names.** `translate` and `balance` carry real signal. That
  is realistic for how Jev gets used, and it also means this is not a measurement
  on opaque classes.
- **NLL is not meaningful here.** When the model is wrong the true class is
  usually reported at exactly 0.00, so the figure is set by the clip constant
  (1e-12) and the error rate rather than by the model's density. It appears in
  the per-experiment reports for completeness; do not quote it.
- **The top of the scale cannot be resolved.** Above 0.98 there are exactly three
  reportable values. A gap there is real; its shape is not measurable.
- **Latency is end-to-end**, including network, and is not model time.
- **Terms.** These numbers come from the direct API, which is waitlisted behind an
  early-access agreement. TypeSafe's public terms contain no clause restricting
  publication of benchmark results, but an early-access agreement is a separate
  document. Confirm before publishing.
