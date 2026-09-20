# How much of this is the dataset's fault?

*A sensitivity analysis on the CLINC150 calibration audit. Reproduce with
`python scripts/label_noise.py`; the machine-readable output is
[`label-noise.json`](label-noise.json).*

## Why this report exists

The audit's first headline was that Jev "reports exactly 1.000 on 62% of
decisions and is wrong 219 of those times." That is what the benchmark says. It
is also, on inspection, substantially unfair.

Looking at those 219 confident errors, they are not scattered across the 150
intents the way genuine model failures would be. **78% of them come from six
pairs of near-synonymous intents**, and on several of those pairs CLINC150's gold
label is contestable — in some cases plainly wrong, with Jev's answer the better
one:

| utterance | CLINC150 says | Jev said |
| --- | --- | --- |
| "when was my last oil change" | `last_maintenance` | `oil_change_when` |
| "give me a recipe for tacos" | `ingredients_list` | `recipe` |
| "how much paid time off do i have left" | `pto_used` | `pto_balance` |
| "set a reminder to call my mom" | `reminder_update` | `reminder` |
| "will late payments hurt my credit" | `improve_credit_score` | `credit_score` |

There is an intent named `oil_change_when`. The question is when the oil was
changed. Counting that as a model error, at a claimed probability of 1.0, and
then publishing it as evidence of overconfidence, would be a measurement
artifact presented as a finding.

## The six pairs

Chosen by inspecting the confident errors, written down in
`scripts/label_noise.py` so the judgement is auditable rather than asserted.
Equivalence is **symmetric**, so the rule cannot be tuned by choosing a
direction.

| pair | gold → said | reverse | why it is contested |
| --- | --- | --- | --- |
| `reminder_update` / `reminder` | 149 | 0 | Nothing is being updated; these create reminders. |
| `accept_reservations` / `restaurant_reservation` | 140 | 0 | "Can I make a reservation at X" is genuinely ambiguous between asking whether reservations exist and asking for one. |
| `improve_credit_score` / `credit_score` | 44 | 0 | Asks what *affects* a score, not how to improve one. |
| `last_maintenance` / `oil_change_when` | 39 | 0 | A more specific intent exists and fits better. |
| `ingredients_list` / `recipe` | 15 | 2 | The utterance asks for a recipe. |
| `pto_used` / `pto_balance` | 8 | 1 | "Left" is a balance, not a usage figure. |

The confusion runs **395 one way against 3 the other**. That asymmetry is itself
a result: Jev is not confused between these intents, it consistently prefers the
broader or more literal one over CLINC150's finer split. Whether that is a model
error or a taxonomy disagreement is a judgement call, which is exactly why both
scorings are published.

## What it costs the headline

398 decisions — **1.77% of the run** — are affected.

| | strict (benchmark-faithful) | six pairs as ties | difference |
| --- | --- | --- | --- |
| accuracy | 92.63% | 94.40% | +1.77% |
| errors | 1,659 | 1,261 | −398 |
| overconfidence | +2.08% | **+0.31%** | −1.77% |
| ECE | 0.0209 | **0.0066** | −0.0143 |
| errors at p = 1.000 | 219 | **48** | −171 |
| error rate at p = 1.000 | 1.57% | 0.34% | −1.23% |

**Two-thirds of the measured miscalibration on this task is annotation
disagreement, not model overconfidence.** Corrected for it, Jev's mean confidence
sits within a third of a percentage point of its accuracy, which is very well
calibrated by any standard.

## Which number to quote

**The strict one.** These pairs were identified *after* seeing which confusions
were expensive, and a post-hoc rule that happens to flatter the subject is its
own form of cherry-picking. Publishing 0.0066 as the headline would be doing to
the benchmark exactly what this repo exists to stop people doing with confidence
numbers.

So: ECE **0.021** on CLINC150 as CLINC150 is written, with this analysis beside
it. The reader can see the pairs, disagree with any of them, and recompute.

## The part that survives

None of this rescues the strongest claim. Even with every contested pair
forgiven, Jev reports a probability of **exactly 1.000** on 62% of all decisions
and is wrong **48** times when it does. A probability of 1.0 asserts that no
other outcome is possible. Forty-eight counterexamples is a smaller number than
219, and it is still not zero, and it cannot be argued away with label noise
because these are errors on pairs nobody contests.

The [E3 gate result](README.md#e3--noul-can-the-assistant-handle-this-2400-decisions-balanced)
is untouched by any of this. It is a different task with `true`/`false` labels
that carry no annotation ambiguity, and it is where the audit's sharpest negative
finding lives.

## The wider point

A calibration study cannot resolve miscalibration smaller than its labels'
error rate. Here that floor is around 1.4 points of ECE — larger than the
difference between a model you would ship on and one you would not.

Anyone publishing ECE on a public benchmark without looking at what the errors
actually are is reporting the dataset's annotation quality as if it were a
property of the model. That is worth more attention than it gets, and it is the
main reason this repo ships the per-decision data rather than only the summary.
