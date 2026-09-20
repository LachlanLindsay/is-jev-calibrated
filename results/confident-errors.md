# What Jev gets wrong when it says it is certain

*Reproduce with `python scripts/confident_errors.py`; machine-readable output in
[`confident-errors.json`](confident-errors.json).*

## A correction first

An earlier version of this report claimed most of the audit's measured
miscalibration was CLINC150 annotation noise — that the gold labels were
contestable and in places "simply wrong", and that forgiving six intent pairs
dropped ECE from 0.021 to 0.0066.

**That was wrong.** It was produced by looking only at the utterances Jev got
wrong, and never checking what CLINC150 labels with the *other* intent in each
pair. Doing that check takes one query and overturns the conclusion: the dataset's
taxonomy is coherent and deliberate, and on 372 of the 395 decisions in question
the gold label is right and the model is wrong.

The strict numbers were correct all along: **accuracy 92.63%, ECE 0.0209, 219
errors at a reported probability of exactly 1.000.**

What follows is what those errors actually are, which turns out to be more
interesting than either the original framing or the mistaken correction.

## The confident errors are systematic

Jev is wrong 219 times while reporting a probability of 1.000. Those errors are
not spread evenly across 150 intents. **74% of them are one of four confusions**,
and all four are the same kind of mistake.

CLINC150 splits certain intents by *pragmatics* rather than by topic. Both sides
of each pair are about the same subject; what separates them is tense, speech
act, or whether a value is being asked about or acted upon:

| CLINC150 distinguishes | one side | the other |
| --- | --- | --- |
| **speech act** | `reminder` — *"what reminders did i have"* | `reminder_update` — *"remind me to call bob"* |
| **possibility vs execution** | `accept_reservations` — *"does olive garden take reservations"* | `restaurant_reservation` — *"reserve a table for 3 at 7"* |
| **value vs mechanism** | `credit_score` — *"show me my credit score"* | `improve_credit_score` — *"steps to raise my credit score"* |
| **tense** | `oil_change_when` — *"when will i need my next oil change"* | `last_maintenance` — *"when did i last change my oil"* |

Jev collapses all four:

| gold | Jev said | errors | at p=1.000 |
| --- | --- | --- | --- |
| `reminder_update` | `reminder` | 149 | 105 |
| `accept_reservations` | `restaurant_reservation` | 140 | 30 |
| `improve_credit_score` | `credit_score` | 44 | 17 |
| `last_maintenance` | `oil_change_when` | 39 | 10 |

**372 forward, 0 reverse.** Not a single error runs the other way. This is not
confusion between two similar labels — that would be roughly symmetric. It is a
systematic failure to represent one specific axis of meaning.

## What that suggests

On these pairs the model appears to be matching *topic* rather than parsing
*intent*. "Remind me to call bob" and "what reminders did i have" are both about
reminders; only one of them asks the assistant to create something. "When was my
last oil change" and "when is my next oil change due" are both about oil changes;
only one is about the past.

The distinction that gets lost is exactly the part that determines **what a
system should actually do** — read or write, look up or act, past or future. For
a model sold on routing and typed decisions in software, that is the load-bearing
part of an intent.

And it loses it *while reporting certainty*. Not low confidence, not a near-tie —
a flat 1.000 on 105 of the 149 `reminder_update` cases. The model is not
uncertain between two readings and picking one. It shows no sign of having
noticed there were two.

## The dataset is not blameless, but nearly

Two pairs do look genuinely mislabelled:

- **`pto_used` / `pto_balance`** (8 errors). *"How much pto do i have left"* is
  labelled `pto_used`, but `pto_balance`'s own rows say exactly that — *"how many
  pto days do i have left"*. Jev is right here.
- **`ingredients_list` / `recipe`** (15 errors, mixed). *"Show me the recipe for
  broccoli cheddar soup"* is labelled `ingredients_list` while `recipe` includes
  *"find a recipe for baked ziti"*. Some of these are annotation errors; others
  ("what does it take to make buttermilk pancakes") are defensible.

Forgiving both moves ECE from 0.0209 to 0.0202 and overconfidence from +2.08% to
+1.98%. **Negligible.** 23 decisions out of 22,500.

So annotation noise in CLINC150 is real but tiny — about 0.1% of the run, not the
two-thirds claimed earlier.

## The methodological lesson

The mistaken version of this report was not a computational error. Every number
in it was correct. The error was in interpretation: I inspected the model's
mistakes, found them concentrated on pairs that *looked* synonymous out of
context, and concluded the benchmark was at fault — without ever sampling the
other intent to see whether the distinction was real.

That is the same failure this repo exists to warn about, pointed at a dataset
instead of a confidence number: a plausible story, assembled from evidence
selected after the fact, and stated with more certainty than the checking
justified. It also flattered the subject under test, which should have been the
signal to check harder rather than publish.

Worth noting that the check which overturned it — *look at what the other label
is used for* — costs one query and would have prevented the whole detour.
