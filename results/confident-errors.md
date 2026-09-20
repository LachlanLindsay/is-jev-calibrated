# What Jev gets wrong when it says it is certain

*Reproduce with `python scripts/confident_errors.py`; machine-readable output in
[`confident-errors.json`](confident-errors.json).*

Jev is wrong 219 times out of 13,977 while reporting a probability of exactly
`1.000`. This is a look at what those errors are, and at whether the dataset or
the model is responsible for them.

## First: is the dataset at fault?

The confident errors are not randomly spread. 74% of them fall on six pairs of
near-synonymous classes, and read in isolation several look like the benchmark
simply mislabelled a row:

    "when was my last oil change"    labelled last_maintenance, model said oil_change_when
    "set a reminder to call my mom"  labelled reminder_update,  model said reminder

That is a tempting conclusion, and it is wrong. It comes from looking only at the
rows the model got wrong, which never shows you the rows that establish what each
class is *for*. Sampling the other class in each pair settles it in one query —
see [the table below](#the-distinctions-being-collapsed). The taxonomy is
deliberate and consistent, and on **372 of the 395** decisions in question the
dataset's label is right and the model's answer is wrong.

Genuine annotation errors come to two class pairs and 23 decisions: 0.1% of the
run, worth 0.0007 of ECE. So the headline numbers stand as measured —
**accuracy 92.63%, ECE 0.0209, 219 errors at `1.000`** — and the interesting
question is what the errors have in common.

## The distinctions being collapsed

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

## A note on method

The "the benchmark is wrong" reading of these errors is easy to reach and hard to
shake, because every piece of evidence supporting it is real. The rows really do
look mislabelled when you read them on their own.

What makes it wrong is the sampling. Inspecting only a model's *errors* shows you
pairs that look interchangeable and never shows you the rows that distinguish
them. The conclusion is a plausible story, assembled from evidence selected after
the fact, that happens to flatter the thing under test — which is reason to check
harder, not to publish.

The check that settles it costs one query: **look at what the other class is used
for.** It is worth running before attributing any error to annotation noise.

## Postscript: what survives full specification

Everything above concerns E1, where classes were described only by their names.
The clean description run (E1c — all 150 classes described by the error-blind
exemplar rule, 7,500 held-out rows) leaves 14 errors at a reported probability
of 1.000. Applying the same discipline as above — check what the dataset uses
the *other* label for before judging — they break down as:

**Five clear gold-label errors, where the model is right:**

| utterance | gold | Jev said | why the gold is wrong |
| --- | --- | --- | --- |
| "allow me to turn on the lights" | `restaurant_reservation` | `smart_home` | indefensible on its face |
| "what is on my to do list" | `reminder` | `todo_list` | `todo_list` rows are literally *"what does my to-do list look like"* |
| "what is my current location" | `share_location` | `current_location` | every `share_location` row names a person to tell; none here |
| "can you call the help desk line for my credit card company" | `replacement_card_duration` | `make_call` | that intent's rows are all "how long until my replacement arrives" |
| "how can i request a new credit card" | `replacement_card_duration` | `new_card` | asks how to request, not how long |

**Two rows of the known `ingredients_list`/`recipe` noise pair**, and roughly
six genuine ambiguities where both labels are defensible ("great talk, thanks"
as `goodbye` vs `thank_you`; "someone used my chase card without my
authorization" as `report_lost_card` vs `report_fraud`; "what time is it in
phoenix" as `timezone` vs `time`). Perhaps one or two are outright model
mistakes.

Two conclusions. First, with classes properly specified, the model's residual
error rate at certainty (0.26%) is pressed against the dataset's own annotation
noise floor — most of what remains is CLINC150's fault or nobody's. Second, the
errors that do remain migrate to where a gate would catch them: on the same
7,500 rows, errors at `1.000` fall from 72 (13% of all errors) to 14 (6%), and
the median confidence on an error drops from 0.76 to 0.69. The model
increasingly fails where it says it is unsure, which is the property a
confidence gate actually needs.

Note the contrast with the main finding of this document, and why the same
method gives opposite verdicts: E1's certainty errors checked out as the
model's fault (372 of 395 gold labels correct); E1c's check out mostly as the
dataset's. The difference is not the method but the model's remaining error
mass — specification fixed the systematic failures, and what is left is the
noise floor.
