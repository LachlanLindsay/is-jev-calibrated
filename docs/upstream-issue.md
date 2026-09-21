Title: 12 mislabelled rows in data_full.json, with evidence from the dataset's own usage

While running an independent calibration audit against CLINC150 (https://github.com/LachlanLindsay/is-jev-calibrated), the
audited model's residual disagreements with the gold labels turned out to
include rows where the gold label contradicts the dataset's own usage. Example:

> "what is on my to do list" is labelled `reminder`, while `todo_list`'s own
> rows include "what does my to-do list look like" and "list my to-do list".

Each row below was checked against the dataset's own utterances for both
labels before being listed — the full list with per-row evidence is published
as machine-readable errata:

- https://github.com/LachlanLindsay/is-jev-calibrated/blob/main/results/label-errata.md
- https://github.com/LachlanLindsay/is-jev-calibrated/blob/main/results/label-errata.jsonl

| utterance | labelled | should be |
| --- | --- | --- |
| allow me to turn on the lights | `restaurant_reservation` | `smart_home` |
| what is on my to do list | `reminder` | `todo_list` |
| what is my current location | `share_location` | `current_location` |
| can you call the help desk line for my credit card company | `replacement_card_duration` | `make_call` |
| how can i request a new credit card | `replacement_card_duration` | `new_card` |
| give me a recipe for tacos | `ingredients_list` | `recipe` |
| what's a good recipe foe tacos | `ingredients_list` | `recipe` |
| show me the recipe for broccoli cheddar soup | `ingredients_list` | `recipe` |
| what kind of pto do i have left | `pto_used` | `pto_balance` |
| how much paid time off do i have left | `pto_used` | `pto_balance` |
| how many days of pto are available | `pto_used` | `pto_balance` |
| how much of my pto is left | `pto_used` | `pto_balance` |

12 rows out of 22,500 in-scope utterances (~0.05%) —
which speaks well of the dataset's overall quality.

We are deliberately **not** proposing changes to `data_full.json`: published
results depend on the file as-is, and errata-on-the-side seems the right
pattern for a frozen benchmark. Filing this so the list is discoverable from
the source; happy to send a PR adding a README pointer instead if you'd prefer.
