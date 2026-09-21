# CLINC150 label errata

12 rows of CLINC150 (Larson et al., EMNLP 2019, CC BY-SA 3.0) whose
gold label contradicts the dataset's **own usage** of the labels involved.
Surfaced by [this calibration audit](https://github.com/LachlanLindsay/is-jev-calibrated) once the audited model's error
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

## Evidence

**"allow me to turn on the lights"** — labelled `restaurant_reservation`, should be
`smart_home`. Nothing about the utterance concerns a restaurant or a reservation; indefensible on its face. The dataset's own `restaurant_reservation` rows:
"you need to call the melt in lakewood for a table for at 2:00, monday, the first week in march" / "hey computer, make a reservation for 2 for my parents at red lobster on wednseday next, 2:00pm if possible" / "i think we should reserve dad's regular table friday for 6 people, use his name, make that close to 4:00 pm at the club".

**"what is on my to do list"** — labelled `reminder`, should be
`todo_list`. todo_list's own rows read 'what does my to-do list look like' and 'list my to-do list'. The dataset's own `reminder` rows:
"tell me my reminders" / "tell me my current reminders list" / "please read my reminders".

**"what is my current location"** — labelled `share_location`, should be
`current_location`. Every share_location row names a person to send the location to; none is named here. current_location's rows read 'please tell me the address of my current location'. The dataset's own `share_location` rows:
"tell lydia and laura where i am located" / "share my location with brad and terry" / "tell john and sam where i am".

**"can you call the help desk line for my credit card company"** — labelled `replacement_card_duration`, should be
`make_call`. replacement_card_duration's rows all ask how long a replacement card takes to arrive; this asks to place a call, which is what make_call's rows do. The dataset's own `replacement_card_duration` rows:
"tell me about today weather" / "can my debit card be replaced quickly" / "how long does visa take to get me a new card".

**"how can i request a new credit card"** — labelled `replacement_card_duration`, should be
`new_card`. Asks how to request a card, not how long a replacement takes; new_card's rows ask exactly this. The dataset's own `replacement_card_duration` rows:
"tell me about today weather" / "can my debit card be replaced quickly" / "how long does visa take to get me a new card".

**"give me a recipe for tacos"** — labelled `ingredients_list`, should be
`recipe`. Asks for a recipe; the dataset's recipe rows read the same way ('find a recipe for baked ziti'), while ingredients_list rows ask what ingredients are needed. The dataset's own `ingredients_list` rows:
"what do you put in chicken noodle casserole" / "what goes into chicken noodle casserole" / "what do i need to make chicken noodle casserole".

**"what's a good recipe foe tacos"** — labelled `ingredients_list`, should be
`recipe`. Asks for a recipe; the dataset's recipe rows read the same way ('find a recipe for baked ziti'), while ingredients_list rows ask what ingredients are needed. The dataset's own `ingredients_list` rows:
"what do you put in chicken noodle casserole" / "what goes into chicken noodle casserole" / "what do i need to make chicken noodle casserole".

**"show me the recipe for broccoli cheddar soup"** — labelled `ingredients_list`, should be
`recipe`. Asks for a recipe; the dataset's recipe rows read the same way ('find a recipe for baked ziti'), while ingredients_list rows ask what ingredients are needed. The dataset's own `ingredients_list` rows:
"what do you put in chicken noodle casserole" / "what goes into chicken noodle casserole" / "what do i need to make chicken noodle casserole".

**"what kind of pto do i have left"** — labelled `pto_used`, should be
`pto_balance`. Asks how much time off remains; pto_balance's own rows read 'how many pto days do i have left', while pto_used's rows ask how much has been taken. The dataset's own `pto_used` rows:
"have i used half of my days off yet" / "go to my job portal, i'd like to know how many days off i have used" / "what amount of days off have i already utilized".

**"how much paid time off do i have left"** — labelled `pto_used`, should be
`pto_balance`. Asks how much time off remains; pto_balance's own rows read 'how many pto days do i have left', while pto_used's rows ask how much has been taken. The dataset's own `pto_used` rows:
"have i used half of my days off yet" / "go to my job portal, i'd like to know how many days off i have used" / "what amount of days off have i already utilized".

**"how many days of pto are available"** — labelled `pto_used`, should be
`pto_balance`. Asks how much time off remains; pto_balance's own rows read 'how many pto days do i have left', while pto_used's rows ask how much has been taken. The dataset's own `pto_used` rows:
"have i used half of my days off yet" / "go to my job portal, i'd like to know how many days off i have used" / "what amount of days off have i already utilized".

**"how much of my pto is left"** — labelled `pto_used`, should be
`pto_balance`. Asks how much time off remains; pto_balance's own rows read 'how many pto days do i have left', while pto_used's rows ask how much has been taken. The dataset's own `pto_used` rows:
"have i used half of my days off yet" / "go to my job portal, i'd like to know how many days off i have used" / "what amount of days off have i already utilized".
