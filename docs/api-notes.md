# What the API actually looks like

The handover flagged the request/response shape as "the single highest-risk
assumption in the repo". It was wrong, and in a way that could not have been
patched by loosening the extraction. This is the record of what was checked, how,
and what it means for the audit. Everything below was verified on 2026-09-19
against the live service and the vendors' own documentation.

## Jev is not an OpenAI-compatible chat model

The previous client posted an OpenAI-style `messages` / `response_format`
request to `https://ai-gateway.vercel.sh/v1/chat/completions` and recovered a
distribution from `top_logprobs` or from JSON in the message content. No part of
that could have worked.

`typesafe-ai/jev` is an **evaluation**-modality model. The gateway's own model
card says so:

```json
{
  "id": "typesafe-ai/jev", "type": "evaluation", "max_tokens": 0,
  "supported_specifications": ["v4"],
  "pricing": { "input": "0.000000042", "output": "0" }
}
```

`max_tokens: 0` is the tell: it does not generate text, so there are no output
tokens to carry logprobs and no message content to parse. Vercel's evaluation
documentation states the restriction directly — evaluation "is not supported
through the OpenAI-compatible, Anthropic-compatible, or Cohere-compatible
endpoints."

A chat-completions body now raises `ExtractionError` with a message that says
where the request should have gone, and a test
(`test_a_chat_completions_body_fails_loudly`) pins that it can never parse as a
decision.

## The two surfaces that do work

Same model, same primitives. Three cosmetic differences: the path, the name of
the yes/no question type, and the casing of the `usage` keys.

| | Vercel AI Gateway | TypeSafe direct |
| --- | --- | --- |
| endpoint | `POST https://ai-gateway.vercel.sh/v1/evaluate` | `POST https://api.typesafe.ai/v1/systemone` |
| model id | `typesafe-ai/jev` | `jev-latest` |
| yes/no type | `boolean`, answered as `probability` | `noul`, answered as `noul` |
| usage keys | `inputTokens` / `outputTokens` | `input_tokens` / `output_tokens` |
| exact cost | `providerMetadata.gateway.cost` | not reported; computed from tokens |
| availability | generally available | waitlisted, early-access agreement |

Both take a shared `state` plus a map of typed `questions`, and answer under
`answers.<question id>`. `--surface` selects between them; the client defaults to
the gateway, because that is the one whose terms let results be published
without checking an agreement first.

**The runs in this repo used the `typesafe` surface**, because that is what the
available key authenticated against. See "Terms" below.

## The three answer shapes, as returned

Captured live from `/v1/systemone`:

```jsonc
// choice
{"type":"choice","choice":"returns","confidence":1.0,
 "probabilities":{"billing":0.0,"shipping":0.0,"returns":1.0}}

// score -- probabilities are keyed by rung INDEX, not rung text
{"type":"score","score":0.18,"confidence":0.73,
 "legend":{"0":"Calm","1":"Mildly annoyed","2":"Very angry"},
 "probabilities":{"0":0.82,"1":0.18,"2":0.0}}

// noul -- a bare scalar, no distribution and no confidence
{"type":"noul","noul":0.97}
```

A Noul is widened to `{"true": p, "false": 1 - p}` so every task kind reaches the
metrics as a distribution.

## `confidence` is not the probability of the answer

This is the one that would most easily have produced a wrong headline.

TypeSafe returns `confidence` on Choice and Score answers, and its documentation
tells users to gate on it. It is **not** the probability assigned to the chosen
option. It is a measure of how peaked the whole distribution is — one minus its
normalised entropy. From the documented example, probabilities
`{0.84, 0.159, 0.001}` are reported with `confidence` 0.596, and

```
1 - H/ln(3) = 1 - 0.4457/1.0986 = 0.5943
```

which is 0.596 up to the rounding in the published probabilities.

So `confidence` answers "how concentrated is this distribution?", not "how likely
is this answer to be right?". **Plotting it against the reliability diagonal
would be a category error** — it has no reason to sit on the diagonal even for a
perfectly calibrated model, and a model could score badly on it while being
flawlessly calibrated.

This audit therefore puts `max(probabilities)` on the reliability diagram, which
is the quantity that should equal the accuracy. Jev's own `confidence` is
recorded alongside it, as `Prediction.reported_confidence`, so the two can be
compared — but it is never the thing being calibrated.

## Probabilities are quantised to 0.01

Every probability comes back rounded to two decimal places, and the rounded
values sum to exactly 1. A deliberately ambiguous 20-option request returned
`{0.0, 0.01, 0.02, 0.04, 0.76}` and nothing between.

Two consequences for the numbers in this repo:

- **The top of the scale cannot be resolved.** Above 0.98 there are exactly two
  reportable values, 0.99 and 1.00. Since most of the mass lands there, the
  reliability diagram's top bin is an instrument limit, not a finding about the
  model. A gap there is real; its *shape* is not measurable.
- **NLL is not meaningful on the 150-way tasks.** When the model is wrong the
  true class is usually reported at exactly 0.00, and `negative_log_likelihood`
  clips at `eps=1e-12`, so each such row contributes `-log(1e-12) ≈ 27.6`. The
  resulting mean is a function of the clip constant and the error rate, not of
  the model's density. It is reported for completeness and should not be quoted.

Brier, ECE, MCE and AUROC are unaffected: they depend on the top probability and
on correctness, both of which survive rounding at this resolution.

## Cost and throughput, measured

The handover estimated "roughly 300 input tokens" per request and "well under a
dollar" for the in-scope set. Both are wrong, because 150 option names and
descriptions travel with every single request:

- **2,496 input tokens** for one 150-option Choice, not ~300.
- **$0.000105 per decision**, so E1's 22,500 rows cost about **$2.36**, and all
  three experiments together about **$5**.
- **~40 decisions/second** at `--concurrency 8`, around 0.23 s per request.

Concurrency is left at 8: TypeSafe's own cookbooks note the public endpoint
rate-limits above roughly eight workers, so the handover's suggested 16 is too
high.

Pricing is confirmed twice over — the gateway model card prices input at
`0.000000042`/token with output free, and the worked example in Vercel's docs
(275 input tokens → `"cost": "0.00001155"`) is exactly `275 x 0.042 / 1e6`.

## Terms

The handover's reason for preferring the gateway was that `typesafe-ai/jev` is
generally available there, outside the waitlisted direct API's early-access
agreement. The key available for this run authenticates only against the direct
API, so the numbers here were taken from `api.typesafe.ai`.

TypeSafe's public terms at <https://typesafe.ai/terms> contain no clause
restricting the publication of benchmark or evaluation results — there is no
mention of benchmarking, publication, or early access anywhere in them. That is
not the same as clearance: an early-access agreement accepted separately could
say otherwise, and only the account holder can check that. **Confirm it before
publishing these numbers.**

`jev-latest` resolved to **`jev-1.13.0`** throughout.
