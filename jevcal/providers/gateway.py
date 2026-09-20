"""Jev, reached through the Vercel AI Gateway's evaluation API.

Why the gateway and not the direct API: at the time of writing the direct
TypeSafe API is waitlisted and sits behind an early-access agreement, whereas
``typesafe-ai/jev`` is generally available through the gateway under ordinary
terms. Auditing the generally-available endpoint keeps the published numbers
free of any early-access restriction, and is also the endpoint most readers can
reproduce against.

**This is not an OpenAI-compatible model.** Jev is an *evaluation*-modality
model: it does not generate text, and Vercel's documentation states that
evaluation "is not supported through the OpenAI-compatible,
Anthropic-compatible, or Cohere-compatible endpoints". It is reached by posting
a shared ``state`` plus a map of typed ``questions`` to ``/v1/evaluate``, and it
answers with a probability distribution over the permitted options. An earlier
version of this client spoke ``/v1/chat/completions`` and could never have
worked; see ``docs/api-notes.md`` for the verification trail.

Because the distribution is a first-class field of the response rather than
something recovered from token logprobs, extraction is unambiguous: a Choice
answer carries ``probabilities``, a Score answer carries ``probabilities`` keyed
by rung index, and a Boolean answer carries a single ``probability`` that the
proposition is true. If the field is missing the request fails loudly rather
than inventing a confidence.

One thing this client will never do is ask the model to *write* a confidence
number into its output. A self-reported number in generated text is a different
object from a calibrated distribution, and auditing it would answer a different
question than the one on the label.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import requests

from ..types import NOUL_LABELS, Example, Prediction, Task
from .base import ProviderError, normalise_distribution

#: The two surfaces that serve Jev's evaluation API. They are the same model and
#: the same primitives with three cosmetic differences: the path, the name of the
#: yes/no question type (``boolean`` on the gateway, ``noul`` on TypeSafe's own
#: API), and the casing of the ``usage`` keys. Verified live on 2026-09-19.
#:
#: Prefer ``gateway``: ``typesafe-ai/jev`` is generally available there under
#: ordinary terms, whereas the direct API is waitlisted behind an early-access
#: agreement. Check what your agreement permits before publishing numbers taken
#: from the ``typesafe`` surface.
Surface = Literal["gateway", "typesafe"]

SURFACES: dict[str, dict[str, str]] = {
    "gateway": {
        "base_url": "https://ai-gateway.vercel.sh/v1",
        "path": "evaluate",
        "model": "typesafe-ai/jev",
        "noul_type": "boolean",
    },
    "typesafe": {
        "base_url": "https://api.typesafe.ai/v1",
        "path": "systemone",
        "model": "jev-latest",
        "noul_type": "noul",
    },
}

DEFAULT_SURFACE: Surface = "gateway"
DEFAULT_BASE_URL = SURFACES["gateway"]["base_url"]
DEFAULT_MODEL = SURFACES["gateway"]["model"]

#: The question id used for the single decision this harness asks per request.
#: The model never sees it; it is only the key the answer comes back under.
QUESTION_ID = "decision"

#: TypeSafe's published price, confirmed against the gateway's own model card
#: (``pricing.input`` = 0.000000042/token, ``pricing.output`` = 0): $0.042 per
#: million input tokens, output free. Used only as a fallback when the gateway
#: does not report an exact cost; nothing depends on it.
INPUT_USD_PER_MTOK = 0.042

#: Where to read the probability from. ``auto`` picks the field that matches the
#: task kind, which is what you want; the explicit values exist so a run can be
#: pinned after ``probe`` has shown what the endpoint actually returned.
ProbabilitySource = Literal["auto", "probabilities", "probability"]




@dataclass
class GatewayProvider:
    """Client for one typed Jev decision via ``POST /v1/evaluate``."""

    surface: Surface = DEFAULT_SURFACE
    model: str = ""
    base_url: str = ""
    api_key: str | None = None
    probability_source: ProbabilitySource = "auto"
    timeout: float = 30.0
    max_retries: int = 4
    keep_raw: bool = False
    zero_data_retention: bool = False
    extra_body: dict[str, Any] = field(default_factory=dict)
    name: str = "gateway"

    def __post_init__(self) -> None:
        if self.surface not in SURFACES:
            raise ProviderError(f"unknown surface {self.surface!r}; expected one of {sorted(SURFACES)}")
        spec = SURFACES[self.surface]
        self.api_key = self.api_key or os.environ.get("AI_GATEWAY_API_KEY") or os.environ.get("JEV_API_KEY")
        if not self.api_key:
            raise ProviderError(
                "no API key: set AI_GATEWAY_API_KEY (or JEV_API_KEY), or run with --provider simulated"
            )
        self.model = self.model or spec["model"]
        self.base_url = os.environ.get("JEV_BASE_URL", self.base_url or spec["base_url"]).rstrip("/")
        self._path = spec["path"]
        self._noul_type = spec["noul_type"]
        self._session = requests.Session()

    def question_type(self, kind: str) -> str:
        return self._noul_type if kind == "noul" else kind

    # ---------------------------------------------------------------- request

    def build_question(self, task: Task) -> dict[str, Any]:
        """Build the typed question object for this task."""
        question: dict[str, Any] = {
            "type": self.question_type(task.kind),
            "instructions": _instructions(task),
        }
        criteria = build_criteria(task)
        if criteria is not None:
            question["criteria"] = criteria
        return question

    def build_payload(self, task: Task, example: Example) -> dict[str, Any]:
        """Build the request body for one typed decision."""
        payload: dict[str, Any] = {
            "model": self.model,
            "state": example.text,
            "questions": {QUESTION_ID: self.build_question(task)},
        }
        if self.zero_data_retention:
            payload["providerOptions"] = {"gateway": {"zeroDataRetention": True}}
        payload.update(self.extra_body)
        return payload

    def raw_call(self, task: Task, example: Example) -> dict[str, Any]:
        """One HTTP round trip, with retries on transient failures."""
        url = f"{self.base_url}/{self._path}"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = self.build_payload(task, example)

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._session.post(url, headers=headers, json=payload, timeout=self.timeout)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise _Transient(f"HTTP {resp.status_code}: {resp.text[:300]}")
                if resp.status_code >= 400:
                    # 4xx other than rate limiting is a bug in the request, not
                    # weather. Retrying it just burns quota.
                    raise ProviderError(f"HTTP {resp.status_code}: {resp.text[:500]}")
                return resp.json()
            except (_Transient, requests.RequestException) as exc:
                last_error = exc
                if attempt == self.max_retries - 1:
                    break
                time.sleep(2.0**attempt)
        raise ProviderError(f"request failed after {self.max_retries} attempts: {last_error}")

    def predict(self, task: Task, example: Example) -> Prediction:
        started = time.perf_counter()
        try:
            body = self.raw_call(task, example)
        except ProviderError as exc:
            return Prediction(example_id=example.id, predicted=None, confidence=None, error=str(exc))
        latency_ms = (time.perf_counter() - started) * 1000.0

        try:
            predicted, distribution = extract_decision(body, task, self.probability_source)
            reported = _reported_confidence(body)
        except ExtractionError as exc:
            return Prediction(
                example_id=example.id,
                predicted=None,
                confidence=None,
                latency_ms=latency_ms,
                error=str(exc),
                raw=body if self.keep_raw else None,
            )

        return Prediction(
            example_id=example.id,
            predicted=predicted,
            confidence=distribution[predicted],
            distribution=distribution,
            latency_ms=latency_ms,
            cost_usd=_cost_usd(body),
            reported_confidence=reported,
            model=str(body.get("model", self.model)),
            raw=body if self.keep_raw else None,
        )


class _Transient(Exception):
    """Retryable failure."""


class ExtractionError(ProviderError):
    """The response did not contain a usable probability distribution."""


def _instructions(task: Task) -> str:
    """The question text shown to the model.

    The option names and their descriptions travel in ``criteria``, so unlike a
    chat prompt this does not restate them: duplicating the label list here
    would just spend tokens.
    """
    lines = [task.question.strip()]
    if task.instructions:
        lines.append(task.instructions.strip())
    return "\n".join(lines)


def build_criteria(task: Task) -> dict[str, str] | list[str] | None:
    """The ``criteria`` field for this task.

    Choice takes a map of option name to description; Score takes the rungs as
    an ordered list, lowest first; Boolean takes an optional ``true``/``false``
    map. When a task spec supplies no descriptions we send the label itself,
    with underscores turned into spaces -- a faithful rendering of the label
    rather than an invented gloss. Descriptions carry real signal, so a task
    that has them should say so in its spec and the writeup should note it.
    """
    described = dict(task.criteria or {})
    if task.kind == "score":
        return [described.get(label, _humanise(label)) for label in task.labels]
    if task.kind == "noul":
        if not described:
            return None
        return {label: described.get(label, _humanise(label)) for label in NOUL_LABELS}
    return {label: described.get(label, _humanise(label)) for label in task.labels}


def _humanise(label: str) -> str:
    return label.replace("_", " ").strip()


# ------------------------------------------------------------------ extraction


def extract_decision(
    body: dict[str, Any], task: Task, source: ProbabilitySource = "auto"
) -> tuple[str, dict[str, float]]:
    """Pull the chosen label and its probability distribution out of a response.

    The evaluation API answers under ``answers.<question id>``:

    ``probabilities``
        Choice and Score answers carry the full distribution. Choice keys it by
        option name; Score keys it by rung *index* as a string (``"0"``,
        ``"1"``, ...), which we map back onto the rubric by position.
    ``probability``
        A Boolean answer carries a single number: the probability that the
        proposition is true. We widen it to ``{"true": p, "false": 1 - p}`` so
        that every task kind reaches the metrics as a distribution.

    Raises :class:`ExtractionError` rather than inventing a confidence, so a
    silent mapping change shows up as failed rows instead of fake numbers.
    """
    answer = _answer(body)
    wanted = _expected_source(task) if source == "auto" else source
    if wanted == "probability":
        dist = _from_probability(answer, task)
    else:
        dist = _from_probabilities(answer, task)

    predicted = max(dist, key=lambda k: dist[k])
    # The response names its own pick. If it disagrees with the argmax of the
    # distribution we were given, something is wrong with our reading of the
    # response and a silent mismatch would corrupt every accuracy number.
    stated = answer.get("choice")
    if isinstance(stated, str) and stated in dist and stated != predicted:
        raise ExtractionError(
            f"response chose {stated!r} but its distribution peaks at {predicted!r}: {dist}"
        )
    return predicted, dist


def _expected_source(task: Task) -> ProbabilitySource:
    return "probability" if task.kind == "noul" else "probabilities"


def _answer(body: dict[str, Any]) -> dict[str, Any]:
    answers = body.get("answers")
    if not isinstance(answers, dict):
        raise ExtractionError(
            "response has no 'answers' object. This endpoint is /v1/evaluate; a "
            "chat-completions style body means the request went somewhere else."
        )
    answer = answers.get(QUESTION_ID)
    if not isinstance(answer, dict):
        raise ExtractionError(
            f"no answer under {QUESTION_ID!r}; response carried {sorted(answers)}"
        )
    return answer


def _from_probabilities(answer: dict[str, Any], task: Task) -> dict[str, float]:
    raw = answer.get("probabilities")
    if not isinstance(raw, dict):
        raise ExtractionError(f"answer has no 'probabilities' map: {answer}")
    if task.kind == "score":
        raw = _index_keys_to_labels(raw, task)
    return _safe_normalise(raw, task)


def _index_keys_to_labels(raw: dict[str, Any], task: Task) -> dict[str, Any]:
    """Score answers key the distribution by rung index, not by rung text."""
    if set(raw) & set(task.labels):
        return raw
    mapped: dict[str, Any] = {}
    for key, value in raw.items():
        try:
            index = int(key)
        except (TypeError, ValueError) as exc:
            raise ExtractionError(f"score probabilities have a non-index key {key!r}") from exc
        if not 0 <= index < len(task.labels):
            raise ExtractionError(
                f"score probability index {index} is outside the {len(task.labels)}-rung rubric"
            )
        mapped[task.labels[index]] = value
    return mapped


def _from_probability(answer: dict[str, Any], task: Task) -> dict[str, float]:
    # ``probability`` on the gateway's evaluation API, ``noul`` on TypeSafe's
    # own. Both carry the same thing: P(the proposition is true).
    value = None
    for key in ("probability", "noul"):
        if key in answer:
            value = answer[key]
            break
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ExtractionError(f"yes/no answer has no numeric 'probability'/'noul': {answer}")
    p = float(value)
    if not 0.0 <= p <= 1.0:
        raise ExtractionError(f"boolean probability {p} is outside [0, 1]")
    return _safe_normalise({"true": p, "false": 1.0 - p}, task)


def _reported_confidence(body: dict[str, Any]) -> float | None:
    """Jev's own ``confidence`` statistic for this answer, if it returned one.

    Choice and Score answers carry it; yes/no answers do not. See the note on
    :attr:`~jevcal.types.Prediction.reported_confidence` for why it is kept
    apart from the probability we actually audit.
    """
    try:
        value = body["answers"][QUESTION_ID].get("confidence")
    except (KeyError, TypeError, AttributeError):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _safe_normalise(dist: dict[str, Any], task: Task) -> dict[str, float]:
    numeric = {
        str(k): float(v)
        for k, v in dist.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    if not numeric:
        raise ExtractionError(f"distribution has no numeric values: {dist}")
    try:
        return normalise_distribution(numeric, task.labels)
    except ValueError as exc:
        raise ExtractionError(str(exc)) from exc


def _cost_usd(body: dict[str, Any]) -> float | None:
    """What the request cost.

    The gateway reports an exact figure under ``providerMetadata.gateway.cost``;
    prefer it over our own arithmetic, which is only a fallback and only covers
    input tokens.
    """
    metadata = body.get("providerMetadata")
    if isinstance(metadata, dict):
        gateway = metadata.get("gateway")
        if isinstance(gateway, dict) and gateway.get("cost") is not None:
            try:
                return float(gateway["cost"])
            except (TypeError, ValueError):
                pass

    usage = body.get("usage")
    if not isinstance(usage, dict):
        return None
    # camelCase on the gateway's evaluation API, snake_case on TypeSafe's own.
    for key in ("inputTokens", "input_tokens", "prompt_tokens"):
        tokens = usage.get(key)
        if tokens is not None:
            return float(tokens) * INPUT_USD_PER_MTOK / 1_000_000
    return None
