"""Jev, reached through the Vercel AI Gateway.

Why the gateway and not the direct API: at the time of writing the direct
TypeSafe API is waitlisted and sits behind an early-access agreement, whereas
``typesafe-ai/jev`` is generally available through the gateway under ordinary
terms. Auditing the generally-available endpoint keeps the published numbers
free of any early-access restriction, and is also the endpoint most readers can
reproduce against.

A note on the confidence field, because it is the entire point of this repo:
Jev natively returns a probability distribution over the permitted options. The
gateway is OpenAI-compatible, so that distribution can surface in more than one
place depending on how the gateway maps it. This client therefore tries several
documented extraction paths and lets you pin one with ``probability_source``.
Run ``python -m jevcal probe`` once against your key: it prints the raw response
body so you can see which path your gateway version actually uses, and fails
loudly rather than guessing if none of them match.

What this client will never do is ask the model to *write* a confidence number
into its output. A self-reported number in generated text is a different object
from a calibrated distribution, and auditing it would answer a different
question than the one on the label.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import requests

from ..types import Example, Prediction, Task
from .base import ProviderError, normalise_distribution

DEFAULT_BASE_URL = "https://ai-gateway.vercel.sh/v1"
DEFAULT_MODEL = "typesafe-ai/jev"

#: TypeSafe's published price at launch: $0.042 per million input tokens,
#: output free. Used only to report what a run cost; nothing depends on it.
INPUT_USD_PER_MTOK = 0.042

ProbabilitySource = Literal["auto", "body", "logprobs", "content"]


@dataclass
class GatewayProvider:
    """OpenAI-compatible client for a typed Jev decision."""

    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    api_key: str | None = None
    probability_source: ProbabilitySource = "auto"
    timeout: float = 30.0
    max_retries: int = 4
    request_logprobs: bool = True
    keep_raw: bool = False
    extra_body: dict[str, Any] = field(default_factory=dict)
    name: str = "gateway"

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("AI_GATEWAY_API_KEY") or os.environ.get("JEV_API_KEY")
        if not self.api_key:
            raise ProviderError(
                "no API key: set AI_GATEWAY_API_KEY (or JEV_API_KEY), or run with --provider simulated"
            )
        self.base_url = os.environ.get("JEV_BASE_URL", self.base_url).rstrip("/")
        self._session = requests.Session()

    # ---------------------------------------------------------------- request

    def build_payload(self, task: Task, example: Example) -> dict[str, Any]:
        """Build the request body for one typed decision."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _system_prompt(task)},
                {"role": "user", "content": example.text},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "decision",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string", "enum": list(task.labels)}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                },
            },
        }
        if self.request_logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = min(len(task.labels), 20)
        payload.update(self.extra_body)
        return payload

    def raw_call(self, task: Task, example: Example) -> dict[str, Any]:
        """One HTTP round trip, with retries on transient failures."""
        url = f"{self.base_url}/chat/completions"
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
            model=str(body.get("model", self.model)),
            raw=body if self.keep_raw else None,
        )


class _Transient(Exception):
    """Retryable failure."""


class ExtractionError(ProviderError):
    """The response did not contain a usable probability distribution."""


def _system_prompt(task: Task) -> str:
    lines = [task.question.strip()]
    if task.instructions:
        lines.append(task.instructions.strip())
    if task.kind == "score":
        lines.append("Answer with one rung of this ordered rubric, lowest first:")
    elif task.kind == "noul":
        lines.append("Answer 'true' or 'false'.")
    else:
        lines.append("Answer with exactly one of these options:")
    if task.kind != "noul":
        lines.append(", ".join(task.labels))
    return "\n".join(lines)


# ------------------------------------------------------------------ extraction


def extract_decision(
    body: dict[str, Any], task: Task, source: ProbabilitySource = "auto"
) -> tuple[str, dict[str, float]]:
    """Pull the chosen label and its probability distribution out of a response.

    Tried in order (``auto``), or exclusively (when pinned):

    ``body``
        A native distribution on the response or choice object, under any of
        ``probabilities`` / ``distribution`` / ``decision.probabilities``. This
        is the path to prefer when the gateway passes Jev's own output through.
    ``logprobs``
        OpenAI-style ``top_logprobs`` on the first content token, exponentiated
        and renormalised over the task's labels.
    ``content``
        A JSON object in the message content carrying a ``probabilities`` map.

    Raises :class:`ExtractionError` rather than inventing a confidence, so a
    silent mapping change shows up as failed rows instead of fake numbers.
    """
    order: tuple[ProbabilitySource, ...] = (
        ("body", "logprobs", "content") if source == "auto" else (source,)
    )
    errors: list[str] = []
    for path in order:
        try:
            dist = _EXTRACTORS[path](body, task)
        except ExtractionError as exc:
            errors.append(f"{path}: {exc}")
            continue
        if dist:
            predicted = max(dist, key=lambda k: dist[k])
            return predicted, dist
    raise ExtractionError(
        "no probability distribution found in the response ("
        + "; ".join(errors)
        + "). Run `python -m jevcal probe` to inspect the raw body, then pin "
        "--probability-source."
    )


def _from_body(body: dict[str, Any], task: Task) -> dict[str, float]:
    containers: list[Any] = [body]
    choices = body.get("choices") or []
    if choices:
        containers.extend([choices[0], choices[0].get("message", {})])
    for container in containers:
        if not isinstance(container, dict):
            continue
        for key in ("probabilities", "distribution", "probs"):
            value = container.get(key)
            if isinstance(value, dict):
                return _safe_normalise(value, task)
        decision = container.get("decision")
        if isinstance(decision, dict):
            for key in ("probabilities", "distribution", "probs"):
                value = decision.get(key)
                if isinstance(value, dict):
                    return _safe_normalise(value, task)
    raise ExtractionError("no distribution field on the response body")


def _from_logprobs(body: dict[str, Any], task: Task) -> dict[str, float]:
    choices = body.get("choices") or []
    if not choices:
        raise ExtractionError("response has no choices")
    logprobs = (choices[0].get("logprobs") or {}).get("content") or []
    if not logprobs:
        raise ExtractionError("no logprobs on the response")

    # Find the first token position that discriminates between the task labels.
    # With a JSON-schema-constrained answer the leading tokens are structural
    # ('{', '"answer"', ':'), and carry no information about the decision.
    for token_info in logprobs:
        tops = token_info.get("top_logprobs") or []
        scores: dict[str, float] = {}
        for entry in tops:
            token = str(entry.get("token", "")).strip().strip('"')
            if not token:
                continue
            matches = [l for l in task.labels if l == token or l.startswith(token)]
            if len(matches) != 1:
                continue
            label = matches[0]
            # Several tokens can map to the same label (casing, leading space);
            # probabilities of disjoint token events add.
            scores[label] = scores.get(label, 0.0) + math.exp(float(entry["logprob"]))
        if len(scores) >= 2 or (len(scores) == 1 and len(task.labels) == 1):
            return _safe_normalise(scores, task)
    raise ExtractionError("no token position discriminated between the task labels")


def _from_content(body: dict[str, Any], task: Task) -> dict[str, float]:
    choices = body.get("choices") or []
    if not choices:
        raise ExtractionError("response has no choices")
    content = (choices[0].get("message") or {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise ExtractionError("message content is empty")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ExtractionError(f"message content is not JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ExtractionError("message content is not a JSON object")
    for key in ("probabilities", "distribution", "probs"):
        value = parsed.get(key)
        if isinstance(value, dict):
            return _safe_normalise(value, task)
    raise ExtractionError("message content has no probabilities map")


_EXTRACTORS = {"body": _from_body, "logprobs": _from_logprobs, "content": _from_content}


def _safe_normalise(dist: dict[str, Any], task: Task) -> dict[str, float]:
    numeric = {str(k): float(v) for k, v in dist.items() if isinstance(v, (int, float))}
    if not numeric:
        raise ExtractionError(f"distribution has no numeric values: {dist}")
    try:
        return normalise_distribution(numeric, task.labels)
    except ValueError as exc:
        raise ExtractionError(str(exc)) from exc


def _cost_usd(body: dict[str, Any]) -> float | None:
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
    if prompt_tokens is None:
        return None
    return float(prompt_tokens) * INPUT_USD_PER_MTOK / 1_000_000
