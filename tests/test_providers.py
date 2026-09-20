"""Provider behaviour: the simulator, and the response parsing for the real one."""

from __future__ import annotations

import math

import pytest

from jevcal.datasets import synthetic_task
from jevcal.providers import build_provider
from jevcal.providers.base import ProviderError, normalise_distribution
from jevcal.providers.gateway import ExtractionError, extract_decision
from jevcal.providers.simulated import SimulatedProvider
from jevcal.types import Example, Task


@pytest.fixture
def choice_task() -> Task:
    return synthetic_task(n=20, kind="choice", n_labels=3, seed=1)


def test_simulated_provider_is_deterministic_for_a_seed(choice_task):
    a = SimulatedProvider(seed=11)
    b = SimulatedProvider(seed=11)
    example = choice_task.examples[0]
    assert a.predict(choice_task, example) == b.predict(choice_task, example)


def test_a_different_seed_gives_a_different_model(choice_task):
    results = {
        SimulatedProvider(seed=s).predict(choice_task, choice_task.examples[0]).confidence
        for s in range(6)
    }
    assert len(results) > 1


def test_simulated_predictions_are_well_formed(choice_task):
    provider = SimulatedProvider(seed=2)
    for example in choice_task.examples:
        p = provider.predict(choice_task, example)
        assert p.ok
        assert p.predicted in choice_task.labels
        assert 0 < p.confidence <= 1
        assert sum(p.distribution.values()) == pytest.approx(1.0)
        # The reported confidence is the mass on the label it actually picked.
        assert max(p.distribution, key=p.distribution.get) == p.predicted
        assert p.distribution[p.predicted] == pytest.approx(p.confidence)


def test_a_score_model_misses_by_one_rung_more_often_than_by_four():
    task = synthetic_task(n=3000, kind="score", n_labels=5, seed=3)
    provider = SimulatedProvider(seed=3)
    distances = []
    for example in task.examples:
        p = provider.predict(task, example)
        distances.append(abs(task.label_index(p.predicted) - task.label_index(example.label)))
    assert distances.count(1) > distances.count(4)


def test_confidence_never_drops_below_chance(choice_task):
    provider = SimulatedProvider(seed=4)
    chance = 1 / len(choice_task.labels)
    assert all(
        provider.predict(choice_task, e).confidence >= chance - 1e-9 for e in choice_task.examples
    )


def test_unknown_provider_names_are_refused():
    with pytest.raises(ProviderError):
        build_provider("gpt5")


def test_gateway_requires_a_key(monkeypatch):
    monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="no API key"):
        build_provider("gateway")


# ------------------------------------------------------------ response parsing


@pytest.fixture
def task() -> Task:
    return Task(
        name="t",
        kind="choice",
        question="q",
        labels=("positive", "negative"),
        examples=(Example(id="1", text="x", label="positive"),),
    )


def _body(answer: dict, **extra) -> dict:
    """A response in the shape both evaluation surfaces actually return.

    Captured from live calls to api.typesafe.ai/v1/systemone on 2026-09-19; the
    gateway's /v1/evaluate differs only in the usage key casing and in calling a
    noul a boolean.
    """
    return {"model": "jev-1.13.0", "answers": {"decision": answer},
            "usage": {"input_tokens": 359, "output_tokens": 38}, **extra}


def test_a_choice_distribution_is_read_from_the_answer(task):
    predicted, dist = extract_decision(
        _body({"type": "choice", "choice": "positive", "confidence": 0.64,
               "probabilities": {"positive": 0.82, "negative": 0.18}}), task)
    assert predicted == "positive"
    assert dist["positive"] == pytest.approx(0.82)


def test_a_score_distribution_is_keyed_by_rung_index_not_rung_text():
    rubric = Task(name="t", kind="score", question="q",
                  labels=("low", "medium", "high"),
                  examples=(Example(id="1", text="x", label="low"),))
    predicted, dist = extract_decision(
        _body({"type": "score", "score": 0.18, "confidence": 0.73,
               "legend": {"0": "low", "1": "medium", "2": "high"},
               "probabilities": {"0": 0.82, "1": 0.18, "2": 0.0}}), rubric)
    assert predicted == "low"
    assert dist == pytest.approx({"low": 0.82, "medium": 0.18, "high": 0.0})


def test_a_score_index_outside_the_rubric_is_an_error():
    rubric = Task(name="t", kind="score", question="q", labels=("low", "high"),
                  examples=(Example(id="1", text="x", label="low"),))
    with pytest.raises(ExtractionError, match="outside"):
        extract_decision(_body({"type": "score", "probabilities": {"0": 0.5, "7": 0.5}}), rubric)


def test_a_noul_scalar_is_widened_to_a_two_sided_distribution():
    """TypeSafe's own API names it 'noul'; the gateway names it 'probability'."""
    gate = Task(name="t", kind="noul", question="q", labels=("true", "false"),
                examples=(Example(id="1", text="x", label="true"),))
    for key in ("noul", "probability"):
        predicted, dist = extract_decision(_body({"type": "noul", key: 0.97}), gate)
        assert predicted == "true"
        assert dist == pytest.approx({"true": 0.97, "false": 0.03})


def test_a_noul_below_a_half_predicts_false():
    gate = Task(name="t", kind="noul", question="q", labels=("true", "false"),
                examples=(Example(id="1", text="x", label="true"),))
    predicted, dist = extract_decision(_body({"type": "noul", "noul": 0.02}), gate)
    assert predicted == "false"
    assert dist["false"] == pytest.approx(0.98)


def test_a_chat_completions_body_fails_loudly(task):
    """The shape this client used to send. It must never parse as a decision."""
    body = {"choices": [{"message": {"content": '{"answer":"positive"}'},
                         "logprobs": {"content": []}}]}
    with pytest.raises(ExtractionError, match="answers"):
        extract_decision(body, task)


def test_a_response_with_no_distribution_fails_loudly(task):
    with pytest.raises(ExtractionError):
        extract_decision(_body({"type": "choice", "choice": "positive"}), task)


def test_a_distribution_that_disagrees_with_the_stated_choice_is_an_error(task):
    """A mismatch means we are misreading the response, not that Jev is wrong."""
    with pytest.raises(ExtractionError, match="peaks at"):
        extract_decision(
            _body({"type": "choice", "choice": "negative",
                   "probabilities": {"positive": 0.82, "negative": 0.18}}), task)


def test_a_distribution_is_restricted_to_the_task_labels_and_renormalised(task):
    _, dist = extract_decision(
        _body({"type": "choice", "choice": "positive",
               "probabilities": {"positive": 0.6, "negative": 0.2, "unrelated": 0.2}}), task)
    assert sum(dist.values()) == pytest.approx(1.0)
    assert dist["positive"] == pytest.approx(0.75)


def test_a_distribution_with_no_mass_on_the_labels_is_an_error(task):
    with pytest.raises(ExtractionError):
        extract_decision(
            _body({"type": "choice", "probabilities": {"other": 1.0}}), task)


def test_pinning_a_source_does_not_silently_fall_through(task):
    """A pinned source must not quietly read the other field instead."""
    with pytest.raises(ExtractionError):
        extract_decision(
            _body({"type": "choice", "probabilities": {"positive": 0.8, "negative": 0.2}}),
            task, source="probability")


# ------------------------------------------------------------------- requests


def test_the_request_sends_the_labels_as_criteria(task, monkeypatch):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-key")
    provider = build_provider("gateway")
    payload = provider.build_payload(task, task.examples[0])
    question = payload["questions"]["decision"]
    assert payload["state"] == "x"
    assert question["type"] == "choice"
    assert set(question["criteria"]) == set(task.labels)


def test_underscored_labels_are_sent_as_readable_descriptions(monkeypatch):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-key")
    t = Task(name="t", kind="choice", question="q",
             labels=("accept_reservations", "card_declined"),
             examples=(Example(id="1", text="x", label="accept_reservations"),))
    provider = build_provider("gateway")
    criteria = provider.build_payload(t, t.examples[0])["questions"]["decision"]["criteria"]
    assert criteria["accept_reservations"] == "accept reservations"


def test_a_spec_supplied_description_wins_over_the_label(monkeypatch):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-key")
    t = Task(name="t", kind="choice", question="q", labels=("a", "b"),
             examples=(Example(id="1", text="x", label="a"),),
             criteria={"a": "the first one"})
    provider = build_provider("gateway")
    criteria = provider.build_payload(t, t.examples[0])["questions"]["decision"]["criteria"]
    assert criteria == {"a": "the first one", "b": "b"}


def test_each_surface_uses_its_own_endpoint_and_yes_no_type_name(monkeypatch):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-key")
    monkeypatch.delenv("JEV_BASE_URL", raising=False)
    gateway = build_provider("gateway", surface="gateway")
    native = build_provider("gateway", surface="typesafe")
    assert gateway.base_url.endswith("ai-gateway.vercel.sh/v1")
    assert gateway.question_type("noul") == "boolean"
    assert native.base_url.endswith("api.typesafe.ai/v1")
    assert native.question_type("noul") == "noul"


def test_the_gateways_exact_cost_is_preferred_over_our_arithmetic(task):
    from jevcal.providers.gateway import _cost_usd

    body = _body({"type": "choice"}, providerMetadata={"gateway": {"cost": "0.00001155"}})
    assert _cost_usd(body) == pytest.approx(0.00001155)


def test_cost_falls_back_to_token_arithmetic_when_unreported(task):
    from jevcal.providers.gateway import _cost_usd

    assert _cost_usd(_body({"type": "choice"})) == pytest.approx(359 * 0.042 / 1e6)
    assert _cost_usd({"usage": {"inputTokens": 359}}) == pytest.approx(359 * 0.042 / 1e6)
