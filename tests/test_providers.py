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


def test_distribution_is_read_from_a_native_field(task):
    body = {"choices": [{"message": {"content": '{"answer":"positive"}'}}],
            "probabilities": {"positive": 0.82, "negative": 0.18}}
    predicted, dist = extract_decision(body, task)
    assert predicted == "positive"
    assert dist["positive"] == pytest.approx(0.82)


def test_distribution_is_read_from_logprobs(task):
    body = {
        "choices": [
            {
                "message": {"content": '{"answer":"negative"}'},
                "logprobs": {
                    "content": [
                        {"token": "{", "top_logprobs": [{"token": "{", "logprob": 0.0}]},
                        {
                            "token": "negative",
                            "top_logprobs": [
                                {"token": "negative", "logprob": math.log(0.7)},
                                {"token": "positive", "logprob": math.log(0.3)},
                            ],
                        },
                    ]
                },
            }
        ]
    }
    predicted, dist = extract_decision(body, task)
    assert predicted == "negative"
    assert dist["negative"] == pytest.approx(0.7)


def test_structural_tokens_are_skipped_when_reading_logprobs(task):
    body = {
        "choices": [
            {
                "logprobs": {
                    "content": [
                        # '"answer"' is a prefix of nothing in the label set and
                        # must not be mistaken for a decision.
                        {"token": '"answer"', "top_logprobs": [{"token": '"answer"', "logprob": 0.0}]},
                        {
                            "token": "pos",
                            "top_logprobs": [
                                {"token": "pos", "logprob": math.log(0.6)},
                                {"token": "neg", "logprob": math.log(0.4)},
                            ],
                        },
                    ]
                }
            }
        ]
    }
    predicted, dist = extract_decision(body, task)
    assert predicted == "positive"
    assert dist["positive"] == pytest.approx(0.6)


def test_distribution_is_read_from_json_in_the_content(task):
    body = {
        "choices": [
            {"message": {"content": '{"answer":"positive","probabilities":{"positive":0.9,"negative":0.1}}'}}
        ]
    }
    predicted, dist = extract_decision(body, task)
    assert predicted == "positive"
    assert dist["positive"] == pytest.approx(0.9)


def test_a_response_with_no_distribution_fails_loudly(task):
    body = {"choices": [{"message": {"content": '{"answer":"positive"}'}}]}
    with pytest.raises(ExtractionError, match="probe"):
        extract_decision(body, task)


def test_pinning_a_source_does_not_silently_fall_through(task):
    body = {"probabilities": {"positive": 0.8, "negative": 0.2}}
    with pytest.raises(ExtractionError):
        extract_decision(body, task, source="logprobs")


def test_a_distribution_is_restricted_to_the_task_labels_and_renormalised(task):
    body = {"probabilities": {"positive": 0.6, "negative": 0.2, "maybe": 0.2}}
    _, dist = extract_decision(body, task)
    assert set(dist) == {"positive", "negative"}
    assert sum(dist.values()) == pytest.approx(1.0)
    assert dist["positive"] == pytest.approx(0.75)


def test_a_distribution_with_no_mass_on_the_labels_is_an_error(task):
    with pytest.raises(ValueError):
        normalise_distribution({"maybe": 1.0}, task.labels)


def test_the_request_constrains_the_answer_to_the_label_set(task, monkeypatch):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-key")
    provider = build_provider("gateway")
    payload = provider.build_payload(task, task.examples[0])
    schema = payload["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["answer"]["enum"] == list(task.labels)
    # The state goes in the user turn; the typed question in the system turn.
    assert payload["messages"][1]["content"] == "x"
    assert "positive" in payload["messages"][0]["content"]
