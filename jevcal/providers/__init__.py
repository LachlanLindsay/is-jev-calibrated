"""Provider registry."""

from __future__ import annotations

from typing import Any

from .base import Provider, ProviderError, normalise_distribution
from .simulated import SimulatedProvider

__all__ = ["Provider", "ProviderError", "normalise_distribution", "SimulatedProvider", "build_provider"]


def build_provider(name: str, **kwargs: Any) -> Provider:
    """Construct a provider by name.

    ``gateway`` is imported lazily so that the offline path never needs
    ``requests`` or an API key.
    """
    if name == "simulated":
        allowed = {"seed", "temperature", "bias", "skill_a", "skill_b", "latency_ms"}
        return SimulatedProvider(**{k: v for k, v in kwargs.items() if k in allowed})
    if name == "gateway":
        from .gateway import GatewayProvider

        allowed = {
            "model",
            "base_url",
            "api_key",
            "probability_source",
            "timeout",
            "max_retries",
            "request_logprobs",
            "keep_raw",
        }
        return GatewayProvider(**{k: v for k, v in kwargs.items() if k in allowed})
    raise ProviderError(f"unknown provider: {name!r} (expected 'gateway' or 'simulated')")
