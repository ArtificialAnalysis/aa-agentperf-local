"""State what one run requested and observed about the served context window.

- `context_is_reduced`, `below_benchmark_context`: the one comparability rule.
- `ContextObservationReason`, `RunContextFacts`: the recorded context facts.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, model_validator

from agentperf_local.common.json_types import JsonObject
from agentperf_local.provenance.benchmark import BENCHMARK_CONTEXT_TOKENS


def below_benchmark_context(context_tokens: int) -> bool:
    """Return whether one context length is shorter than the full benchmark context."""
    return context_tokens < BENCHMARK_CONTEXT_TOKENS


def context_is_reduced(requested_tokens: int, observed_tokens: int | None) -> bool:
    """Return whether a run failed to prove the full benchmark context.

    A reduced run is never comparable and never eligible for a ranked submission.
    An unobserved context counts as reduced, because the server would not prove it.
    """
    if below_benchmark_context(requested_tokens):
        return True
    return observed_tokens is None or below_benchmark_context(observed_tokens)


class ContextObservationReason(StrEnum):
    """Name how this run's served-context observation was resolved."""

    REPORTED = "reported"
    NOT_PROBED = "not-probed"
    ENDPOINT_UNREACHABLE = "endpoint-unreachable"
    HTTP_ERROR = "http-error"
    MALFORMED_RESPONSE = "malformed-response"
    MODEL_NOT_LISTED = "model-not-listed"
    CONTEXT_NOT_REPORTED = "context-not-reported"
    NON_POSITIVE_CONTEXT = "non-positive-context"


class RunContextFacts(BaseModel, frozen=True):
    """Store what this run requested and observed about the served context window."""

    requested_tokens: int
    observed_tokens: int | None
    observed_reason: ContextObservationReason

    @model_validator(mode="after")
    def check_invariants(self) -> Self:
        """Reject impossible context claims."""
        if self.requested_tokens <= 0 or self.requested_tokens > BENCHMARK_CONTEXT_TOKENS:
            raise ValueError(f"requested context must be between 1 and {BENCHMARK_CONTEXT_TOKENS} tokens")
        if self.observed_tokens is not None and self.observed_tokens <= 0:
            raise ValueError("observed context must be positive or None")
        if (self.observed_tokens is not None) != (self.observed_reason is ContextObservationReason.REPORTED):
            raise ValueError("observed_reason must be 'reported' exactly when a context observation exists")
        return self

    @property
    def reduced(self) -> bool:
        """Return whether this run failed to prove the full benchmark context."""
        return context_is_reduced(self.requested_tokens, self.observed_tokens)

    def to_json(self) -> JsonObject:
        """Return the context facts for the summary config block."""
        return {
            "requested_tokens": self.requested_tokens,
            "observed_tokens": self.observed_tokens,
            "observed_reason": self.observed_reason.value,
            "full_benchmark_tokens": BENCHMARK_CONTEXT_TOKENS,
            "reduced": self.reduced,
        }
