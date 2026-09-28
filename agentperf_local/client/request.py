"""Build OpenAI-compatible completion request bodies."""

from typing import Annotated, Self

from pydantic import BaseModel, Field, PositiveInt, model_validator

from agentperf_local.common.json_types import JsonObject, JsonValue

DEFAULT_MAX_OUTPUT_TOKENS = 16_384


class CompletionRequest(BaseModel, frozen=True):
    """Describe one streamed chat completion."""

    messages: Annotated[tuple[JsonObject, ...], Field(min_length=1)]
    model: Annotated[str, Field(min_length=1)]
    tools: tuple[JsonObject, ...] = ()
    max_tokens: PositiveInt = DEFAULT_MAX_OUTPUT_TOKENS
    reasoning_effort: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    # Ask the server to keep generating past end-of-sequence up to max_tokens, so the
    # output length is the request's choice rather than the model's. Honoured by
    # llama.cpp, SGLang, and vLLM on the chat completions endpoint.
    ignore_eos: bool = False
    extra_headers: tuple[tuple[str, str], ...] = ()
    extra_body: tuple[tuple[str, JsonValue], ...] = ()

    @model_validator(mode="after")
    def check_invariants(self) -> Self:
        """Reject values that cannot form a useful request."""
        if self.temperature is not None and self.temperature < 0:
            raise ValueError("temperature must not be negative")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError("top_p must be greater than zero and at most one")
        return self

    def body(self) -> JsonObject:
        """Return the JSON body sent to an OpenAI-compatible endpoint."""
        body: JsonObject = {
            "model": self.model,
            "messages": list(self.messages),
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": self.max_tokens,
        }
        if self.tools:
            body["tools"] = list(self.tools)
        if self.reasoning_effort is not None:
            body["reasoning_effort"] = self.reasoning_effort
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.top_p is not None:
            body["top_p"] = self.top_p
        if self.ignore_eos:
            body["ignore_eos"] = True
        for key, value in self.extra_body:
            body[key] = value
        return body

    def headers(self) -> dict[str, str]:
        """Return additional HTTP headers."""
        return dict(self.extra_headers)
