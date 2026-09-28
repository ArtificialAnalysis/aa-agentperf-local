"""Calculate request metrics after streaming ends."""

import math

from pydantic import BaseModel

from agentperf_local.client.protocol import CompletionResult
from agentperf_local.common.json_types import JsonValue, json_object_or_none
from agentperf_local.metrics.decode import StreamChunk, decode_sse_reads
from agentperf_local.metrics.response import (
    ResponseChannels,
    flatten_output_text,
    output_timestamp_range,
    parse_response_channels,
)
from agentperf_local.metrics.tokenization import TokenCounter


class RequestMetrics(BaseModel, frozen=True):
    """Hold measurements and decoded output for one request."""

    started_at: float
    ended_at: float
    time_to_first_byte: float | None
    time_to_first_token: float | None
    generation_time: float | None
    prompt_tokens: int | None
    output_tokens: int
    server_output_tokens: int | None
    channels: ResponseChannels
    chunks: tuple[StreamChunk, ...]
    aborted: bool
    cached_prompt_tokens: int | None = None

    @property
    def duration(self) -> float:
        """Return total request duration in seconds."""
        return self.ended_at - self.started_at

    @property
    def reported_output_tokens(self) -> int:
        """Return the server's completion count when it reports one, else the local count."""
        return self.server_output_tokens if self.server_output_tokens is not None else self.output_tokens

    @property
    def uncached_prompt_tokens(self) -> int | None:
        """Return uncached prompt tokens when the server reports cache accounting."""
        if self.prompt_tokens is None or self.cached_prompt_tokens is None:
            return None
        return self.prompt_tokens - self.cached_prompt_tokens


def _optional_token_count(value: JsonValue | None, field: str) -> int | None:
    if value is None:
        return None
    # Some servers serialize whole token counts as JSON floats, so 7.0 means 7.
    # Fractional and negative counts stay errors.
    if isinstance(value, float) and value.is_integer() and value >= 0:
        return int(value)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"server usage {field} must be a non-negative integer or null")
    return value


def _usage(chunks: tuple[StreamChunk, ...]) -> tuple[int | None, int | None, int | None]:
    for chunk in reversed(chunks):
        raw_usage = chunk.data.get("usage")
        if raw_usage is None:
            continue
        usage = json_object_or_none(raw_usage)
        if usage is None:
            raise ValueError("server usage must be an object or null")
        prompt_tokens = _optional_token_count(usage.get("prompt_tokens"), "prompt_tokens")
        prompt_details_value = usage.get("prompt_tokens_details")
        prompt_details = json_object_or_none(prompt_details_value)
        if prompt_details_value is not None and prompt_details is None:
            raise ValueError("server usage prompt_tokens_details must be an object or null")
        cached_prompt_tokens = (
            _optional_token_count(prompt_details.get("cached_tokens"), "prompt_tokens_details.cached_tokens")
            if prompt_details is not None
            else None
        )
        if prompt_tokens is not None and cached_prompt_tokens is not None and cached_prompt_tokens > prompt_tokens:
            raise ValueError("server usage cached prompt tokens must not exceed prompt tokens")
        return (
            prompt_tokens,
            _optional_token_count(usage.get("completion_tokens"), "completion_tokens"),
            cached_prompt_tokens,
        )
    return None, None, None


def _validate_timestamps(result: CompletionResult, started_at: float, ended_at: float) -> None:
    if not math.isfinite(started_at) or not math.isfinite(ended_at):
        raise ValueError("request timestamps must be finite")
    if ended_at < started_at:
        raise ValueError("ended_at must not precede started_at")
    previous = started_at
    for read in result.reads:
        if not math.isfinite(read.timestamp):
            raise ValueError("raw read timestamps must be finite")
        if read.timestamp < previous:
            raise ValueError("raw read timestamps must be monotonic and not precede request start")
        if read.timestamp > ended_at:
            raise ValueError("raw read timestamps must not follow request end")
        previous = read.timestamp


def measure_request(
    result: CompletionResult,
    *,
    started_at: float,
    ended_at: float,
    token_counter: TokenCounter,
) -> RequestMetrics:
    """Decode one response and calculate its request metrics."""
    _validate_timestamps(result, started_at, ended_at)
    chunks = decode_sse_reads(result.reads)
    channels = parse_response_channels(chunks)
    prompt_tokens, server_output_tokens, cached_prompt_tokens = _usage(chunks)
    first_read = result.reads[0].timestamp if result.reads else None
    first_output, last_output = output_timestamp_range(chunks)
    return RequestMetrics(
        started_at=started_at,
        ended_at=ended_at,
        time_to_first_byte=first_read - started_at if first_read is not None else None,
        time_to_first_token=first_output - started_at if first_output is not None else None,
        generation_time=last_output - first_output if first_output is not None and last_output is not None else None,
        prompt_tokens=prompt_tokens,
        output_tokens=token_counter.count(flatten_output_text(chunks)),
        server_output_tokens=server_output_tokens,
        channels=channels,
        chunks=chunks,
        aborted=result.aborted,
        cached_prompt_tokens=cached_prompt_tokens,
    )
