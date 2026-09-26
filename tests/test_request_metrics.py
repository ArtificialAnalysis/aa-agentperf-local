"""Pin post-close generation timing semantics."""

from __future__ import annotations

import orjson
import pytest

from agentperf_local.client.protocol import CompletionResult, RawRead
from agentperf_local.common.json_types import JsonObject
from agentperf_local.metrics.request import measure_request
from tests.token_counter import CharacterCounter


def _event(delta: JsonObject) -> bytes:
    payload: JsonObject = {
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": None,
            }
        ]
    }
    return b"data: " + orjson.dumps(payload) + b"\n\n"


def _usage_event(
    prompt_tokens: int | float | str | bool,
    completion_tokens: int | float | str | bool,
    *,
    cached_tokens: int | float | str | bool | None = None,
) -> bytes:
    usage: JsonObject = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
    if cached_tokens is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
    payload: JsonObject = {
        "choices": [],
        "usage": usage,
    }
    return b"data: " + orjson.dumps(payload) + b"\n\n"


@pytest.mark.parametrize(
    ("reads", "expected_generation_time"),
    [
        (
            (RawRead(timestamp=10.2, data=_event({"content": "one"})),),
            None,
        ),
        (
            (
                RawRead(timestamp=10.1, data=_event({"role": "assistant", "content": ""})),
                RawRead(timestamp=10.2, data=_event({"content": "one"})),
                RawRead(timestamp=10.4, data=_event({"tool_calls": [{"index": 0, "id": "call-1"}]})),
                RawRead(timestamp=10.6, data=_event({"reasoning": "two"})),
            ),
            0.4,
        ),
        (
            (
                RawRead(
                    timestamp=10.2,
                    data=_event({"content": "one"}) + _event({"content": "two"}),
                ),
            ),
            0.0,
        ),
    ],
    ids=("one-output-event", "first-to-last-output", "two-events-in-one-read"),
)
def test_generation_time_uses_first_to_last_visible_output(
    reads: tuple[RawRead, ...],
    expected_generation_time: float | None,
) -> None:
    metrics = measure_request(
        CompletionResult(reads=reads, aborted=False),
        started_at=10.0,
        ended_at=11.0,
        token_counter=CharacterCounter(),
    )

    if expected_generation_time is None:
        assert metrics.generation_time is None
    else:
        assert metrics.generation_time == pytest.approx(expected_generation_time)


@pytest.mark.parametrize(
    ("started_at", "ended_at", "read_timestamps"),
    [
        (float("nan"), 11.0, ()),
        (10.0, float("inf"), ()),
        (10.0, 11.0, (9.9,)),
        (10.0, 11.0, (11.1,)),
        (10.0, 11.0, (10.5, 10.4)),
        (10.0, 11.0, (float("nan"),)),
    ],
    ids=("non-finite-start", "non-finite-end", "before-start", "after-end", "out-of-order", "non-finite-read"),
)
def test_rejects_invalid_request_timestamps(
    started_at: float,
    ended_at: float,
    read_timestamps: tuple[float, ...],
) -> None:
    reads = tuple(RawRead(timestamp=timestamp, data=_event({"content": "one"})) for timestamp in read_timestamps)

    with pytest.raises(ValueError):
        measure_request(
            CompletionResult(reads=reads, aborted=False),
            started_at=started_at,
            ended_at=ended_at,
            token_counter=CharacterCounter(),
        )


@pytest.mark.parametrize(
    ("prompt_tokens", "completion_tokens"),
    [
        (-1, 1),
        (1, -1),
        (True, 1),
        (1, "invalid"),
        (1, 7.5),
        (-1.0, 1),
    ],
)
def test_rejects_invalid_server_usage(
    prompt_tokens: int | float | str | bool,
    completion_tokens: int | float | str | bool,
) -> None:
    result = CompletionResult(
        reads=(RawRead(timestamp=10.5, data=_usage_event(prompt_tokens, completion_tokens)),),
        aborted=False,
    )

    with pytest.raises(ValueError):
        measure_request(
            result,
            started_at=10.0,
            ended_at=11.0,
            token_counter=CharacterCounter(),
        )


def test_accepts_integral_float_server_usage() -> None:
    result = CompletionResult(
        reads=(RawRead(timestamp=10.5, data=_usage_event(11.0, 7.0)),),
        aborted=False,
    )

    metrics = measure_request(
        result,
        started_at=10.0,
        ended_at=11.0,
        token_counter=CharacterCounter(),
    )

    assert metrics.prompt_tokens == 11
    assert metrics.server_output_tokens == 7
    assert metrics.cached_prompt_tokens is None
    assert metrics.uncached_prompt_tokens is None


def test_reads_cached_prompt_token_usage_after_stream_close() -> None:
    result = CompletionResult(
        reads=(RawRead(timestamp=10.5, data=_usage_event(11, 7, cached_tokens=8)),),
        aborted=False,
    )

    metrics = measure_request(
        result,
        started_at=10.0,
        ended_at=11.0,
        token_counter=CharacterCounter(),
    )

    assert metrics.cached_prompt_tokens == 8
    assert metrics.uncached_prompt_tokens == 3


@pytest.mark.parametrize("cached_tokens", [-1, 12, True, "invalid", 7.5])
def test_rejects_invalid_cached_prompt_token_usage(cached_tokens: int | float | str | bool) -> None:
    result = CompletionResult(
        reads=(RawRead(timestamp=10.5, data=_usage_event(11, 7, cached_tokens=cached_tokens)),),
        aborted=False,
    )

    with pytest.raises(ValueError):
        measure_request(
            result,
            started_at=10.0,
            ended_at=11.0,
            token_counter=CharacterCounter(),
        )
