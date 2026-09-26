"""Reconstruct response channels from decoded stream events."""

from dataclasses import dataclass

import orjson

from agentperf_local.common.json_types import JsonObject, JsonValue, json_object_or_none
from agentperf_local.metrics.decode import StreamChunk

INVALID_TOOL_CALL_INDEX = -1


@dataclass(frozen=True, slots=True)
class ToolCall:
    """Hold one reconstructed function call."""

    index: int
    identifier: str | None
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class ResponseChannels:
    """Hold separated model output channels."""

    content: str
    reasoning: str
    tool_calls: tuple[ToolCall, ...]
    finish_reason: str | None


@dataclass(slots=True)
class _ToolCallParts:
    identifier: str | None = None
    name: str = ""
    arguments: str = ""


def _first_choice(data: JsonObject) -> JsonObject | None:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    return json_object_or_none(choices[0])


def _tool_index(value: JsonValue | None) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return INVALID_TOOL_CALL_INDEX


def _stream_error_message(error: JsonValue) -> str:
    detail = json_object_or_none(error)
    if detail is not None:
        message = detail.get("message")
        if isinstance(message, str) and message:
            return message
    return orjson.dumps(error).decode()


def _reject_stream_error(chunk: StreamChunk) -> None:
    # A server that reports an error after finish_reason has truncated the turn.
    # Both client backends share this decode path, so both fail the same way.
    error = chunk.data.get("error")
    if error is None:
        return
    raise ValueError(f"server stream reported an error: {_stream_error_message(error)}")


def parse_response_channels(chunks: tuple[StreamChunk, ...]) -> ResponseChannels:
    """Separate content, reasoning, and tool calls, and reject a reported stream error."""
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_parts: dict[int, _ToolCallParts] = {}
    finish_reason: str | None = None
    for chunk in chunks:
        _reject_stream_error(chunk)
        choice = _first_choice(chunk.data)
        if choice is None:
            continue
        reason = choice.get("finish_reason")
        if isinstance(reason, str):
            finish_reason = reason
        delta = json_object_or_none(choice.get("delta"))
        if delta is None:
            continue
        content = delta.get("content")
        if isinstance(content, str):
            content_parts.append(content)
        reasoning = delta.get("reasoning")
        if not isinstance(reasoning, str) or not reasoning:
            fallback_reasoning = delta.get("reasoning_content")
            if isinstance(fallback_reasoning, str):
                reasoning = fallback_reasoning
        if isinstance(reasoning, str):
            reasoning_parts.append(reasoning)
        tool_calls = delta.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for raw_call in tool_calls:
            call = json_object_or_none(raw_call)
            if call is None:
                continue
            index = _tool_index(call.get("index"))
            parts = tool_parts.setdefault(index, _ToolCallParts())
            identifier = call.get("id")
            if isinstance(identifier, str):
                parts.identifier = identifier
            function = json_object_or_none(call.get("function"))
            if function is None:
                continue
            name = function.get("name")
            if isinstance(name, str):
                parts.name = name
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                parts.arguments += arguments
    calls = tuple(
        ToolCall(
            index=index,
            identifier=parts.identifier,
            name=parts.name,
            arguments=parts.arguments,
        )
        for index, parts in sorted(tool_parts.items())
    )
    return ResponseChannels(
        content="".join(content_parts),
        reasoning="".join(reasoning_parts),
        tool_calls=calls,
        finish_reason=finish_reason,
    )


def flatten_output_text(chunks: tuple[StreamChunk, ...]) -> str:
    """Join output deltas in source order for local token counting."""
    parts: list[str] = []
    for chunk in chunks:
        choice = _first_choice(chunk.data)
        delta = json_object_or_none(choice.get("delta")) if choice is not None else None
        if delta is None:
            continue
        content = delta.get("content")
        if isinstance(content, str):
            parts.append(content)
        reasoning = delta.get("reasoning")
        if not isinstance(reasoning, str) or not reasoning:
            fallback_reasoning = delta.get("reasoning_content")
            if isinstance(fallback_reasoning, str):
                reasoning = fallback_reasoning
        if isinstance(reasoning, str):
            parts.append(reasoning)
        tool_calls = delta.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for raw_call in tool_calls:
            call = json_object_or_none(raw_call)
            function = json_object_or_none(call.get("function")) if call is not None else None
            if function is None:
                continue
            name = function.get("name")
            if isinstance(name, str):
                parts.append(name)
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                parts.append(arguments)
    return "".join(parts)


def _has_visible_output(chunk: StreamChunk) -> bool:
    choice = _first_choice(chunk.data)
    delta = json_object_or_none(choice.get("delta")) if choice is not None else None
    if delta is None:
        return False
    for field in ("content", "reasoning", "reasoning_content"):
        value = delta.get(field)
        if isinstance(value, str) and value:
            return True
    tool_calls = delta.get("tool_calls")
    if not isinstance(tool_calls, list):
        return False
    for raw_call in tool_calls:
        call = json_object_or_none(raw_call)
        function = json_object_or_none(call.get("function")) if call is not None else None
        if function is None:
            continue
        name = function.get("name")
        arguments = function.get("arguments")
        if (isinstance(name, str) and name) or (isinstance(arguments, str) and arguments):
            return True
    return False


def output_timestamp_range(chunks: tuple[StreamChunk, ...]) -> tuple[float | None, float | None]:
    """Return the first output timestamp and the last when another event follows."""
    first: float | None = None
    last: float | None = None
    for chunk in chunks:
        if not _has_visible_output(chunk):
            continue
        if first is None:
            first = chunk.timestamp
        else:
            last = chunk.timestamp
    return first, last
