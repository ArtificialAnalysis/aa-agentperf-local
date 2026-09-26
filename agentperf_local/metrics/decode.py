"""Decode timestamped SSE bytes after a response closes."""

from dataclasses import dataclass

import orjson

from agentperf_local.client.protocol import RawRead
from agentperf_local.common.json_types import JsonObject, normalize_json_object

LINE_FEED = b"\n"
CARRIAGE_RETURN = b"\r"
CARRIAGE_RETURN_LINE_FEED = b"\r\n"
EVENT_SEPARATOR = b"\n\n"


@dataclass(frozen=True, slots=True)
class StreamChunk:
    """Hold one decoded SSE data event."""

    timestamp: float
    data: JsonObject


@dataclass(slots=True)
class _LineTerminators:
    """Rewrite SSE line terminators as line feeds across read boundaries."""

    pending_carriage_return: bool = False

    def normalize(self, data: bytes) -> bytes:
        """Return one read whose CR and CRLF terminators became line feeds."""
        prefix = b""
        if self.pending_carriage_return:
            self.pending_carriage_return = False
            prefix = LINE_FEED
            data = data.removeprefix(LINE_FEED)
        if CARRIAGE_RETURN not in data:
            # Almost every server ends lines with a bare line feed; skip the rewrites.
            return prefix + data
        if data.endswith(CARRIAGE_RETURN):
            # A carriage return at a read boundary may still be half of a CRLF pair,
            # so hold it back until the next read or the end of the stream decides.
            self.pending_carriage_return = True
            data = data[:-1]
        normalized = data.replace(CARRIAGE_RETURN_LINE_FEED, LINE_FEED).replace(CARRIAGE_RETURN, LINE_FEED)
        return prefix + normalized

    def flush(self) -> bytes:
        """Return the terminator held back at the last read boundary."""
        if not self.pending_carriage_return:
            return b""
        self.pending_carriage_return = False
        return LINE_FEED


def _data_payload(event: bytes) -> bytes | None:
    data_lines: list[bytes] = []
    for line in event.split(LINE_FEED):
        if line == b"data":
            data_lines.append(b"")
        elif line.startswith(b"data:"):
            data_lines.append(line[5:].lstrip(b" "))
    if not data_lines:
        return None
    return LINE_FEED.join(data_lines)


def _decode_payload(payload: bytes) -> JsonObject | None:
    if not payload or payload.rstrip() == b"[DONE]":
        return None
    try:
        decoded: object = orjson.loads(payload)
        return normalize_json_object(decoded)
    except (orjson.JSONDecodeError, ValueError) as error:
        raise ValueError("SSE data event must contain a JSON object or [DONE]") from error


def _append_event(block: bytes, timestamp: float, chunks: list[StreamChunk]) -> None:
    payload = _data_payload(block)
    data = _decode_payload(payload) if payload is not None else None
    if data is not None:
        chunks.append(StreamChunk(timestamp=timestamp, data=data))


def _drain_events(buffer: bytearray, timestamp: float, chunks: list[StreamChunk]) -> None:
    while (index := buffer.find(EVENT_SEPARATOR)) >= 0:
        block = bytes(buffer[:index])
        del buffer[: index + len(EVENT_SEPARATOR)]
        _append_event(block, timestamp, chunks)


def _append_closing_event(buffer: bytearray, timestamp: float, chunks: list[StreamChunk]) -> None:
    remainder = bytes(buffer)
    if not remainder.strip():
        return
    # A server may end the stream right after its last event line without sending
    # the blank line that would separate a following event. That event is complete;
    # a payload cut off mid-line is not, and still fails to decode below.
    if not remainder.endswith(LINE_FEED):
        raise ValueError("SSE stream ended with an incomplete event")
    _append_event(remainder, timestamp, chunks)


def decode_sse_reads(reads: tuple[RawRead, ...]) -> tuple[StreamChunk, ...]:
    """Decode complete SSE events from raw network reads."""
    if not reads:
        return ()
    buffer = bytearray()
    chunks: list[StreamChunk] = []
    terminators = _LineTerminators()
    for read in reads:
        buffer.extend(terminators.normalize(read.data))
        _drain_events(buffer, read.timestamp, chunks)
    last_timestamp = reads[-1].timestamp
    buffer.extend(terminators.flush())
    _drain_events(buffer, last_timestamp, chunks)
    _append_closing_event(buffer, last_timestamp, chunks)
    return tuple(chunks)
