"""Test the raw streaming client and post-close metrics contract."""

from __future__ import annotations

import asyncio
from importlib.util import find_spec
from time import perf_counter

import orjson
import pytest

from agentperf_local.client.backends import CLIENT_BACKENDS, ClientBackend, streaming_client
from agentperf_local.client.endpoint import MEASURED_USER_AGENT, normalize_base_url
from agentperf_local.client.protocol import CompletionClient, CompletionError, CompletionResult, RawRead
from agentperf_local.client.request import CompletionRequest
from agentperf_local.common.json_types import JsonObject
from agentperf_local.metrics.decode import decode_sse_reads
from agentperf_local.metrics.request import measure_request
from agentperf_local.metrics.response import parse_response_channels
from agentperf_local.replay.fidelity import ExpectedToolCall, evaluate_tool_fidelity
from agentperf_local.workload.schema import parse_json_object
from tests.localhost_sse import CapturedRequest, LocalSseServer
from tests.token_counter import CharacterCounter

HTTP_BAD_REQUEST = 400
HTTP_TEMPORARY_REDIRECT = 307
ABORT_TEST_TIMEOUT_SECONDS = 0.25
STALLED_RESPONSE_SECONDS = 1.0
STREAM_DELAY_SECONDS = 0.03
READ_TIMEOUT_SECONDS = 0.1
READ_TIMEOUT_TEST_DEADLINE_SECONDS = 0.5
KEEP_ALIVE_REQUEST_COUNT = 2
HAS_RUSTCORE = find_spec("agentperf_local_rustcore") is not None
# Hop-by-hop and length headers belong to the transport, not to client identity.
UNCOMPARED_HEADERS = frozenset(("host", "content-length", "connection"))
REQUIRES_RUSTCORE = pytest.mark.skipif(not HAS_RUSTCORE, reason="agentperf_local_rustcore is not installed")
EVERY_BACKEND = pytest.mark.parametrize(
    "backend",
    [
        pytest.param("python", id="python"),
        pytest.param("rust", id="rust", marks=REQUIRES_RUSTCORE),
    ],
)


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("http://localhost", "http://localhost"),
        ("http://localhost:8080/v1/", "http://localhost:8080/v1"),
        ("https://[::1]:8443/v1/", "https://[::1]:8443/v1"),
    ],
)
def test_normalizes_supported_endpoint_base_urls(base_url: str, expected: str) -> None:
    assert normalize_base_url(base_url) == expected


@pytest.mark.parametrize(
    "base_url",
    [
        "",
        "localhost:8080/v1",
        "ftp://localhost/v1",
        "http:///v1",
        "http://user:secret@localhost/v1",
        "http://localhost/v1?token=secret",
        "http://localhost/v1#fragment",
        "http://local host/v1",
        "http://localhost:0/v1",
        "http://localhost:65536/v1",
        "http://localhost/v1/chat/completions",
    ],
)
def test_rejects_ambiguous_endpoint_base_urls(base_url: str) -> None:
    with pytest.raises(ValueError):
        normalize_base_url(base_url)


def _sse(payload: JsonObject) -> bytes:
    return b"data: " + orjson.dumps(payload) + b"\n\n"


def _chunk(delta: JsonObject, *, finish_reason: str | None = None) -> JsonObject:
    return {
        "id": "response",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "model",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _usage() -> JsonObject:
    return {
        "id": "response",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "model",
        "choices": [],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
            "prompt_tokens_details": {"cached_tokens": 5},
        },
    }


def _events() -> tuple[bytes, ...]:
    return (
        _sse(_chunk({"role": "assistant", "content": ""})),
        _sse(_chunk({"reasoning": "think "})),
        _sse(_chunk({"reasoning": "", "reasoning_content": "carefully"})),
        _sse(_chunk({"content": "answer"})),
        _sse(_chunk({"tool_calls": [{"index": 0, "id": "call-1"}]})),
        _sse(
            _chunk(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"name": "lookup", "arguments": '{"q":'},
                        }
                    ]
                }
            )
        ),
        _sse(_chunk({"tool_calls": [{"index": 0, "function": {"arguments": '"x"}'}}]})),
        _sse(_chunk({}, finish_reason="tool_calls")),
        _sse(_usage()),
        b"data: [DONE]\n\n",
    )


def _request() -> CompletionRequest:
    return CompletionRequest(
        messages=({"role": "user", "content": "hello"},),
        model="model",
        max_tokens=32,
        extra_headers=(("x-replay", "yes"),),
    )


def test_request_applies_an_explicit_tool_choice() -> None:
    """A caller can select a framework-compatible tool policy."""
    with_tools = CompletionRequest(
        messages=({"role": "user", "content": "hi"},),
        model="m",
        tools=({"type": "function", "function": {"name": "f"}},),
        ignore_eos=True,
        extra_body=(("tool_choice", "none"),),
    ).body()
    assert with_tools["ignore_eos"] is True
    assert with_tools["tool_choice"] == "none"
    assert (
        "tool_choice"
        not in CompletionRequest(
            messages=({"role": "user", "content": "hi"},),
            model="m",
            tools=({"type": "function", "function": {"name": "f"}},),
        ).body()
    )
    assert (
        "tool_choice"
        not in CompletionRequest(messages=({"role": "user", "content": "hi"},), model="m", ignore_eos=True).body()
    )


def _client(backend: ClientBackend, base_url: str, *, timeout_seconds: float = 2.0) -> CompletionClient:
    """Build one single-connection client through the factory a run uses."""
    return streaming_client(
        backend, base_url=base_url, api_key=None, timeout_seconds=timeout_seconds, max_connections=1
    )


def test_fragmented_multi_event_sse_reconstructs_channels_usage_and_metrics() -> None:
    events = _events()
    split_at = len(events[1]) // 2
    reads = (
        RawRead(timestamp=10.1, data=events[0] + events[1][:split_at]),
        RawRead(timestamp=10.2, data=events[1][split_at:] + events[2] + events[3]),
        RawRead(timestamp=10.3, data=events[4] + events[5]),
        RawRead(timestamp=10.4, data=events[6] + events[7]),
        RawRead(timestamp=10.5, data=events[8] + events[9]),
    )
    result = CompletionResult(reads=reads, aborted=False)

    chunks = decode_sse_reads(reads)
    channels = parse_response_channels(chunks)
    metrics = measure_request(
        result,
        started_at=10.0,
        ended_at=10.8,
        token_counter=CharacterCounter(),
    )

    assert len(chunks) == len(events) - 1
    assert chunks[1].timestamp == pytest.approx(10.2)
    assert channels.content == "answer"
    assert channels.reasoning == "think carefully"
    assert channels.finish_reason == "tool_calls"
    assert len(channels.tool_calls) == 1
    assert channels.tool_calls[0].identifier == "call-1"
    assert channels.tool_calls[0].name == "lookup"
    assert channels.tool_calls[0].arguments == '{"q":"x"}'
    assert metrics.time_to_first_byte == pytest.approx(0.1)
    assert metrics.time_to_first_token == pytest.approx(0.2)
    assert metrics.prompt_tokens == 11
    assert metrics.cached_prompt_tokens == 5
    assert metrics.uncached_prompt_tokens == 6
    assert metrics.server_output_tokens == 7
    assert metrics.output_tokens == len('think carefullyanswerlookup{"q":"x"}')
    assert metrics.duration == pytest.approx(0.8)


@pytest.mark.parametrize(
    "event",
    [
        b"data: not-json\n\n",
        b"data: []\n\n",
        b'data: {"unfinished":true}',
        b'data: {"unfinished":\n',
    ],
    ids=("invalid-json", "non-object-json", "incomplete-event", "truncated-final-payload"),
)
def test_decoder_rejects_malformed_sse_data(event: bytes) -> None:
    with pytest.raises(ValueError):
        decode_sse_reads((RawRead(timestamp=1.0, data=event),))


@pytest.mark.parametrize("terminator", [b"\n", b"\r\n", b"\r"], ids=("lf", "crlf", "cr"))
def test_decoder_accepts_every_sse_line_terminator(terminator: bytes) -> None:
    payload = orjson.dumps(_chunk({"content": "answer"}, finish_reason="stop"))
    reads = (RawRead(timestamp=1.0, data=b"data: " + payload + terminator + terminator),)

    channels = parse_response_channels(decode_sse_reads(reads))

    assert channels.content == "answer"
    assert channels.finish_reason == "stop"


def test_decoder_joins_a_crlf_terminator_split_across_reads() -> None:
    event = b"data: " + orjson.dumps(_chunk({"content": "answer"}, finish_reason="stop")) + b"\r\n\r\n"
    split_at = len(event) - 1
    reads = (
        RawRead(timestamp=1.0, data=event[:split_at]),
        RawRead(timestamp=1.1, data=event[split_at:]),
    )

    assert parse_response_channels(decode_sse_reads(reads)).content == "answer"


@pytest.mark.parametrize(
    "final_event",
    [b"data: [DONE]\n", b'data: {"choices":[{"delta":{"content":"!"},"finish_reason":"stop"}]}\n'],
    ids=("done-sentinel", "data-event"),
)
def test_decoder_accepts_a_final_event_without_its_blank_line(final_event: bytes) -> None:
    reads = (RawRead(timestamp=1.0, data=_sse(_chunk({"content": "answer"})) + final_event),)

    assert parse_response_channels(decode_sse_reads(reads)).content.startswith("answer")


@pytest.mark.parametrize(
    ("error", "expected_detail"),
    [
        ({"message": "worker crashed", "type": "server_error"}, "worker crashed"),
        ({"type": "server_error"}, '{"type":"server_error"}'),
    ],
    ids=("with-message", "without-message"),
)
def test_midstream_server_error_fails_the_turn(error: JsonObject, expected_detail: str) -> None:
    reads = (
        RawRead(timestamp=1.0, data=_sse(_chunk({"content": "partial"}, finish_reason="stop"))),
        RawRead(timestamp=1.1, data=_sse({"error": error})),
    )

    with pytest.raises(ValueError) as failure:
        measure_request(
            CompletionResult(reads=reads, aborted=False),
            started_at=0.9,
            ended_at=1.2,
            token_counter=CharacterCounter(),
        )

    assert str(failure.value) == f"server stream reported an error: {expected_detail}"


def test_tool_registration_metadata_does_not_start_token_timing() -> None:
    registration = _sse(_chunk({"tool_calls": [{"index": 0, "id": "call-1"}]}))
    arguments = _sse(_chunk({"tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]}))
    result = CompletionResult(
        reads=(
            RawRead(timestamp=20.1, data=registration),
            RawRead(timestamp=20.3, data=arguments),
        ),
        aborted=False,
    )

    metrics = measure_request(
        result,
        started_at=20.0,
        ended_at=20.5,
        token_counter=CharacterCounter(),
    )

    assert metrics.time_to_first_token == pytest.approx(0.3)


@pytest.mark.parametrize("index", [None, True, "0", -1], ids=("missing", "boolean", "text", "negative"))
def test_malformed_tool_call_indices_fail_transport_fidelity(index: int | str | bool | None) -> None:
    call: JsonObject = {"function": {"name": "lookup", "arguments": '{"q":"x"}'}}
    if index is not None:
        call["index"] = index
    chunks = decode_sse_reads(
        (
            RawRead(
                timestamp=1.0,
                data=_sse(_chunk({"tool_calls": [call]}, finish_reason="tool_calls")),
            ),
        )
    )
    channels = parse_response_channels(chunks)

    report = evaluate_tool_fidelity(
        channels.tool_calls,
        (ExpectedToolCall(name="lookup", arguments={"q": "x"}),),
        finish_reason=channels.finish_reason,
    )

    assert report.transport_valid is False
    assert "non_contiguous_indices" in report.issues


async def test_python_client_streams_raw_bytes_from_localhost() -> None:
    events = _events()
    async with LocalSseServer(events, inter_chunk_delay_seconds=STREAM_DELAY_SECONDS) as server:
        client = streaming_client(
            "python", base_url=server.base_url, api_key="secret", timeout_seconds=2.0, max_connections=1
        )
        try:
            result = await client.complete(_request())
        finally:
            await client.close()

    assert result.aborted is False
    assert result.reads
    assert b"".join(read.data for read in result.reads) == b"".join(events)
    assert [read.timestamp for read in result.reads] == sorted(read.timestamp for read in result.reads)
    assert parse_response_channels(decode_sse_reads(result.reads)).content == "answer"
    assert len(server.requests) == 1
    captured = server.requests[0]
    body = parse_json_object(captured.body, "captured request")
    assert captured.method == "POST"
    assert captured.path == "/v1/chat/completions"
    assert captured.headers["authorization"] == "Bearer secret"
    assert captured.headers["x-replay"] == "yes"
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}


async def test_python_client_abort_does_not_wait_for_another_read() -> None:
    abort = asyncio.Event()

    async def abort_while_body_is_stalled() -> None:
        await asyncio.sleep(STREAM_DELAY_SECONDS)
        abort.set()

    async with LocalSseServer(_events(), first_chunk_delay_seconds=STALLED_RESPONSE_SECONDS) as server:
        client = _client("python", server.base_url)
        try:
            result, _ = await asyncio.wait_for(
                asyncio.gather(client.complete(_request(), abort), abort_while_body_is_stalled()),
                ABORT_TEST_TIMEOUT_SECONDS,
            )
        finally:
            await client.close()

    assert result.aborted is True
    assert result.reads == ()


async def test_python_client_reports_partial_reads_after_midstream_abort() -> None:
    abort = asyncio.Event()

    async def abort_after_first_read() -> None:
        await asyncio.sleep(STREAM_DELAY_SECONDS / 2)
        abort.set()

    events = _events()
    async with LocalSseServer(events, inter_chunk_delay_seconds=STREAM_DELAY_SECONDS) as server:
        client = _client("python", server.base_url)
        try:
            result, _ = await asyncio.gather(client.complete(_request(), abort), abort_after_first_read())
        finally:
            await client.close()

    assert result.aborted is True
    assert 0 < len(result.reads) < len(events)


@EVERY_BACKEND
async def test_clients_normalize_http_status_error(backend: ClientBackend) -> None:
    async with LocalSseServer((b'{"error":"bad request"}',), status=HTTP_BAD_REQUEST) as server:
        client = _client(backend, server.base_url)
        try:
            with pytest.raises(CompletionError) as error:
                await client.complete(_request())
        finally:
            await client.close()

    assert error.value.status_code == HTTP_BAD_REQUEST
    assert error.value.response_body == '{"error":"bad request"}'


@EVERY_BACKEND
async def test_clients_normalize_read_timeout(backend: ClientBackend) -> None:
    async with LocalSseServer(_events(), first_chunk_delay_seconds=STALLED_RESPONSE_SECONDS) as server:
        client = _client(backend, server.base_url, timeout_seconds=READ_TIMEOUT_SECONDS)
        try:
            with pytest.raises(CompletionError) as error:
                await asyncio.wait_for(
                    client.complete(_request()),
                    READ_TIMEOUT_TEST_DEADLINE_SECONDS,
                )
        finally:
            await client.close()

    assert str(error.value)
    assert error.value.status_code is None
    assert error.value.response_body is None


@EVERY_BACKEND
async def test_clients_never_follow_endpoint_redirects(backend: ClientBackend) -> None:
    async with LocalSseServer(_events()) as redirect_target:
        location = f"{redirect_target.base_url}/chat/completions"
        async with LocalSseServer(
            (),
            status=HTTP_TEMPORARY_REDIRECT,
            response_headers=(("Location", location),),
        ) as server:
            client = _client(backend, server.base_url)
            try:
                with pytest.raises(CompletionError) as error:
                    await client.complete(_request())
            finally:
                await client.close()

    assert error.value.status_code == HTTP_TEMPORARY_REDIRECT
    assert len(server.requests) == 1
    assert not redirect_target.requests


@EVERY_BACKEND
async def test_clients_request_identity_transfer_encoding(backend: ClientBackend) -> None:
    async with LocalSseServer(_events()) as server:
        client = _client(backend, server.base_url)
        try:
            await client.complete(_request())
        finally:
            await client.close()

    assert len(server.requests) == 1
    assert server.requests[0].headers["accept-encoding"] == "identity"


@REQUIRES_RUSTCORE
async def test_rust_client_read_timestamps_follow_start_on_a_kept_alive_connection() -> None:
    async with LocalSseServer(_events(), keep_alive=True) as server:
        client = _client("rust", server.base_url)
        try:
            for _ in range(KEEP_ALIVE_REQUEST_COUNT):
                started_at = perf_counter()
                result = await client.complete(_request())
                ended_at = perf_counter()
                metrics = measure_request(
                    result,
                    started_at=started_at,
                    ended_at=ended_at,
                    token_counter=CharacterCounter(),
                )

                assert result.reads
                assert result.reads[0].timestamp >= started_at
                assert metrics.time_to_first_byte is not None
                assert metrics.time_to_first_byte >= 0
        finally:
            await client.close()

    assert server.connections == 1
    assert len(server.requests) == KEEP_ALIVE_REQUEST_COUNT


@REQUIRES_RUSTCORE
async def test_rust_client_honors_preset_abort_while_body_is_stalled() -> None:
    abort = asyncio.Event()
    abort.set()

    async with LocalSseServer(_events(), first_chunk_delay_seconds=STALLED_RESPONSE_SECONDS) as server:
        client = _client("rust", server.base_url)
        try:
            result = await asyncio.wait_for(client.complete(_request(), abort), ABORT_TEST_TIMEOUT_SECONDS)
        finally:
            await client.close()

    assert result.aborted is True
    assert result.reads == ()


def _compared_headers(captured: CapturedRequest) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(header for header in captured.raw_headers if header[0] not in UNCOMPARED_HEADERS))


@REQUIRES_RUSTCORE
async def test_rust_and_python_clients_send_identical_request_headers() -> None:
    request = CompletionRequest(
        messages=({"role": "user", "content": "hello"},),
        model="model",
        max_tokens=32,
        # Overriding a default header must replace it, not add a second copy.
        extra_headers=(("accept", "text/event-stream"), ("x-replay", "yes")),
    )
    async with LocalSseServer(_events()) as server:
        for backend in CLIENT_BACKENDS:
            client = streaming_client(
                backend, base_url=server.base_url, api_key="secret", timeout_seconds=2.0, max_connections=1
            )
            try:
                await client.complete(request)
            finally:
                await client.close()

    assert len(server.requests) == len(CLIENT_BACKENDS)
    python_headers, rust_headers = (_compared_headers(captured) for captured in server.requests)
    assert python_headers == rust_headers
    assert dict(python_headers)["user-agent"] == MEASURED_USER_AGENT
    assert [name for name, _ in python_headers].count("accept") == 1
    assert [name for name, _ in rust_headers].count("accept") == 1


@REQUIRES_RUSTCORE
async def test_rust_and_python_clients_have_equivalent_raw_results() -> None:
    events = _events()
    async with LocalSseServer(events, inter_chunk_delay_seconds=STREAM_DELAY_SECONDS) as server:
        python_client = _client("python", server.base_url)
        rust_client = _client("rust", server.base_url)
        try:
            python_started_at = perf_counter()
            python_result = await python_client.complete(_request())
            python_ended_at = perf_counter()
            rust_started_at = perf_counter()
            rust_result = await rust_client.complete(_request())
            rust_ended_at = perf_counter()
        finally:
            await python_client.close()
            await rust_client.close()

    assert python_result.aborted is rust_result.aborted is False
    assert b"".join(read.data for read in python_result.reads) == b"".join(events)
    assert b"".join(read.data for read in rust_result.reads) == b"".join(events)
    python_chunks = decode_sse_reads(python_result.reads)
    rust_chunks = decode_sse_reads(rust_result.reads)
    assert [chunk.data for chunk in python_chunks] == [chunk.data for chunk in rust_chunks]
    assert parse_response_channels(python_chunks) == parse_response_channels(rust_chunks)
    python_metrics = measure_request(
        python_result,
        started_at=python_started_at,
        ended_at=python_ended_at,
        token_counter=CharacterCounter(),
    )
    rust_metrics = measure_request(
        rust_result,
        started_at=rust_started_at,
        ended_at=rust_ended_at,
        token_counter=CharacterCounter(),
    )
    assert rust_metrics.prompt_tokens == python_metrics.prompt_tokens
    assert rust_metrics.cached_prompt_tokens == python_metrics.cached_prompt_tokens
    assert rust_metrics.uncached_prompt_tokens == python_metrics.uncached_prompt_tokens
    assert rust_metrics.output_tokens == python_metrics.output_tokens
    assert rust_metrics.server_output_tokens == python_metrics.server_output_tokens
    assert rust_metrics.channels == python_metrics.channels
    assert rust_metrics.aborted is python_metrics.aborted is False
    for metrics in (python_metrics, rust_metrics):
        assert metrics.time_to_first_byte is not None
        assert 0 <= metrics.time_to_first_byte <= metrics.duration
        assert metrics.time_to_first_token is not None
        assert 0 <= metrics.time_to_first_token <= metrics.duration
    assert [parse_json_object(request.body, "captured request") for request in server.requests] == [
        _request().body(),
        _request().body(),
    ]
    assert [request.body for request in server.requests] == [orjson.dumps(_request().body())] * 2
    assert [request.headers["content-type"] for request in server.requests] == ["application/json"] * 2
