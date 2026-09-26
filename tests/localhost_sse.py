"""Provide a deterministic localhost SSE server for public client tests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import orjson

from tests.sse_ignore_eos import ignore_eos_max_tokens_of, parse_request, rewrite_event

HTTP_OK = 200
HTTP_BAD_REQUEST = 400
HTTP_NOT_FOUND = 404
OLLAMA_VERSION = "0.12.0"
OLLAMA_IDENTITY_PATHS = ("/api/version", "/api/tags")
SERVER_CLOSE_TIMEOUT_SECONDS = 1.0
SSE_OK_RESPONSE = (
    b'data: {"choices":[{"delta":{"content":"o"},"finish_reason":null}]}\n\n'
    b'data: {"choices":[{"delta":{"content":"k"},"finish_reason":"stop"}]}\n\n'
    b'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}\n\n'
    b"data: [DONE]\n\n"
)


@dataclass(frozen=True, slots=True)
class CapturedRequest:
    """Store one request received by the localhost server."""

    method: str
    path: str
    headers: dict[str, str]
    raw_headers: tuple[tuple[str, str], ...]
    body: bytes
    # The body parsed once, so the server and the tests read the same request facts.
    json: dict[str, object] | None

    @property
    def asks_ignore_eos(self) -> bool:
        """Return whether this completion request carried the ignore_eos field."""
        return self.json is not None and self.json.get("ignore_eos") is True


class LocalSseServer:
    """Serve deterministic SSE bytes over an ephemeral localhost socket."""

    def __init__(
        self,
        chunks: tuple[bytes, ...],
        *,
        responses: tuple[tuple[bytes, ...], ...] | None = None,
        status: int = HTTP_OK,
        response_headers: tuple[tuple[str, str], ...] = (),
        first_chunk_delay_seconds: float = 0.0,
        inter_chunk_delay_seconds: float = 0.0,
        keep_alive: bool = False,
        models: tuple[str, ...] = (),
        served_context_tokens: int | None = None,
        honours_ignore_eos: bool = True,
        rejects_ignore_eos: bool = False,
        answers_as_ollama: bool = False,
        answers_ollama_version_only: bool = False,
        ollama_trickle_seconds: float = 0.0,
    ) -> None:
        self._responses = responses if responses is not None else (chunks,)
        # A server that drops ignore_eos, as Ollama does, serves the script unchanged; a
        # strict gateway refuses any request that carries the field.
        self._honours_ignore_eos = honours_ignore_eos
        self._rejects_ignore_eos = rejects_ignore_eos
        # Ollama answers both identity paths; an impostor answers /api/version alone.
        self._answers_as_ollama = answers_as_ollama
        self._answers_ollama_version = answers_as_ollama or answers_ollama_version_only
        # A positive delay sends the identity answers one byte at a time.
        self._ollama_trickle_seconds = ollama_trickle_seconds
        if not self._responses:
            raise ValueError("localhost SSE server needs at least one response")
        self._models = models
        self._served_context_tokens = served_context_tokens
        self._status = status
        self._response_headers = response_headers
        self._first_chunk_delay_seconds = first_chunk_delay_seconds
        self._inter_chunk_delay_seconds = inter_chunk_delay_seconds
        self._keep_alive = keep_alive
        self._server: asyncio.AbstractServer | None = None
        self._tasks: list[asyncio.Task[None]] = []
        self.requests: list[CapturedRequest] = []
        self._response_number = 0
        self.connections = 0
        self.base_url = ""

    async def __aenter__(self) -> LocalSseServer:
        self._server = await asyncio.start_server(self._connected, "127.0.0.1", 0)
        sockets = self._server.sockets
        if not sockets:
            raise RuntimeError("localhost SSE server did not bind a socket")
        address: object = sockets[0].getsockname()
        if not isinstance(address, tuple) or len(address) < 2 or not isinstance(address[1], int):
            raise RuntimeError("localhost SSE server returned an invalid address")
        self.base_url = f"http://127.0.0.1:{address[1]}/v1"
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        for task in self._tasks:
            if not task.done():
                task.cancel()
        if self._tasks:
            await asyncio.wait_for(asyncio.gather(*self._tasks, return_exceptions=True), SERVER_CLOSE_TIMEOUT_SECONDS)

    def _connected(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.connections += 1
        self._tasks.append(asyncio.create_task(self._serve(reader, writer)))

    def _models_response(self) -> bytes:
        """Answer GET /models the way an OpenAI-compatible server lists what it serves."""
        entries: list[dict[str, str | dict[str, int]]] = []
        for model in self._models:
            entry: dict[str, str | dict[str, int]] = {"id": model, "object": "model"}
            if self._served_context_tokens is not None:
                entry["meta"] = {"n_ctx": self._served_context_tokens}
            entries.append(entry)
        return orjson.dumps({"object": "list", "data": entries})

    def _canned_response(self, request: CapturedRequest) -> bytes | None:
        """Return the JSON answer for a request that consumes no scripted stream, else None.

        The TUI's server check lands on GET /models, the Ollama check on GET /api/version
        and GET /api/tags, and a strict gateway refuses a completion that carries ignore_eos
        before any generation happens.
        """
        if request.method == "GET" and request.path.endswith("/models"):
            return _json_response(self._models_response(), keep_alive=self._keep_alive)
        if request.method == "GET" and request.path in OLLAMA_IDENTITY_PATHS:
            if request.path == "/api/version" and self._answers_ollama_version:
                return _json_response(orjson.dumps({"version": OLLAMA_VERSION}), keep_alive=self._keep_alive)
            if request.path == "/api/tags" and self._answers_as_ollama:
                tags = {"models": [{"name": model} for model in self._models]}
                return _json_response(orjson.dumps(tags), keep_alive=self._keep_alive)
            body = orjson.dumps({"error": "not found"})
            return _json_response(body, status=HTTP_NOT_FOUND, keep_alive=self._keep_alive)
        if self._rejects_ignore_eos and request.asks_ignore_eos:
            body = orjson.dumps({"error": {"message": "unknown field: ignore_eos"}})
            return _json_response(body, status=HTTP_BAD_REQUEST, keep_alive=self._keep_alive)
        return None

    async def _write_canned(self, writer: asyncio.StreamWriter, canned: bytes, *, trickle: bool) -> None:
        """Write one canned answer, a byte at a time when this server trickles its Ollama answers."""
        if not (trickle and self._ollama_trickle_seconds):
            writer.write(canned)
            await writer.drain()
            return
        for index in range(len(canned)):
            writer.write(canned[index : index + 1])
            await writer.drain()
            await asyncio.sleep(self._ollama_trickle_seconds)

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                request = await _read_request(reader)
                self.requests.append(request)
                canned = self._canned_response(request)
                if canned is not None:
                    await self._write_canned(writer, canned, trickle=request.path in OLLAMA_IDENTITY_PATHS)
                    if not self._keep_alive:
                        return
                    continue
                response_chunks = self._responses[min(self._response_number, len(self._responses) - 1)]
                self._response_number += 1
                if self._honours_ignore_eos:
                    response_chunks = _honour_ignore_eos(request.json, response_chunks)
                writer.write(
                    _response_head(
                        self._status,
                        sum(len(chunk) for chunk in response_chunks),
                        self._response_headers,
                        keep_alive=self._keep_alive,
                    )
                )
                await writer.drain()
                if self._first_chunk_delay_seconds:
                    await asyncio.sleep(self._first_chunk_delay_seconds)
                for index, chunk in enumerate(response_chunks):
                    if index and self._inter_chunk_delay_seconds:
                        await asyncio.sleep(self._inter_chunk_delay_seconds)
                    writer.write(chunk)
                    await writer.drain()
                if not self._keep_alive:
                    return
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionError:
                pass


def _json_response(body: bytes, *, keep_alive: bool, status: int = HTTP_OK) -> bytes:
    """Frame one JSON body as a complete HTTP response."""
    connection = "keep-alive" if keep_alive else "close"
    head = (
        f"HTTP/1.1 {status} \r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"Connection: {connection}\r\n"
        "\r\n"
    ).encode("ascii")
    return head + body


def _honour_ignore_eos(request: dict[str, object] | None, chunks: tuple[bytes, ...]) -> tuple[bytes, ...]:
    """Answer an ignore_eos request the way a real server does.

    A server told to ignore end-of-sequence generates exactly max_tokens and finishes
    with "length". The scripted content is kept; the closing finish reason and the
    usage count are rewritten so the replay sees the length it asked for.
    """
    max_tokens = ignore_eos_max_tokens_of(request)
    if max_tokens is None:
        return chunks
    # A chunk may bundle several events joined by blank lines; rewrite each and rejoin.
    return tuple(b"\n\n".join(rewrite_event(event, max_tokens) for event in chunk.split(b"\n\n")) for chunk in chunks)


async def _read_request(reader: asyncio.StreamReader) -> CapturedRequest:
    head = await reader.readuntil(b"\r\n\r\n")
    lines = head.removesuffix(b"\r\n\r\n").split(b"\r\n")
    request_parts = lines[0].decode("ascii").split()
    if len(request_parts) != 3:
        raise ValueError("localhost server received an invalid HTTP request line")
    raw_headers: list[tuple[str, str]] = []
    for line in lines[1:]:
        name, separator, value = line.partition(b":")
        if not separator:
            raise ValueError("localhost server received an invalid HTTP header")
        raw_headers.append((name.decode("ascii").lower(), value.decode("latin-1").strip()))
    headers = dict(raw_headers)
    content_length = int(headers.get("content-length", "0"))
    body = await reader.readexactly(content_length)
    return CapturedRequest(
        method=request_parts[0],
        path=request_parts[1],
        headers=headers,
        raw_headers=tuple(raw_headers),
        body=body,
        json=parse_request(body) if body else None,
    )


def _response_head(
    status: int,
    content_length: int,
    headers: tuple[tuple[str, str], ...],
    *,
    keep_alive: bool,
) -> bytes:
    reason = "OK" if status == HTTP_OK else "Bad Request"
    additional_headers = "".join(f"{name}: {value}\r\n" for name, value in headers)
    connection = "keep-alive" if keep_alive else "close"
    return (
        f"HTTP/1.1 {status} {reason}\r\n"
        "Content-Type: text/event-stream\r\n"
        f"Content-Length: {content_length}\r\n"
        f"{additional_headers}"
        f"Connection: {connection}\r\n"
        "\r\n"
    ).encode("ascii")
