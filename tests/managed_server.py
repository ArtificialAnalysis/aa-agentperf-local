"""Serve the minimal model-list endpoint used by managed deployment tests."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import orjson

from tests.sse_ignore_eos import ignore_eos_max_tokens, rewrite_event

DEFAULT_SERVED_CONTEXT_TOKENS = 65536
# One short streamed answer: two visible tokens, a usage block, then the terminator.
# Each event is flushed after a pause so every turn has a measurable decode window.
SSE_EVENTS = (
    b'data: {"choices":[{"delta":{"content":"o"},"finish_reason":null}]}\n\n',
    b'data: {"choices":[{"delta":{"content":"k"},"finish_reason":"stop"}]}\n\n',
    b'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}\n\n',
    b"data: [DONE]\n\n",
)
EVENT_DELAY_SECONDS = 0.005
STARTUP_LINES = {
    "metal": "ggml_metal_init: Metal fixture; offloaded 1/1 layers to GPU",
    "cuda": "using device CUDA0 (fixture); CUDA0 model buffer size = 1 MiB; offloaded 1/1 layers to GPU",
    "rocm": "using device ROCm0 (fixture); ROCm0 model buffer size = 1 MiB; offloaded 1/1 layers to GPU",
    "vulkan": "using device Vulkan0 (fixture); Vulkan0 model buffer size = 1 MiB; offloaded 1/1 layers to GPU",
}
SPLASH_STARTUP_LINE = "20:59:43 Kernel policy for GPU family 10 with 20 cores."


def _events_for(body: bytes, honors_ignore_eos: bool) -> tuple[bytes, ...]:
    """Answer an ignore_eos request the way a real server does: the full length, ending on "length"."""
    max_tokens = ignore_eos_max_tokens(body)
    if max_tokens is None or not honors_ignore_eos:
        return SSE_EVENTS
    # Each scripted event already carries its trailing blank line; rewrite the body and restore it.
    return tuple(rewrite_event(event.removesuffix(b"\n\n"), max_tokens) + b"\n\n" for event in SSE_EVENTS)


def _handler(
    model_alias: str, served_context_tokens: int | None, backend: str, honors_ignore_eos: bool
) -> type[BaseHTTPRequestHandler]:
    class ModelHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path != "/v1/chat/completions":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for event in _events_for(body, honors_ignore_eos):
                self.wfile.write(event)
                self.wfile.flush()
                time.sleep(EVENT_DELAY_SECONDS)

        def do_GET(self) -> None:
            if backend == "splash" and self.path == "/status":
                # Splash lists its models without a context and reports it on its status endpoint.
                status = {} if served_context_tokens is None else {"maximum_context_tokens": served_context_tokens}
                self._send_json(orjson.dumps(status))
                return
            if self.path != "/v1/models":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            model: dict[str, object] = {"id": model_alias, "object": "model"}
            if backend == "splash":
                model["owned_by"] = "splash"
            elif served_context_tokens is not None:
                # llama.cpp reports the served context under meta; SGLang reports max_model_len.
                if backend in ("sglang", "vllm"):
                    model["max_model_len"] = served_context_tokens
                else:
                    model["meta"] = {"n_ctx": served_context_tokens}
            self._send_json(orjson.dumps({"object": "list", "data": [model]}))

        def _send_json(self, body: bytes) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    return ModelHandler


def main(argv: Sequence[str] | None = None) -> int:
    """Run the local model-list fixture until its parent terminates it."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--offloaded", default="1/1")
    parser.add_argument("--ctx-size", type=int, default=DEFAULT_SERVED_CONTEXT_TOKENS)
    parser.add_argument("--hide-ctx", action="store_true", help="omit the served context like a silent server")
    parser.add_argument("--platform", choices=tuple(STARTUP_LINES), default="metal")
    parser.add_argument("--device")
    parser.add_argument("--backend", default="llama-cpp", choices=("llama-cpp", "sglang", "vllm", "splash"))
    parser.add_argument("--token-pool", type=int, default=DEFAULT_SERVED_CONTEXT_TOKENS)
    parser.add_argument("--drop-ignore-eos", action="store_true", help="stop early like a server without ignore_eos")
    namespace = parser.parse_args(argv)
    port = namespace.port
    model_alias = namespace.alias
    offloaded = namespace.offloaded
    context_tokens = namespace.ctx_size
    if not isinstance(port, int) or isinstance(port, bool):
        raise RuntimeError("test port was not parsed as an integer")
    if not isinstance(model_alias, str):
        raise RuntimeError("test model alias was not parsed as text")
    if not isinstance(offloaded, str):
        raise RuntimeError("test offload count was not parsed as text")
    if not isinstance(context_tokens, int) or isinstance(context_tokens, bool):
        raise RuntimeError("test context size was not parsed as an integer")
    backend = namespace.backend
    token_pool = namespace.token_pool
    if not isinstance(backend, str) or not isinstance(token_pool, int) or isinstance(token_pool, bool):
        raise RuntimeError("test backend or token pool was not parsed")
    served_context_tokens = None if namespace.hide_ctx else context_tokens
    if backend == "sglang":
        print("Detected platform cuda; capture target decode CUDA graph", file=sys.stderr, flush=True)
        print(f"max_total_num_tokens={token_pool}, max_running_requests=1", file=sys.stderr, flush=True)
    elif backend == "vllm":
        print("Automatically detected platform cuda.", file=sys.stderr, flush=True)
        print(f"GPU KV cache size: {token_pool} tokens", file=sys.stderr, flush=True)
    elif backend == "splash":
        print(SPLASH_STARTUP_LINE, file=sys.stderr, flush=True)
    else:
        print(STARTUP_LINES[namespace.platform].replace("1/1", offloaded), file=sys.stderr, flush=True)
    server = ThreadingHTTPServer(
        ("127.0.0.1", port), _handler(model_alias, served_context_tokens, backend, not namespace.drop_ignore_eos)
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
