"""Rewrite a scripted SSE response the way a server that honours ignore_eos does.

A request with ignore_eos generates exactly max_tokens and finishes on "length". Both
localhost test servers share this transform so each emulates a real server the same way.
"""

from __future__ import annotations

import orjson


def parse_request(request_body: bytes) -> dict[str, object] | None:
    """Return the request's JSON object, or None when the body is not one."""
    try:
        request = orjson.loads(request_body)
    except orjson.JSONDecodeError:
        return None
    return request if isinstance(request, dict) else None


def ignore_eos_max_tokens(request_body: bytes) -> int | None:
    """Return the request's max_tokens when it asked to ignore end-of-sequence, else None."""
    return ignore_eos_max_tokens_of(parse_request(request_body))


def ignore_eos_max_tokens_of(request: dict[str, object] | None) -> int | None:
    """Read the ignore_eos budget from an already parsed request."""
    if request is None or request.get("ignore_eos") is not True:
        return None
    max_tokens = request.get("max_tokens")
    return max_tokens if isinstance(max_tokens, int) else None


def rewrite_event(event: bytes, max_tokens: int) -> bytes:
    """Rewrite one `data: {...}` SSE event to finish on length with max_tokens of usage.

    A line that is not a JSON data event (a blank separator, `data: [DONE]`) is returned
    unchanged. The event carries no trailing blank line; the caller frames the events.
    """
    if not event.startswith(b"data: {"):
        return event
    data = orjson.loads(event[len(b"data: ") :])
    for choice in data.get("choices") or ():
        if choice.get("finish_reason") is not None:
            choice["finish_reason"] = "length"
    usage = data.get("usage")
    if isinstance(usage, dict):
        usage["completion_tokens"] = max_tokens
        usage["total_tokens"] = usage.get("prompt_tokens", 0) + max_tokens
    return b"data: " + orjson.dumps(data)
