"""Serve a deterministic fake of the submission API on an ephemeral localhost port."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import orjson
from pydantic import BaseModel

from agentperf_local.common.json_types import JsonObject, normalize_json_object

VALID_TOKEN = "aa-test-token"
SUBMISSION_ID = "sub_abcdefghijklmnopqrstuvwxyz"
ALLOWLISTED_REVISION = "2" * 40


class CapturedSubmission(BaseModel, frozen=True):
    """Store one request the fake service accepted or refused."""

    headers: dict[str, str]
    body: JsonObject
    status: int


@dataclass(slots=True)
class FakeSubmissionService:
    """Hold the fake service's state across requests."""

    body_cap_bytes: int | None = None
    rate_limited: bool = False
    require_token: bool = False
    response_release: threading.Event | None = None
    seen_digests: dict[str, str] = field(default_factory=dict)
    captured: list[CapturedSubmission] = field(default_factory=list)
    status_answers: dict[str, JsonObject] = field(default_factory=dict)


def _error(handler: BaseHTTPRequestHandler, status: HTTPStatus, code: str, message: str, detail: JsonObject) -> None:
    body = orjson.dumps({"error": {"code": code, "message": message, "detail": detail}})
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _json(handler: BaseHTTPRequestHandler, status: HTTPStatus, payload: JsonObject) -> None:
    body = orjson.dumps(payload)
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _handler(service: FakeSubmissionService) -> type[BaseHTTPRequestHandler]:
    class SubmissionHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path != "/v1/submissions":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            if service.response_release is not None:
                service.response_release.wait()
            headers = {name.lower(): value for name, value in self.headers.items()}
            if service.body_cap_bytes is not None and length > service.body_cap_bytes:
                service.captured.append(CapturedSubmission(headers=headers, body={}, status=413))
                _error(self, HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "bundle_too_large", "too large", {})
                return
            if service.rate_limited:
                service.captured.append(CapturedSubmission(headers=headers, body={}, status=429))
                _error(self, HTTPStatus.TOO_MANY_REQUESTS, "rate_limited", "slow down", {"retry_after_seconds": 30})
                return
            token = headers.get("authorization", "").removeprefix("Bearer ")
            if service.require_token and token != VALID_TOKEN:
                service.captured.append(CapturedSubmission(headers=headers, body={}, status=401))
                _error(self, HTTPStatus.UNAUTHORIZED, "auth_required", "token missing or invalid", {})
                return
            body = normalize_json_object(orjson.loads(raw))
            service.captured.append(CapturedSubmission(headers=headers, body=body, status=0))
            manifest = body.get("bundle_manifest")
            digest = manifest.get("aggregate_payload_digest") if isinstance(manifest, dict) else None
            key = digest if isinstance(digest, str) else ""
            if key in service.seen_digests:
                service.captured[-1] = CapturedSubmission(headers=headers, body=body, status=200)
                _json(self, HTTPStatus.OK, {"submission_id": service.seen_digests[key], "status": "queued"})
                return
            service.seen_digests[key] = SUBMISSION_ID
            service.captured[-1] = CapturedSubmission(headers=headers, body=body, status=202)
            _json(self, HTTPStatus.ACCEPTED, {"submission_id": SUBMISSION_ID, "status": "queued"})

        def do_GET(self) -> None:
            if self.path == "/v1/revisions":
                _json(
                    self,
                    HTTPStatus.OK,
                    {
                        "minimum_revision": ALLOWLISTED_REVISION,
                        "revisions": [{"sha": ALLOWLISTED_REVISION, "committed_at": "2026-09-01T00:00:00Z"}],
                        "window_days": 14,
                    },
                )
                return
            prefix = "/v1/submissions/"
            if self.path.startswith(prefix):
                submission_id = self.path.removeprefix(prefix)
                answer = service.status_answers.get(submission_id)
                if answer is None:
                    _error(self, HTTPStatus.NOT_FOUND, "not_found", "no such submission", {})
                    return
                _json(self, HTTPStatus.OK, answer)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *args: object) -> None:
            return

    return SubmissionHandler


class LocalSubmissionServer:
    """Run the fake service in a background thread for the life of a with-block."""

    def __init__(self, service: FakeSubmissionService | None = None) -> None:
        self.service = service if service is not None else FakeSubmissionService()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(self.service))
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        """Return the loopback base URL of the fake service."""
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> LocalSubmissionServer:
        self._thread.start()
        return self

    def __exit__(self, exception_type: object, exception: object, traceback: object) -> None:
        if self.service.response_release is not None:
            self.service.response_release.set()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)
