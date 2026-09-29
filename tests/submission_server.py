"""Serve a deterministic fake of the submission API, and of GitHub's commit lookup, on a localhost port."""

from __future__ import annotations

import hashlib
import re
import threading
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import orjson
from pydantic import BaseModel

from agentperf_local.common.json_types import JsonObject, JsonValue, normalize_json_object
from agentperf_local.common.models import read_record
from agentperf_local.submission.contract import SubmissionRequest
from agentperf_local.submission.spec import validate_against_spec

VALID_TOKEN = "aa-test-token"
SUBMISSION_ID = "sub_abcdefghijklmnopqrstuvwxyz"
ALLOWLISTED_REVISION = "2" * 40
GITHUB_COMMIT_PATH = re.compile(r"^/repos/[^/]+/[^/]+/commits/(?P<ref>[^/]+)$")
SHORT_COMMIT = re.compile(r"^[0-9a-f]{7,40}$")
FULL_COMMIT_LENGTH = 40


class CapturedSubmission(BaseModel, frozen=True):
    """Store one request the fake service accepted or refused."""

    headers: dict[str, str]
    body: JsonObject
    status: int


@dataclass(slots=True)
class FakeSubmissionService:
    """Hold the fake service's state across requests."""

    require_token: bool = False
    response_release: threading.Event | None = None
    # The canonical content accepted for each run_id; the service keys submissions on it.
    accepted_content: dict[str, bytes] = field(default_factory=dict)
    captured: list[CapturedSubmission] = field(default_factory=list)
    status_answers: dict[str, JsonObject] = field(default_factory=dict)
    looked_up_refs: list[str] = field(default_factory=list)


def fake_commit(ref: str) -> str:
    """Return the full commit the fake GitHub names for a short commit or a tag."""
    if SHORT_COMMIT.fullmatch(ref) is not None:
        return ref.ljust(FULL_COMMIT_LENGTH, "0")
    return hashlib.sha1(ref.encode(), usedforsecurity=False).hexdigest()


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
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            if service.response_release is not None:
                service.response_release.wait()
            headers = {name.lower(): value for name, value in self.headers.items()}
            token = headers.get("authorization", "").removeprefix("Bearer ")
            if service.require_token and token != VALID_TOKEN:
                service.captured.append(CapturedSubmission(headers=headers, body={}, status=401))
                _error(self, HTTPStatus.UNAUTHORIZED, "auth_required", "token missing or invalid", {})
                return
            body = normalize_json_object(orjson.loads(raw))
            try:
                request = read_record(SubmissionRequest, raw, "request")
                validate_against_spec(raw)
            except ValueError as error:
                service.captured.append(CapturedSubmission(headers=headers, body=body, status=422))
                reasons: list[JsonValue] = []
                reasons.extend(str(error).splitlines())
                detail: JsonObject = {"reasons": reasons}
                _error(
                    self, HTTPStatus.UNPROCESSABLE_ENTITY, "validation_failed", "submission failed validation", detail
                )
                return
            # Formatting and key order do not change the content.
            content = orjson.dumps(body, option=orjson.OPT_SORT_KEYS)
            accepted = service.accepted_content.get(request.run_id)
            if accepted is not None and accepted != content:
                service.captured.append(CapturedSubmission(headers=headers, body=body, status=409))
                _error(self, HTTPStatus.CONFLICT, "idempotency_conflict", "different content", {"reasons": []})
                return
            status = HTTPStatus.OK if accepted is not None else HTTPStatus.ACCEPTED
            service.accepted_content[request.run_id] = content
            service.captured.append(CapturedSubmission(headers=headers, body=body, status=status))
            _json(self, status, {"submission_id": SUBMISSION_ID, "status": "accepted"})

        def do_GET(self) -> None:
            commit_path = GITHUB_COMMIT_PATH.fullmatch(self.path)
            if commit_path is not None:
                ref = commit_path.group("ref")
                service.looked_up_refs.append(ref)
                _json(self, HTTPStatus.OK, {"sha": fake_commit(ref)})
                return
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
                answer = service.status_answers.get(self.path.removeprefix(prefix))
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
