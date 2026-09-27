"""Send a prepared bundle to the submission service and read back its status.

Public surface: submit_bundle, submit_bundle_async, fetch_submission_status,
fetch_revision_allowlist, check_revision_allowlist, revision_advice,
read_submit_token, SubmissionReceipt, SubmissionStatus, RevisionAllowlist,
RevisionCheck, SubmissionError, and the notice, URL, and token constants.
"""

from __future__ import annotations

import asyncio
import base64
import os
import re
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path

import httpx
import orjson
from pydantic import BaseModel

from agentperf_local import __version__
from agentperf_local.client.endpoint import url_is_cleartext_remote
from agentperf_local.common.json_fields import decode_json_object, lenient_string
from agentperf_local.common.json_types import JsonObject, JsonValue
from agentperf_local.provenance.benchmark import PRODUCER_CLIENT_NAME, SourceProvenance
from agentperf_local.replay.config import API_KEY_ENV_PATTERN
from agentperf_local.submission.bundle import (
    ARTIFACT_CONTRACTS,
    ValidatedSubmissionBundle,
    validate_submission_bundle,
)
from agentperf_local.submission.private_audit import PRIVATE_AUDIT_RETENTION_DAYS

SUBMIT_BASE_URL = "https://submit.artificialanalysis.ai"
SUBMIT_TOKEN_ENV = "AGENTPERF_SUBMIT_TOKEN"
SUBMISSIONS_PATH = "/v1/submissions"
REVISIONS_PATH = "/v1/revisions"
MAX_REQUEST_BYTES = 32 * 1024**2
SUBMIT_TIMEOUT_SECONDS = 60.0
STATUS_TIMEOUT_SECONDS = 20.0
REVISIONS_TIMEOUT_SECONDS = 10.0
UPLOAD_CHUNK_BYTES = 64 * 1024
# Each complete or padded three-byte base64 group produces four bytes.
BASE64_INPUT_GROUP_BYTES = 3
BASE64_OUTPUT_GROUP_BYTES = 4
USER_AGENT = f"agentperf-local/{__version__}"
# The wording the user agreed to travels with the request; bump the version when it changes.
PRIVATE_AUDIT_NOTICE_VERSION = "2026-09-03"
PRIVATE_AUDIT_NOTICE = (
    "Artificial Analysis may publish aggregate results and sanitized turn timings.\n"
    f"Hardware and verification evidence stays private and is deleted within {PRIVATE_AUDIT_RETENTION_DAYS} days.\n"
    "Prompts, responses, credentials, local paths, hostnames, serial numbers, and endpoint URLs are never sent.\n"
    "Failed checks are accepted as self-reported. A copy stays on this computer; retries do not duplicate it."
)
SUBMISSION_ID_PATTERN = re.compile(r"^sub_[a-z2-7]{26}$")
SUBMISSION_STATUSES = frozenset(("queued", "validating", "accepted", "rejected"))
TRUST_TIERS = frozenset(("verified", "community-self-reported"))
CLEARTEXT_TOKEN_MESSAGE = "refusing to send a submit token over cleartext http to a non-loopback host"
type UploadProgress = Callable[[int, int], None]


class SubmissionError(RuntimeError):
    """Describe one refused request in the service's own error codes."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        status_code: int | None = None,
        reasons: tuple[str, ...] = (),
        retry_after_seconds: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.reasons = reasons
        self.retry_after_seconds = retry_after_seconds


class SubmissionReceipt(BaseModel, frozen=True):
    """Store the service's answer to one upload."""

    submission_id: str
    status: str
    created: bool
    bytes_sent: int


class _PreparedUpload(BaseModel, frozen=True):
    """Hold one encoded request shared by synchronous and asynchronous transports."""

    url: str
    headers: dict[str, str]
    encoded: bytes


class SubmissionReason(BaseModel, frozen=True):
    """Store one coded reason for a tier placement or a rejection."""

    code: str
    message: str


class SubmissionStatus(BaseModel, frozen=True):
    """Store one submission's current state as the service reports it."""

    submission_id: str
    status: str
    trust_tier: str | None
    reasons: tuple[SubmissionReason, ...]
    submitted_at: str | None
    source_revision: str | None
    public_url: str | None

    def to_json(self) -> JsonObject:
        """Return the status as JSON data."""
        reasons: list[JsonValue] = [{"code": reason.code, "message": reason.message} for reason in self.reasons]
        return {
            "submission_id": self.submission_id,
            "status": self.status,
            "trust_tier": self.trust_tier,
            "reasons": reasons,
            "submitted_at": self.submitted_at,
            "source_revision": self.source_revision,
            "public_url": self.public_url,
        }


class RevisionAllowlist(BaseModel, frozen=True):
    """Store the commit window the service currently accepts for the verified tier."""

    revisions: tuple[str, ...]
    window_days: int | None


class RevisionCheck(BaseModel, frozen=True):
    """Store the outcome of the advisory allowlist check before a run or an upload."""

    reachable: bool
    advice: str | None


def read_submit_token(name: str) -> str | None:
    """Read the submit token from the named variable; unset or blank means an anonymous submission."""
    if not API_KEY_ENV_PATTERN.fullmatch(name):
        raise ValueError("token environment variable names use letters, digits, and underscores")
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return value.strip()


def _reject_cleartext_token(base_url: str, token: str | None) -> None:
    if token is not None and url_is_cleartext_remote(base_url):
        raise ValueError(CLEARTEXT_TOKEN_MESSAGE)


def _headers(token: str | None) -> dict[str, str]:
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _json_body(response: httpx.Response) -> JsonObject:
    try:
        return decode_json_object(response.content, "response body is not a JSON object")
    except ValueError as error:
        raise SubmissionError(
            f"the service answered {response.status_code} without a JSON body",
            code="invalid_response",
            status_code=response.status_code,
        ) from error


def _error_from_response(response: httpx.Response) -> SubmissionError:
    """Turn one error envelope into a typed error; an envelope-less answer keeps the status."""
    try:
        body = _json_body(response)
    except SubmissionError:
        body = {}
    raw_envelope = body.get("error")
    envelope: JsonObject = raw_envelope if isinstance(raw_envelope, dict) else {}
    raw_detail = envelope.get("detail")
    detail: JsonObject = raw_detail if isinstance(raw_detail, dict) else {}
    raw_reasons = detail.get("reasons")
    reasons: tuple[str, ...] = ()
    if isinstance(raw_reasons, list):
        reasons = tuple(reason for reason in raw_reasons if isinstance(reason, str))
    raw_retry = detail.get("retry_after_seconds")
    retry_after_seconds = raw_retry if isinstance(raw_retry, int) and not isinstance(raw_retry, bool) else None
    code = lenient_string(envelope, "code") or f"http_{response.status_code}"
    message = lenient_string(envelope, "message") or f"the service answered {response.status_code}"
    return SubmissionError(
        f"{code}: {message}",
        code=code,
        status_code=response.status_code,
        reasons=reasons,
        retry_after_seconds=retry_after_seconds,
    )


def _request(
    method: str,
    url: str,
    *,
    label: str,
    headers: dict[str, str],
    timeout_seconds: float,
    transport: httpx.BaseTransport | None,
    content: Iterator[bytes] | None = None,
) -> httpx.Response:
    """Send one request; a transport failure becomes a typed error that names no address."""
    with httpx.Client(timeout=timeout_seconds, transport=transport, follow_redirects=False) as client:
        try:
            return client.request(method, url, headers=headers, content=content)
        except httpx.HTTPError as error:
            raise SubmissionError(
                f"the {label} request failed before the service answered: {type(error).__name__}",
                code="transport_error",
            ) from error


async def _async_request(
    method: str,
    url: str,
    *,
    label: str,
    headers: dict[str, str],
    timeout_seconds: float,
    transport: httpx.AsyncBaseTransport | None,
    content: AsyncIterator[bytes] | None = None,
) -> httpx.Response:
    """Send one cancellable request and keep transport details out of errors."""
    async with httpx.AsyncClient(timeout=timeout_seconds, transport=transport, follow_redirects=False) as client:
        try:
            return await client.request(method, url, headers=headers, content=content)
        except httpx.HTTPError as error:
            raise SubmissionError(
                f"the {label} request failed before the service answered: {type(error).__name__}",
                code="transport_error",
            ) from error


def _receipt(response: httpx.Response, bytes_sent: int) -> SubmissionReceipt:
    body = _json_body(response)
    submission_id = lenient_string(body, "submission_id")
    status = lenient_string(body, "status")
    if submission_id is None or SUBMISSION_ID_PATTERN.fullmatch(submission_id) is None:
        raise SubmissionError(
            "the service accepted the upload without a well-formed submission identifier",
            code="invalid_response",
            status_code=response.status_code,
        )
    if status is None or status not in SUBMISSION_STATUSES:
        raise SubmissionError(
            "the service accepted the upload without a known status",
            code="invalid_response",
            status_code=response.status_code,
        )
    return SubmissionReceipt(
        submission_id=submission_id,
        status=status,
        created=response.status_code == httpx.codes.ACCEPTED,
        bytes_sent=bytes_sent,
    )


def build_submission_body(bundle: ValidatedSubmissionBundle) -> bytes:
    """Encode the validated bundle exactly as the service expects it.

    The files travel as the exact bytes the manifest digests; the acknowledgement
    records which notice wording the user agreed to before these bytes left the disk.
    The size cap is checked from the manifest before any file is expanded.
    """
    base64_bytes = sum(
        ((artifact.byte_size + BASE64_INPUT_GROUP_BYTES - 1) // BASE64_INPUT_GROUP_BYTES) * BASE64_OUTPUT_GROUP_BYTES
        for artifact in bundle.manifest.artifacts
    )
    if base64_bytes > MAX_REQUEST_BYTES:
        raise SubmissionError(
            f"the base64-encoded bundle files total {base64_bytes} bytes, "
            f"more than the {MAX_REQUEST_BYTES}-byte request cap",
            code="bundle_too_large",
        )
    files: JsonObject = {
        contract.filename: base64.b64encode(encoded).decode("ascii")
        for contract, encoded in zip(ARTIFACT_CONTRACTS, bundle.artifact_bytes, strict=True)
    }
    body: JsonObject = {
        "bundle_manifest": bundle.manifest.to_json(),
        "files": files,
        "private_audit_acknowledgement": {
            "acknowledged": True,
            "notice_version": PRIVATE_AUDIT_NOTICE_VERSION,
        },
        "client": {"name": PRODUCER_CLIENT_NAME, "version": __version__},
    }
    encoded = orjson.dumps(body)
    if len(encoded) > MAX_REQUEST_BYTES:
        raise SubmissionError(
            f"the encoded bundle is {len(encoded)} bytes, above the {MAX_REQUEST_BYTES}-byte request cap",
            code="bundle_too_large",
        )
    return encoded


def _prepared_upload(
    bundle: Path | ValidatedSubmissionBundle,
    base_url: str,
    token: str | None,
) -> _PreparedUpload:
    _reject_cleartext_token(base_url, token)
    validated = validate_submission_bundle(bundle) if isinstance(bundle, Path) else bundle
    encoded = build_submission_body(validated)
    headers = _headers(token)
    headers["Content-Type"] = "application/json"
    headers["Content-Length"] = str(len(encoded))
    return _PreparedUpload(
        url=f"{base_url.rstrip('/')}{SUBMISSIONS_PATH}",
        headers=headers,
        encoded=encoded,
    )


def _chunks(encoded: bytes, progress: UploadProgress | None) -> Iterator[bytes]:
    sent = 0
    for start in range(0, len(encoded), UPLOAD_CHUNK_BYTES):
        chunk = encoded[start : start + UPLOAD_CHUNK_BYTES]
        yield chunk
        sent += len(chunk)
        if progress is not None:
            progress(sent, len(encoded))


async def _async_chunks(encoded: bytes, progress: UploadProgress | None) -> AsyncIterator[bytes]:
    for chunk in _chunks(encoded, progress):
        yield chunk


def _submission_result(response: httpx.Response, bytes_sent: int) -> SubmissionReceipt:
    if response.status_code in (httpx.codes.ACCEPTED, httpx.codes.OK):
        return _receipt(response, bytes_sent)
    raise _error_from_response(response)


def submit_bundle(
    bundle: Path | ValidatedSubmissionBundle,
    *,
    base_url: str = SUBMIT_BASE_URL,
    token: str | None = None,
    progress: UploadProgress | None = None,
    timeout_seconds: float = SUBMIT_TIMEOUT_SECONDS,
    transport: httpx.BaseTransport | None = None,
) -> SubmissionReceipt:
    """Send one locally validated bundle in one request and return the receipt.

    The request is synchronous. If it is interrupted, running it again with the same
    bundle is the recovery: the service keys submissions on the aggregate digest. A
    path is validated here; callers that already need its parsed fields may pass that
    validated value to avoid reading the files twice.
    """
    prepared = _prepared_upload(bundle, base_url, token)
    response = _request(
        "POST",
        prepared.url,
        label="submission",
        headers=prepared.headers,
        timeout_seconds=timeout_seconds,
        transport=transport,
        content=_chunks(prepared.encoded, progress),
    )
    return _submission_result(response, len(prepared.encoded))


async def submit_bundle_async(
    bundle: Path | ValidatedSubmissionBundle,
    *,
    base_url: str = SUBMIT_BASE_URL,
    token: str | None = None,
    progress: UploadProgress | None = None,
    timeout_seconds: float = SUBMIT_TIMEOUT_SECONDS,
    transport: httpx.AsyncBaseTransport | None = None,
) -> SubmissionReceipt:
    """Send one bundle without hiding cancellation in an executor thread."""
    prepared = await asyncio.to_thread(_prepared_upload, bundle, base_url, token)
    response = await _async_request(
        "POST",
        prepared.url,
        label="submission",
        headers=prepared.headers,
        timeout_seconds=timeout_seconds,
        transport=transport,
        content=_async_chunks(prepared.encoded, progress),
    )
    return _submission_result(response, len(prepared.encoded))


def _reasons(data: JsonObject) -> tuple[SubmissionReason, ...]:
    raw = data.get("reasons")
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise SubmissionError("the service reported reasons in an unknown shape", code="invalid_response")
    reasons: list[SubmissionReason] = []
    for item in raw:
        if not isinstance(item, dict):
            raise SubmissionError("the service reported a reason in an unknown shape", code="invalid_response")
        code = lenient_string(item, "code")
        if code is None:
            raise SubmissionError("the service reported a reason without a code", code="invalid_response")
        reasons.append(SubmissionReason(code=code, message=lenient_string(item, "message") or code))
    return tuple(reasons)


def fetch_submission_status(
    submission_id: str,
    *,
    base_url: str = SUBMIT_BASE_URL,
    token: str | None = None,
    timeout_seconds: float = STATUS_TIMEOUT_SECONDS,
    transport: httpx.BaseTransport | None = None,
) -> SubmissionStatus:
    """Read one submission's status, tier, and reason codes."""
    if SUBMISSION_ID_PATTERN.fullmatch(submission_id) is None:
        raise ValueError("submission identifiers look like sub_ followed by 26 base32 characters")
    _reject_cleartext_token(base_url, token)
    response = _request(
        "GET",
        f"{base_url.rstrip('/')}{SUBMISSIONS_PATH}/{submission_id}",
        label="status",
        headers=_headers(token),
        timeout_seconds=timeout_seconds,
        transport=transport,
    )
    if response.status_code != httpx.codes.OK:
        raise _error_from_response(response)
    body = _json_body(response)
    status = lenient_string(body, "status")
    trust_tier = lenient_string(body, "trust_tier")
    if lenient_string(body, "submission_id") != submission_id or status not in SUBMISSION_STATUSES:
        raise SubmissionError("the service answered with an unknown submission status", code="invalid_response")
    if trust_tier is not None and trust_tier not in TRUST_TIERS:
        raise SubmissionError("the service answered with an unknown trust tier", code="invalid_response")
    return SubmissionStatus(
        submission_id=submission_id,
        status=status,
        trust_tier=trust_tier,
        reasons=_reasons(body),
        submitted_at=lenient_string(body, "submitted_at"),
        source_revision=lenient_string(body, "source_revision"),
        public_url=lenient_string(body, "public_url"),
    )


def fetch_revision_allowlist(
    *,
    base_url: str = SUBMIT_BASE_URL,
    timeout_seconds: float = REVISIONS_TIMEOUT_SECONDS,
    transport: httpx.BaseTransport | None = None,
) -> RevisionAllowlist:
    """Read the public commit allowlist; the service check stays authoritative."""
    response = _request(
        "GET",
        f"{base_url.rstrip('/')}{REVISIONS_PATH}",
        label="allowlist",
        headers=_headers(None),
        timeout_seconds=timeout_seconds,
        transport=transport,
    )
    if response.status_code != httpx.codes.OK:
        raise _error_from_response(response)
    body = _json_body(response)
    raw_revisions = body.get("revisions")
    revisions: list[str] = []
    if isinstance(raw_revisions, list):
        for entry in raw_revisions:
            sha = lenient_string(entry, "sha") if isinstance(entry, dict) else None
            if sha is not None:
                revisions.append(sha)
    window = body.get("window_days")
    return RevisionAllowlist(
        revisions=tuple(revisions),
        window_days=window if isinstance(window, int) and not isinstance(window, bool) else None,
    )


def revision_advice(allowlist: RevisionAllowlist, provenance: SourceProvenance) -> str | None:
    """Say, before a run, why this client build could not reach the verified tier.

    An empty allowlist means the service admits no commit to verified yet. No client
    update could change that, so there is nothing to advise.
    """
    if not allowlist.revisions:
        return None
    if provenance.source_state != "clean":
        return (
            f"the client source tree is {provenance.source_state}; only a clean allowlisted commit can reach verified"
        )
    if provenance.source_revision is None or provenance.source_revision not in allowlist.revisions:
        window = "" if allowlist.window_days is None else f" {allowlist.window_days}-day"
        return (
            f"commit {provenance.source_revision} is outside the service's{window} allowlist window; "
            "update the client to reach verified"
        )
    return None


def check_revision_allowlist(
    provenance: SourceProvenance,
    *,
    base_url: str = SUBMIT_BASE_URL,
    transport: httpx.BaseTransport | None = None,
) -> RevisionCheck:
    """Run the advisory allowlist check; an unreachable service is an outcome, not an error."""
    try:
        allowlist = fetch_revision_allowlist(base_url=base_url, transport=transport)
    except (SubmissionError, ValueError):
        return RevisionCheck(reachable=False, advice=None)
    return RevisionCheck(reachable=True, advice=revision_advice(allowlist, provenance))
