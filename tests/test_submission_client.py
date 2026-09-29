"""Exercise the upload path against a fake submission service on loopback."""

import asyncio
from pathlib import Path

import httpx
import orjson
import pytest

from agentperf_local.cli import main
from agentperf_local.provenance.benchmark import SourceProvenance
from agentperf_local.submission.builder import encode_submission
from agentperf_local.submission.client import (
    CLEARTEXT_TOKEN_MESSAGE,
    MAX_REQUEST_BYTES,
    RevisionAllowlist,
    SubmissionError,
    fetch_revision_allowlist,
    fetch_submission_status,
    revision_advice,
    submit_body,
    submit_body_async,
)
from agentperf_local.submission.notice import PRIVACY_NOTICE_VERSION
from tests.submission_body import sample_submission
from tests.submission_server import (
    ALLOWLISTED_REVISION,
    SUBMISSION_ID,
    VALID_TOKEN,
    FakeSubmissionService,
    LocalSubmissionServer,
)

TOKEN_ENV = "AGENTPERF_TEST_SUBMIT_TOKEN"
CANCELLATION_TEST_TIMEOUT_SECONDS = 5.0
DIFFERENT_WALL_DURATION_MS = 2_000.0


def test_submit_sends_exact_bytes_and_keys_retries_on_run_id() -> None:
    encoded = encode_submission(sample_submission())
    changed = encode_submission(sample_submission(wall_duration_ms=DIFFERENT_WALL_DURATION_MS))
    progress: list[tuple[int, int]] = []

    with LocalSubmissionServer(FakeSubmissionService(require_token=True)) as server:
        receipt = submit_body(
            encoded,
            base_url=server.base_url,
            token=VALID_TOKEN,
            progress=lambda sent, total: progress.append((sent, total)),
        )
        again = submit_body(encoded, base_url=server.base_url, token=VALID_TOKEN)
        with pytest.raises(SubmissionError, match="idempotency_conflict") as conflict:
            submit_body(changed, base_url=server.base_url, token=VALID_TOKEN)
        with pytest.raises(SubmissionError, match="auth_required") as refused:
            submit_body(encoded, base_url=server.base_url, token="wrong-token")
    captured = server.service.captured

    assert receipt.submission_id == again.submission_id == SUBMISSION_ID
    assert receipt.created and not again.created
    assert receipt.status == again.status == "accepted"
    assert progress[-1] == (len(encoded), len(encoded))
    assert [entry.status for entry in captured] == [202, 200, 409, 401]
    assert captured[0].headers["authorization"] == f"Bearer {VALID_TOKEN}"
    assert captured[0].headers["content-length"] == str(len(encoded))
    assert captured[0].headers["user-agent"].startswith("agentperf-local/")
    assert captured[0].body == orjson.loads(encoded)
    assert (conflict.value.status_code, refused.value.status_code) == (409, 401)


async def test_async_submit_propagates_cancellation() -> None:
    encoded = encode_submission(sample_submission())
    started = asyncio.Event()
    cancelled = asyncio.Event()
    release = asyncio.Event()

    async def wait_for_cancel(request: httpx.Request) -> httpx.Response:
        started.set()
        try:
            await release.wait()
        finally:
            cancelled.set()
        return httpx.Response(202, json={"submission_id": SUBMISSION_ID, "status": "accepted"})

    async with asyncio.timeout(CANCELLATION_TEST_TIMEOUT_SECONDS):
        task = asyncio.create_task(
            submit_body_async(encoded, base_url="http://agentperf.test", transport=httpx.MockTransport(wait_for_cancel))
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert cancelled.is_set()


def test_submit_surfaces_the_service_reasons_for_a_refused_body() -> None:
    refused = orjson.loads(encode_submission(sample_submission()))
    refused["trust_tier"] = "verified"

    with LocalSubmissionServer() as server, pytest.raises(SubmissionError, match="validation_failed") as error:
        submit_body(orjson.dumps(refused), base_url=server.base_url)

    assert error.value.status_code == 422
    assert any("trust_tier" in reason for reason in error.value.reasons)


def test_submit_refuses_an_oversized_body_and_a_cleartext_token_before_any_request() -> None:
    with LocalSubmissionServer() as server:
        with pytest.raises(SubmissionError, match="request cap"):
            submit_body(b" " * (MAX_REQUEST_BYTES + 1), base_url=server.base_url)
        assert server.service.captured == []
    with pytest.raises(ValueError, match=CLEARTEXT_TOKEN_MESSAGE):
        submit_body(encode_submission(sample_submission()), base_url="http://submit.example/", token=VALID_TOKEN)


def test_status_and_allowlist_round_trip() -> None:
    service = FakeSubmissionService()
    service.status_answers[SUBMISSION_ID] = {
        "submission_id": SUBMISSION_ID,
        "status": "queued",
        "reasons": [{"code": "resubmit_required", "message": "Resubmit this run to add its BigQuery row."}],
        "submitted_at": "2026-09-29T00:00:00Z",
    }
    release = SourceProvenance(client_version="0.3.0", source_revision=ALLOWLISTED_REVISION, source_state="release")
    stale = SourceProvenance(client_version="0.3.0", source_revision="3" * 40, source_state="clean")
    dirty = SourceProvenance(client_version="0.3.0", source_revision=ALLOWLISTED_REVISION, source_state="dirty")

    with LocalSubmissionServer(service) as server:
        status = fetch_submission_status(SUBMISSION_ID, base_url=server.base_url)
        allowlist = fetch_revision_allowlist(base_url=server.base_url)
        with pytest.raises(SubmissionError, match="not_found"):
            fetch_submission_status("sub_zzzzzzzzzzzzzzzzzzzzzzzzzz", base_url=server.base_url)
    with pytest.raises(ValueError, match="base32"):
        fetch_submission_status("not-an-id")

    assert status.status == "queued"
    assert [reason.code for reason in status.reasons] == ["resubmit_required"]
    assert status.to_json()["submitted_at"] == "2026-09-29T00:00:00Z"
    assert allowlist == RevisionAllowlist(revisions=(ALLOWLISTED_REVISION,), window_days=14)
    assert revision_advice(allowlist, release) is None
    assert "outside" in (revision_advice(allowlist, stale) or "")
    assert "dirty" in (revision_advice(allowlist, dirty) or "")
    assert revision_advice(RevisionAllowlist(revisions=(), window_days=14), stale) is None


def test_submit_cli_requires_acknowledgement_and_never_prints_the_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    submission_path = tmp_path / "submission.json"
    submission_path.write_bytes(encode_submission(sample_submission()))
    monkeypatch.setenv(TOKEN_ENV, VALID_TOKEN)

    with LocalSubmissionServer(FakeSubmissionService(require_token=True)) as server:
        base = ["submit", str(submission_path), "--base-url", server.base_url, "--token-env", TOKEN_ENV]
        assert main(base) == 1
        refused = capsys.readouterr()
        assert main([*base, "--yes"]) == 0
        accepted = capsys.readouterr()
        status_code = main(
            ["submission-status", SUBMISSION_ID, "--base-url", server.base_url, "--token-env", TOKEN_ENV]
        )
        status_output = capsys.readouterr()
        monkeypatch.delenv(TOKEN_ENV)
        assert main([*base, "--yes"]) == 1
        anonymous = capsys.readouterr()

    assert "needs --yes" in refused.err
    assert server.service.captured[0].status == 202
    response = orjson.loads(accepted.out)
    assert response["submission_id"] == SUBMISSION_ID
    assert response["upload_performed"] is True
    assert response["created"] is True
    assert response["privacy_notice_version"] == PRIVACY_NOTICE_VERSION
    assert "private storage indefinitely" in accepted.err
    assert VALID_TOKEN not in accepted.out + accepted.err
    assert status_code == 1
    assert "not_found" in status_output.err
    assert "submitting anonymously" in anonymous.err
    assert "auth_required" in anonymous.err
