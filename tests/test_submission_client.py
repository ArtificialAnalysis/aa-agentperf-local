"""Exercise the upload path against a fake submission service on loopback."""

import asyncio
import base64
from pathlib import Path

import httpx
import orjson
import pytest

from agentperf_local.cli import main
from agentperf_local.provenance.benchmark import SourceProvenance
from agentperf_local.submission.bundle import AGGREGATE_FILENAME, build_submission_bundle, write_submission_bundle
from agentperf_local.submission.client import (
    CLEARTEXT_TOKEN_MESSAGE,
    PRIVATE_AUDIT_NOTICE_VERSION,
    RevisionAllowlist,
    SubmissionError,
    fetch_revision_allowlist,
    fetch_submission_status,
    revision_advice,
    submit_bundle,
    submit_bundle_async,
)
from tests.submission_server import (
    ALLOWLISTED_REVISION,
    SUBMISSION_ID,
    VALID_TOKEN,
    FakeSubmissionService,
    LocalSubmissionServer,
)
from tests.test_submission import _private_summary, _write_bound_results

TOKEN_ENV = "AGENTPERF_TEST_SUBMIT_TOKEN"
CANCELLATION_TEST_TIMEOUT_SECONDS = 5.0


def _prepared_bundle(tmp_path: Path) -> Path:
    results_dir = _write_bound_results(tmp_path, _private_summary())
    bundle_dir = tmp_path / "bundle"
    write_submission_bundle(bundle_dir, build_submission_bundle(results_dir))
    return bundle_dir


def test_submit_sends_exact_bytes_once_and_reports_progress(tmp_path: Path) -> None:
    bundle_dir = _prepared_bundle(tmp_path)
    progress: list[tuple[int, int]] = []

    with LocalSubmissionServer(FakeSubmissionService(require_token=True)) as server:
        receipt = submit_bundle(
            bundle_dir,
            base_url=server.base_url,
            token=VALID_TOKEN,
            progress=lambda sent, total: progress.append((sent, total)),
        )
        again = submit_bundle(bundle_dir, base_url=server.base_url, token=VALID_TOKEN)
        with pytest.raises(SubmissionError, match="auth_required") as refused:
            submit_bundle(bundle_dir, base_url=server.base_url, token="wrong-token")
    captured = server.service.captured

    assert receipt.submission_id == again.submission_id == SUBMISSION_ID
    assert receipt.created and not again.created
    assert receipt.status == again.status == "queued"
    assert progress[-1] == (receipt.bytes_sent, receipt.bytes_sent)
    assert [entry.status for entry in captured] == [202, 200, 401]
    assert captured[0].headers["authorization"] == f"Bearer {VALID_TOKEN}"
    assert captured[0].headers["content-length"] == str(receipt.bytes_sent)
    assert captured[0].headers["user-agent"].startswith("agentperf-local/")
    files = captured[0].body["files"]
    assert isinstance(files, dict)
    encoded_aggregate = files[AGGREGATE_FILENAME]
    assert isinstance(encoded_aggregate, str)
    assert base64.b64decode(encoded_aggregate) == (bundle_dir / AGGREGATE_FILENAME).read_bytes()
    assert captured[0].body["private_audit_acknowledgement"] == {
        "acknowledged": True,
        "notice_version": PRIVATE_AUDIT_NOTICE_VERSION,
    }
    assert refused.value.status_code == 401


async def test_async_submit_propagates_cancellation(tmp_path: Path) -> None:
    bundle_dir = _prepared_bundle(tmp_path)
    started = asyncio.Event()
    cancelled = asyncio.Event()
    release = asyncio.Event()

    async def wait_for_cancel(request: httpx.Request) -> httpx.Response:
        started.set()
        try:
            await release.wait()
        finally:
            cancelled.set()
        return httpx.Response(202, json={"submission_id": SUBMISSION_ID, "status": "queued"})

    async with asyncio.timeout(CANCELLATION_TEST_TIMEOUT_SECONDS):
        task = asyncio.create_task(
            submit_bundle_async(
                bundle_dir, base_url="http://agentperf.test", transport=httpx.MockTransport(wait_for_cancel)
            )
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert cancelled.is_set()


@pytest.mark.parametrize(
    ("service", "code", "status_code"),
    [
        (FakeSubmissionService(body_cap_bytes=1), "bundle_too_large", 413),
        (FakeSubmissionService(rate_limited=True), "rate_limited", 429),
    ],
    ids=("too-large", "rate-limited"),
)
def test_submit_surfaces_service_refusals_with_their_codes(
    tmp_path: Path,
    service: FakeSubmissionService,
    code: str,
    status_code: int,
) -> None:
    bundle_dir = _prepared_bundle(tmp_path)

    with LocalSubmissionServer(service) as server:
        with pytest.raises(SubmissionError, match=code) as error:
            submit_bundle(bundle_dir, base_url=server.base_url)

    assert error.value.status_code == status_code
    assert error.value.code == code
    if code == "rate_limited":
        assert error.value.retry_after_seconds == 30


def test_submit_validates_locally_before_any_request_and_refuses_cleartext_tokens(tmp_path: Path) -> None:
    bundle_dir = _prepared_bundle(tmp_path)
    aggregate_path = bundle_dir / AGGREGATE_FILENAME
    aggregate_path.write_bytes(aggregate_path.read_bytes() + b" ")

    with LocalSubmissionServer() as server:
        with pytest.raises(ValueError, match="bytes do not match"):
            submit_bundle(bundle_dir, base_url=server.base_url)
        assert server.service.captured == []
    with pytest.raises(ValueError, match=CLEARTEXT_TOKEN_MESSAGE):
        submit_bundle(bundle_dir, base_url="http://submit.example/", token=VALID_TOKEN)


def test_status_and_allowlist_round_trip(tmp_path: Path) -> None:
    service = FakeSubmissionService()
    service.status_answers[SUBMISSION_ID] = {
        "submission_id": SUBMISSION_ID,
        "status": "accepted",
        "trust_tier": "community-self-reported",
        "reasons": [{"code": "power_coverage_insufficient", "message": "Power coverage was below 0.95."}],
        "submitted_at": "2026-09-02T00:00:00Z",
        "source_revision": ALLOWLISTED_REVISION,
        "public_url": "https://artificialanalysis.ai/agentperf/sub",
    }
    clean = SourceProvenance(client_version="0.1.0", source_revision=ALLOWLISTED_REVISION, source_state="clean")
    stale = SourceProvenance(client_version="0.1.0", source_revision="3" * 40, source_state="clean")
    dirty = SourceProvenance(client_version="0.1.0", source_revision=ALLOWLISTED_REVISION, source_state="dirty")

    with LocalSubmissionServer(service) as server:
        status = fetch_submission_status(SUBMISSION_ID, base_url=server.base_url)
        allowlist = fetch_revision_allowlist(base_url=server.base_url)
        with pytest.raises(SubmissionError, match="not_found"):
            fetch_submission_status("sub_zzzzzzzzzzzzzzzzzzzzzzzzzz", base_url=server.base_url)
    with pytest.raises(ValueError, match="base32"):
        fetch_submission_status("not-an-id")

    assert status.trust_tier == "community-self-reported"
    assert [reason.code for reason in status.reasons] == ["power_coverage_insufficient"]
    assert status.to_json()["public_url"] == "https://artificialanalysis.ai/agentperf/sub"
    assert allowlist == RevisionAllowlist(revisions=(ALLOWLISTED_REVISION,), window_days=14)
    assert revision_advice(allowlist, clean) is None
    assert "outside" in (revision_advice(allowlist, stale) or "")
    assert "dirty" in (revision_advice(allowlist, dirty) or "")
    empty_allowlist = RevisionAllowlist(revisions=(), window_days=14)
    assert revision_advice(empty_allowlist, stale) is None


def test_submit_cli_requires_acknowledgement_and_never_prints_the_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bundle_dir = _prepared_bundle(tmp_path)
    monkeypatch.setenv(TOKEN_ENV, VALID_TOKEN)

    with LocalSubmissionServer(FakeSubmissionService(require_token=True)) as server:
        base = ["submit", str(bundle_dir), "--base-url", server.base_url, "--token-env", TOKEN_ENV]
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
    assert response["notice_version"] == PRIVATE_AUDIT_NOTICE_VERSION
    assert "Hardware and verification evidence stays private" in accepted.err
    assert VALID_TOKEN not in accepted.out + accepted.err
    assert status_code == 1
    assert "not_found" in status_output.err
    assert "submitting anonymously" in anonymous.err
    assert "auth_required" in anonymous.err
