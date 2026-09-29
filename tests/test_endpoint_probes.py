"""Cover the served-context and ignore_eos probes and the policy they bind a run to."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from agentperf_local.cli import main
from agentperf_local.deployment.endpoint_probes import (
    IGNORE_EOS_PROBE_OUTPUT_TOKENS,
    OLLAMA_RECORDED_POLICY_WARNING,
    IgnoreEosSupport,
    is_ollama_endpoint,
    probe_ignore_eos,
    probe_served_context_tokens,
)
from agentperf_local.deployment.qualification import QUALIFICATION_PROBE_IDS
from agentperf_local.provenance.context import ContextObservationReason
from tests.localhost_sse import SSE_OK_RESPONSE, LocalSseServer
from tests.replay_workload import write_replay_workload

MODEL = "served-model"
# One byte per read arrives well inside any per-read timeout, so only an overall deadline ends it.
OLLAMA_TRICKLE_BYTE_SECONDS = 0.05
OLLAMA_TEST_DEADLINE_SECONDS = 0.5
# The detection may overrun its deadline by connection teardown, never by the whole trickle.
OLLAMA_TEST_ELAPSED_LIMIT_SECONDS = 1.0

# A stream that already fills the probe's budget on its own, as a long-winded model does.
SSE_FILLED_BUDGET_RESPONSE = (
    b'data: {"choices":[{"delta":{"content":"o"},"finish_reason":null}]}\n\n'
    b'data: {"choices":[{"delta":{"content":"k"},"finish_reason":"length"}]}\n\n'
    + b'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":%d,"total_tokens":%d}}\n\n'
    % (IGNORE_EOS_PROBE_OUTPUT_TOKENS, IGNORE_EOS_PROBE_OUTPUT_TOKENS + 3)
    + b"data: [DONE]\n\n"
)
# A stream that stops without reporting usage, so only the finish reason speaks.
SSE_NO_USAGE_RESPONSE = (
    b'data: {"choices":[{"delta":{"content":"o"},"finish_reason":null}]}\n\n'
    b'data: {"choices":[{"delta":{"content":"k"},"finish_reason":"stop"}]}\n\n'
    b"data: [DONE]\n\n"
)


@pytest.mark.parametrize(
    ("chunks", "honours_ignore_eos", "rejects_ignore_eos", "status", "expected_support", "expected_requests"),
    (
        pytest.param(SSE_OK_RESPONSE, True, False, 200, IgnoreEosSupport.HONOURED, 2, id="honoring"),
        pytest.param(SSE_NO_USAGE_RESPONSE, True, False, 200, IgnoreEosSupport.HONOURED, 2, id="honoring-no-usage"),
        pytest.param(SSE_OK_RESPONSE, False, False, 200, IgnoreEosSupport.IGNORED, 1, id="dropping"),
        pytest.param(SSE_OK_RESPONSE, True, True, 200, IgnoreEosSupport.IGNORED, 2, id="strict"),
        pytest.param(SSE_FILLED_BUDGET_RESPONSE, False, False, 200, IgnoreEosSupport.UNDETERMINED, 2, id="long-winded"),
        pytest.param(SSE_OK_RESPONSE, True, False, 500, IgnoreEosSupport.UNDETERMINED, 2, id="unusable"),
    ),
)
async def test_probe_reads_ignore_eos_support_and_binds_the_policy(
    chunks: bytes,
    honours_ignore_eos: bool,
    rejects_ignore_eos: bool,
    status: int,
    expected_support: IgnoreEosSupport,
    expected_requests: int,
) -> None:
    """A dropped ignore_eos is the only outcome that gives up the exact policy."""
    async with LocalSseServer(
        (chunks,), honours_ignore_eos=honours_ignore_eos, rejects_ignore_eos=rejects_ignore_eos, status=status
    ) as stub:
        result = await probe_ignore_eos(stub.base_url, MODEL, "python")

    assert result.support is expected_support
    assert result.output_token_policy == ("recorded" if expected_support is IgnoreEosSupport.IGNORED else "exact")
    # The control request is sent only when the first answer did not settle it, and every
    # probe request streams, as the replay's turns do.
    assert [request.asks_ignore_eos for request in stub.requests] == [True, False][:expected_requests]
    assert all(request.body_json is not None and request.body_json.get("stream") is True for request in stub.requests)


@pytest.mark.parametrize(
    ("served_context_tokens", "asked_model", "expected_tokens", "expected_reason"),
    (
        pytest.param(131_072, MODEL, 131_072, ContextObservationReason.REPORTED, id="reported"),
        pytest.param(131_072, "unknown-model", None, ContextObservationReason.MODEL_NOT_LISTED, id="unlisted"),
        pytest.param(None, MODEL, None, ContextObservationReason.CONTEXT_NOT_REPORTED, id="silent"),
        # A non-positive n_ctx is recorded as unobserved, never crashed on.
        pytest.param(0, MODEL, None, ContextObservationReason.NON_POSITIVE_CONTEXT, id="non-positive"),
    ),
)
async def test_context_probe_reports_the_served_length_or_why_it_holds_none(
    served_context_tokens: int | None,
    asked_model: str,
    expected_tokens: int | None,
    expected_reason: ContextObservationReason,
) -> None:
    async with LocalSseServer(
        (SSE_OK_RESPONSE,), models=(MODEL,), served_context_tokens=served_context_tokens
    ) as server:
        # The probe is a blocking GET; the server answers on this event loop.
        result = await asyncio.to_thread(probe_served_context_tokens, server.base_url, asked_model)

    assert result.observed_tokens == expected_tokens
    assert result.reason is expected_reason


async def test_probes_of_an_unreachable_server_record_no_answer() -> None:
    """A server that never answers gets one ignore_eos attempt, not two, and no context."""
    async with LocalSseServer((SSE_OK_RESPONSE,)) as stub:
        closed_url = stub.base_url
    result = await probe_ignore_eos(closed_url, MODEL, "python")
    context = await asyncio.to_thread(probe_served_context_tokens, closed_url, MODEL)

    assert result.support is IgnoreEosSupport.UNDETERMINED
    assert result.observed is None
    assert context.observed_tokens is None
    assert context.reason is ContextObservationReason.ENDPOINT_UNREACHABLE


@pytest.mark.parametrize(
    ("answers_as_ollama", "policy_args"),
    (
        pytest.param(False, (), id="unnamed-server-default-policy"),
        pytest.param(True, ("--output-token-policy", "exact"), id="ollama-explicit-exact"),
    ),
)
async def test_run_refuses_exact_policy_when_the_server_drops_ignore_eos(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], answers_as_ollama: bool, policy_args: tuple[str, ...]
) -> None:
    """The CLI names the recorded policy instead of failing every turn against Ollama-like servers."""
    manifest_path = write_replay_workload(tmp_path / "workload", name="Probe refusal test")
    output_dir = tmp_path / "results"

    async with LocalSseServer(
        (SSE_OK_RESPONSE,), honours_ignore_eos=False, answers_as_ollama=answers_as_ollama
    ) as server:
        status = await asyncio.to_thread(
            main,
            [
                "run",
                str(manifest_path),
                "--base-url",
                server.base_url,
                "--model",
                "drops-ignore-eos",
                "--output-dir",
                str(output_dir),
                "--client",
                "python",
                *policy_args,
            ],
        )

    captured = capsys.readouterr()
    assert status == 1
    assert "server ignores ignore_eos" in captured.err
    assert "--output-token-policy recorded" in captured.err
    # One probe request settled it; no turn ran and no binding was written.
    assert [request.method for request in server.requests if request.method == "POST"] == ["POST"]
    assert not (output_dir / "measurement.json").exists()


@pytest.mark.parametrize("margin_args", ((), ("--output-token-margin", "16")), ids=("default-margin", "margin"))
async def test_run_warns_early_and_switches_to_recorded_against_ollama(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], margin_args: tuple[str, ...]
) -> None:
    """An Ollama server is named before any generation, and the default policy becomes recorded.

    A recorded-only option such as the output token margin is then accepted.
    """
    manifest_path = write_replay_workload(tmp_path / "workload", name="Ollama warning test")
    output_dir = tmp_path / "results"

    async with LocalSseServer((SSE_OK_RESPONSE,), honours_ignore_eos=False, answers_as_ollama=True) as server:
        status = await asyncio.to_thread(
            main,
            [
                "run",
                str(manifest_path),
                "--base-url",
                server.base_url,
                "--model",
                "ollama-model",
                "--output-dir",
                str(output_dir),
                "--client",
                "python",
                *margin_args,
            ],
        )

    captured = capsys.readouterr()
    assert status == 0, captured.err
    assert f"warning: {OLLAMA_RECORDED_POLICY_WARNING}" in captured.err
    # The identity check came first, and no POST asks for ignore_eos: the qualification
    # probes, then the replay's own turn.
    assert server.requests[0].path == "/api/version"
    posts = [request for request in server.requests if request.method == "POST"]
    assert len(posts) == len(QUALIFICATION_PROBE_IDS) + 1
    assert not any(post.asks_ignore_eos for post in posts)


@pytest.mark.parametrize(
    ("answers_as_ollama", "version_only", "trickle_seconds", "expected"),
    (
        pytest.param(True, False, 0.0, True, id="ollama"),
        pytest.param(False, True, 0.0, False, id="version-only-impostor"),
        pytest.param(True, False, OLLAMA_TRICKLE_BYTE_SECONDS, False, id="trickling-ollama"),
        pytest.param(False, False, 0.0, False, id="openai-only"),
    ),
)
async def test_ollama_detection_needs_both_identity_answers_within_one_deadline(
    answers_as_ollama: bool, version_only: bool, trickle_seconds: float, expected: bool
) -> None:
    async with LocalSseServer(
        (SSE_OK_RESPONSE,),
        models=(MODEL,),
        keep_alive=True,
        answers_as_ollama=answers_as_ollama,
        answers_ollama_version_only=version_only,
        ollama_trickle_seconds=trickle_seconds,
    ) as server:
        started = time.monotonic()
        detected = await asyncio.to_thread(
            is_ollama_endpoint, server.base_url, deadline_seconds=OLLAMA_TEST_DEADLINE_SECONDS
        )
        elapsed = time.monotonic() - started

    assert detected is expected
    assert elapsed < OLLAMA_TEST_ELAPSED_LIMIT_SECONDS
