"""Exercise one attached run's submission end to end: record, prepare, check, and send."""

from pathlib import Path

import orjson
import pytest

from agentperf_local.cli import main
from agentperf_local.common.json_types import JsonObject, normalize_json_object
from agentperf_local.common.models import read_record
from agentperf_local.provenance.benchmark import BENCHMARK_CONTEXT_TOKENS
from agentperf_local.submission.contract import SubmissionRequest
from agentperf_local.submission.framework_commit import GITHUB_API_URL_ENV
from tests.attached_run import (
    API_KEY_VALUE,
    CACHED_PROMPT_TOKENS,
    FRAMEWORK_VERSION,
    MODEL,
    RELEASE_REVISION,
    record_attached_run,
)
from tests.submission_server import LocalSubmissionServer, fake_commit


def _read_json(path: Path) -> JsonObject:
    return normalize_json_object(orjson.loads(path.read_bytes()))


async def test_an_attached_run_is_prepared_checked_and_sent_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    submission_path = tmp_path / "submission.json"
    with LocalSubmissionServer() as service:
        monkeypatch.setenv(GITHUB_API_URL_ENV, service.base_url)
        results_dir = await record_attached_run(tmp_path, monkeypatch, capsys)
        prepared = main(["prepare-submission", str(results_dir), "--output", str(submission_path)])
        prepare_captured = capsys.readouterr()
        assert prepared == 0, prepare_captured.err
        prepare_output = normalize_json_object(orjson.loads(prepare_captured.out))
        first = main(["submit", str(submission_path), "--base-url", service.base_url, "--yes"])
        first_output = normalize_json_object(orjson.loads(capsys.readouterr().out))
        again = main(["submit", str(submission_path), "--base-url", service.base_url, "--yes"])
        again_output = normalize_json_object(orjson.loads(capsys.readouterr().out))
        tampered_path = tmp_path / "tampered.json"
        tampered = _read_json(submission_path)
        tampered["trust_tier"] = "verified"
        tampered_path.write_bytes(orjson.dumps(tampered))
        refused = main(["submit", str(tampered_path), "--base-url", service.base_url, "--yes"])
        refusal = capsys.readouterr().err

    encoded = submission_path.read_bytes()
    request = read_record(SubmissionRequest, encoded, "submission")
    deployment = request.deployment
    assert (prepared, first, again, refused) == (0, 0, 0, 1)
    assert prepare_output["deployment_mode"] == "attached"
    assert (first_output["created"], again_output["created"]) == (True, False)
    assert [captured.status for captured in service.service.captured] == [202, 200]
    assert "trust_tier" in refusal
    assert deployment.deployment_mode == "attached"
    assert deployment.framework_commit == fake_commit(f"v{FRAMEWORK_VERSION}")
    assert deployment.server_launch_command == (
        'vllm serve "$LOCAL_DIR_1"/qwen3.8 --api-key "$API_KEY" --max-model-len 65536'
    )
    assert request.client.source_state == "release"
    assert request.client.source_revision == RELEASE_REVISION
    assert request.benchmark.observed_context_tokens == BENCHMARK_CONTEXT_TOKENS
    assert request.hardware.platform_family == "macos"
    assert request.hardware.accelerator.product == "Apple M5 Pro"
    assert request.power is None
    assert [(turn.task_ordinal, turn.turn_in_task) for turn in request.turns] == [(0, 0), (1, 0)]
    assert all(turn.cached_input_tokens == CACHED_PROMPT_TOKENS for turn in request.turns)
    assert len(request.qualification.outcomes) == 5
    for private in (b"prompt 0", str(tmp_path).encode(), API_KEY_VALUE.encode(), b"hf_secret", MODEL.encode()):
        assert private not in encoded


def _set(data: JsonObject, path: tuple[str, ...], value: str | int | bool | None) -> None:
    target = data
    for key in path[:-1]:
        child = target[key]
        assert isinstance(child, dict)
        target = child
    target[path[-1]] = value


@pytest.mark.parametrize(
    ("filename", "path", "value", "message"),
    [
        ("summary.json", ("config", "output_tokens", "policy"), "fixed", "fixed output token policy"),
        ("summary.json", ("config", "output_tokens", "fallback"), 1000, "Input should be 16384"),
        ("summary.json", ("config", "cache_isolation", "enabled"), False, "Input should be True"),
        ("summary.json", ("config", "base_url"), "http://gpu-box:8080/v1", "ran on this computer"),
        ("measurement.json", ("observed_context_tokens",), None, "did not report its context window"),
        ("measurement.json", ("observed_context_tokens",), 32_768, "below the 65,536 tokens"),
        ("measurement.json", ("producer", "source_state"), "dirty", "uncommitted changes"),
        ("attached-server.json", ("framework_version",), "0.12.0", "changed after the run started"),
    ],
)
async def test_prepare_refuses_a_run_the_service_would_refuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    filename: str,
    path: tuple[str, ...],
    value: str | int | bool | None,
    message: str,
) -> None:
    with LocalSubmissionServer() as service:
        monkeypatch.setenv(GITHUB_API_URL_ENV, service.base_url)
        results_dir = await record_attached_run(tmp_path, monkeypatch, capsys)
        data = _read_json(results_dir / filename)
        _set(data, path, value)
        (results_dir / filename).write_bytes(orjson.dumps(data))
        status = main(["prepare-submission", str(results_dir), "--output", str(tmp_path / "submission.json")])

    assert status == 1
    assert message in capsys.readouterr().err
    assert not (tmp_path / "submission.json").exists()
