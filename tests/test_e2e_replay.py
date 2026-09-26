"""Replay an eight-task manifest through the public runner and report writer."""

import asyncio
from pathlib import Path

import orjson
import pytest

from agentperf_local.cli import main
from agentperf_local.common.identity import sha256_bytes
from agentperf_local.provenance.benchmark import (
    BENCHMARK_CONTEXT_TOKENS,
    SourceProvenance,
    SubmissionContext,
    create_measurement_binding,
    workload_digest,
    write_measurement_binding,
)
from agentperf_local.provenance.context import ContextObservationReason, RunContextFacts
from agentperf_local.provenance.hardware import HardwareSnapshot
from agentperf_local.provenance.hardware_facts import AcceleratorSnapshot
from agentperf_local.replay.config import RunConfig
from agentperf_local.replay.runner import run_manifest
from agentperf_local.reports.reporting import write_run_artifacts
from agentperf_local.submission.bundle import validate_submission_bundle
from agentperf_local.workload.schema import parse_json_object
from tests.localhost_sse import LocalSseServer
from tests.replay_workload import write_replay_workload
from tests.token_counter import CharacterCounter

TASK_COUNT = 8
LOGICAL_CPU_COUNT = 8
HOST_MEMORY_GIB = 64
ACCELERATOR_MEMORY_GIB = 32
BYTES_PER_GIB = 1024**3
SOURCE_REVISION_HEX_DIGITS = 40
API_KEY_ENV = "AGENTPERF_TEST_API_KEY"
API_KEY = "private-test-key"
WORKLOAD_NAME = "eight-task-e2e"
# Each event is flushed separately with a real gap between them, so every turn has
# a first-to-last-token window that normalization can actually measure.
SSE_EVENTS = (
    b'data: {"choices":[{"delta":{"content":"o"},"finish_reason":null}]}\n\n',
    b'data: {"choices":[{"delta":{"content":"k"},"finish_reason":"stop"}]}\n\n',
    b'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}\n\n',
    b"data: [DONE]\n\n",
)
EVENT_DELAY_SECONDS = 0.005
RUN_ID = "8f5b2f2e-4c3a-4d6e-9b1a-2c3d4e5f6a7b"


async def test_eight_task_manifest_replays_once_and_validates_public_bundle(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_path = write_replay_workload(tmp_path / "workload", name=WORKLOAD_NAME, task_count=TASK_COUNT)

    async with LocalSseServer(SSE_EVENTS, inter_chunk_delay_seconds=EVENT_DELAY_SECONDS) as server:
        config = RunConfig(
            base_url=server.base_url,
            model="local-model",
            client_backend="python",
            cache_isolation=False,
        )
        result = await run_manifest(manifest_path, config, token_counter=CharacterCounter())

    # A proven full context keeps this run submittable; the reduced marking is covered elsewhere.
    run_context = RunContextFacts(
        requested_tokens=BENCHMARK_CONTEXT_TOKENS,
        observed_tokens=BENCHMARK_CONTEXT_TOKENS,
        observed_reason=ContextObservationReason.REPORTED,
    )
    artifacts = write_run_artifacts(result, tmp_path / "results", config, run_context=run_context, run_id=RUN_ID)
    bodies = [parse_json_object(request.body, "captured request") for request in server.requests]
    summary = parse_json_object(artifacts.summary.read_bytes(), artifacts.summary)

    assert result.success is True
    assert [task.task_id for task in result.tasks] == [f"task-{index}" for index in range(TASK_COUNT)]
    assert [turn.turn_id for turn in result.turns] == [f"task-{index}:0000" for index in range(TASK_COUNT)]
    assert len(server.requests) == TASK_COUNT
    assert [body.get("messages") for body in bodies] == [
        [{"role": "user", "content": f"prompt {index}"}] for index in range(TASK_COUNT)
    ]
    assert len(artifacts.turns.read_bytes().splitlines()) == TASK_COUNT
    assert all(
        turn.metrics is not None and turn.metrics.generation_time is not None and turn.metrics.generation_time > 0
        for turn in result.turns
    )
    assert summary["success"] is True
    assert summary["output_tokens_per_second"] is not None
    assert summary["failed_turn_ids"] == []
    assert summary["artifacts"] == {
        "turns": "turns.jsonl",
        "tasks": "tasks.json",
        "tools": "tools.json",
        "failures": "failures.json",
    }

    context = SubmissionContext(
        suite_id="aa-agentic-gpu-smoke",
        suite_epoch="2026-q4",
        suite_digest=workload_digest(manifest_path),
        model_semantics_id="synthetic-model-v1",
        model_artifact_digest=sha256_bytes(b"synthetic-model-artifact"),
        runtime_id="synthetic-runtime-v1",
    )
    hardware = HardwareSnapshot(
        operating_system="Linux",
        operating_system_version="test-os",
        kernel_version="test-kernel",
        architecture="x86_64",
        cpu_model="test-cpu",
        logical_cpu_count=LOGICAL_CPU_COUNT,
        memory_bytes=HOST_MEMORY_GIB * BYTES_PER_GIB,
        accelerators=(
            AcceleratorSnapshot(
                vendor="NVIDIA",
                name="Synthetic GPU",
                memory_bytes=ACCELERATOR_MEMORY_GIB * BYTES_PER_GIB,
                core_count=None,
                driver_version="590.42",
                api="CUDA",
            ),
        ),
        warnings=(),
    )
    binding = create_measurement_binding(
        context,
        manifest_path,
        config.model,
        hardware,
        SourceProvenance(
            client_version="0.1.0",
            source_revision="1" * SOURCE_REVISION_HEX_DIGITS,
            source_state="clean",
        ),
        observed_context_tokens=BENCHMARK_CONTEXT_TOKENS,
        run_id=RUN_ID,
    )
    write_measurement_binding(tmp_path / "results" / "measurement.json", binding)

    bundle_dir = tmp_path / "submission-preview"
    assert main(["prepare-submission", str(tmp_path / "results"), "--output-dir", str(bundle_dir)]) == 0
    prepare_output = orjson.loads(capsys.readouterr().out)
    assert prepare_output["upload_performed"] is False
    validated = validate_submission_bundle(bundle_dir)
    assert validated.aggregate_payload_digest == prepare_output["aggregate_payload_digest"]
    assert b"prompt 0" not in b"".join(path.read_bytes() for path in bundle_dir.iterdir())


async def test_cli_progress_keeps_frames_on_stderr_and_final_json_on_stdout(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(API_KEY_ENV, API_KEY)
    manifest_path = write_replay_workload(tmp_path / "workload", name=WORKLOAD_NAME, task_count=TASK_COUNT)
    output_dir = tmp_path / "results"

    async with LocalSseServer(SSE_EVENTS, inter_chunk_delay_seconds=EVENT_DELAY_SECONDS) as server:
        status = await asyncio.to_thread(
            main,
            [
                "run",
                str(manifest_path),
                "--base-url",
                server.base_url,
                "--model",
                "local-model",
                "--output-dir",
                str(output_dir),
                "--client",
                "python",
                "--api-key-env",
                API_KEY_ENV,
                "--no-cache-isolation",
                "--tool-choice",
                "none",
                "--progress",
            ],
        )

    captured = capsys.readouterr()
    output = orjson.loads(captured.out)
    assert status == 0
    assert output["success"] is True
    # The pre-run probe asked this attached server for its context; the test server
    # lists no models, so the run records None with the reason and stays non-comparable.
    assert output["context"] == {
        "requested_tokens": BENCHMARK_CONTEXT_TOKENS,
        "observed_tokens": None,
        "observed_reason": "model-not-listed",
        "full_benchmark_tokens": BENCHMARK_CONTEXT_TOKENS,
        "reduced": True,
    }
    assert "will be recorded as non-comparable" in captured.err
    assert output["artifacts"]["summary"] == str(output_dir / "summary.json")
    assert "PREFLIGHT  task" in captured.err
    assert "COMPLETE  task" in captured.err
    assert all(request.headers["authorization"] == f"Bearer {API_KEY}" for request in server.requests)
    summary = orjson.loads((output_dir / "summary.json").read_bytes())
    assert summary["config"]["sampling"]["extra_body"]["tool_choice"] == "none"
    assert API_KEY not in captured.out
    assert API_KEY not in captured.err
    assert "prompt 0" not in captured.err
