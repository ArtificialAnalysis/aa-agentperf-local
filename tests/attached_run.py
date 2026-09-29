"""Record one submittable attached run through the real `run` command, against local fakes."""

import asyncio
from pathlib import Path

import pytest

from agentperf_local.cli import main
from agentperf_local.common.units import BYTES_PER_GIB
from agentperf_local.provenance.benchmark import BENCHMARK_CONTEXT_TOKENS, SourceProvenance
from agentperf_local.provenance.hardware import HardwareSnapshot
from agentperf_local.provenance.hardware_facts import AcceleratorSnapshot
from tests.localhost_sse import LocalSseServer
from tests.replay_workload import write_replay_workload

MODEL = "local-model"
RELEASE_REVISION = "3" * 40
CACHED_PROMPT_TOKENS = 1
API_KEY_VALUE = "sk-must-not-leave"
# The server's version names a release, so the client asks GitHub for the tag's commit.
FRAMEWORK_VERSION = "0.11.0"
EVENT_DELAY_SECONDS = 0.005
SSE_EVENTS = (
    b'data: {"choices":[{"delta":{"content":"o"},"finish_reason":null}]}\n\n',
    b'data: {"choices":[{"delta":{"content":"k"},"finish_reason":"stop"}]}\n\n',
    b'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5,'
    b'"prompt_tokens_details":{"cached_tokens":1}}}\n\n',
    b"data: [DONE]\n\n",
)
ATTACHED_SERVER = """\
model_release_slug: qwen3-8-27b
hf_repository: Qwen/Qwen3.8-27B
hf_revision: {hf_revision}
framework: vllm
framework_version: '{framework_version}'
accelerator_backend: metal
server_launch_command: >-
  HF_TOKEN=hf_secret {model_dir}/venv/bin/vllm serve {model_dir}/qwen3.8
  --api-key {api_key} --max-model-len 65536
"""


def _apple_host() -> HardwareSnapshot:
    return HardwareSnapshot(
        operating_system="Darwin",
        operating_system_version="26.5.1",
        kernel_version="25.5.0",
        architecture="arm64",
        cpu_model="Apple M5 Pro",
        logical_cpu_count=18,
        memory_bytes=64 * BYTES_PER_GIB,
        accelerators=(
            AcceleratorSnapshot(
                vendor="Apple",
                name="Apple M5 Pro",
                memory_bytes=None,
                core_count=20,
                driver_version=None,
                api="Metal",
                memory_is_unified=True,
            ),
        ),
        warnings=(),
    )


def _release_build() -> SourceProvenance:
    return SourceProvenance(client_version="0.3.0", source_revision=RELEASE_REVISION, source_state="release")


async def record_attached_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> Path:
    """Replay two one-turn tasks against a local server described by --attached-server, and return the results."""
    monkeypatch.setattr("agentperf_local.cli.options.collect_hardware_snapshot", _apple_host)
    monkeypatch.setattr("agentperf_local.provenance.benchmark.collect_source_provenance", _release_build)
    manifest_path = write_replay_workload(tmp_path / "workload", name="submission-e2e", task_count=2)
    server_file = tmp_path / "server.yaml"
    server_file.write_text(
        ATTACHED_SERVER.format(
            hf_revision="a" * 40,
            framework_version=FRAMEWORK_VERSION,
            model_dir=tmp_path / "models",
            api_key=API_KEY_VALUE,
        )
    )
    results_dir = tmp_path / "results"
    async with LocalSseServer(
        SSE_EVENTS,
        models=(MODEL,),
        served_context_tokens=BENCHMARK_CONTEXT_TOKENS,
        inter_chunk_delay_seconds=EVENT_DELAY_SECONDS,
    ) as endpoint:
        status = await asyncio.to_thread(
            main,
            [
                "run",
                str(manifest_path),
                "--base-url",
                endpoint.base_url,
                "--model",
                MODEL,
                "--output-dir",
                str(results_dir),
                "--attached-server",
                str(server_file),
                "--client",
                "python",
            ],
        )
    captured = capsys.readouterr()
    assert status == 0, captured.err
    return results_dir
