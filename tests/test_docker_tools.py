"""Test Docker tool replay command construction and lifecycle."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from pydantic import BaseModel

from agentperf_local.tools.docker import (
    DOCKER_CLEANUP_TIMEOUT_SECONDS,
    DOCKER_START_TIMEOUT_SECONDS,
    TOOL_FAILURE_RETURN_CODE,
    DockerEnvironmentVariable,
    DockerToolEnvironment,
    DockerToolEnvironmentConfig,
    docker_image_present,
    docker_server_architecture,
    native_swebench_image,
)
from agentperf_local.workload.schema import RecordedToolCall

CONTAINER_ID = "container-123"


class _DockerCall(BaseModel, frozen=True):
    """Hold one command the fake Docker dispatcher received."""

    command: list[str]
    options: dict[str, object]


def _install_fake_docker(
    monkeypatch: pytest.MonkeyPatch,
    *,
    run_error: Exception | None = None,
    execute_error: Exception | None = None,
    execute_returncode: int = 0,
    execute_output: str = "",
    query_error: Exception | None = None,
    server_architecture: str = "",
) -> list[_DockerCall]:
    """Replace subprocess.run with a Docker dispatcher and return the calls it records."""
    calls: list[_DockerCall] = []

    def fake_run(command: list[str], **options: object) -> subprocess.CompletedProcess[str]:
        calls.append(_DockerCall(command=command, options=options))
        if command[:2] == ["docker", "run"]:
            if run_error is not None:
                raise run_error
            return subprocess.CompletedProcess(command, 0, stdout=f"{CONTAINER_ID}\n", stderr="")
        if command[:2] == ["docker", "exec"]:
            if execute_error is not None:
                raise execute_error
            return subprocess.CompletedProcess(command, execute_returncode, stdout=execute_output, stderr="")
        if command[1:2] == ["version"]:
            if query_error is not None:
                raise query_error
            return subprocess.CompletedProcess(command, 0, stdout=f"{server_architecture}\n", stderr="")
        if command[1:3] == ["image", "inspect"]:
            if query_error is not None:
                raise query_error
            return subprocess.CompletedProcess(command, 0, stdout="sha256:0123456789ab\n", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def test_docker_environment_variable_rejects_invalid_name() -> None:
    with pytest.raises(ValueError):
        DockerEnvironmentVariable(name="BAD=NAME", value="value")


def test_docker_tool_environment_starts_executes_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = _install_fake_docker(monkeypatch, execute_returncode=7, execute_output="tool output")

    environment = DockerToolEnvironment(
        DockerToolEnvironmentConfig(
            image="agentperf-pinchbench:latest",
            workspace=tmp_path,
            interpreter=("bash", "-c"),
            command_timeout_seconds=3.0,
            environment=(DockerEnvironmentVariable(name="DEMO", value="yes"),),
            run_args=(),
        )
    )
    result = environment.execute(RecordedToolCall(duration_ms=10.0, action={"command": "cat /workspace/input.txt"}))
    environment.cleanup()

    run_call, execute_call, stop_call, remove_call = calls
    run_command = run_call.command

    assert run_command[:5] == ["docker", "run", "-d", "--name", environment.container_name]
    assert run_command[run_command.index("--network") : run_command.index("--network") + 2] == [
        "--network",
        "none",
    ]
    assert run_command[run_command.index("-v") : run_command.index("-v") + 2] == [
        "-v",
        f"{tmp_path.resolve()}:/workspace",
    ]
    assert run_command[-3:] == ["agentperf-pinchbench:latest", "sleep", "2h"]
    assert run_call.options["timeout"] == DOCKER_START_TIMEOUT_SECONDS
    assert execute_call.command == [
        "docker",
        "exec",
        "-w",
        "/workspace",
        "-e",
        "DEMO=yes",
        CONTAINER_ID,
        "bash",
        "-c",
        "cat /workspace/input.txt",
    ]
    assert execute_call.options["timeout"] == 3.0
    assert stop_call.command == ["docker", "stop", CONTAINER_ID]
    assert stop_call.options["timeout"] == DOCKER_CLEANUP_TIMEOUT_SECONDS
    assert remove_call.command == ["docker", "rm", "-f", CONTAINER_ID]
    assert remove_call.options["timeout"] == DOCKER_CLEANUP_TIMEOUT_SECONDS
    assert result.returncode == 7
    assert result.output == "tool output"
    assert result.duration_ms >= 0


@pytest.mark.parametrize(
    "error",
    [
        subprocess.TimeoutExpired(["docker", "exec"], 0.1, output=b"partial output"),
        OSError("synthetic Docker failure"),
    ],
    ids=("timeout", "os-error"),
)
def test_docker_tool_environment_reports_command_failures(
    monkeypatch: pytest.MonkeyPatch,
    error: OSError | subprocess.TimeoutExpired,
) -> None:
    _install_fake_docker(monkeypatch, execute_error=error)
    environment = DockerToolEnvironment(DockerToolEnvironmentConfig(image="synthetic:latest"))
    try:
        result = environment.execute(RecordedToolCall(duration_ms=1.0, action={"command": "pwd"}))
    finally:
        environment.cleanup()

    assert result.returncode == TOOL_FAILURE_RETURN_CODE
    assert type(error).__name__ in result.exception_info
    if isinstance(error, subprocess.TimeoutExpired):
        assert result.output == "partial output"
    else:
        assert result.output == ""


def test_docker_tool_environment_attempts_cleanup_after_start_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake_docker(monkeypatch, run_error=subprocess.CalledProcessError(1, ["docker", "run"]))

    with pytest.raises(subprocess.CalledProcessError):
        DockerToolEnvironment(DockerToolEnvironmentConfig(image="synthetic:latest"))

    assert calls[-1].command[:3] == ["docker", "rm", "-f"]


@pytest.mark.parametrize(
    ("image", "architecture", "expected"),
    [
        (
            "docker.io/swebench/sweb.eval.x86_64.django_1776_django-15851:latest",
            "arm64",
            "docker.io/swebench/sweb.eval.arm64.django_1776_django-15851:latest",
        ),
        (
            "docker.io/swebench/sweb.eval.arm64.django_1776_django-15851:latest",
            "x86_64",
            "docker.io/swebench/sweb.eval.x86_64.django_1776_django-15851:latest",
        ),
        # An unreadable daemon architecture keeps the recorded image.
        (
            "docker.io/swebench/sweb.eval.x86_64.django-15851:latest",
            None,
            "docker.io/swebench/sweb.eval.x86_64.django-15851:latest",
        ),
        # A digest names one exact image, so the recorded architecture stands.
        (
            "docker.io/swebench/sweb.eval.x86_64.django-15851@sha256:" + "a" * 64,
            "arm64",
            "docker.io/swebench/sweb.eval.x86_64.django-15851@sha256:" + "a" * 64,
        ),
        # Only the repository segment carries the architecture.
        (
            "registry.invalid/sweb.eval.x86_64.mirror/sweb.eval.x86_64.django-15851:latest",
            "arm64",
            "registry.invalid/sweb.eval.x86_64.mirror/sweb.eval.arm64.django-15851:latest",
        ),
        ("example.invalid/swe/sanitized-task:latest", "arm64", "example.invalid/swe/sanitized-task:latest"),
    ],
)
def test_native_swebench_image_rewrites_only_a_tagged_repository_segment(
    image: str, architecture: str | None, expected: str
) -> None:
    assert native_swebench_image(image, architecture) == expected


@pytest.mark.parametrize(
    ("reported", "expected"),
    [("aarch64", "arm64"), ("amd64", "x86_64"), ("riscv64", None)],
)
def test_docker_server_architecture_maps_the_daemon_answer(
    monkeypatch: pytest.MonkeyPatch, reported: str, expected: str | None
) -> None:
    calls = _install_fake_docker(monkeypatch, server_architecture=reported)

    assert docker_server_architecture("docker") == expected
    assert calls[0].command == ["docker", "version", "--format", "{{.Server.Arch}}"]


def test_docker_server_architecture_returns_none_without_a_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_docker(monkeypatch, query_error=OSError("docker is not installed"))

    assert docker_server_architecture("docker") is None


@pytest.mark.parametrize(
    ("query_error", "expected"),
    [(None, True), (subprocess.CalledProcessError(1, ["docker"]), False)],
)
def test_docker_image_present_reports_only_the_local_store(
    monkeypatch: pytest.MonkeyPatch, query_error: Exception | None, expected: bool
) -> None:
    calls = _install_fake_docker(monkeypatch, query_error=query_error)

    assert docker_image_present("example.invalid/task:latest", executable="docker") is expected
    # A registry lookup would prove the wrong thing, so only the local store is asked.
    assert [call.command[1:3] for call in calls] == [["image", "inspect"]]
