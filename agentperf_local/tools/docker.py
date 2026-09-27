"""Run recorded shell tool calls in isolated Docker containers."""

from __future__ import annotations

import contextlib
import os
import re
import subprocess
import uuid
from pathlib import Path
from time import perf_counter
from typing import Self

from pydantic import BaseModel, model_validator

from agentperf_local.workload.schema import RecordedToolCall

DEFAULT_DOCKER_COMMAND_TIMEOUT_SECONDS = 30.0
DEFAULT_DOCKER_CONTAINER_TIMEOUT = "2h"
DOCKER_START_TIMEOUT_SECONDS = 120.0
DOCKER_CLEANUP_TIMEOUT_SECONDS = 60.0
TOOL_FAILURE_RETURN_CODE = -1
CONTAINER_NAME_HEX_DIGITS = 8
DOCKER_EXECUTABLE_ENV = "AGENTPERF_LOCAL_DOCKER_EXECUTABLE"
DOCKER_QUERY_TIMEOUT_SECONDS = 30.0
# SWE-bench names the architecture in the repository segment, as sweb.eval.<arch>.<task>.
SWEBENCH_IMAGE_PREFIX = "sweb.eval."
SWEBENCH_IMAGE_ARCHITECTURE_PATTERN = re.compile(rf"{re.escape(SWEBENCH_IMAGE_PREFIX)}(?:x86_64|arm64)\.")
IMAGE_ARCHITECTURE_SEGMENTS = {"amd64": "x86_64", "x86_64": "x86_64", "arm64": "arm64", "aarch64": "arm64"}


class DockerEnvironmentVariable(BaseModel, frozen=True):
    """Store one environment variable passed to a tool container."""

    name: str
    value: str

    @model_validator(mode="after")
    def check_invariants(self) -> Self:
        if not self.name or "=" in self.name:
            raise ValueError("Docker environment variable names must not be empty or contain '='")
        return self


class DockerToolEnvironmentConfig(BaseModel, frozen=True):
    """Configure one isolated live tool environment."""

    image: str
    cwd: str = "/workspace"
    network: str | None = "none"
    executable: str = "docker"
    command_timeout_seconds: float = DEFAULT_DOCKER_COMMAND_TIMEOUT_SECONDS
    container_timeout: str = DEFAULT_DOCKER_CONTAINER_TIMEOUT
    interpreter: tuple[str, ...] = ("bash", "-c")
    workspace: Path | None = None
    run_args: tuple[str, ...] = ("--rm",)
    environment: tuple[DockerEnvironmentVariable, ...] = ()

    @model_validator(mode="after")
    def check_invariants(self) -> Self:
        if not self.image:
            raise ValueError("Docker tool image must not be empty")
        if not self.interpreter:
            raise ValueError("Docker tool interpreter must not be empty")
        if self.command_timeout_seconds <= 0:
            raise ValueError("Docker tool command timeout must be positive")
        return self


class ToolExecutionResult(BaseModel, frozen=True):
    """Store the result of one live tool command."""

    output: str
    returncode: int
    duration_ms: float
    exception_info: str = ""


class DockerToolEnvironment:
    """Own one Docker container used for sequential tool calls."""

    def __init__(self, config: DockerToolEnvironmentConfig) -> None:
        self.config = config
        self.container_name = f"agentperf-local-{uuid.uuid4().hex[:CONTAINER_NAME_HEX_DIGITS]}"
        self.container_id: str | None = None
        self._start()

    def _start_command(self) -> list[str]:
        command = [
            self.config.executable,
            "run",
            "-d",
            "--name",
            self.container_name,
            "-w",
            self.config.cwd,
            *self.config.run_args,
        ]
        if self.config.network:
            command.extend(["--network", self.config.network])
        if self.config.workspace is not None:
            command.extend(["-v", f"{self.config.workspace.resolve()}:{self.config.cwd}"])
        command.extend([self.config.image, "sleep", self.config.container_timeout])
        return command

    def _start(self) -> None:
        try:
            result = subprocess.run(
                self._start_command(),
                capture_output=True,
                text=True,
                timeout=DOCKER_START_TIMEOUT_SECONDS,
                check=True,
            )
        except (OSError, subprocess.SubprocessError):
            self._remove_container(self.container_name)
            raise
        self.container_id = result.stdout.strip()
        if not self.container_id:
            self._remove_container(self.container_name)
            raise RuntimeError("Docker did not return a container ID")

    def _execute_command(self, command: str) -> list[str]:
        if self.container_id is None:
            raise RuntimeError("Docker tool environment is not running")

        docker_command = [self.config.executable, "exec", "-w", self.config.cwd]
        for variable in self.config.environment:
            docker_command.extend(["-e", f"{variable.name}={variable.value}"])
        docker_command.extend([self.container_id, *self.config.interpreter, command])
        return docker_command

    def execute(self, call: RecordedToolCall) -> ToolExecutionResult:
        """Execute the shell command stored in one recorded tool call."""
        command = call.command
        if command is None:
            raise ValueError("live Docker tool replay only supports actions with a string command")

        start = perf_counter()
        try:
            result = subprocess.run(
                self._execute_command(command),
                text=True,
                timeout=self.config.command_timeout_seconds,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            return ToolExecutionResult(
                output=_timeout_output(error),
                returncode=TOOL_FAILURE_RETURN_CODE,
                duration_ms=(perf_counter() - start) * 1000.0,
                exception_info=f"{type(error).__name__}: {error}",
            )
        except (OSError, subprocess.SubprocessError) as error:
            return ToolExecutionResult(
                output="",
                returncode=TOOL_FAILURE_RETURN_CODE,
                duration_ms=(perf_counter() - start) * 1000.0,
                exception_info=f"{type(error).__name__}: {error}",
            )

        return ToolExecutionResult(
            output=result.stdout,
            returncode=result.returncode,
            duration_ms=(perf_counter() - start) * 1000.0,
        )

    def _stop_container(self, container: str) -> bool:
        try:
            result = subprocess.run(
                [self.config.executable, "stop", container],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=DOCKER_CLEANUP_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return result.returncode == 0

    def _remove_container(self, container: str) -> None:
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            subprocess.run(
                [self.config.executable, "rm", "-f", container],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=DOCKER_CLEANUP_TIMEOUT_SECONDS,
                check=False,
            )

    def cleanup(self) -> None:
        """Stop the container and remove it when Docker will not do so."""
        container = self.container_id
        if container is None:
            return
        try:
            stopped = self._stop_container(container)
            if not stopped or "--rm" not in self.config.run_args:
                self._remove_container(container)
        finally:
            self.container_id = None

    def __enter__(self) -> DockerToolEnvironment:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.cleanup()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.cleanup()


def _timeout_output(error: subprocess.TimeoutExpired) -> str:
    output = error.output or ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output


def docker_executable_from_env(default: str = "docker") -> str:
    """Return the configured Docker-compatible executable."""
    return os.getenv(DOCKER_EXECUTABLE_ENV, default)


def _docker_query(command: list[str]) -> str | None:
    """Run one short Docker query and return its trimmed output, or None when it fails."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=DOCKER_QUERY_TIMEOUT_SECONDS,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def docker_server_architecture(executable: str = "docker") -> str | None:
    """Return the Docker daemon architecture as a SWE-bench image segment.

    The daemon decides which image runs without emulation, so this asks the daemon
    instead of the host. A remote DOCKER_HOST or a Linux virtual machine can run a
    different architecture from the process that starts the run. An unreadable or
    unknown answer returns None, and the caller then keeps the recorded image.
    """
    reported = _docker_query([executable, "version", "--format", "{{.Server.Arch}}"])
    if reported is None:
        return None
    return IMAGE_ARCHITECTURE_SEGMENTS.get(reported.lower())


def native_swebench_image(image: str, architecture: str | None) -> str:
    """Point a SWE-bench image at the architecture that runs without emulation.

    Emulated containers make tool timings meaningless, and SWE-bench publishes the
    same task under one image name per architecture. Only the last path segment
    holds the architecture, so a registry or namespace that repeats the pattern
    stays untouched. A digest names one exact image, so a rewritten tag would still
    pull the recorded architecture; such a reference is left alone.
    """
    if architecture is None or "@" in image:
        return image
    prefix, separator, name = image.rpartition("/")
    native = SWEBENCH_IMAGE_ARCHITECTURE_PATTERN.sub(f"{SWEBENCH_IMAGE_PREFIX}{architecture}.", name, count=1)
    return prefix + separator + native


def docker_image_present(image: str, *, executable: str = "docker") -> bool:
    """Report whether Docker already holds this image locally.

    A run asks before it starts, and only a local image answers the question it
    cares about. An image the registry merely offers is still pulled by the first
    container start, which happens inside the measured window and behind the
    container start timeout, so a registry check would prove the wrong thing.
    """
    return _docker_query([executable, "image", "inspect", "--format", "{{.Id}}", image]) is not None
