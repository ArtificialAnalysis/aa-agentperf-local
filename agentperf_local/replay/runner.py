"""Replay manifest tasks once in source order."""

from __future__ import annotations

import asyncio
import math
import shutil
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter, time
from typing import Literal, Protocol

from agentperf_local.client.backends import ClientBackend, streaming_client
from agentperf_local.client.protocol import CompletionClient, CompletionResult
from agentperf_local.client.request import CompletionRequest
from agentperf_local.common.units import MILLISECONDS_PER_SECOND
from agentperf_local.metrics.request import RequestMetrics, measure_request
from agentperf_local.metrics.tokenization import TiktokenTokenCounter, TokenCounter
from agentperf_local.replay.cache_isolation import isolate_messages, resolve_cache_namespace
from agentperf_local.replay.config import RunConfig, SamplingSettings
from agentperf_local.replay.fidelity import LENGTH_FINISH_REASON, evaluate_tool_fidelity, generated_whole_budget
from agentperf_local.tools.docker import (
    DockerToolEnvironment,
    DockerToolEnvironmentConfig,
    ToolExecutionResult,
    docker_image_present,
    docker_server_architecture,
    native_swebench_image,
)
from agentperf_local.tools.shell import shell_executable
from agentperf_local.workload.schema import ManifestTask, RecordedToolCall, TraceRow, load_manifest, load_trace

SEQUENTIAL_MAX_CONNECTIONS = 1

type ReplayDelaySource = Literal["none", "recorded", "scale_recorded", "live"]
type Sleep = Callable[[float], Awaitable[None]]


class LiveToolEnvironment(Protocol):
    """Execute recorded calls and release one live environment."""

    def execute(self, call: RecordedToolCall) -> ToolExecutionResult:
        """Execute one recorded tool call."""
        ...

    def cleanup(self) -> None:
        """Release the live environment."""
        ...


type LiveToolFactory = Callable[[DockerToolEnvironmentConfig], LiveToolEnvironment]


class ImageProbe(Protocol):
    """Report whether Docker already holds one image."""

    def __call__(self, image: str, *, executable: str) -> bool:
        """Return whether the image is ready to start without a pull."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class _LiveDocker:
    """Hold what one live run needs to know about its Docker daemon."""

    executable: str
    architecture: str | None
    image_present: ImageProbe


@dataclass(frozen=True, slots=True, kw_only=True)
class RunStartedBoundary:
    """Describe immutable work before the first request starts."""

    tasks: int
    turns: int

    def __post_init__(self) -> None:
        """Reject impossible replay plans."""
        if self.tasks < 0 or self.turns < 0:
            raise ValueError("run plan counts must be non-negative")


def _validate_turn_position(*, task: int, tasks: int, task_turn: int, task_turns: int, turn: int, turns: int) -> None:
    """Reject a turn position that cannot exist, for either end of a turn."""
    for current, total, label in (
        (task, tasks, "task"),
        (task_turn, task_turns, "task turn"),
        (turn, turns, "turn"),
    ):
        if total < 1 or current < 1 or current > total:
            raise ValueError(f"{label} must be between one and its total")


@dataclass(frozen=True, slots=True, kw_only=True)
class TurnStartedBoundary:
    """Describe the request about to be sent: where it sits in the run and how large it is.

    The size is the count the recording carries, not one the served model produced,
    because nothing has tokenized the prompt yet. Its tokenizer may count differently,
    so every screen wording treats this number as approximate.
    """

    task: int
    tasks: int
    task_turn: int
    task_turns: int
    turn: int
    turns: int
    recorded_prompt_tokens: int | None

    def __post_init__(self) -> None:
        """Reject an impossible position or a negative prompt size."""
        _validate_turn_position(
            task=self.task,
            tasks=self.tasks,
            task_turn=self.task_turn,
            task_turns=self.task_turns,
            turn=self.turn,
            turns=self.turns,
        )
        if self.recorded_prompt_tokens is not None and self.recorded_prompt_tokens < 0:
            raise ValueError("recorded prompt tokens must be non-negative")


@dataclass(frozen=True, slots=True, kw_only=True)
class TurnCompletedBoundary:
    """Describe one completed turn without response content or identifiers."""

    task: int
    tasks: int
    task_turn: int
    task_turns: int
    turn: int
    turns: int
    elapsed_ms: float
    time_to_first_token_ms: float | None
    e2e_latency_ms: float | None
    # Post-close decode facts: the output count the turn reports and its first-to-last-token
    # window. Together they give the per-turn decode speed without any response content.
    output_tokens: int | None
    generation_time_ms: float | None
    success: bool

    def __post_init__(self) -> None:
        """Reject inconsistent or non-finite turn progress."""
        _validate_turn_position(
            task=self.task,
            tasks=self.tasks,
            task_turn=self.task_turn,
            task_turns=self.task_turns,
            turn=self.turn,
            turns=self.turns,
        )
        for value in (self.elapsed_ms, self.time_to_first_token_ms, self.e2e_latency_ms, self.generation_time_ms):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("boundary durations must be finite and non-negative")
        if (
            self.time_to_first_token_ms is not None
            and self.e2e_latency_ms is not None
            and self.time_to_first_token_ms > self.e2e_latency_ms
        ):
            raise ValueError("time to first token must not exceed end-to-end latency")
        if (
            self.generation_time_ms is not None
            and self.e2e_latency_ms is not None
            and self.generation_time_ms > self.e2e_latency_ms
        ):
            raise ValueError("generation time must not exceed end-to-end latency")
        if self.output_tokens is not None and self.output_tokens < 0:
            raise ValueError("output tokens must be non-negative")


@dataclass(frozen=True, slots=True, kw_only=True)
class RunFinishedBoundary:
    """Describe a completed finite replay without private run fields."""

    completed_tasks: int
    tasks: int
    completed_turns: int
    turns: int
    elapsed_ms: float
    success: bool

    def __post_init__(self) -> None:
        """Reject inconsistent final progress."""
        for completed, total, label in (
            (self.completed_tasks, self.tasks, "completed tasks"),
            (self.completed_turns, self.turns, "completed turns"),
        ):
            if total < 0 or completed < 0 or completed > total:
                raise ValueError(f"{label} must be between zero and its total")
        if not math.isfinite(self.elapsed_ms) or self.elapsed_ms < 0:
            raise ValueError("elapsed_ms must be finite and non-negative")
        if self.success and (self.completed_tasks != self.tasks or self.completed_turns != self.turns):
            raise ValueError("a successful run must complete every task and turn")


type RunBoundaryEvent = RunStartedBoundary | TurnStartedBoundary | TurnCompletedBoundary | RunFinishedBoundary


class RunObserver(Protocol):
    """Receive coarse events outside the streaming and post-close decode path."""

    def on_boundary(self, event: RunBoundaryEvent) -> None:
        """Handle one privacy-contained boundary event."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class CompositeRunObserver:
    """Send each boundary to observers in a fixed order."""

    observers: tuple[RunObserver, ...]

    def __post_init__(self) -> None:
        """Require at least one destination."""
        if not self.observers:
            raise ValueError("a composite observer requires at least one observer")

    def on_boundary(self, event: RunBoundaryEvent) -> None:
        """Notify every destination in order."""
        for observer in self.observers:
            observer.on_boundary(event)


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolReplayResult:
    """Store how one recorded tool call was replayed."""

    call: RecordedToolCall
    replayed_duration_ms: float
    delay_source: ReplayDelaySource
    command: str | None = None
    returncode: int | None = None
    returncode_matches_recorded: bool | None = None
    exception_info: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class TurnResult:
    """Store one attempted model turn and its tool replay."""

    turn_id: str
    task_id: str
    conversation_idx: int
    request: CompletionRequest
    started_at: float
    ended_at: float
    metrics: RequestMetrics | None
    error: str | None
    recorded_prompt_tokens: int | None
    recorded_completion_tokens: int | None
    target_output_tokens: int | None
    recorded_tool_delay_ms: float
    tool_replays: tuple[ToolReplayResult, ...]

    @property
    def success(self) -> bool:
        """Return whether model and tool activity completed without errors."""
        return (
            self.error is None
            and self.metrics is not None
            and not self.metrics.aborted
            and all(not replay.exception_info for replay in self.tool_replays)
        )

    @property
    def replayed_tool_delay_ms(self) -> float:
        """Return the total replayed tool duration after this turn."""
        return sum(replay.replayed_duration_ms for replay in self.tool_replays)


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskResult:
    """Store the ordered turns from one manifest task."""

    task_id: str
    trace_path: Path
    turns: tuple[TurnResult, ...]

    @property
    def success(self) -> bool:
        """Return whether every task turn succeeded."""
        return all(turn.success for turn in self.turns)


@dataclass(frozen=True, slots=True, kw_only=True)
class RunResult:
    """Store all task results from one finite replay."""

    manifest_path: Path
    model: str
    client_backend: ClientBackend
    cache_namespace: str | None
    started_at: float
    ended_at: float
    measured_duration_seconds: float
    observer_duration_seconds: float
    observer_enabled: bool
    tasks: tuple[TaskResult, ...]

    def __post_init__(self) -> None:
        """Reject invalid run clocks and observer metadata."""
        values = (
            self.started_at,
            self.ended_at,
            self.measured_duration_seconds,
            self.observer_duration_seconds,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("run clocks and durations must be finite")
        if self.ended_at < self.started_at:
            raise ValueError("run ended_at must not precede started_at")
        if self.measured_duration_seconds < 0 or self.observer_duration_seconds < 0:
            raise ValueError("run durations must be non-negative")
        if not self.observer_enabled and self.observer_duration_seconds != 0:
            raise ValueError("observer duration requires an enabled observer")

    @property
    def turns(self) -> tuple[TurnResult, ...]:
        """Return all turns in manifest order."""
        return tuple(turn for task in self.tasks for turn in task.turns)

    @property
    def success(self) -> bool:
        """Return whether every task succeeded."""
        return all(task.success for task in self.tasks)


@dataclass(frozen=True, slots=True, kw_only=True)
class _PreparedTask:
    task: ManifestTask
    trace_path: Path
    rows: tuple[TraceRow, ...]
    live_environment: DockerToolEnvironmentConfig | None


def _resolved_under(root: Path, relative: Path, message: str) -> Path:
    """Resolve a relative path under an already resolved root, and reject anything that escapes it."""
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(message)
    return resolved


def _prepare_tasks(manifest_path: Path, config: RunConfig, image_probe: ImageProbe) -> tuple[_PreparedTask, ...]:
    manifest = load_manifest(manifest_path)
    if not manifest.tasks:
        raise ValueError("manifest contains no tasks")
    manifest_root = manifest_path.parent.resolve()
    live_docker: _LiveDocker | None = None
    prepared: list[_PreparedTask] = []
    task_ids: set[str] = set()
    for task in manifest.tasks:
        if task.task_id in task_ids:
            raise ValueError(f"manifest contains duplicate task_id: {task.task_id}")
        task_ids.add(task.task_id)
        if task.trace.is_absolute():
            raise ValueError(f"task {task.task_id} trace must be relative to its manifest")
        trace_path = _resolved_under(
            manifest_root, task.trace, f"task {task.task_id} trace must remain inside its manifest directory"
        )
        rows = tuple(load_trace(trace_path))
        if len(rows) != task.model_calls:
            raise ValueError(
                f"task {task.task_id} manifest records {task.model_calls} model calls but trace contains {len(rows)}"
            )
        if any(row.task_id != task.task_id for row in rows):
            raise ValueError(f"task {task.task_id} trace contains a different task_id")
        if any(not row.messages for row in rows):
            raise ValueError(f"task {task.task_id} trace contains a turn with no messages")
        if config.output_token_policy == "exact" and any(_recorded_target(row) is None for row in rows):
            raise ValueError(
                f"task {task.task_id} trace has a turn with no recorded output length; "
                "the exact output policy needs one for every turn"
            )
        tool_calls = sum(len(row.recorded_tool_calls_after) for row in rows)
        if tool_calls != task.tool_calls:
            raise ValueError(
                f"task {task.task_id} manifest records {task.tool_calls} tool calls but trace contains {tool_calls}"
            )
        # Resolving the live environment here fails a misconfigured task before the
        # first request, instead of discarding the inference already replayed.
        live_environment = None
        if config.tool_mode == "live" and task.tool_calls > 0:
            if live_docker is None:
                live_docker = _resolve_live_docker(config, image_probe)
            live_environment = _live_environment_config(task, config, live_docker, manifest_root)
        prepared.append(_PreparedTask(task=task, trace_path=trace_path, rows=rows, live_environment=live_environment))
    return tuple(prepared)


def _create_client(config: RunConfig) -> CompletionClient:
    return streaming_client(
        config.client_backend,
        base_url=config.base_url,
        api_key=config.api_key,
        timeout_seconds=config.request_timeout_seconds,
        max_connections=SEQUENTIAL_MAX_CONNECTIONS,
    )


def _recorded_target(row: TraceRow) -> int | None:
    target = row.target_output_tokens
    return target if target is not None else row.recorded_completion_tokens


def _max_output_tokens(row: TraceRow, config: RunConfig) -> int:
    # A trace can cap the request without inventing a recorded normalization target.
    if row.max_output_tokens is not None:
        return min(config.max_output_tokens, row.max_output_tokens)
    if config.output_token_policy == "fixed":
        return config.max_output_tokens
    target = _recorded_target(row)
    if target is None:
        # Unreachable under exact: _prepare_tasks refuses a suite whose rows carry no target.
        return config.max_output_tokens
    # exact forbids a margin (RunConfig.__post_init__), so this is the bare target there.
    return max(1, min(config.max_output_tokens, target + config.output_token_margin))


def _request(
    row: TraceRow,
    config: RunConfig,
    sampling: SamplingSettings,
    cache_namespace: str | None,
) -> CompletionRequest:
    messages = [message.to_dict() for message in row.messages]
    if cache_namespace is not None:
        messages = isolate_messages(messages, cache_namespace)
    return CompletionRequest(
        messages=tuple(messages),
        model=config.model,
        tools=tuple(tool.to_dict() for tool in row.tools),
        max_tokens=_max_output_tokens(row, config),
        reasoning_effort=config.reasoning_effort,
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        ignore_eos=config.output_token_policy == "exact",
        extra_body=sampling.extra_body,
    )


def _simulated_tool_replays(row: TraceRow, config: RunConfig) -> tuple[ToolReplayResult, ...]:
    replays: list[ToolReplayResult] = []
    for call in row.recorded_tool_calls_after:
        command = shell_executable(call.command)
        if config.tool_mode == "none":
            replays.append(
                ToolReplayResult(
                    call=call,
                    replayed_duration_ms=0.0,
                    delay_source="none",
                    command=command,
                )
            )
            continue
        replays.append(
            ToolReplayResult(
                call=call,
                replayed_duration_ms=call.duration_ms * config.tool_delay_scale,
                delay_source="recorded" if config.tool_delay_scale == 1.0 else "scale_recorded",
                command=command,
            )
        )
    return tuple(replays)


def _resolve_live_docker(config: RunConfig, image_probe: ImageProbe) -> _LiveDocker:
    """Ask the Docker daemon once for the facts every live task in this run shares."""
    executable = config.docker_executable()
    # An explicit image is used verbatim, so its architecture is never in question.
    architecture = docker_server_architecture(executable) if config.live_tool_image is None else None
    return _LiveDocker(executable=executable, architecture=architecture, image_present=image_probe)


def _live_environment_config(
    task: ManifestTask,
    config: RunConfig,
    live_docker: _LiveDocker,
    manifest_root: Path,
) -> DockerToolEnvironmentConfig:
    specification = task.tool_environment
    # An explicit --live-tool-image names one image on purpose; only a recorded image
    # follows the architecture that runs here without emulation.
    if config.live_tool_image is not None:
        image = config.live_tool_image
    elif specification is not None:
        image = native_swebench_image(specification.image, live_docker.architecture)
    else:
        raise ValueError(f"task {task.task_id} has no Docker image for live tool replay")
    if not live_docker.image_present(image, executable=live_docker.executable):
        raise ValueError(f"task {task.task_id} Docker image is not present locally; pull it first: {image}")

    cwd = specification.cwd if specification is not None else "/workspace"
    interpreter = specification.interpreter if specification is not None else ("bash", "-c")
    network = (
        config.live_network if config.live_network is not None else specification.network if specification else "none"
    )
    workspace_mount = specification.workspace_mount if specification is not None else cwd == "/workspace"
    workspace = None
    if workspace_mount:
        if config.live_workspace_root is None:
            raise ValueError(f"task {task.task_id} requires a workspace mount; configure live_workspace_root")
        if "/" in task.task_id or "\\" in task.task_id or task.task_id in {".", ".."}:
            raise ValueError(f"task_id cannot be used as a workspace directory: {task.task_id!r}")
        workspace_root = config.live_workspace_root.resolve()
        workspace = _resolved_under(
            workspace_root,
            Path(task.task_id),
            f"task {task.task_id} workspace must remain inside live_workspace_root",
        )
        _stage_workspace(task, manifest_root, workspace)
    return DockerToolEnvironmentConfig(
        image=image,
        cwd=cwd,
        network=network,
        executable=live_docker.executable,
        command_timeout_seconds=config.live_timeout_seconds,
        interpreter=interpreter,
        workspace=workspace,
    )


def _stage_workspace(task: ManifestTask, manifest_root: Path, destination: Path) -> None:
    """Give the task an empty workspace, or a fresh copy of the fixture its manifest ships."""
    specification = task.tool_environment
    workspace_path = specification.workspace_path if specification is not None else None
    if workspace_path is None:
        destination.mkdir(parents=True, exist_ok=True)
        return
    source = _resolved_under(
        manifest_root,
        workspace_path,
        f"task {task.task_id} workspace source must remain inside its manifest directory",
    )
    if not source.is_dir():
        raise ValueError(f"task {task.task_id} workspace source does not exist: {source}")
    # Both paths are resolved before staging. Keep the working copy separate from its inputs.
    if source.is_relative_to(destination) or destination.is_relative_to(source):
        raise ValueError(f"task {task.task_id} workspace source and destination must not overlap")
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def _live_tool_replays(
    row: TraceRow,
    environment: LiveToolEnvironment,
) -> tuple[ToolReplayResult, ...]:
    replays: list[ToolReplayResult] = []
    for call in row.recorded_tool_calls_after:
        started_at = perf_counter()
        try:
            result = environment.execute(call)
        except Exception as error:
            replays.append(
                ToolReplayResult(
                    call=call,
                    replayed_duration_ms=(perf_counter() - started_at) * MILLISECONDS_PER_SECOND,
                    delay_source="live",
                    command=shell_executable(call.command),
                    returncode_matches_recorded=None,
                    exception_info=f"{type(error).__name__}: {error}",
                )
            )
            continue
        recorded_returncode = call.recorded_returncode
        replays.append(
            ToolReplayResult(
                call=call,
                replayed_duration_ms=result.duration_ms,
                delay_source="live",
                command=shell_executable(call.command),
                returncode=result.returncode,
                returncode_matches_recorded=(
                    result.returncode == recorded_returncode if recorded_returncode is not None else None
                ),
                exception_info=result.exception_info,
            )
        )
    return tuple(replays)


async def _replay_tools(
    row: TraceRow,
    config: RunConfig,
    environment: LiveToolEnvironment | None,
    sleep: Sleep,
) -> tuple[ToolReplayResult, ...]:
    if not row.recorded_tool_calls_after:
        return ()
    if config.tool_mode == "live":
        if environment is None:
            raise RuntimeError("live tool environment was not created")
        return _live_tool_replays(row, environment)

    replays = _simulated_tool_replays(row, config)
    replayed_duration_ms = sum(replay.replayed_duration_ms for replay in replays)
    if config.tool_mode == "recorded" and replayed_duration_ms > 0:
        await sleep(replayed_duration_ms / MILLISECONDS_PER_SECOND)
    return replays


def _exact_length_error(metrics: RequestMetrics, max_output_tokens: int) -> str | None:
    """Require a turn under the exact policy to have generated the length it was given.

    The request asked the server to ignore end-of-sequence, so a "length" finish is the
    only correct outcome: anything else means the server stopped on its own and the
    turn measured a shorter output than every other run of this suite. Tool-call
    fidelity is not read under this policy, because generation past the model's own
    stop corrupts its actions by design; the qualification probes cover that instead.
    """
    finish_reason = metrics.channels.finish_reason
    if generated_whole_budget(finish_reason, metrics.server_output_tokens, max_output_tokens):
        return None
    if finish_reason != LENGTH_FINISH_REASON:
        return f"exact output policy: server finished with {finish_reason!r} instead of generating the full length"
    return (
        f"exact output policy: server reported {metrics.server_output_tokens} output tokens "
        f"for a {max_output_tokens}-token request"
    )


def _spent_output_budget(metrics: RequestMetrics, max_output_tokens: int) -> bool:
    """Report whether the response used every output token this run allowed it.

    The server's own count settles it when the endpoint reports usage. Without usage
    the locally decoded count stands in, and it can only undercount, so a response
    this misses stays a plain finish_reason mismatch.
    """
    observed = metrics.server_output_tokens if metrics.server_output_tokens is not None else metrics.output_tokens
    return observed >= max_output_tokens


def _verify_turn(config: RunConfig, metrics: RequestMetrics, max_output_tokens: int) -> str | None:
    """Return why a completed turn is unacceptable under the run's output policy, or None.

    The exact policy pins every turn to a length and generates it whole, so the check is
    that length; the other policies let the model choose when to stop, so the check is F0
    tool-call transport, with a cap-truncated action excused.
    """
    if metrics.aborted:
        return "request aborted"
    if config.output_token_policy == "exact":
        return _exact_length_error(metrics, max_output_tokens)
    transport = evaluate_tool_fidelity(
        metrics.channels.tool_calls,
        (),
        finish_reason=metrics.channels.finish_reason,
        output_capped=_spent_output_budget(metrics, max_output_tokens),
    )
    return None if transport.transport_valid else "response failed F0 transport validity"


async def _run_turn(
    row: TraceRow,
    config: RunConfig,
    sampling: SamplingSettings,
    cache_namespace: str | None,
    client: CompletionClient,
    token_counter: TokenCounter,
    environment: LiveToolEnvironment | None,
    sleep: Sleep,
) -> TurnResult:
    request = _request(row, config, sampling, cache_namespace)
    started_at = perf_counter()
    completion: CompletionResult | None = None
    metrics: RequestMetrics | None = None
    error: str | None = None
    try:
        completion = await client.complete(request)
    except Exception as request_error:
        error = f"{type(request_error).__name__}: {request_error}"
    ended_at = perf_counter()
    if completion is not None:
        try:
            metrics = measure_request(
                completion,
                started_at=started_at,
                ended_at=ended_at,
                token_counter=token_counter,
            )
        except Exception as measurement_error:
            error = f"{type(measurement_error).__name__}: {measurement_error}"
        if metrics is not None:
            error = _verify_turn(config, metrics, request.max_tokens)
    # A failed request produced no tool calls to replay, so pacing after it would
    # only add delay that no measured turn accounts for.
    tool_replays = () if error is not None else await _replay_tools(row, config, environment, sleep)
    return TurnResult(
        turn_id=row.turn_id,
        task_id=row.task_id,
        conversation_idx=row.conversation_idx,
        request=request,
        started_at=started_at,
        ended_at=ended_at,
        metrics=metrics,
        error=error,
        recorded_prompt_tokens=row.recorded_prompt_tokens,
        recorded_completion_tokens=row.recorded_completion_tokens,
        target_output_tokens=row.target_output_tokens,
        recorded_tool_delay_ms=row.simulated_tool_delay_ms_after,
        tool_replays=tool_replays,
    )


async def run_manifest(
    manifest_path: Path,
    config: RunConfig,
    *,
    client: CompletionClient | None = None,
    token_counter: TokenCounter | None = None,
    sleep: Sleep = asyncio.sleep,
    live_tool_factory: LiveToolFactory = DockerToolEnvironment,
    live_image_probe: ImageProbe = docker_image_present,
    observer: RunObserver | None = None,
) -> RunResult:
    """Replay every manifest turn exactly once and close the client."""
    prepared_tasks = await asyncio.to_thread(_prepare_tasks, manifest_path, config, live_image_probe)
    counter = token_counter if token_counter is not None else await asyncio.to_thread(TiktokenTokenCounter)
    cache_namespace = resolve_cache_namespace(config.cache_namespace) if config.cache_isolation else None
    sampling = config.sampling()
    completion_client = client if client is not None else _create_client(config)
    started_at = time()
    started_monotonic = perf_counter()
    observer_duration_seconds = 0.0
    task_results: list[TaskResult] = []

    def elapsed_ms() -> float:
        return (perf_counter() - started_monotonic - observer_duration_seconds) * MILLISECONDS_PER_SECOND

    def notify(event: RunBoundaryEvent) -> None:
        nonlocal observer_duration_seconds
        if observer is None:
            return
        observer_started = perf_counter()
        observer.on_boundary(event)
        observer_duration_seconds += perf_counter() - observer_started

    total_turns = sum(len(prepared.rows) for prepared in prepared_tasks)
    completed_turns = 0
    try:
        notify(RunStartedBoundary(tasks=len(prepared_tasks), turns=total_turns))
        for task_ordinal, prepared in enumerate(prepared_tasks, start=1):
            environment: LiveToolEnvironment | None = None
            if prepared.live_environment is not None:
                environment = live_tool_factory(prepared.live_environment)
            turns: list[TurnResult] = []
            try:
                for task_turn_ordinal, row in enumerate(prepared.rows, start=1):
                    notify(
                        TurnStartedBoundary(
                            task=task_ordinal,
                            tasks=len(prepared_tasks),
                            task_turn=task_turn_ordinal,
                            task_turns=len(prepared.rows),
                            turn=completed_turns + 1,
                            turns=total_turns,
                            recorded_prompt_tokens=row.recorded_prompt_tokens,
                        )
                    )
                    turn = await _run_turn(
                        row,
                        config,
                        sampling,
                        cache_namespace,
                        completion_client,
                        counter,
                        environment,
                        sleep,
                    )
                    turns.append(turn)
                    completed_turns += 1
                    metrics = turn.metrics
                    notify(
                        TurnCompletedBoundary(
                            task=task_ordinal,
                            tasks=len(prepared_tasks),
                            task_turn=task_turn_ordinal,
                            task_turns=len(prepared.rows),
                            turn=completed_turns,
                            turns=total_turns,
                            elapsed_ms=elapsed_ms(),
                            time_to_first_token_ms=(
                                metrics.time_to_first_token * MILLISECONDS_PER_SECOND
                                if metrics is not None and metrics.time_to_first_token is not None
                                else None
                            ),
                            e2e_latency_ms=(
                                metrics.duration * MILLISECONDS_PER_SECOND if metrics is not None else None
                            ),
                            output_tokens=metrics.reported_output_tokens if metrics is not None else None,
                            generation_time_ms=(
                                metrics.generation_time * MILLISECONDS_PER_SECOND
                                if metrics is not None and metrics.generation_time is not None
                                else None
                            ),
                            success=turn.success,
                        )
                    )
            finally:
                if environment is not None:
                    environment.cleanup()
            task_results.append(
                TaskResult(
                    task_id=prepared.task.task_id,
                    trace_path=prepared.trace_path,
                    turns=tuple(turns),
                )
            )
    finally:
        await completion_client.close()
    run_success = all(task.success for task in task_results)
    elapsed_before_finish_seconds = perf_counter() - started_monotonic - observer_duration_seconds
    notify(
        RunFinishedBoundary(
            completed_tasks=len(task_results),
            tasks=len(prepared_tasks),
            completed_turns=completed_turns,
            turns=total_turns,
            elapsed_ms=elapsed_before_finish_seconds * MILLISECONDS_PER_SECOND,
            success=run_success,
        )
    )
    ended_monotonic = perf_counter()
    return RunResult(
        manifest_path=manifest_path,
        model=config.model,
        client_backend=config.client_backend,
        cache_namespace=cache_namespace,
        started_at=started_at,
        # A backwards wall-clock step must not discard a finished run. Reported
        # durations come from perf_counter and are unaffected by this clamp.
        ended_at=max(time(), started_at),
        measured_duration_seconds=ended_monotonic - started_monotonic - observer_duration_seconds,
        observer_duration_seconds=observer_duration_seconds,
        observer_enabled=observer is not None,
        tasks=tuple(task_results),
    )
