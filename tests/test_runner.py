"""Exercise finite replay through the public runner surface."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter, sleep

import pytest

from agentperf_local.client.protocol import CompletionResult, RawRead
from agentperf_local.client.request import CompletionRequest
from agentperf_local.replay.config import OutputTokenPolicy, RunConfig, ToolReplayMode
from agentperf_local.replay.runner import (
    ImageProbe,
    LiveToolEnvironment,
    RunBoundaryEvent,
    RunFinishedBoundary,
    RunStartedBoundary,
    TurnCompletedBoundary,
    TurnStartedBoundary,
    run_manifest,
)
from agentperf_local.tools.docker import DockerToolEnvironmentConfig, ToolExecutionResult
from agentperf_local.workload.schema import (
    DockerToolEnvironmentSpec,
    ManifestTask,
    RecordedToolCall,
    ReplayManifest,
    RequestMessage,
    TraceRow,
    write_manifest,
    write_trace,
)

SSE_RESPONSE = (
    b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n'
    b'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2}}\n\n'
    b"data: [DONE]\n\n"
)
LENGTH_CAPPED_RESPONSE = (
    b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"length"}]}\n\n'
    b'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2}}\n\n'
    b"data: [DONE] \n\n"
)
# A call the cap cut mid-arguments: the streamed text stops where the cut fell.
TOOL_CALL_LENGTH_CAPPED_RESPONSE = (
    b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function",'
    b'"function":{"name":"bash","arguments":"{\\"command\\":\\"rg ne"}}]},"finish_reason":"length"}]}\n\n'
    b'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":7}}\n\n'
    b"data: [DONE]\n\n"
)
SYNTHETIC_FAILURE_CALL = 2
# The fake responses below stop on their own, which only the recorded policy allows;
# the exact policy has its own tests further down.
ISOLATION_OFF_CONFIG = RunConfig(
    base_url="http://127.0.0.1:1/v1",
    model="local-model",
    client_backend="python",
    cache_isolation=False,
    output_token_policy="recorded",
)


@dataclass
class _FakeClient:
    response: bytes = SSE_RESPONSE
    failure_call: int | None = SYNTHETIC_FAILURE_CALL
    requests: list[CompletionRequest] = field(default_factory=list)
    closed: bool = False

    async def complete(
        self,
        request: CompletionRequest,
        abort: asyncio.Event | None = None,
    ) -> CompletionResult:
        self.requests.append(request)
        if self.failure_call is not None and len(self.requests) == self.failure_call:
            raise RuntimeError("synthetic request failure")
        return CompletionResult(
            reads=(RawRead(timestamp=perf_counter(), data=self.response),),
            aborted=abort.is_set() if abort is not None else False,
        )

    async def close(self) -> None:
        self.closed = True


@dataclass
class _FakeTokenCounter:
    texts: list[str] = field(default_factory=list)

    def count(self, text: str) -> int:
        self.texts.append(text)
        return len(text)


@dataclass
class _FakeLiveEnvironment:
    config: DockerToolEnvironmentConfig
    calls: list[RecordedToolCall] = field(default_factory=list)
    cleaned: bool = False

    def execute(self, call: RecordedToolCall) -> ToolExecutionResult:
        self.calls.append(call)
        return ToolExecutionResult(output="synthetic", returncode=0, duration_ms=3.0)

    def cleanup(self) -> None:
        self.cleaned = True


@dataclass
class _FakeLiveFactory:
    configs: list[DockerToolEnvironmentConfig] = field(default_factory=list)
    environments: list[_FakeLiveEnvironment] = field(default_factory=list)

    def __call__(self, config: DockerToolEnvironmentConfig) -> LiveToolEnvironment:
        environment = _FakeLiveEnvironment(config=config)
        self.configs.append(config)
        self.environments.append(environment)
        return environment


@dataclass
class _RecordingObserver:
    events: list[RunBoundaryEvent] = field(default_factory=list)

    def on_boundary(self, event: RunBoundaryEvent) -> None:
        self.events.append(event)


@dataclass
class _FailingObserver:
    def on_boundary(self, event: RunBoundaryEvent) -> None:
        raise RuntimeError("synthetic observer failure")


@dataclass
class _FinalDelayObserver:
    delay_seconds: float
    finished: bool = False

    def on_boundary(self, event: RunBoundaryEvent) -> None:
        if isinstance(event, RunFinishedBoundary):
            sleep(self.delay_seconds)
            self.finished = True


@pytest.mark.parametrize(
    ("tool_mode", "output_policy", "expected_max_tokens", "expected_sleeps", "expected_tool_durations"),
    [
        ("none", "recorded", [7, 8, 9], [], [0.0, 0.0]),
        ("recorded", "recorded", [7, 8, 9], [0.005, 0.010], [5.0, 10.0]),
        ("live", "recorded", [7, 8, 9], [], [3.0, 3.0]),
        ("none", "fixed", [12, 12, 12], [], [0.0, 0.0]),
    ],
)
async def test_run_manifest_executes_every_turn_once_in_source_order(
    tmp_path: Path,
    tool_mode: ToolReplayMode,
    output_policy: OutputTokenPolicy,
    expected_max_tokens: list[int],
    expected_sleeps: list[float],
    expected_tool_durations: list[float],
) -> None:
    manifest_path = _write_replay(tmp_path)
    client = _FakeClient()
    token_counter = _FakeTokenCounter()
    sleeps: list[float] = []
    live_factory = _FakeLiveFactory()
    observer = _RecordingObserver()

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    config = RunConfig(
        base_url="http://127.0.0.1:1/v1",
        model="local-model",
        client_backend="python",
        output_token_policy=output_policy,
        max_output_tokens=12,
        output_token_margin=2,
        cache_namespace="1 2 3",
        tool_mode=tool_mode,
        tool_delay_scale=0.5,
    )

    result = await run_manifest(
        manifest_path,
        config,
        client=client,
        token_counter=token_counter,
        sleep=fake_sleep,
        live_tool_factory=live_factory,
        live_image_probe=_accept_any_image,
        observer=observer,
    )

    assert [task.task_id for task in result.tasks] == ["task_demo", "task_other"]
    assert [turn.turn_id for turn in result.turns] == ["task_demo:0000", "task_demo:0001", "task_other:0000"]
    assert [request.messages[0]["content"] for request in client.requests] == [
        "1 2 3\nPerformance replay cache namespace. Ignore the digits above.\n\nfirst",
        "1 2 3\nPerformance replay cache namespace. Ignore the digits above.\n\nsecond",
        "1 2 3\nPerformance replay cache namespace. Ignore the digits above.\n\nthird",
    ]
    assert [request.max_tokens for request in client.requests] == expected_max_tokens
    assert [request.temperature for request in client.requests] == [0.7, 0.7, 0.7]
    assert [request.top_p for request in client.requests] == [0.8, 0.8, 0.8]
    assert [request.extra_body for request in client.requests] == [
        (("top_k", 20), ("min_p", 0.0)),
        (("top_k", 20), ("min_p", 0.0)),
        (("top_k", 20), ("min_p", 0.0)),
    ]
    assert [turn.error for turn in result.turns] == [None, "RuntimeError: synthetic request failure", None]
    assert [turn.metrics.output_tokens if turn.metrics is not None else None for turn in result.turns] == [2, None, 2]
    assert [replay.replayed_duration_ms for turn in result.turns for replay in turn.tool_replays] == pytest.approx(
        expected_tool_durations
    )
    assert sleeps == pytest.approx(expected_sleeps)
    assert token_counter.texts == ["ok", "ok"]
    assert client.closed is True
    assert result.success is False
    assert result.ended_at >= result.started_at
    assert result.measured_duration_seconds >= 0
    assert result.observer_duration_seconds >= 0
    assert result.observer_enabled is True
    assert observer.events[0] == RunStartedBoundary(tasks=2, turns=3)
    # Every request is announced before it is sent and closed after it returns.
    assert [type(event) for event in observer.events] == [
        RunStartedBoundary,
        *(TurnStartedBoundary, TurnCompletedBoundary) * 3,
        RunFinishedBoundary,
    ]
    turn_events = [event for event in observer.events if isinstance(event, TurnCompletedBoundary)]
    assert [event.turn for event in turn_events] == [1, 2, 3]
    assert [event.success for event in turn_events] == [True, False, True]
    assert all(event.elapsed_ms >= 0 for event in turn_events)
    assert all(event.output_tokens is not None for event in turn_events if event.success)
    assert all(
        event.generation_time_ms is None
        or event.e2e_latency_ms is None
        or event.generation_time_ms <= event.e2e_latency_ms
        for event in turn_events
    )
    assert isinstance(observer.events[-1], RunFinishedBoundary)
    assert observer.events[-1].success is False
    assert observer.events[-1].completed_turns == 3
    assert "task_demo" not in repr(observer.events)
    if tool_mode == "live":
        assert [config.image for config in live_factory.configs] == ["synthetic-tool:latest", "synthetic-tool:latest"]
        assert [len(environment.calls) for environment in live_factory.environments] == [1, 1]
        assert all(environment.cleaned for environment in live_factory.environments)
    else:
        assert live_factory.environments == []


@pytest.mark.parametrize(
    "build",
    [
        lambda: RunConfig(base_url="http://localhost", model="m", max_output_tokens=0),
        lambda: RunConfig(base_url="http://localhost", model="m", cache_isolation=False, cache_namespace="x"),
        lambda: RunConfig(base_url="http://localhost", model="m", top_p=0.0),
        lambda: RunConfig(
            base_url="http://localhost",
            model="m",
            tool_mode="none",
            live_tool_image="unused:latest",
        ),
    ],
)
def test_run_config_rejects_invalid_combinations(build: Callable[[], RunConfig]) -> None:
    with pytest.raises(ValueError):
        build()


async def test_run_manifest_rejects_protocol_incomplete_responses(tmp_path: Path) -> None:
    incomplete_response = b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
    client = _FakeClient(response=incomplete_response, failure_call=None)

    result = await run_manifest(
        _write_replay(tmp_path),
        ISOLATION_OFF_CONFIG,
        client=client,
        token_counter=_FakeTokenCounter(),
    )

    assert result.success is False
    assert [turn.error for turn in result.turns] == ["response failed F0 transport validity"] * len(result.turns)
    assert client.closed is True


@pytest.mark.parametrize(
    ("max_output_tokens", "expected_errors"),
    # The response reports 7 output tokens. Allowing exactly 7 means the run's own cap
    # cut the action short; allowing 9 means the model stopped early on its own.
    [(7, [None, None, None]), (9, ["response failed F0 transport validity"] * 3)],
    ids=("spent-the-budget", "stopped-under-the-budget"),
)
async def test_an_action_cut_short_by_the_output_cap_is_not_a_failed_turn(
    tmp_path: Path,
    max_output_tokens: int,
    expected_errors: list[str | None],
) -> None:
    client = _FakeClient(response=TOOL_CALL_LENGTH_CAPPED_RESPONSE, failure_call=None)

    result = await run_manifest(
        _write_replay(tmp_path),
        RunConfig(
            base_url="http://127.0.0.1:1/v1",
            model="local-model",
            client_backend="python",
            cache_isolation=False,
            output_token_policy="fixed",
            max_output_tokens=max_output_tokens,
        ),
        client=client,
        token_counter=_FakeTokenCounter(),
    )

    assert [turn.error for turn in result.turns] == expected_errors
    assert [turn.metrics.channels.finish_reason for turn in result.turns if turn.metrics is not None] == ["length"] * 3


def _exact_response(completion_tokens: int, finish_reason: str) -> bytes:
    """Emulate a server answering an ignore_eos request with the given count and finish."""
    return (
        b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"' + finish_reason.encode() + b'"}]}\n\n'
        b'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":'
        + str(completion_tokens).encode()
        + b"}}\n\n"
        b"data: [DONE]\n\n"
    )


EXACT_CONFIG = RunConfig(
    base_url="http://127.0.0.1:1/v1",
    model="local-model",
    client_backend="python",
    cache_isolation=False,
)


def _write_single_task_manifest(tmp_path: Path, rows: list[TraceRow]) -> Path:
    """Write a one-task manifest around the given trace rows and return its path."""
    write_trace(tmp_path / "traces" / "only.jsonl", rows)
    manifest_path = tmp_path / "manifest.json"
    write_manifest(
        manifest_path,
        ReplayManifest(
            name="exact-test",
            source="synthetic",
            tasks=[
                ManifestTask(
                    task_id="task_demo",
                    trace=Path("traces/only.jsonl"),
                    source_recording=Path("only.json"),
                    model_calls=len(rows),
                    tool_calls=sum(len(row.recorded_tool_calls_after) for row in rows),
                    total_recorded_tool_delay_ms=0.0,
                )
            ],
        ),
    )
    return manifest_path


async def test_exact_policy_asks_for_each_recorded_length_with_eos_ignored(tmp_path: Path) -> None:
    """The default policy pins every request to its recorded length; the fake serves exactly that."""
    client = _FakeClient(response=_exact_response(7, "length"), failure_call=None)

    result = await run_manifest(_write_replay(tmp_path), EXACT_CONFIG, client=client, token_counter=_FakeTokenCounter())

    assert EXACT_CONFIG.output_token_policy == "exact"
    assert [request.max_tokens for request in client.requests] == [5, 6, 7]
    assert all(request.body()["ignore_eos"] is True for request in client.requests)
    assert len(result.turns) == 3


@pytest.mark.parametrize(
    ("completion_tokens", "finish_reason", "expected_error"),
    [
        (7, "length", None),
        (7, "stop", "exact output policy: server finished with 'stop' instead of generating the full length"),
        (3, "length", "exact output policy: server reported 3 output tokens for a 7-token request"),
    ],
    ids=("full-length", "stopped-early", "under-reported"),
)
async def test_exact_policy_requires_the_full_length_from_every_turn(
    tmp_path: Path, completion_tokens: int, finish_reason: str, expected_error: str | None
) -> None:
    manifest_path = _write_single_task_manifest(
        tmp_path, [_trace_row(task_id="task_demo", index=0, content="only", target_tokens=7)]
    )
    client = _FakeClient(response=_exact_response(completion_tokens, finish_reason), failure_call=None)

    result = await run_manifest(manifest_path, EXACT_CONFIG, client=client, token_counter=_FakeTokenCounter())

    assert [turn.error for turn in result.turns] == [expected_error]
    assert result.success is (expected_error is None)


async def test_exact_policy_refuses_a_suite_without_recorded_lengths(tmp_path: Path) -> None:
    """A turn with no target would fall back to the cap and generate freely; refuse before any request."""
    rows = [
        TraceRow(
            turn_id="task_demo:0000",
            task_id="task_demo",
            conversation_id="task_demo",
            conversation_idx=0,
            messages=[RequestMessage(role="user", content="no target")],
            simulated_tool_delay_ms_after=0.0,
        )
    ]
    manifest_path = _write_single_task_manifest(tmp_path, rows)
    client = _FakeClient(failure_call=None)

    with pytest.raises(ValueError, match="no recorded output length"):
        await run_manifest(manifest_path, EXACT_CONFIG, client=client, token_counter=_FakeTokenCounter())

    assert client.requests == []


async def test_length_capped_turns_succeed_with_padded_done_sentinel(tmp_path: Path) -> None:
    client = _FakeClient(response=LENGTH_CAPPED_RESPONSE, failure_call=None)

    result = await run_manifest(
        _write_replay(tmp_path),
        ISOLATION_OFF_CONFIG,
        client=client,
        token_counter=_FakeTokenCounter(),
    )

    assert [turn.error for turn in result.turns] == [None, None, None]
    assert [turn.metrics.channels.finish_reason for turn in result.turns if turn.metrics is not None] == ["length"] * 3
    assert result.success is True


@dataclass
class _AnnouncementObserver:
    """Record how many requests the client had sent when each turn was announced."""

    client: _FakeClient
    announcements: list[tuple[int, int | None]] = field(default_factory=list)

    def on_boundary(self, event: RunBoundaryEvent) -> None:
        if isinstance(event, TurnStartedBoundary):
            self.announcements.append((len(self.client.requests), event.recorded_prompt_tokens))


async def test_turn_is_announced_with_its_recorded_size_before_the_request_is_sent(tmp_path: Path) -> None:
    client = _FakeClient(failure_call=None)
    observer = _AnnouncementObserver(client=client)

    await run_manifest(
        _write_replay(tmp_path),
        ISOLATION_OFF_CONFIG,
        client=client,
        token_counter=_FakeTokenCounter(),
        observer=observer,
    )

    # Each announcement names the size of a request that has not been sent yet.
    assert observer.announcements == [(0, 10), (1, 12), (2, 14)]


async def test_closes_client_when_start_observer_fails(tmp_path: Path) -> None:
    client = _FakeClient()

    with pytest.raises(RuntimeError, match="synthetic observer failure"):
        await run_manifest(
            _write_replay(tmp_path),
            ISOLATION_OFF_CONFIG,
            client=client,
            token_counter=_FakeTokenCounter(),
            observer=_FailingObserver(),
        )

    assert client.closed is True
    assert client.requests == []


async def test_final_boundary_is_included_in_observer_duration(tmp_path: Path) -> None:
    client = _FakeClient()
    observer_delay_seconds = 0.01
    observer = _FinalDelayObserver(delay_seconds=observer_delay_seconds)

    result = await run_manifest(
        _write_replay(tmp_path),
        ISOLATION_OFF_CONFIG,
        client=client,
        token_counter=_FakeTokenCounter(),
        observer=observer,
    )

    assert observer.finished is True
    assert result.observer_duration_seconds >= observer_delay_seconds


async def test_live_replay_creates_one_scoped_workspace_per_task(tmp_path: Path) -> None:
    manifest_path = _write_replay(tmp_path / "workload", workspace_mounts=(True, True))
    workspace_root = tmp_path / "workspaces"
    live_factory = _FakeLiveFactory()

    result = await run_manifest(
        manifest_path,
        RunConfig(
            base_url="http://127.0.0.1:1/v1",
            model="local-model",
            client_backend="python",
            cache_isolation=False,
            tool_mode="live",
            live_workspace_root=workspace_root,
        ),
        client=_FakeClient(),
        token_counter=_FakeTokenCounter(),
        live_tool_factory=live_factory,
        live_image_probe=_accept_any_image,
    )

    assert [config.workspace for config in live_factory.configs] == [
        workspace_root / "task_demo",
        workspace_root / "task_other",
    ]
    assert all(config.workspace is not None and config.workspace.is_dir() for config in live_factory.configs)
    assert all(environment.cleaned for environment in live_factory.environments)
    assert result.success is False


async def test_live_replay_stages_manifest_workspace_per_task(tmp_path: Path) -> None:
    workload_root = tmp_path / "workload"
    source = workload_root / "fixtures" / "task_demo"
    source.mkdir(parents=True)
    (source / "input.txt").write_text("clean input", encoding="utf-8")
    workspace_root = tmp_path / "workspaces"
    stale_workspace = workspace_root / "task_demo"
    stale_workspace.mkdir(parents=True)
    (stale_workspace / "stale.txt").write_text("old output", encoding="utf-8")
    live_factory = _FakeLiveFactory()

    await run_manifest(
        _write_replay(workload_root, workspace_mounts=(True, False), first_workspace_path=Path("fixtures/task_demo")),
        RunConfig(
            base_url="http://127.0.0.1:1/v1",
            model="local-model",
            client_backend="python",
            cache_isolation=False,
            tool_mode="live",
            live_workspace_root=workspace_root,
        ),
        client=_FakeClient(),
        token_counter=_FakeTokenCounter(),
        live_tool_factory=live_factory,
        live_image_probe=_accept_any_image,
    )

    assert live_factory.configs[0].workspace == workspace_root / "task_demo"
    assert (workspace_root / "task_demo" / "input.txt").read_text(encoding="utf-8") == "clean input"
    assert not (workspace_root / "task_demo" / "stale.txt").exists()
    assert (source / "input.txt").read_text(encoding="utf-8") == "clean input"


@pytest.mark.parametrize(
    ("source_path", "workspace_path"),
    [
        (Path("fixtures/task_demo"), Path("fixtures")),
        (Path("fixtures/task_demo/inputs"), Path("fixtures")),
        (Path("fixtures/task_demo"), Path("fixtures/task_demo/generated")),
    ],
    ids=("same-directory", "destination-contains-source", "source-contains-destination"),
)
async def test_live_replay_rejects_overlapping_workspaces_without_changing_inputs(
    tmp_path: Path,
    source_path: Path,
    workspace_path: Path,
) -> None:
    source = tmp_path / source_path
    source.mkdir(parents=True)
    input_path = source / "input.txt"
    input_path.write_text("original input", encoding="utf-8")
    manifest_path = _write_replay(tmp_path, workspace_mounts=(True, False), first_workspace_path=source_path)
    client = _FakeClient()
    live_factory = _FakeLiveFactory()

    with pytest.raises(ValueError, match="workspace source and destination must not overlap"):
        await run_manifest(
            manifest_path,
            RunConfig(
                base_url="http://127.0.0.1:1/v1",
                model="local-model",
                client_backend="python",
                cache_isolation=False,
                tool_mode="live",
                live_workspace_root=tmp_path / workspace_path,
            ),
            client=client,
            token_counter=_FakeTokenCounter(),
            live_tool_factory=live_factory,
            live_image_probe=_accept_any_image,
        )

    assert input_path.read_text(encoding="utf-8") == "original input"
    assert list(source.iterdir()) == [input_path]
    assert client.requests == []
    assert live_factory.configs == []


async def test_failed_turns_skip_tool_replay(tmp_path: Path) -> None:
    client = _FakeClient(failure_call=1)
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    result = await run_manifest(
        _write_replay(tmp_path),
        RunConfig(
            base_url="http://127.0.0.1:1/v1",
            model="local-model",
            client_backend="python",
            cache_isolation=False,
            output_token_policy="recorded",
            tool_mode="recorded",
            tool_delay_scale=1.0,
        ),
        client=client,
        token_counter=_FakeTokenCounter(),
        sleep=fake_sleep,
    )

    assert [turn.error for turn in result.turns] == ["RuntimeError: synthetic request failure", None, None]
    assert [len(turn.tool_replays) for turn in result.turns] == [0, 0, 1]
    assert sleeps == pytest.approx([0.020])


async def test_manifest_without_tasks_fails_before_any_request(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest_path, ReplayManifest(name="empty", source="synthetic", tasks=[]))
    client = _FakeClient()

    with pytest.raises(ValueError, match="manifest contains no tasks"):
        await run_manifest(
            manifest_path,
            RunConfig(
                base_url="http://127.0.0.1:1/v1",
                model="local-model",
                client_backend="python",
                cache_isolation=False,
            ),
            client=client,
            token_counter=_FakeTokenCounter(),
        )

    assert client.requests == []


def _accept_any_image(image: str, *, executable: str) -> bool:
    """Accept every image so live tests never reach a Docker daemon."""
    return True


def _reject_any_image(image: str, *, executable: str) -> bool:
    """Report every image as missing, without reaching a Docker daemon."""
    return False


@pytest.mark.parametrize(
    ("workspace_mounts", "image_probe", "message"),
    (
        pytest.param((False, True), _accept_any_image, "requires a workspace mount", id="later-task-mount"),
        pytest.param((False, False), _reject_any_image, "not present locally", id="missing-image"),
    ),
)
async def test_live_replay_refuses_a_setup_problem_before_any_request(
    tmp_path: Path,
    workspace_mounts: tuple[bool, bool],
    image_probe: ImageProbe,
    message: str,
) -> None:
    manifest_path = _write_replay(tmp_path / "workload", workspace_mounts=workspace_mounts)
    client = _FakeClient()
    live_factory = _FakeLiveFactory()

    with pytest.raises(ValueError, match=message):
        await run_manifest(
            manifest_path,
            RunConfig(
                base_url="http://127.0.0.1:1/v1",
                model="local-model",
                client_backend="python",
                cache_isolation=False,
                tool_mode="live",
            ),
            client=client,
            token_counter=_FakeTokenCounter(),
            live_tool_factory=live_factory,
            live_image_probe=image_probe,
        )

    assert client.requests == []
    assert live_factory.environments == []


async def test_live_replay_runs_the_swebench_image_for_the_daemon_architecture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _write_replay(tmp_path / "workload", image="docker.io/swebench/sweb.eval.x86_64.demo:latest")
    monkeypatch.setattr("agentperf_local.replay.runner.docker_server_architecture", lambda executable: "arm64")
    live_factory = _FakeLiveFactory()

    await run_manifest(
        manifest_path,
        RunConfig(
            base_url="http://127.0.0.1:1/v1",
            model="local-model",
            client_backend="python",
            cache_isolation=False,
            tool_mode="live",
        ),
        client=_FakeClient(),
        token_counter=_FakeTokenCounter(),
        live_tool_factory=live_factory,
        live_image_probe=_accept_any_image,
    )

    assert [config.image for config in live_factory.configs] == [
        "docker.io/swebench/sweb.eval.arm64.demo:latest",
        "docker.io/swebench/sweb.eval.arm64.demo:latest",
    ]


def _write_replay(
    root: Path,
    *,
    workspace_mounts: tuple[bool, bool] = (False, False),
    first_workspace_path: Path | None = None,
    image: str = "synthetic-tool:latest",
) -> Path:
    traces = root / "traces"
    first_rows = [
        _trace_row(
            task_id="task_demo",
            index=0,
            content="first",
            target_tokens=5,
            tool_call=RecordedToolCall(
                duration_ms=10.0,
                step=1,
                action_index=0,
                recorded_returncode=0,
                tool_name="bash",
                action={"command": "rg -n needle file.txt"},
            ),
        ),
        _trace_row(task_id="task_demo", index=1, content="second", target_tokens=6),
    ]
    second_rows = [
        _trace_row(
            task_id="task_other",
            index=0,
            content="third",
            target_tokens=7,
            tool_call=RecordedToolCall(
                duration_ms=20.0,
                step=8,
                action_index=0,
                recorded_returncode=0,
                tool_name="bash",
                action={"command": "rg -n another file.txt"},
            ),
        )
    ]
    write_trace(traces / "first.jsonl", first_rows)
    write_trace(traces / "second.jsonl", second_rows)
    first_environment = _environment(workspace_mounts[0], image, workspace_path=first_workspace_path)
    second_environment = _environment(workspace_mounts[1], image)
    manifest = ReplayManifest(
        name="runner-test",
        source="synthetic",
        tasks=[
            ManifestTask(
                task_id="task_demo",
                trace=Path("traces/first.jsonl"),
                source_recording=Path("first.json"),
                model_calls=2,
                tool_calls=1,
                total_recorded_tool_delay_ms=10.0,
                tool_environment=first_environment,
            ),
            ManifestTask(
                task_id="task_other",
                trace=Path("traces/second.jsonl"),
                source_recording=Path("second.json"),
                model_calls=1,
                tool_calls=1,
                total_recorded_tool_delay_ms=20.0,
                tool_environment=second_environment,
            ),
        ],
    )
    manifest_path = root / "manifest.json"
    write_manifest(manifest_path, manifest)
    return manifest_path


def _environment(workspace_mount: bool, image: str, *, workspace_path: Path | None = None) -> DockerToolEnvironmentSpec:
    return DockerToolEnvironmentSpec(
        image=image,
        cwd="/workspace",
        interpreter=("bash", "-lc"),
        workspace_mount=workspace_mount,
        workspace_path=workspace_path,
    )


def _trace_row(
    *,
    task_id: str,
    index: int,
    content: str,
    target_tokens: int,
    tool_call: RecordedToolCall | None = None,
) -> TraceRow:
    tool_calls = [tool_call] if tool_call is not None else []
    return TraceRow(
        turn_id=f"{task_id}:{index:04d}",
        task_id=task_id,
        conversation_id=task_id,
        conversation_idx=index,
        messages=[RequestMessage(role="user", content=content)],
        target_output_tokens=target_tokens,
        recorded_prompt_tokens=target_tokens * 2,
        recorded_completion_tokens=target_tokens,
        recorded_total_tokens=target_tokens * 3,
        simulated_tool_delay_ms_after=sum(call.duration_ms for call in tool_calls),
        recorded_tool_calls_after=tool_calls,
    )
