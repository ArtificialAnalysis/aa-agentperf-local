"""Exercise report generation through its public artifact surface."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr

from agentperf_local.client.request import CompletionRequest
from agentperf_local.common.json_types import JsonObject, JsonValue
from agentperf_local.metrics.request import RequestMetrics
from agentperf_local.metrics.response import ResponseChannels, ToolCall
from agentperf_local.provenance.benchmark import BENCHMARK_CONTEXT_TOKENS
from agentperf_local.provenance.context import ContextObservationReason, RunContextFacts
from agentperf_local.replay.config import RunConfig
from agentperf_local.replay.runner import RunResult, TaskResult, ToolReplayResult, TurnResult
from agentperf_local.reports.reporting import (
    REPORT_VERSION,
    SHORT_OUTPUT_RATIO_WARNING_THRESHOLD,
    ArtifactPaths,
    normalize_output_length,
    turn_decode_tokens_per_second,
    write_run_artifacts,
)
from agentperf_local.workload.schema import RecordedToolCall, parse_json_object
from tests.file_modes import has_mode

RUN_ID = "8f5b2f2e-4c3a-4d6e-9b1a-2c3d4e5f6a7b"
FULL_RUN_CONTEXT = RunContextFacts(
    requested_tokens=BENCHMARK_CONTEXT_TOKENS,
    observed_tokens=BENCHMARK_CONTEXT_TOKENS,
    observed_reason=ContextObservationReason.REPORTED,
)


def test_write_run_artifacts_emits_versioned_deterministic_summaries(tmp_path: Path) -> None:
    config = RunConfig(
        base_url="http://127.0.0.1:8000/v1",
        model="local-model",
        api_key=SecretStr("never-serialize-this-key"),
        client_backend="python",
        output_token_policy="recorded",
        max_output_tokens=20,
        output_token_margin=2,
        cache_namespace="1 2 3",
        tool_mode="fixed_delay",
    )
    request = CompletionRequest(
        messages=({"role": "user", "content": "sanitized"},),
        model=config.model,
        max_tokens=12,
        temperature=0.7,
        top_p=0.8,
    )
    metrics = RequestMetrics(
        started_at=10.0,
        ended_at=11.0,
        time_to_first_byte=0.1,
        time_to_first_token=0.2,
        generation_time=0.4,
        prompt_tokens=8,
        output_tokens=4,
        server_output_tokens=4,
        channels=ResponseChannels(content="tiny", reasoning="", tool_calls=(), finish_reason="stop"),
        chunks=(),
        aborted=False,
        cached_prompt_tokens=6,
    )
    recorded_call = RecordedToolCall(
        duration_ms=100.0,
        step=1,
        action_index=0,
        recorded_returncode=0,
        tool_call_id="call_1",
        tool_name="bash",
        action={"command": "rg -n needle file.txt"},
    )
    successful_turn = TurnResult(
        turn_id="task_demo:0000",
        task_id="task_demo",
        conversation_idx=0,
        request=request,
        started_at=10.0,
        ended_at=11.0,
        metrics=metrics,
        error=None,
        recorded_prompt_tokens=10,
        recorded_completion_tokens=10,
        target_output_tokens=10,
        recorded_tool_delay_ms=100.0,
        tool_replays=(
            ToolReplayResult(
                call=recorded_call,
                replayed_duration_ms=100.0,
                delay_source="recorded",
                command="rg",
                returncode=0,
                returncode_matches_recorded=True,
            ),
        ),
    )
    failed_turn = TurnResult(
        turn_id="task_demo:0001",
        task_id="task_demo",
        conversation_idx=1,
        request=request,
        started_at=11.0,
        ended_at=11.1,
        metrics=None,
        error="RuntimeError: synthetic failure",
        recorded_prompt_tokens=20,
        recorded_completion_tokens=6,
        target_output_tokens=6,
        recorded_tool_delay_ms=0.0,
        tool_replays=(),
    )
    result = RunResult(
        manifest_path=Path("fixtures/manifest.json"),
        model=config.model,
        client_backend=config.client_backend,
        cache_namespace=config.cache_namespace,
        started_at=100.0,
        ended_at=102.0,
        measured_duration_seconds=2.0,
        observer_duration_seconds=0.0,
        observer_enabled=False,
        tasks=(
            TaskResult(
                task_id="task_demo",
                trace_path=Path("fixtures/traces/task_demo.jsonl"),
                turns=(successful_turn, failed_turn),
            ),
        ),
    )

    paths = write_run_artifacts(result, tmp_path, config, run_context=FULL_RUN_CONTEXT, run_id=RUN_ID)
    first_contents = {path: path.read_bytes() for path in _paths(paths)}
    with pytest.raises(FileExistsError, match="must not replace"):
        write_run_artifacts(result, tmp_path, config, run_context=FULL_RUN_CONTEXT, run_id=RUN_ID)

    assert [path.name for path in _paths(paths)] == [
        "turns.jsonl",
        "tasks.json",
        "tools.json",
        "failures.json",
        "summary.json",
    ]
    assert first_contents == {path: path.read_bytes() for path in _paths(paths)}
    assert b"never-serialize-this-key" not in b"".join(first_contents.values())
    assert not (tmp_path / "tasks.csv").exists()
    assert all(has_mode(path, 0o600) for path in _paths(paths))

    turn_rows = [parse_json_object(line, paths.turns) for line in paths.turns.read_bytes().splitlines() if line.strip()]
    assert [row["version"] for row in turn_rows] == [REPORT_VERSION, REPORT_VERSION]
    assert [row["turn_id"] for row in turn_rows] == ["task_demo:0000", "task_demo:0001"]
    first_normalization = _object(turn_rows[0]["normalization"])
    first_timing = _object(turn_rows[0]["timing"])
    first_tokens = _object(turn_rows[0]["tokens"])
    assert first_timing["generation_ms"] == pytest.approx(400.0)
    assert first_normalization["normalized_generation_ms"] == pytest.approx(1200.0)
    assert first_normalization["normalized_e2e_latency_ms"] == pytest.approx(1800.0)
    assert first_tokens["server_cached_prompt_tokens"] == 6
    assert first_tokens["server_uncached_prompt_tokens"] == 2
    warning = _object(first_normalization["warning"])
    assert warning["observed_to_target_ratio"] == pytest.approx(0.4)
    assert warning["threshold"] == SHORT_OUTPUT_RATIO_WARNING_THRESHOLD
    tool_call = _objects(turn_rows[0], "tool_calls")[0]
    assert tool_call["delay_source"] == "recorded"
    assert tool_call["returncode_matches_recorded"] is True
    # Report v1 keeps the timing-profile keys as null after the profiled mode left.
    assert tool_call["profile_key"] is None

    tasks = parse_json_object(paths.tasks.read_bytes(), paths.tasks)
    task = _objects(tasks, "tasks")[0]
    task_totals = _object(task["totals"])
    assert task_totals["turns"] == 2
    assert task_totals["successful_turns"] == 1
    assert task_totals["failed_turns"] == 1
    assert task_totals["short_output_warnings"] == 1
    assert task_totals["total_agentic_replay_ms"] == pytest.approx(1100.0)
    assert task_totals["total_server_cached_prompt_tokens"] == 6
    assert task_totals["total_server_uncached_prompt_tokens"] == 2

    tools = parse_json_object(paths.tools.read_bytes(), paths.tools)
    overall_tools = _object(tools["overall"])
    assert overall_tools["calls"] == 1
    assert overall_tools["recorded_duration_ms"] == pytest.approx(100.0)
    assert overall_tools["replayed_duration_ms"] == pytest.approx(100.0)
    assert overall_tools["returncode_matches"] == 1
    assert _objects(tools, "by_delay_source")[0]["key"] == "recorded"

    failures = parse_json_object(paths.failures.read_bytes(), paths.failures)
    failure = _objects(failures, "failures")[0]
    assert failure["turn_id"] == "task_demo:0001"
    assert failure["request_error"] == "RuntimeError: synthetic failure"

    summary = parse_json_object(paths.summary.read_bytes(), paths.summary)
    summary_totals = _object(summary["totals"])
    summary_config = _object(summary["config"])
    assert summary["version"] == REPORT_VERSION
    assert summary["success"] is False
    assert summary["wall_duration_ms"] == pytest.approx(2000.0)
    assert summary["measured_duration_ms"] == pytest.approx(2000.0)
    assert summary["observer"] == {"enabled": False, "duration_ms": 0.0, "ranked_eligible": False}
    tool_replay = _object(summary_config["tool_replay"])
    assert (tool_replay["profile"], tool_replay["profile_statistic"]) == (None, None)
    assert summary_config["context"] == {
        "requested_tokens": BENCHMARK_CONTEXT_TOKENS,
        "observed_tokens": BENCHMARK_CONTEXT_TOKENS,
        "observed_reason": "reported",
        "full_benchmark_tokens": BENCHMARK_CONTEXT_TOKENS,
        "reduced": False,
    }
    assert summary["length_finished_turn_ids"] == []
    assert summary_totals == task_totals
    assert _object(summary_config["cache_isolation"])["namespace"] == "1 2 3"
    assert summary["short_output_warning_turn_ids"] == ["task_demo:0000"]
    assert summary["failed_turn_ids"] == ["task_demo:0001"]
    # Three decode intervals cover four output tokens across a 0.4 second window.
    assert summary["output_tokens_per_second"] == pytest.approx(7.5)
    assert summary["end_to_end_output_tokens_per_second"] == pytest.approx(4.0)

    blocked = tmp_path / "blocked"
    blocked.mkdir()
    existing = blocked / "tools.json"
    existing.write_text("prior attempt")
    with pytest.raises(FileExistsError, match="tools.json"):
        write_run_artifacts(result, blocked, config, run_context=FULL_RUN_CONTEXT, run_id=RUN_ID)
    assert tuple(path.name for path in blocked.iterdir()) == ("tools.json",)
    assert existing.read_text() == "prior attempt"

    target = tmp_path / "artifact-target"
    target.mkdir()
    link = tmp_path / "artifact-link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="not a symbolic link"):
        write_run_artifacts(result, link, config, run_context=FULL_RUN_CONTEXT, run_id=RUN_ID)
    assert not tuple(target.iterdir())


@pytest.mark.parametrize(
    ("generation_time", "expected_throughput"),
    [(0.5, 18.0), (None, None), (0.0, None)],
    ids=("decode-window", "no-decode-window", "single-read-window"),
)
def test_summary_reports_throughput_and_counts_length_capped_turns(
    tmp_path: Path,
    generation_time: float | None,
    expected_throughput: float | None,
) -> None:
    config = RunConfig(
        base_url="http://127.0.0.1:8000/v1",
        model="local-model",
        client_backend="python",
        cache_isolation=False,
    )
    result = RunResult(
        manifest_path=Path("fixtures/manifest.json"),
        model=config.model,
        client_backend=config.client_backend,
        cache_namespace=config.cache_namespace,
        started_at=100.0,
        ended_at=102.0,
        measured_duration_seconds=2.0,
        observer_duration_seconds=0.0,
        observer_enabled=False,
        tasks=(
            TaskResult(
                task_id="task_demo",
                trace_path=Path("fixtures/traces/task_demo.jsonl"),
                turns=(
                    _turn(index=0, finish_reason="stop", output_tokens=12, generation_time=generation_time),
                    _turn(index=1, finish_reason="length", output_tokens=8, generation_time=generation_time),
                ),
            ),
        ),
    )

    paths = write_run_artifacts(result, tmp_path, config, run_context=FULL_RUN_CONTEXT, run_id=RUN_ID)
    summary = parse_json_object(paths.summary.read_bytes(), paths.summary)
    distributions = _object(summary["latency_distributions_ms"])

    assert summary["success"] is True
    assert _object(summary["totals"])["successful_turns"] == 2
    assert _object(distributions["e2e"])["count"] == 2
    assert _object(distributions["time_to_first_token"])["count"] == 2
    # A truncation-polluted run stays visibly polluted even though every turn succeeded.
    assert summary["length_finished_turn_ids"] == ["task_demo:0001"]
    if expected_throughput is None:
        assert summary["output_tokens_per_second"] is None
    else:
        assert summary["output_tokens_per_second"] == pytest.approx(expected_throughput)
    assert summary["end_to_end_output_tokens_per_second"] == pytest.approx(10.0)


def test_summary_excludes_buffered_tool_calls_from_decode_throughput(tmp_path: Path) -> None:
    """A parsed tool call does not contribute an unobservable decode window."""
    config = RunConfig(
        base_url="http://127.0.0.1:8000/v1",
        model="local-model",
        client_backend="python",
        cache_isolation=False,
    )
    result = RunResult(
        manifest_path=Path("fixtures/manifest.json"),
        model=config.model,
        client_backend=config.client_backend,
        cache_namespace=config.cache_namespace,
        started_at=100.0,
        ended_at=102.0,
        measured_duration_seconds=2.0,
        observer_duration_seconds=0.0,
        observer_enabled=False,
        tasks=(
            TaskResult(
                task_id="task_demo",
                trace_path=Path("fixtures/traces/task_demo.jsonl"),
                turns=(
                    _turn(
                        index=0,
                        finish_reason="tool_calls",
                        output_tokens=1000,
                        generation_time=0.01,
                        tool_call=True,
                    ),
                    _turn(index=1, finish_reason="length", output_tokens=10, generation_time=1.0),
                ),
            ),
        ),
    )

    paths = write_run_artifacts(result, tmp_path, config, run_context=FULL_RUN_CONTEXT, run_id=RUN_ID)
    summary = parse_json_object(paths.summary.read_bytes(), paths.summary)

    assert summary["output_tokens_per_second"] == pytest.approx(9.0)
    assert summary["end_to_end_output_tokens_per_second"] == pytest.approx(505.0)


def test_reduced_and_unobserved_contexts_mark_the_summary_non_comparable() -> None:
    reported = ContextObservationReason.REPORTED
    assert RunContextFacts(requested_tokens=32_768, observed_tokens=32_768, observed_reason=reported).reduced is True
    assert (
        RunContextFacts(
            requested_tokens=BENCHMARK_CONTEXT_TOKENS,
            observed_tokens=None,
            observed_reason=ContextObservationReason.CONTEXT_NOT_REPORTED,
        ).reduced
        is True
    )
    assert (
        RunContextFacts(
            requested_tokens=BENCHMARK_CONTEXT_TOKENS,
            observed_tokens=BENCHMARK_CONTEXT_TOKENS,
            observed_reason=reported,
        ).reduced
        is False
    )
    reduced = RunContextFacts(requested_tokens=32_768, observed_tokens=32_768, observed_reason=reported).to_json()
    assert reduced == {
        "requested_tokens": 32_768,
        "observed_tokens": 32_768,
        "observed_reason": "reported",
        "full_benchmark_tokens": BENCHMARK_CONTEXT_TOKENS,
        "reduced": True,
    }


@pytest.mark.parametrize(
    ("observed_tokens", "observed_reason"),
    [
        (None, ContextObservationReason.REPORTED),
        (BENCHMARK_CONTEXT_TOKENS, ContextObservationReason.NOT_PROBED),
    ],
    ids=("reported-without-observation", "observation-without-reported"),
)
def test_context_facts_reject_an_observation_reason_mismatch(
    observed_tokens: int | None,
    observed_reason: ContextObservationReason,
) -> None:
    with pytest.raises(ValueError, match="observed_reason"):
        RunContextFacts(
            requested_tokens=BENCHMARK_CONTEXT_TOKENS,
            observed_tokens=observed_tokens,
            observed_reason=observed_reason,
        )


@pytest.mark.parametrize(
    ("generation_ms", "observed_output_tokens", "expected_code"),
    [
        (400.0, 4, "observed_output_below_target"),
        (0.0, 4, "decode_window_unmeasurable"),
        (400.0, 1, "decode_window_unmeasurable"),
        (None, 0, "decode_window_unmeasurable"),
    ],
    ids=("measurable-but-short", "single-read-window", "single-output-token", "no-visible-output"),
)
def test_unmeasurable_decode_windows_warn_instead_of_normalizing(
    generation_ms: float | None,
    observed_output_tokens: int,
    expected_code: str,
) -> None:
    report = normalize_output_length(
        e2e_latency_ms=1000.0,
        generation_ms=generation_ms,
        observed_output_tokens=observed_output_tokens,
        target_output_tokens=10,
    )

    assert report.warning is not None
    assert report.warning.code == expected_code
    if expected_code == "decode_window_unmeasurable":
        assert report.normalized_generation_ms is None
        assert report.normalized_e2e_latency_ms is None
        assert report.generation_ms_per_token is None
    else:
        assert report.normalized_generation_ms is not None
        assert report.normalized_e2e_latency_ms is not None


def _turn(
    *,
    index: int,
    finish_reason: str,
    output_tokens: int,
    generation_time: float | None,
    tool_call: bool = False,
) -> TurnResult:
    tool_calls = (ToolCall(index=0, identifier="call-1", name="lookup", arguments='{"q":"x"}'),) if tool_call else ()
    return TurnResult(
        turn_id=f"task_demo:{index:04d}",
        task_id="task_demo",
        conversation_idx=index,
        request=CompletionRequest(
            messages=({"role": "user", "content": "sanitized"},),
            model="local-model",
            max_tokens=output_tokens,
        ),
        started_at=10.0,
        ended_at=11.0,
        metrics=RequestMetrics(
            started_at=10.0,
            ended_at=11.0,
            time_to_first_byte=0.1,
            time_to_first_token=0.2,
            generation_time=generation_time,
            prompt_tokens=8,
            output_tokens=output_tokens,
            server_output_tokens=output_tokens,
            channels=ResponseChannels(content="tiny", reasoning="", tool_calls=tool_calls, finish_reason=finish_reason),
            chunks=(),
            aborted=False,
        ),
        error=None,
        recorded_prompt_tokens=10,
        recorded_completion_tokens=output_tokens,
        target_output_tokens=output_tokens,
        recorded_tool_delay_ms=0.0,
        tool_replays=(),
    )


def _object(value: JsonValue) -> JsonObject:
    assert isinstance(value, dict)
    return value


def _objects(data: JsonObject, key: str) -> list[JsonObject]:
    value = data[key]
    assert isinstance(value, list)
    assert all(isinstance(item, dict) for item in value)
    return [item for item in value if isinstance(item, dict)]


def _paths(paths: ArtifactPaths) -> tuple[Path, ...]:
    return (paths.turns, paths.tasks, paths.tools, paths.failures, paths.summary)


@pytest.mark.parametrize(
    ("output_tokens", "generation_time_ms", "expected"),
    [
        (101, 200.0, 500.0),
        (2, 1_000.0, 1.0),
        (1, 200.0, None),
        (0, 200.0, None),
        (101, 0.0, None),
        (None, 200.0, None),
        (101, None, None),
    ],
)
def test_turn_decode_speed_counts_one_interval_fewer_than_its_tokens(
    output_tokens: int | None, generation_time_ms: float | None, expected: float | None
) -> None:
    assert turn_decode_tokens_per_second(output_tokens, generation_time_ms) == expected
