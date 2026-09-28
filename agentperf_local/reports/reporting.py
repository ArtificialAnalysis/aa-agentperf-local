"""Build and write versioned replay reports."""

from __future__ import annotations

import statistics
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Literal

import orjson
from pydantic import BaseModel

from agentperf_local.client.endpoint import MEASURED_TRANSPORT_POLICY_ID
from agentperf_local.common.durable_files import NewFile, commit_new_file_set, validate_new_file_paths
from agentperf_local.common.identity import validate_run_id
from agentperf_local.common.json_records import json_record
from agentperf_local.common.json_types import JsonObject, JsonValue, pretty_json_bytes
from agentperf_local.common.statistics import P50_PERCENTILE, P95_PERCENTILE, percentile
from agentperf_local.common.units import MILLISECONDS_PER_SECOND
from agentperf_local.provenance.context import RunContextFacts
from agentperf_local.replay.cache_isolation import CacheIsolationMetadata, cache_isolation_metadata
from agentperf_local.replay.config import RunConfig, SamplingSettings
from agentperf_local.replay.fidelity import LENGTH_FINISH_REASON
from agentperf_local.replay.runner import ReplayDelaySource, RunResult, ToolReplayResult, TurnResult

REPORT_VERSION = 1
SHORT_OUTPUT_RATIO_WARNING_THRESHOLD = 0.5
OUTPUT_LENGTH_NORMALIZATION_POLICY = "observed-decode-time-v1"
# A first-to-last-token window spans one decode interval fewer than the tokens
# it covers, so a turn needs two output tokens before any per-token time exists.
MIN_MEASURABLE_OUTPUT_TOKENS = 2

type ShortOutputWarningCode = Literal["observed_output_below_target", "decode_window_unmeasurable"]

TURNS_FILENAME = "turns.jsonl"
TASKS_FILENAME = "tasks.json"
TOOLS_FILENAME = "tools.json"
FAILURES_FILENAME = "failures.json"
SUMMARY_FILENAME = "summary.json"


class ShortOutputWarning(BaseModel, frozen=True):
    """Describe output that cannot support a trustworthy normalization."""

    observed_output_tokens: int
    target_output_tokens: int
    observed_to_target_ratio: float
    threshold: float = SHORT_OUTPUT_RATIO_WARNING_THRESHOLD
    code: ShortOutputWarningCode = "observed_output_below_target"

    @property
    def message(self) -> str:
        """Return why this turn is unfit for normalized reporting."""
        if self.code == "decode_window_unmeasurable":
            return "the turn has no measurable first-to-last output token window"
        return f"observed output is less than {self.threshold:.0%} of the normalization target"

    def to_json(self) -> JsonObject:
        """Return the warning as JSON data."""
        return {
            "code": self.code,
            "message": self.message,
            "observed_output_tokens": self.observed_output_tokens,
            "target_output_tokens": self.target_output_tokens,
            "observed_to_target_ratio": self.observed_to_target_ratio,
            "threshold": self.threshold,
        }


class NormalizationReport(BaseModel, frozen=True):
    """Store output-length-normalized timing values."""

    observed_output_tokens: int | None
    target_output_tokens: int | None
    normalized_decode_tokens: int | None
    generation_ms_per_token: float | None
    normalized_generation_ms: float | None
    normalized_e2e_latency_ms: float | None
    warning: ShortOutputWarning | None

    def to_json(self) -> JsonObject:
        """Return normalization data."""
        return {
            "observed_output_tokens": self.observed_output_tokens,
            "target_output_tokens": self.target_output_tokens,
            "normalized_decode_tokens": self.normalized_decode_tokens,
            "generation_ms_per_token": self.generation_ms_per_token,
            "normalized_generation_ms": self.normalized_generation_ms,
            "normalized_e2e_latency_ms": self.normalized_e2e_latency_ms,
            "warning": self.warning.to_json() if self.warning is not None else None,
        }


class TimingReport(BaseModel, frozen=True):
    """Store request timing values in milliseconds."""

    e2e_latency_ms: float | None
    time_to_first_byte_ms: float | None
    time_to_first_token_ms: float | None
    generation_ms: float | None

    def to_json(self) -> JsonObject:
        """Return timing data."""
        return json_record(self)


class TokenUsageReport(BaseModel, frozen=True):
    """Store recorded, server, and locally counted token usage."""

    server_prompt_tokens: int | None
    server_output_tokens: int | None
    local_output_tokens: int | None
    recorded_prompt_tokens: int | None
    recorded_completion_tokens: int | None
    target_output_tokens: int | None
    server_cached_prompt_tokens: int | None = None
    server_uncached_prompt_tokens: int | None = None

    def to_json(self) -> JsonObject:
        """Return token usage data."""
        return json_record(self)


class ToolCallReport(BaseModel, frozen=True):
    """Store one recorded and replayed tool call."""

    tool_name: str | None
    tool_call_id: str | None
    command: str | None
    step: int | None
    action_index: int | None
    recorded_duration_ms: float
    replayed_duration_ms: float
    delay_source: ReplayDelaySource
    recorded_returncode: int | None
    returncode: int | None
    returncode_matches_recorded: bool | None
    exception_info: str

    def to_json(self) -> JsonObject:
        """Return tool replay data.

        profile_key stays as null so the report v1 shape outlives the removed profiled mode.
        """
        return {**json_record(self), "profile_key": None}


class TurnReport(BaseModel, frozen=True):
    """Store report fields for one replay turn."""

    turn_id: str
    task_id: str
    conversation_idx: int
    success: bool
    error: str | None
    requested_max_output_tokens: int
    timing: TimingReport
    tokens: TokenUsageReport
    normalization: NormalizationReport
    recorded_tool_delay_ms: float
    replayed_tool_delay_ms: float
    tool_calls: tuple[ToolCallReport, ...]
    response_chunks: int
    response_tool_calls: int
    finish_reason: str | None
    aborted: bool

    def to_json(self) -> JsonObject:
        """Return one versioned turn record."""
        return {
            "version": REPORT_VERSION,
            "kind": "turn",
            "turn_id": self.turn_id,
            "task_id": self.task_id,
            "conversation_idx": self.conversation_idx,
            "success": self.success,
            "error": self.error,
            "requested_max_output_tokens": self.requested_max_output_tokens,
            "timing": self.timing.to_json(),
            "tokens": self.tokens.to_json(),
            "normalization": self.normalization.to_json(),
            "recorded_tool_delay_ms": self.recorded_tool_delay_ms,
            "replayed_tool_delay_ms": self.replayed_tool_delay_ms,
            "tool_calls": [tool.to_json() for tool in self.tool_calls],
            "response_chunks": self.response_chunks,
            "response_tool_calls": self.response_tool_calls,
            "finish_reason": self.finish_reason,
            "aborted": self.aborted,
        }


class TotalsReport(BaseModel, frozen=True):
    """Store totals shared by task and run summaries."""

    turns: int
    successful_turns: int
    failed_turns: int
    short_output_warnings: int
    tool_calls: int
    total_inference_latency_ms: float
    total_recorded_tool_delay_ms: float
    total_replayed_tool_delay_ms: float
    total_agentic_replay_ms: float
    total_normalized_generation_ms: float
    total_normalized_inference_latency_ms: float
    total_normalized_agentic_replay_ms: float
    total_server_prompt_tokens: int
    total_server_output_tokens: int
    total_local_output_tokens: int
    total_recorded_prompt_tokens: int
    total_recorded_completion_tokens: int
    total_server_cached_prompt_tokens: int | None = None
    total_server_uncached_prompt_tokens: int | None = None

    def to_json(self) -> JsonObject:
        """Return aggregate totals."""
        return json_record(self)


class TaskReport(BaseModel, frozen=True):
    """Store aggregate data for one replay task."""

    task_id: str
    totals: TotalsReport
    failed_turn_ids: tuple[str, ...]

    def to_json(self) -> JsonObject:
        """Return one task summary."""
        return json_record(self)


class ToolGroupReport(BaseModel, frozen=True):
    """Store aggregate replay data for one tool grouping."""

    key: str
    calls: int
    recorded_duration_ms: float
    replayed_duration_ms: float
    returncode_comparisons: int
    returncode_matches: int
    returncode_mismatches: int
    failures: int

    def to_json(self) -> JsonObject:
        """Return one tool aggregate."""
        return json_record(self)


class ToolSummary(BaseModel, frozen=True):
    """Store overall and grouped tool replay totals."""

    overall: ToolGroupReport
    by_delay_source: tuple[ToolGroupReport, ...]
    by_tool: tuple[ToolGroupReport, ...]
    by_command: tuple[ToolGroupReport, ...]

    def to_json(self) -> JsonObject:
        """Return the versioned tool summary."""
        return {
            "version": REPORT_VERSION,
            "kind": "tool_summary",
            "overall": self.overall.to_json(),
            "by_delay_source": [group.to_json() for group in self.by_delay_source],
            "by_tool": [group.to_json() for group in self.by_tool],
            "by_command": [group.to_json() for group in self.by_command],
        }


class DistributionReport(BaseModel, frozen=True):
    """Store a small latency distribution."""

    count: int
    mean: float | None
    p50: float | None
    p95: float | None

    def to_json(self) -> JsonObject:
        """Return distribution data."""
        return json_record(self)


class FailureReport(BaseModel, frozen=True):
    """Store one failed turn and its error details."""

    task_id: str
    turn_id: str
    request_error: str | None
    tool_errors: tuple[str, ...]

    def to_json(self) -> JsonObject:
        """Return failure data."""
        return json_record(self)


class ArtifactPaths(BaseModel, frozen=True):
    """Point to every artifact written for one run."""

    turns: Path
    tasks: Path
    tools: Path
    failures: Path
    summary: Path


class _ToolAccumulator(BaseModel):
    calls: int = 0
    recorded_duration_ms: float = 0.0
    replayed_duration_ms: float = 0.0
    returncode_comparisons: int = 0
    returncode_matches: int = 0
    returncode_mismatches: int = 0
    failures: int = 0

    def add(self, tool: ToolCallReport) -> None:
        self.calls += 1
        self.recorded_duration_ms += tool.recorded_duration_ms
        self.replayed_duration_ms += tool.replayed_duration_ms
        if tool.returncode_matches_recorded is not None:
            self.returncode_comparisons += 1
            if tool.returncode_matches_recorded:
                self.returncode_matches += 1
            else:
                self.returncode_mismatches += 1
        if tool.exception_info:
            self.failures += 1

    def report(self, key: str) -> ToolGroupReport:
        return ToolGroupReport(
            key=key,
            calls=self.calls,
            recorded_duration_ms=self.recorded_duration_ms,
            replayed_duration_ms=self.replayed_duration_ms,
            returncode_comparisons=self.returncode_comparisons,
            returncode_matches=self.returncode_matches,
            returncode_mismatches=self.returncode_mismatches,
            failures=self.failures,
        )


def _warning(observed_output_tokens: int | None, target_output_tokens: int | None) -> ShortOutputWarning | None:
    if observed_output_tokens is None or target_output_tokens is None or target_output_tokens <= 0:
        return None
    ratio = observed_output_tokens / target_output_tokens
    if ratio >= SHORT_OUTPUT_RATIO_WARNING_THRESHOLD:
        return None
    return ShortOutputWarning(
        observed_output_tokens=observed_output_tokens,
        target_output_tokens=target_output_tokens,
        observed_to_target_ratio=ratio,
    )


def _unmeasurable_warning(observed_output_tokens: int, target_output_tokens: int | None) -> ShortOutputWarning:
    resolved_target = observed_output_tokens if target_output_tokens is None else target_output_tokens
    return ShortOutputWarning(
        observed_output_tokens=observed_output_tokens,
        target_output_tokens=resolved_target,
        observed_to_target_ratio=observed_output_tokens / resolved_target if resolved_target > 0 else 0.0,
        code="decode_window_unmeasurable",
    )


def _unnormalized(
    observed_output_tokens: int | None,
    target_output_tokens: int | None,
    warning: ShortOutputWarning | None,
) -> NormalizationReport:
    return NormalizationReport(
        observed_output_tokens=observed_output_tokens,
        target_output_tokens=target_output_tokens,
        normalized_decode_tokens=None,
        generation_ms_per_token=None,
        normalized_generation_ms=None,
        normalized_e2e_latency_ms=None,
        warning=warning,
    )


def normalize_output_length(
    *,
    e2e_latency_ms: float | None,
    generation_ms: float | None,
    observed_output_tokens: int | None,
    target_output_tokens: int | None,
) -> NormalizationReport:
    """Normalize one turn to its recorded output-token target."""
    if observed_output_tokens is None or e2e_latency_ms is None:
        return _unnormalized(
            observed_output_tokens,
            target_output_tokens,
            _warning(observed_output_tokens, target_output_tokens),
        )
    if generation_ms is None or generation_ms <= 0.0 or observed_output_tokens < MIN_MEASURABLE_OUTPUT_TOKENS:
        # A turn read in one network read, or one that emitted no visible output at
        # all, has no decode window. Normalizing it would divide a real end-to-end
        # latency by a zero-length window, so the turn carries a warning instead.
        return _unnormalized(
            observed_output_tokens,
            target_output_tokens,
            _unmeasurable_warning(observed_output_tokens, target_output_tokens),
        )

    warning = _warning(observed_output_tokens, target_output_tokens)
    resolved_target = observed_output_tokens if target_output_tokens is None else target_output_tokens
    observed_decode_tokens = observed_output_tokens - 1
    target_decode_tokens = max(resolved_target - 1, 0)
    generation_ms_per_token = generation_ms / observed_decode_tokens
    normalized_generation_ms = generation_ms_per_token * target_decode_tokens
    normalized_e2e_latency_ms = max(0.0, e2e_latency_ms - generation_ms + normalized_generation_ms)
    return NormalizationReport(
        observed_output_tokens=observed_output_tokens,
        target_output_tokens=resolved_target,
        normalized_decode_tokens=target_decode_tokens,
        generation_ms_per_token=generation_ms_per_token,
        normalized_generation_ms=normalized_generation_ms,
        normalized_e2e_latency_ms=normalized_e2e_latency_ms,
        warning=warning,
    )


def _tool_report(replay: ToolReplayResult) -> ToolCallReport:
    call = replay.call
    return ToolCallReport(
        tool_name=call.tool_name,
        tool_call_id=call.tool_call_id,
        command=replay.command,
        step=call.step,
        action_index=call.action_index,
        recorded_duration_ms=call.duration_ms,
        replayed_duration_ms=replay.replayed_duration_ms,
        delay_source=replay.delay_source,
        recorded_returncode=call.recorded_returncode,
        returncode=replay.returncode,
        returncode_matches_recorded=replay.returncode_matches_recorded,
        exception_info=replay.exception_info,
    )


def _turn_report(turn: TurnResult) -> TurnReport:
    metrics = turn.metrics
    e2e_latency_ms = metrics.duration * MILLISECONDS_PER_SECOND if metrics is not None else None
    time_to_first_byte_ms = (
        metrics.time_to_first_byte * MILLISECONDS_PER_SECOND
        if metrics is not None and metrics.time_to_first_byte is not None
        else None
    )
    time_to_first_token_ms = (
        metrics.time_to_first_token * MILLISECONDS_PER_SECOND
        if metrics is not None and metrics.time_to_first_token is not None
        else None
    )
    generation_ms = (
        metrics.generation_time * MILLISECONDS_PER_SECOND
        if metrics is not None and metrics.generation_time is not None
        else None
    )
    observed_output_tokens = None
    if metrics is not None:
        observed_output_tokens = (
            metrics.server_output_tokens if metrics.server_output_tokens is not None else metrics.output_tokens
        )
    target_output_tokens = (
        turn.target_output_tokens if turn.target_output_tokens is not None else turn.recorded_completion_tokens
    )
    tools = tuple(_tool_report(replay) for replay in turn.tool_replays)
    return TurnReport(
        turn_id=turn.turn_id,
        task_id=turn.task_id,
        conversation_idx=turn.conversation_idx,
        success=turn.success,
        error=turn.error,
        requested_max_output_tokens=turn.request.max_tokens,
        timing=TimingReport(
            e2e_latency_ms=e2e_latency_ms,
            time_to_first_byte_ms=time_to_first_byte_ms,
            time_to_first_token_ms=time_to_first_token_ms,
            generation_ms=generation_ms,
        ),
        tokens=TokenUsageReport(
            server_prompt_tokens=metrics.prompt_tokens if metrics is not None else None,
            server_output_tokens=metrics.server_output_tokens if metrics is not None else None,
            local_output_tokens=metrics.output_tokens if metrics is not None else None,
            recorded_prompt_tokens=turn.recorded_prompt_tokens,
            recorded_completion_tokens=turn.recorded_completion_tokens,
            target_output_tokens=target_output_tokens,
            server_cached_prompt_tokens=metrics.cached_prompt_tokens if metrics is not None else None,
            server_uncached_prompt_tokens=metrics.uncached_prompt_tokens if metrics is not None else None,
        ),
        normalization=normalize_output_length(
            e2e_latency_ms=e2e_latency_ms,
            generation_ms=generation_ms,
            observed_output_tokens=observed_output_tokens,
            target_output_tokens=target_output_tokens,
        ),
        recorded_tool_delay_ms=turn.recorded_tool_delay_ms,
        replayed_tool_delay_ms=turn.replayed_tool_delay_ms,
        tool_calls=tools,
        response_chunks=len(metrics.chunks) if metrics is not None else 0,
        response_tool_calls=len(metrics.channels.tool_calls) if metrics is not None else 0,
        finish_reason=metrics.channels.finish_reason if metrics is not None else None,
        aborted=metrics.aborted if metrics is not None else False,
    )


def _optional_int_total(values: tuple[int | None, ...]) -> int:
    return sum(value for value in values if value is not None)


def _complete_optional_int_total(values: tuple[int | None, ...]) -> int | None:
    """Sum counts only when every turn reports one."""
    if not values or any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def _totals(turns: tuple[TurnReport, ...]) -> TotalsReport:
    successful = tuple(turn for turn in turns if turn.success)
    inference_latency = sum(turn.timing.e2e_latency_ms or 0.0 for turn in successful)
    recorded_tool_delay = sum(turn.recorded_tool_delay_ms for turn in turns)
    replayed_tool_delay = sum(turn.replayed_tool_delay_ms for turn in turns)
    # Agentic replay time pairs inference with the tool pacing that followed it,
    # so both terms must come from the same population of successful turns.
    successful_replayed_tool_delay = sum(turn.replayed_tool_delay_ms for turn in successful)
    normalized_generation = sum(turn.normalization.normalized_generation_ms or 0.0 for turn in successful)
    normalized_inference = sum(turn.normalization.normalized_e2e_latency_ms or 0.0 for turn in successful)
    normalized_agentic = sum(
        (turn.normalization.normalized_e2e_latency_ms or 0.0) + turn.replayed_tool_delay_ms
        for turn in successful
        if turn.normalization.normalized_e2e_latency_ms is not None
    )
    return TotalsReport(
        turns=len(turns),
        successful_turns=sum(turn.success for turn in turns),
        failed_turns=sum(not turn.success for turn in turns),
        short_output_warnings=sum(turn.normalization.warning is not None for turn in turns),
        tool_calls=sum(len(turn.tool_calls) for turn in turns),
        total_inference_latency_ms=inference_latency,
        total_recorded_tool_delay_ms=recorded_tool_delay,
        total_replayed_tool_delay_ms=replayed_tool_delay,
        total_agentic_replay_ms=inference_latency + successful_replayed_tool_delay,
        total_normalized_generation_ms=normalized_generation,
        total_normalized_inference_latency_ms=normalized_inference,
        total_normalized_agentic_replay_ms=normalized_agentic,
        total_server_prompt_tokens=_optional_int_total(tuple(turn.tokens.server_prompt_tokens for turn in successful)),
        total_server_output_tokens=_optional_int_total(tuple(turn.tokens.server_output_tokens for turn in successful)),
        total_local_output_tokens=_optional_int_total(tuple(turn.tokens.local_output_tokens for turn in successful)),
        total_recorded_prompt_tokens=_optional_int_total(tuple(turn.tokens.recorded_prompt_tokens for turn in turns)),
        total_recorded_completion_tokens=_optional_int_total(
            tuple(turn.tokens.recorded_completion_tokens for turn in turns)
        ),
        total_server_cached_prompt_tokens=_complete_optional_int_total(
            tuple(turn.tokens.server_cached_prompt_tokens for turn in successful)
        ),
        total_server_uncached_prompt_tokens=_complete_optional_int_total(
            tuple(turn.tokens.server_uncached_prompt_tokens for turn in successful)
        ),
    )


def _task_reports(result: RunResult, turns: tuple[TurnReport, ...]) -> tuple[TaskReport, ...]:
    reports: list[TaskReport] = []
    offset = 0
    for task in result.tasks:
        task_turns = turns[offset : offset + len(task.turns)]
        offset += len(task.turns)
        reports.append(
            TaskReport(
                task_id=task.task_id,
                totals=_totals(task_turns),
                failed_turn_ids=tuple(turn.turn_id for turn in task_turns if not turn.success),
            )
        )
    return tuple(reports)


def _tool_groups(
    tools: tuple[ToolCallReport, ...],
    key: Callable[[ToolCallReport], str],
) -> tuple[ToolGroupReport, ...]:
    accumulators: dict[str, _ToolAccumulator] = {}
    for tool in tools:
        group_key = key(tool)
        accumulator = accumulators.setdefault(group_key, _ToolAccumulator())
        accumulator.add(tool)
    return tuple(accumulators[group_key].report(group_key) for group_key in sorted(accumulators))


def _tool_summary(turns: tuple[TurnReport, ...]) -> ToolSummary:
    tools = tuple(tool for turn in turns for tool in turn.tool_calls)
    overall = _ToolAccumulator()
    for tool in tools:
        overall.add(tool)
    return ToolSummary(
        overall=overall.report("all"),
        by_delay_source=_tool_groups(tools, lambda tool: tool.delay_source),
        by_tool=_tool_groups(tools, lambda tool: tool.tool_name or "unknown"),
        by_command=_tool_groups(tools, lambda tool: tool.command or "unknown"),
    )


def summarize_distribution(values: tuple[float, ...]) -> DistributionReport:
    """Summarize latency values the one way the summary, the aggregate, and the evidence all use."""
    return DistributionReport(
        count=len(values),
        mean=statistics.fmean(values) if values else None,
        p50=percentile(values, P50_PERCENTILE),
        p95=percentile(values, P95_PERCENTILE),
    )


class RunTimingSummary(BaseModel, frozen=True):
    """Store headline turn timings shared by the summary writer and the TUI."""

    ttft_p50_ms: float | None
    e2e_p50_ms: float | None
    failed_turns: int
    total_turns: int


def run_timing_summary(result: RunResult) -> RunTimingSummary:
    """Summarize successful-turn latency medians and the failed-turn count."""
    ttft_values: list[float] = []
    e2e_values: list[float] = []
    failed_turns = 0
    for turn in result.turns:
        metrics = turn.metrics
        if not turn.success or metrics is None:
            failed_turns += 1
            continue
        if metrics.time_to_first_token is not None:
            ttft_values.append(metrics.time_to_first_token * MILLISECONDS_PER_SECOND)
        e2e_values.append(metrics.duration * MILLISECONDS_PER_SECOND)
    return RunTimingSummary(
        ttft_p50_ms=percentile(tuple(ttft_values), P50_PERCENTILE),
        e2e_p50_ms=percentile(tuple(e2e_values), P50_PERCENTILE),
        failed_turns=failed_turns,
        total_turns=len(result.turns),
    )


def pooled_decode_tokens_per_second(windows: Iterable[tuple[int, float]]) -> float | None:
    """Pool measurable (output tokens, generation seconds) windows into one decode speed.

    The measured window runs from the first token to the last, so each window covers
    one decode interval fewer than the tokens its turn produced. Returns None without
    any window.
    """
    decode_tokens = 0
    generation_seconds = 0.0
    for output_tokens, seconds in windows:
        decode_tokens += output_tokens - 1
        generation_seconds += seconds
    if generation_seconds == 0.0:
        return None
    return decode_tokens / generation_seconds


def run_output_tokens_per_second(result: RunResult) -> float | None:
    """Return decode throughput across directly streamed text turns.

    Servers may buffer generated tool calls while parsing them. Their full completion
    token count then covers work performed before the first visible tool-call chunk,
    so the client decode window cannot measure that turn. Exact-output requests disable
    tool calls and remain measurable.
    """
    windows: list[tuple[int, float]] = []
    for turn in result.turns:
        metrics = turn.metrics
        if metrics is None or not turn.success:
            continue
        if metrics.channels.tool_calls:
            continue
        generation_time = metrics.generation_time
        if generation_time is None or generation_time <= 0.0:
            continue
        if metrics.reported_output_tokens < MIN_MEASURABLE_OUTPUT_TOKENS:
            continue
        windows.append((metrics.reported_output_tokens, generation_time))
    return pooled_decode_tokens_per_second(windows)


def run_end_to_end_output_tokens_per_second(result: RunResult) -> float | None:
    """Return output throughput over complete successful request durations."""
    output_tokens = 0
    duration_seconds = 0.0
    for turn in result.turns:
        metrics = turn.metrics
        if metrics is None or not turn.success:
            continue
        output_tokens += metrics.reported_output_tokens
        duration_seconds += metrics.duration
    if duration_seconds <= 0.0:
        return None
    return output_tokens / duration_seconds


def turn_decode_tokens_per_second(output_tokens: int | None, generation_time_ms: float | None) -> float | None:
    """Return one turn's decode speed over its first-to-last-token window, or None when unmeasurable.

    This applies the run-level rule to a single turn: the window covers one decode
    interval fewer than the tokens produced, and a turn below the measurable minimum
    has no window at all.
    """
    if output_tokens is None or generation_time_ms is None or generation_time_ms <= 0.0:
        return None
    if output_tokens < MIN_MEASURABLE_OUTPUT_TOKENS:
        return None
    return (output_tokens - 1) / (generation_time_ms / MILLISECONDS_PER_SECOND)


def _failure_reports(turns: tuple[TurnReport, ...]) -> tuple[FailureReport, ...]:
    return tuple(
        FailureReport(
            task_id=turn.task_id,
            turn_id=turn.turn_id,
            request_error=turn.error,
            tool_errors=tuple(tool.exception_info for tool in turn.tool_calls if tool.exception_info),
        )
        for turn in turns
        if not turn.success
    )


def _sampling_json(sampling: SamplingSettings) -> JsonObject:
    extra_body: JsonObject = dict(sampling.extra_body)
    return {
        "preset": sampling.preset,
        "temperature": sampling.temperature,
        "top_p": sampling.top_p,
        "extra_body": extra_body,
    }


def _config_json(config: RunConfig, cache: CacheIsolationMetadata, run_context: RunContextFacts) -> JsonObject:
    return {
        "base_url": config.base_url,
        "model": config.model,
        "client_backend": config.client_backend,
        "transport_policy_id": MEASURED_TRANSPORT_POLICY_ID,
        "context": run_context.to_json(),
        "request_timeout_seconds": config.request_timeout_seconds,
        "output_tokens": {
            "policy": config.output_token_policy,
            "fallback": config.max_output_tokens,
            "margin": config.output_token_margin,
        },
        "sampling": _sampling_json(config.sampling()),
        "reasoning_effort": config.reasoning_effort,
        "cache_isolation": cache.to_dict(),
        "tool_replay": {
            "mode": config.tool_mode,
            "delay_scale": config.tool_delay_scale,
            # The profiled mode is gone; these stay as null so the report v1 shape does not change.
            "profile": None,
            "profile_statistic": None,
            "live_image_override": config.live_tool_image,
            "live_network_override": config.live_network,
        },
    }


def _json_array(values: tuple[JsonObject, ...]) -> list[JsonValue]:
    result: list[JsonValue] = []
    result.extend(values)
    return result


def _json_bytes(data: JsonObject) -> bytes:
    return pretty_json_bytes(data)


def _jsonl_bytes(rows: tuple[JsonObject, ...]) -> bytes:
    options = orjson.OPT_APPEND_NEWLINE | orjson.OPT_SORT_KEYS
    return b"".join(orjson.dumps(row, option=options) for row in rows)


def _artifact_paths(output_dir: Path) -> ArtifactPaths:
    return ArtifactPaths(
        turns=output_dir / TURNS_FILENAME,
        tasks=output_dir / TASKS_FILENAME,
        tools=output_dir / TOOLS_FILENAME,
        failures=output_dir / FAILURES_FILENAME,
        summary=output_dir / SUMMARY_FILENAME,
    )


def validate_run_artifact_output(output_dir: Path) -> None:
    """Reject an output directory that could replace run artifacts."""
    paths = _artifact_paths(output_dir)
    validate_new_file_paths((paths.turns, paths.tasks, paths.tools, paths.failures, paths.summary))


def write_run_artifacts(
    result: RunResult,
    output_dir: Path,
    config: RunConfig,
    *,
    run_context: RunContextFacts,
    run_id: str,
) -> ArtifactPaths:
    """Write deterministic turn, task, tool, failure, and run reports.

    Every summary states its context facts; a reduced or unproven context marks the
    run non-comparable here, and the public packaging gates read exactly this block.
    The run identifier was minted before inference; the packaging gates require the
    summary to carry the same one as the measurement binding.
    """
    validate_run_id(run_id, "run_id")
    if result.model != config.model or result.client_backend != config.client_backend:
        raise ValueError("run result does not match the reporting configuration")
    paths = _artifact_paths(output_dir)
    validate_run_artifact_output(output_dir)
    turns = tuple(_turn_report(turn) for turn in result.turns)
    tasks = _task_reports(result, turns)
    tools = _tool_summary(turns)
    failures = _failure_reports(turns)
    totals = _totals(turns)
    cache = cache_isolation_metadata(result.cache_namespace)
    successful_turns = tuple(turn for turn in turns if turn.success)
    e2e_values = tuple(
        turn.timing.e2e_latency_ms for turn in successful_turns if turn.timing.e2e_latency_ms is not None
    )
    ttft_values = tuple(
        turn.timing.time_to_first_token_ms
        for turn in successful_turns
        if turn.timing.time_to_first_token_ms is not None
    )
    normalized_values = tuple(
        turn.normalization.normalized_e2e_latency_ms
        for turn in successful_turns
        if turn.normalization.normalized_e2e_latency_ms is not None
    )

    tasks_data: JsonObject = {
        "version": REPORT_VERSION,
        "kind": "task_summaries",
        "tasks": _json_array(tuple(task.to_json() for task in tasks)),
    }
    failures_data: JsonObject = {
        "version": REPORT_VERSION,
        "kind": "failures",
        "failures": _json_array(tuple(failure.to_json() for failure in failures)),
    }
    summary_data: JsonObject = {
        "version": REPORT_VERSION,
        "kind": "run_summary",
        "run_id": run_id,
        "manifest": str(result.manifest_path),
        "started_at": result.started_at,
        "ended_at": result.ended_at,
        "wall_duration_ms": (result.measured_duration_seconds + result.observer_duration_seconds)
        * MILLISECONDS_PER_SECOND,
        "measured_duration_ms": result.measured_duration_seconds * MILLISECONDS_PER_SECOND,
        "observer": {
            "enabled": result.observer_enabled,
            "duration_ms": result.observer_duration_seconds * MILLISECONDS_PER_SECOND,
            # New v1 reports never claim overall ranking eligibility. The reader accepts historical true values.
            "ranked_eligible": False,
        },
        "success": result.success,
        "output_tokens_per_second": run_output_tokens_per_second(result),
        "end_to_end_output_tokens_per_second": run_end_to_end_output_tokens_per_second(result),
        "config": _config_json(config, cache, run_context),
        "totals": totals.to_json(),
        "latency_distributions_ms": {
            "e2e": summarize_distribution(e2e_values).to_json(),
            "time_to_first_token": summarize_distribution(ttft_values).to_json(),
            "normalized_e2e": summarize_distribution(normalized_values).to_json(),
        },
        "short_output_warning_turn_ids": [turn.turn_id for turn in turns if turn.normalization.warning is not None],
        # Length-capped turns already fail the public methodology gate; naming them here
        # keeps a truncation-polluted run visibly polluted without changing run success.
        "length_finished_turn_ids": [turn.turn_id for turn in turns if turn.finish_reason == LENGTH_FINISH_REASON],
        "failed_turn_ids": [turn.turn_id for turn in turns if not turn.success],
        "artifacts": {
            "turns": paths.turns.name,
            "tasks": paths.tasks.name,
            "tools": paths.tools.name,
            "failures": paths.failures.name,
        },
    }
    commit_new_file_set(
        (
            NewFile(path=paths.turns, data=_jsonl_bytes(tuple(turn.to_json() for turn in turns))),
            NewFile(path=paths.tasks, data=_json_bytes(tasks_data)),
            NewFile(path=paths.tools, data=_json_bytes(tools.to_json())),
            NewFile(path=paths.failures, data=_json_bytes(failures_data)),
        ),
        NewFile(path=paths.summary, data=_json_bytes(summary_data)),
    )
    return paths
