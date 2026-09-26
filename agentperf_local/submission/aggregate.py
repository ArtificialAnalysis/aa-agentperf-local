"""Build the allowlisted public aggregate a submission uploads."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import orjson

from agentperf_local.client.endpoint import MEASURED_TRANSPORT_POLICY_ID
from agentperf_local.common.identity import sha256_bytes, sha256_file, validate_digest, validate_run_id
from agentperf_local.common.json_fields import (
    decode_json_object,
    optional_integer,
    optional_number,
    optional_string,
    required_boolean,
    required_integer,
    required_number,
    required_object,
    required_string,
)
from agentperf_local.common.json_records import json_record
from agentperf_local.common.json_types import JsonObject
from agentperf_local.provenance.benchmark import (
    BENCHMARK_CONTEXT_TOKENS,
    MEASUREMENT_BINDING_FILENAME,
    MeasurementBinding,
    SourceProvenance,
    SubmissionContext,
    load_measurement_binding,
    workload_digest,
)
from agentperf_local.provenance.context import context_is_reduced
from agentperf_local.provenance.hardware import PublicHardwareProfile, public_hardware_profile
from agentperf_local.reports.reporting import (
    FAILURES_FILENAME,
    SUMMARY_FILENAME,
    TASKS_FILENAME,
    TOOLS_FILENAME,
    TURNS_FILENAME,
    normalize_output_length,
    summarize_distribution,
)
from agentperf_local.workload.schema import load_manifest, load_trace

# Version 2 added the run identifier minted before inference; the v1 schema stays for old aggregates.
PUBLIC_SUBMISSION_VERSION = 2
SUPPORTED_SUMMARY_VERSION = 1
PUBLIC_PRIVACY_PROFILE = "public-minimal-v1"
SELF_REPORTED_TRUST_TIER = "community-self-reported"
# Observers run between turns, after a response closes, and their time is excluded
# from the measured duration. A run stays packageable while that excluded time is a
# small fraction of the measured window; the public aggregate states the fraction's
# numerator so the receiver can apply a stricter rule of its own.
OBSERVER_OVERHEAD_TOLERANCE_FRACTION = 0.01
REASONING_EFFORT_PATTERN = re.compile(r"^[A-Za-z0-9-]+$")
DURATION_CONSISTENCY_TOLERANCE_MS = 0.000001
# Independent writers add the same durations in different orders, so a few rounding steps must still agree.
DURATION_ASSOCIATION_RELATIVE_TOLERANCE = 1e-9
PRIVATE_FIELD_CATEGORIES = (
    "prompts and model responses",
    "tool arguments, outputs, commands, and exception text",
    "endpoint URLs and credentials",
    "local paths, hostnames, serial numbers, and stable device identifiers",
    "raw cache namespaces, nonces, and challenge secrets",
    "raw environment variables, process lists, and command output",
)


def _durations_agree(left: float, right: float) -> bool:
    """Compare two duration sums that independent writers may associate differently."""
    return math.isclose(
        left,
        right,
        rel_tol=DURATION_ASSOCIATION_RELATIVE_TOLERANCE,
        abs_tol=DURATION_CONSISTENCY_TOLERANCE_MS,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class TotalsTurn:
    """Hold the per-turn values that public totals add up."""

    success: bool
    has_short_output_warning: bool
    tool_calls: int
    e2e_latency_ms: float
    recorded_tool_delay_ms: float
    replayed_tool_delay_ms: float
    normalized_generation_ms: float
    normalized_e2e_latency_ms: float
    server_prompt_tokens: int
    server_output_tokens: int
    local_output_tokens: int


@dataclass(frozen=True, slots=True, kw_only=True)
class PublicTotals:
    """Store public aggregate counters and durations."""

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

    def __post_init__(self) -> None:
        """Reject impossible aggregate values."""
        if any(value < 0 for value in self._counter_values()):
            raise ValueError("summary totals counters must be non-negative")
        if any(not math.isfinite(value) or value < 0 for value in self._duration_values()):
            raise ValueError("summary total durations must be finite and non-negative")
        if self.successful_turns + self.failed_turns != self.turns:
            raise ValueError("successful_turns and failed_turns must add up to turns")
        if self.short_output_warnings > self.turns:
            raise ValueError("short_output_warnings must not exceed turns")

    def _counter_values(self) -> tuple[int, ...]:
        """Return every integer counter in a fixed order."""
        return (
            self.turns,
            self.successful_turns,
            self.failed_turns,
            self.short_output_warnings,
            self.tool_calls,
            self.total_server_prompt_tokens,
            self.total_server_output_tokens,
            self.total_local_output_tokens,
        )

    def _duration_values(self) -> tuple[float, ...]:
        """Return every duration in a fixed order."""
        return (
            self.total_inference_latency_ms,
            self.total_recorded_tool_delay_ms,
            self.total_replayed_tool_delay_ms,
            self.total_agentic_replay_ms,
            self.total_normalized_generation_ms,
            self.total_normalized_inference_latency_ms,
            self.total_normalized_agentic_replay_ms,
        )

    def matches(self, other: PublicTotals) -> bool:
        """Compare counters exactly and durations within the association tolerance."""
        if self._counter_values() != other._counter_values():
            return False
        return all(
            _durations_agree(left, right)
            for left, right in zip(self._duration_values(), other._duration_values(), strict=True)
        )

    @classmethod
    def from_turns(cls, turns: tuple[TotalsTurn, ...]) -> PublicTotals:
        """Add up per-turn values in turn order.

        The aggregate and the sanitized evidence both call this. Their totals must agree
        to the exact float, so there is one summation and one addition order.
        """
        successful = tuple(turn for turn in turns if turn.success)
        inference_latency = sum(turn.e2e_latency_ms for turn in successful)
        replayed_tool_delay = sum(turn.replayed_tool_delay_ms for turn in turns)
        return cls(
            turns=len(turns),
            successful_turns=len(successful),
            failed_turns=len(turns) - len(successful),
            short_output_warnings=sum(turn.has_short_output_warning for turn in turns),
            tool_calls=sum(turn.tool_calls for turn in turns),
            total_inference_latency_ms=inference_latency,
            total_recorded_tool_delay_ms=sum(turn.recorded_tool_delay_ms for turn in turns),
            total_replayed_tool_delay_ms=replayed_tool_delay,
            total_agentic_replay_ms=inference_latency + replayed_tool_delay,
            total_normalized_generation_ms=sum(turn.normalized_generation_ms for turn in successful),
            total_normalized_inference_latency_ms=sum(turn.normalized_e2e_latency_ms for turn in successful),
            total_normalized_agentic_replay_ms=sum(
                turn.normalized_e2e_latency_ms + turn.replayed_tool_delay_ms for turn in successful
            ),
            total_server_prompt_tokens=sum(turn.server_prompt_tokens for turn in successful),
            total_server_output_tokens=sum(turn.server_output_tokens for turn in successful),
            total_local_output_tokens=sum(turn.local_output_tokens for turn in successful),
        )

    @classmethod
    def from_json(cls, data: JsonObject) -> PublicTotals:
        """Build public totals from a private run summary."""
        source = "summary.totals"
        return cls(
            turns=required_integer(data, "turns", source),
            successful_turns=required_integer(data, "successful_turns", source),
            failed_turns=required_integer(data, "failed_turns", source),
            short_output_warnings=required_integer(data, "short_output_warnings", source),
            tool_calls=required_integer(data, "tool_calls", source),
            total_inference_latency_ms=required_number(data, "total_inference_latency_ms", source),
            total_recorded_tool_delay_ms=required_number(data, "total_recorded_tool_delay_ms", source),
            total_replayed_tool_delay_ms=required_number(data, "total_replayed_tool_delay_ms", source),
            total_agentic_replay_ms=required_number(data, "total_agentic_replay_ms", source),
            total_normalized_generation_ms=required_number(data, "total_normalized_generation_ms", source),
            total_normalized_inference_latency_ms=required_number(
                data, "total_normalized_inference_latency_ms", source
            ),
            total_normalized_agentic_replay_ms=required_number(data, "total_normalized_agentic_replay_ms", source),
            total_server_prompt_tokens=required_integer(data, "total_server_prompt_tokens", source),
            total_server_output_tokens=required_integer(data, "total_server_output_tokens", source),
            total_local_output_tokens=required_integer(data, "total_local_output_tokens", source),
        )

    def to_json(self) -> JsonObject:
        """Return aggregate results as JSON data."""
        return json_record(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class PublicDistribution:
    """Store one public latency distribution."""

    count: int
    mean: float | None
    p50: float | None
    p95: float | None

    def __post_init__(self) -> None:
        """Reject impossible distribution values."""
        if self.count < 0:
            raise ValueError("distribution count must be non-negative")
        values = (self.mean, self.p50, self.p95)
        if any(value is not None and (not math.isfinite(value) or value < 0) for value in values):
            raise ValueError("distribution values must be finite and non-negative")
        if self.count == 0 and any(value is not None for value in values):
            raise ValueError("an empty distribution must not have statistics")
        if self.count > 0 and any(value is None for value in values):
            raise ValueError("a non-empty distribution must have all statistics")
        if self.p50 is not None and self.p95 is not None and self.p50 > self.p95:
            raise ValueError("distribution p50 must not exceed p95")

    @classmethod
    def from_values(cls, values: tuple[float, ...]) -> PublicDistribution:
        """Summarize latency values exactly as the run summary does."""
        report = summarize_distribution(values)
        return cls(count=report.count, mean=report.mean, p50=report.p50, p95=report.p95)

    @classmethod
    def from_json(cls, data: JsonObject, source: str) -> PublicDistribution:
        """Build one distribution from a private summary."""
        return cls(
            count=required_integer(data, "count", source),
            mean=optional_number(data, "mean", source),
            p50=optional_number(data, "p50", source),
            p95=optional_number(data, "p95", source),
        )

    def to_json(self) -> JsonObject:
        """Return the distribution as JSON data."""
        return json_record(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class PublicLatencyDistributions:
    """Store public latency distributions."""

    e2e: PublicDistribution
    time_to_first_token: PublicDistribution
    normalized_e2e: PublicDistribution

    @classmethod
    def from_json(cls, data: JsonObject) -> PublicLatencyDistributions:
        """Build latency distributions from a private summary."""
        source = "summary.latency_distributions_ms"
        return cls(
            e2e=PublicDistribution.from_json(required_object(data, "e2e", source), f"{source}.e2e"),
            time_to_first_token=PublicDistribution.from_json(
                required_object(data, "time_to_first_token", source), f"{source}.time_to_first_token"
            ),
            normalized_e2e=PublicDistribution.from_json(
                required_object(data, "normalized_e2e", source), f"{source}.normalized_e2e"
            ),
        )

    def to_json(self) -> JsonObject:
        """Return all distributions as JSON data."""
        return json_record(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class PublicRunPolicy:
    """Store non-sensitive settings that affect comparability."""

    client_backend: str
    transport_policy_id: str
    context_requested_tokens: int
    context_observed_tokens: int | None
    context_full_benchmark_tokens: int
    context_reduced: bool
    output_token_policy: str
    max_output_tokens: int
    output_token_margin: int
    sampling_preset: str
    temperature: float | None
    top_p: float | None
    top_k: int | None
    min_p: float | None
    reasoning_effort: str | None
    cache_isolation_enabled: bool
    cache_isolation_mode: str
    cache_namespace_digits: int
    tool_replay_mode: str
    tool_delay_scale: float
    tool_profile_statistic: str | None

    def __post_init__(self) -> None:
        """Reject unsupported or impossible public policy values."""
        if self.client_backend not in {"python", "rust"}:
            raise ValueError("client_backend must be python or rust")
        if self.transport_policy_id != MEASURED_TRANSPORT_POLICY_ID:
            raise ValueError("transport_policy_id is not supported")
        if self.context_full_benchmark_tokens != BENCHMARK_CONTEXT_TOKENS:
            raise ValueError(f"context.full_benchmark_tokens must be {BENCHMARK_CONTEXT_TOKENS}")
        if not 0 < self.context_requested_tokens <= self.context_full_benchmark_tokens:
            raise ValueError("context.requested_tokens must be positive and within the full benchmark context")
        if self.context_observed_tokens is not None and self.context_observed_tokens <= 0:
            raise ValueError("context.observed_tokens must be positive or null")
        if self.context_reduced != context_is_reduced(self.context_requested_tokens, self.context_observed_tokens):
            raise ValueError("context.reduced must match the requested and observed context")
        if self.output_token_policy not in {"exact", "fixed", "recorded"}:
            raise ValueError("output token policy must be exact, fixed, or recorded")
        if self.max_output_tokens <= 0 or self.output_token_margin < 0:
            raise ValueError("output token settings are invalid")
        if self.sampling_preset not in {"custom", "standard"}:
            raise ValueError("sampling preset must be custom or standard")
        if self.temperature is not None and self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError("top_p must be greater than zero and at most one")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.min_p is not None and not 0 <= self.min_p <= 1:
            raise ValueError("min_p must be between zero and one")
        if self.reasoning_effort is not None and (
            len(self.reasoning_effort) > 32 or REASONING_EFFORT_PATTERN.fullmatch(self.reasoning_effort) is None
        ):
            raise ValueError("reasoning_effort must be a short portable value")
        if self.cache_isolation_mode not in {"none", "run_namespace_prefix"}:
            raise ValueError("cache isolation mode is not supported")
        if self.cache_namespace_digits < 0:
            raise ValueError("cache namespace digits must be non-negative")
        if self.cache_isolation_enabled != (self.cache_isolation_mode != "none"):
            raise ValueError("cache isolation fields are inconsistent")
        if self.tool_replay_mode not in {"none", "recorded"}:
            raise ValueError("public submission v1 supports only none or recorded tool replay")
        if not math.isfinite(self.tool_delay_scale) or self.tool_delay_scale < 0:
            raise ValueError("tool delay scale must be finite and non-negative")
        if self.tool_profile_statistic is not None:
            raise ValueError("public submission v1 does not support tool timing profiles")

    @classmethod
    def from_json(cls, data: JsonObject) -> PublicRunPolicy:
        """Select only public run policy fields."""
        source = "summary.config"
        output_tokens = required_object(data, "output_tokens", source)
        sampling = required_object(data, "sampling", source)
        extra_body = required_object(sampling, "extra_body", f"{source}.sampling")
        cache = required_object(data, "cache_isolation", source)
        tools = required_object(data, "tool_replay", source)
        context = required_object(data, "context", source)
        return cls(
            client_backend=required_string(data, "client_backend", source),
            transport_policy_id=required_string(data, "transport_policy_id", source),
            context_requested_tokens=required_integer(context, "requested_tokens", f"{source}.context"),
            context_observed_tokens=optional_integer(context, "observed_tokens", f"{source}.context"),
            context_full_benchmark_tokens=required_integer(context, "full_benchmark_tokens", f"{source}.context"),
            context_reduced=required_boolean(context, "reduced", f"{source}.context"),
            output_token_policy=required_string(output_tokens, "policy", f"{source}.output_tokens"),
            max_output_tokens=required_integer(output_tokens, "fallback", f"{source}.output_tokens"),
            output_token_margin=required_integer(output_tokens, "margin", f"{source}.output_tokens"),
            sampling_preset=required_string(sampling, "preset", f"{source}.sampling"),
            temperature=optional_number(sampling, "temperature", f"{source}.sampling"),
            top_p=optional_number(sampling, "top_p", f"{source}.sampling"),
            top_k=optional_integer(extra_body, "top_k", f"{source}.sampling.extra_body"),
            min_p=optional_number(extra_body, "min_p", f"{source}.sampling.extra_body"),
            reasoning_effort=optional_string(data, "reasoning_effort", source),
            cache_isolation_enabled=required_boolean(cache, "enabled", f"{source}.cache_isolation"),
            cache_isolation_mode=required_string(cache, "mode", f"{source}.cache_isolation"),
            cache_namespace_digits=required_integer(cache, "namespace_digits", f"{source}.cache_isolation"),
            tool_replay_mode=required_string(tools, "mode", f"{source}.tool_replay"),
            tool_delay_scale=required_number(tools, "delay_scale", f"{source}.tool_replay"),
            tool_profile_statistic=optional_string(tools, "profile_statistic", f"{source}.tool_replay"),
        )

    def to_json(self) -> JsonObject:
        """Return the public run policy as JSON data."""
        return {
            "client_backend": self.client_backend,
            "transport_policy_id": self.transport_policy_id,
            "context": {
                "requested_tokens": self.context_requested_tokens,
                "observed_tokens": self.context_observed_tokens,
                "full_benchmark_tokens": self.context_full_benchmark_tokens,
                "reduced": self.context_reduced,
            },
            "output_tokens": {
                "policy": self.output_token_policy,
                "fallback": self.max_output_tokens,
                "margin": self.output_token_margin,
            },
            "sampling": {
                "preset": self.sampling_preset,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "min_p": self.min_p,
            },
            "reasoning_effort": self.reasoning_effort,
            "cache_isolation": {
                "enabled": self.cache_isolation_enabled,
                "mode": self.cache_isolation_mode,
                "namespace_digits": self.cache_namespace_digits,
            },
            "tool_replay": {
                "mode": self.tool_replay_mode,
                "delay_scale": self.tool_delay_scale,
                "profile_statistic": self.tool_profile_statistic,
            },
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class PublicRunResult:
    """Store aggregate run data safe for the public profile."""

    success: bool
    wall_duration_ms: float
    observer_duration_ms: float
    totals: PublicTotals
    latency_distributions_ms: PublicLatencyDistributions
    policy: PublicRunPolicy

    def __post_init__(self) -> None:
        """Reject failed or internally inconsistent public results.

        The observer rule lives here so the packaging gate and the bundle validator
        apply the same one: observers run between turns and their time is excluded
        from the measured window, and a run stays packageable while that excluded
        time is a small fraction of what was measured.
        """
        if not math.isfinite(self.wall_duration_ms) or self.wall_duration_ms < 0:
            raise ValueError("wall_duration_ms must be finite and non-negative")
        if not math.isfinite(self.observer_duration_ms) or self.observer_duration_ms < 0:
            raise ValueError("observer_duration_ms must be finite and non-negative")
        if self.observer_duration_ms > self.wall_duration_ms:
            raise ValueError("observer_duration_ms must not exceed wall_duration_ms")
        measured_duration_ms = self.wall_duration_ms - self.observer_duration_ms
        if self.observer_duration_ms > measured_duration_ms * OBSERVER_OVERHEAD_TOLERANCE_FRACTION:
            raise ValueError(
                f"observer overhead exceeds {OBSERVER_OVERHEAD_TOLERANCE_FRACTION:.0%} of the measured duration; "
                "the run is not eligible for public packaging"
            )
        if self.success != (self.totals.failed_turns == 0):
            raise ValueError("run success and failed turn count are inconsistent")
        if not self.success:
            raise ValueError("failed runs cannot become public submissions")
        distribution_counts = (
            self.latency_distributions_ms.e2e.count,
            self.latency_distributions_ms.time_to_first_token.count,
            self.latency_distributions_ms.normalized_e2e.count,
        )
        if any(count != self.totals.successful_turns for count in distribution_counts):
            raise ValueError("every latency distribution count must equal successful turns")

    def to_json(self) -> JsonObject:
        """Return public run data as JSON data."""
        return json_record(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class ValidatedEvidenceTurn:
    """Store only turn fields needed to recompute public aggregates."""

    turn_id: str
    task_id: str
    success: bool
    aborted: bool
    error: str | None
    finish_reason: str | None
    response_chunks: int
    response_tool_calls: int
    has_tool_failure: bool
    e2e_latency_ms: float
    time_to_first_byte_ms: float
    time_to_first_token_ms: float
    generation_ms: float
    normalized_generation_ms: float
    normalized_e2e_latency_ms: float
    has_short_output_warning: bool
    recorded_tool_delay_ms: float
    replayed_tool_delay_ms: float
    tool_calls: int
    server_prompt_tokens: int
    server_output_tokens: int
    local_output_tokens: int
    server_cached_prompt_tokens: int | None
    server_uncached_prompt_tokens: int | None
    observed_output_tokens: int
    target_output_tokens: int

    def totals_turn(self) -> TotalsTurn:
        """Return the values this turn adds to public totals."""
        return TotalsTurn(
            success=self.success,
            has_short_output_warning=self.has_short_output_warning,
            tool_calls=self.tool_calls,
            e2e_latency_ms=self.e2e_latency_ms,
            recorded_tool_delay_ms=self.recorded_tool_delay_ms,
            replayed_tool_delay_ms=self.replayed_tool_delay_ms,
            normalized_generation_ms=self.normalized_generation_ms,
            normalized_e2e_latency_ms=self.normalized_e2e_latency_ms,
            server_prompt_tokens=self.server_prompt_tokens,
            server_output_tokens=self.server_output_tokens,
            local_output_tokens=self.local_output_tokens,
        )

    @classmethod
    def from_json(cls, data: JsonObject, source: str) -> ValidatedEvidenceTurn:
        """Select aggregate evidence from one private turn."""
        if required_integer(data, "version", source) != SUPPORTED_SUMMARY_VERSION:
            raise ValueError(f"{source}.version is not supported")
        if required_string(data, "kind", source) != "turn":
            raise ValueError(f"{source}.kind must be turn")
        timing = required_object(data, "timing", source)
        normalization = required_object(data, "normalization", source)
        tokens = required_object(data, "tokens", source)
        warning = normalization.get("warning")
        if warning is not None and not isinstance(warning, dict):
            raise ValueError(f"{source}.normalization.warning must be an object or null")
        tool_calls = data.get("tool_calls")
        if not isinstance(tool_calls, list) or not all(isinstance(tool, dict) for tool in tool_calls):
            raise ValueError(f"{source}.tool_calls must contain objects")
        tool_objects = tuple(tool for tool in tool_calls if isinstance(tool, dict))
        error = data.get("error")
        if error is not None and not isinstance(error, str):
            raise ValueError(f"{source}.error must be text or null")
        has_tool_failure = False
        for index, tool in enumerate(tool_objects):
            exception_info = tool.get("exception_info")
            if not isinstance(exception_info, str):
                raise ValueError(f"{source}.tool_calls[{index}].exception_info must be text")
            returncode_matches = tool.get("returncode_matches_recorded")
            if returncode_matches is not None and not isinstance(returncode_matches, bool):
                raise ValueError(f"{source}.tool_calls[{index}].returncode_matches_recorded must be boolean or null")
            has_tool_failure = has_tool_failure or bool(exception_info) or returncode_matches is False
        return cls(
            turn_id=required_string(data, "turn_id", source),
            task_id=required_string(data, "task_id", source),
            success=required_boolean(data, "success", source),
            aborted=required_boolean(data, "aborted", source),
            error=error,
            finish_reason=optional_string(data, "finish_reason", source),
            response_chunks=required_integer(data, "response_chunks", source),
            response_tool_calls=required_integer(data, "response_tool_calls", source),
            has_tool_failure=has_tool_failure,
            e2e_latency_ms=required_number(timing, "e2e_latency_ms", f"{source}.timing"),
            time_to_first_byte_ms=required_number(timing, "time_to_first_byte_ms", f"{source}.timing"),
            time_to_first_token_ms=required_number(timing, "time_to_first_token_ms", f"{source}.timing"),
            generation_ms=required_number(timing, "generation_ms", f"{source}.timing"),
            normalized_generation_ms=required_number(
                normalization, "normalized_generation_ms", f"{source}.normalization"
            ),
            normalized_e2e_latency_ms=required_number(
                normalization, "normalized_e2e_latency_ms", f"{source}.normalization"
            ),
            has_short_output_warning=warning is not None,
            recorded_tool_delay_ms=required_number(data, "recorded_tool_delay_ms", source),
            replayed_tool_delay_ms=required_number(data, "replayed_tool_delay_ms", source),
            tool_calls=len(tool_objects),
            server_prompt_tokens=required_integer(tokens, "server_prompt_tokens", f"{source}.tokens"),
            server_output_tokens=required_integer(tokens, "server_output_tokens", f"{source}.tokens"),
            local_output_tokens=required_integer(tokens, "local_output_tokens", f"{source}.tokens"),
            server_cached_prompt_tokens=optional_integer(tokens, "server_cached_prompt_tokens", f"{source}.tokens"),
            server_uncached_prompt_tokens=optional_integer(tokens, "server_uncached_prompt_tokens", f"{source}.tokens"),
            observed_output_tokens=required_integer(normalization, "observed_output_tokens", f"{source}.normalization"),
            target_output_tokens=required_integer(normalization, "target_output_tokens", f"{source}.normalization"),
        )

    def _rejected(self, reason: str) -> ValueError:
        """Return the standard rejection error for this turn."""
        return ValueError(f"turn {self.turn_id} cannot join a public submission: {reason}")

    def __post_init__(self) -> None:
        """Validate one turn as public benchmark evidence."""
        if not self.success or self.aborted or self.error is not None:
            raise self._rejected("public turn evidence must be successful, not aborted, and error-free")
        if not self.finish_reason:
            raise self._rejected("successful turn evidence must contain a finish reason")
        if self.response_chunks <= 0:
            raise self._rejected("successful turn evidence must contain response chunks")
        if self.response_tool_calls < 0:
            raise self._rejected("response_tool_calls must be non-negative")
        if self.has_tool_failure:
            raise self._rejected("public turn evidence must not contain tool failures")
        if self.has_short_output_warning:
            raise self._rejected("short or unmeasurable output (short_output_warning)")
        numbers = (
            self.e2e_latency_ms,
            self.time_to_first_byte_ms,
            self.time_to_first_token_ms,
            self.generation_ms,
            self.normalized_generation_ms,
            self.normalized_e2e_latency_ms,
            self.recorded_tool_delay_ms,
            self.replayed_tool_delay_ms,
        )
        if any(value < 0 for value in numbers):
            raise self._rejected("turn evidence durations must be non-negative")
        # The sanitized evidence stage repeats these checks, so the first gate must reject the same turns.
        if self.time_to_first_byte_ms > self.e2e_latency_ms:
            raise self._rejected("time to first byte must not exceed end-to-end latency")
        if self.time_to_first_token_ms > self.e2e_latency_ms:
            raise self._rejected("time to first token must not exceed end-to-end latency")
        if self.generation_ms > self.e2e_latency_ms:
            raise self._rejected("generation time must not exceed end-to-end latency")
        counts = (
            self.server_prompt_tokens,
            self.server_output_tokens,
            self.local_output_tokens,
            self.observed_output_tokens,
            self.target_output_tokens,
        )
        if any(value < 0 for value in counts):
            raise self._rejected("turn evidence token counts must be non-negative")
        cache_counts = (self.server_cached_prompt_tokens, self.server_uncached_prompt_tokens)
        if (cache_counts[0] is None) != (cache_counts[1] is None):
            raise self._rejected("cached and uncached prompt token counts must be reported together")
        if any(value is not None and value < 0 for value in cache_counts):
            raise self._rejected("cache token counts must be non-negative")
        if (
            self.server_cached_prompt_tokens is not None
            and self.server_uncached_prompt_tokens is not None
            and self.server_cached_prompt_tokens + self.server_uncached_prompt_tokens != self.server_prompt_tokens
        ):
            raise self._rejected("cached and uncached prompt tokens must add up to prompt tokens")
        if self.observed_output_tokens != self.server_output_tokens:
            raise self._rejected("normalization observed tokens must match server output tokens")
        if self.observed_output_tokens <= 0 or self.target_output_tokens <= 0:
            raise self._rejected("normalization observed and target token counts must be positive")
        expected = normalize_output_length(
            e2e_latency_ms=self.e2e_latency_ms,
            generation_ms=self.generation_ms,
            observed_output_tokens=self.observed_output_tokens,
            target_output_tokens=self.target_output_tokens,
        )
        if expected.normalized_generation_ms != self.normalized_generation_ms:
            raise self._rejected("normalized generation timing does not match raw turn evidence")
        if expected.normalized_e2e_latency_ms != self.normalized_e2e_latency_ms:
            raise self._rejected("normalized end-to-end timing does not match raw turn evidence")
        if (expected.warning is not None) != self.has_short_output_warning:
            raise self._rejected("short-output warning does not match raw turn evidence")


def load_validated_evidence_turns(path: Path) -> tuple[ValidatedEvidenceTurn, ...]:
    """Load allowlisted aggregate fields from private turn evidence."""
    turns: list[ValidatedEvidenceTurn] = []
    with path.open("rb") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            data = decode_json_object(raw_line, f"invalid turn JSON at {path}:{line_number}")
            turns.append(ValidatedEvidenceTurn.from_json(data, f"turns[{line_number - 1}]"))
    if not turns:
        raise ValueError("turns.jsonl must contain at least one turn")
    turn_ids = tuple(turn.turn_id for turn in turns)
    if len(turn_ids) != len(set(turn_ids)):
        raise ValueError("turns.jsonl contains duplicate turn IDs")
    return tuple(turns)


def _load_artifact(path: Path, expected_kind: str) -> JsonObject:
    data = decode_json_object(path.read_bytes(), f"invalid {expected_kind} JSON: {path}")
    if required_integer(data, "version", expected_kind) != SUPPORTED_SUMMARY_VERSION:
        raise ValueError(f"{expected_kind}.version is not supported")
    if required_string(data, "kind", expected_kind) != expected_kind:
        raise ValueError(f"{expected_kind}.kind must be {expected_kind}")
    return data


def _validate_failure_artifact(results_dir: Path) -> None:
    data = _load_artifact(results_dir / FAILURES_FILENAME, "failures")
    failures = data.get("failures")
    if not isinstance(failures, list):
        raise ValueError("failures.failures must be an array")
    if failures:
        raise ValueError("public submissions must not contain failures")


def _validate_task_artifact(
    results_dir: Path,
    manifest_task_ids: tuple[str, ...],
    turns: tuple[ValidatedEvidenceTurn, ...],
) -> None:
    data = _load_artifact(results_dir / TASKS_FILENAME, "task_summaries")
    task_values = data.get("tasks")
    if not isinstance(task_values, list) or not all(isinstance(task, dict) for task in task_values):
        raise ValueError("task_summaries.tasks must contain objects")
    task_objects = tuple(task for task in task_values if isinstance(task, dict))
    task_ids = tuple(required_string(task, "task_id", "task_summaries.tasks") for task in task_objects)
    if task_ids != manifest_task_ids:
        raise ValueError("task summary order does not match the bound workload")
    for task in task_objects:
        task_id = required_string(task, "task_id", "task_summaries.tasks")
        failed_turn_ids = task.get("failed_turn_ids")
        if not isinstance(failed_turn_ids, list) or failed_turn_ids:
            raise ValueError("public task summaries must contain no failed turn IDs")
        task_totals = PublicTotals.from_json(required_object(task, "totals", "task_summaries.tasks"))
        evidence_totals = PublicTotals.from_turns(
            tuple(turn.totals_turn() for turn in turns if turn.task_id == task_id)
        )
        if not task_totals.matches(evidence_totals):
            raise ValueError("task summary totals do not match turn evidence")


def _validate_tool_artifact(results_dir: Path, totals: PublicTotals) -> None:
    data = _load_artifact(results_dir / TOOLS_FILENAME, "tool_summary")
    overall = required_object(data, "overall", "tool_summary")
    if required_integer(overall, "calls", "tool_summary.overall") != totals.tool_calls:
        raise ValueError("tool summary call count does not match turn evidence")
    if required_integer(overall, "failures", "tool_summary.overall") != 0:
        raise ValueError("public submissions must not contain tool failures")
    if required_integer(overall, "returncode_mismatches", "tool_summary.overall") != 0:
        raise ValueError("public submissions must not contain tool return-code mismatches")
    # tools.json adds every call once while the summary adds per-turn subtotals, so the two orders differ.
    recorded_duration = required_number(overall, "recorded_duration_ms", "tool_summary.overall")
    replayed_duration = required_number(overall, "replayed_duration_ms", "tool_summary.overall")
    if not _durations_agree(recorded_duration, totals.total_recorded_tool_delay_ms):
        raise ValueError("tool summary recorded duration does not match turn evidence")
    if not _durations_agree(replayed_duration, totals.total_replayed_tool_delay_ms):
        raise ValueError("tool summary replayed duration does not match turn evidence")


def _validate_evidence(
    results_dir: Path,
    manifest_path: Path,
    totals: PublicTotals,
    distributions: PublicLatencyDistributions,
) -> None:
    turns = load_validated_evidence_turns(results_dir / TURNS_FILENAME)
    manifest = load_manifest(manifest_path)
    manifest_root = manifest_path.resolve().parent
    expected_turn_ids = tuple(row.turn_id for task in manifest.tasks for row in load_trace(manifest_root / task.trace))
    if tuple(turn.turn_id for turn in turns) != expected_turn_ids:
        raise ValueError("turns.jsonl is missing, reordered, or does not belong to the bound workload")
    # The runner and this recomputation add the same per-turn values in the same order, so they must agree exactly.
    if PublicTotals.from_turns(tuple(turn.totals_turn() for turn in turns)) != totals:
        raise ValueError("summary totals do not match turns.jsonl evidence")
    successful = tuple(turn for turn in turns if turn.success)
    recomputed = PublicLatencyDistributions(
        e2e=PublicDistribution.from_values(tuple(turn.e2e_latency_ms for turn in successful)),
        time_to_first_token=PublicDistribution.from_values(tuple(turn.time_to_first_token_ms for turn in successful)),
        normalized_e2e=PublicDistribution.from_values(tuple(turn.normalized_e2e_latency_ms for turn in successful)),
    )
    if recomputed != distributions:
        raise ValueError("summary latency distributions do not match turns.jsonl evidence")
    _validate_task_artifact(results_dir, tuple(task.task_id for task in manifest.tasks), turns)
    _validate_tool_artifact(results_dir, totals)
    _validate_failure_artifact(results_dir)


@dataclass(frozen=True, slots=True, kw_only=True)
class ArtifactDigests:
    """Bind the public result to exact private evidence files."""

    summary: str
    turns: str
    tasks: str
    tools: str
    failures: str
    measurement: str

    @classmethod
    def from_results_dir(cls, results_dir: Path) -> ArtifactDigests:
        """Hash every evidence artifact produced by the runner."""
        return cls(
            summary=sha256_file(results_dir / SUMMARY_FILENAME),
            turns=sha256_file(results_dir / TURNS_FILENAME),
            tasks=sha256_file(results_dir / TASKS_FILENAME),
            tools=sha256_file(results_dir / TOOLS_FILENAME),
            failures=sha256_file(results_dir / FAILURES_FILENAME),
            measurement=sha256_file(results_dir / MEASUREMENT_BINDING_FILENAME),
        )

    def to_json(self) -> JsonObject:
        """Return content identifiers for private evidence."""
        return json_record(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class PublicSubmission:
    """Store one deterministic public submission aggregate."""

    run_id: str
    context: SubmissionContext
    hardware: PublicHardwareProfile | None
    producer: SourceProvenance
    artifacts: ArtifactDigests
    run: PublicRunResult
    payload_digest: str
    version: int = PUBLIC_SUBMISSION_VERSION

    def __post_init__(self) -> None:
        """Validate the payload digest when the aggregate already carries one.

        The empty digest belongs to the draft built to compute it.
        Managed runs apply the single-accelerator rule upstream.
        """
        validate_run_id(self.run_id, "run_id")
        if self.payload_digest:
            validate_digest(self.payload_digest, "payload_digest")

    def _payload_json(self) -> JsonObject:
        return {
            "run_id": self.run_id,
            "benchmark": self.context.to_json(),
            "producer": self.producer.to_json(),
            "hardware": None if self.hardware is None else self.hardware.to_json(),
            "private_evidence_digests": self.artifacts.to_json(),
            "run": self.run.to_json(),
            "trust_tier": SELF_REPORTED_TRUST_TIER,
        }

    def to_json(self) -> JsonObject:
        """Return the content-addressed public envelope."""
        return {
            "version": self.version,
            "kind": "agentperf_local_public_submission",
            "privacy_profile": PUBLIC_PRIVACY_PROFILE,
            "payload_digest": self.payload_digest,
            "payload": self._payload_json(),
            "excluded_private_categories": list(PRIVATE_FIELD_CATEGORIES),
        }


def _load_summary(path: Path) -> JsonObject:
    summary = decode_json_object(path.read_bytes(), f"invalid summary JSON: {path}")
    if required_integer(summary, "version", "summary") != SUPPORTED_SUMMARY_VERSION:
        raise ValueError("summary.version is not supported")
    if required_string(summary, "kind", "summary") != "run_summary":
        raise ValueError("summary.kind must be run_summary")
    return summary


def _payload_digest(payload: JsonObject) -> str:
    canonical = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
    return sha256_bytes(canonical)


def _validate_binding(binding: MeasurementBinding, summary: JsonObject) -> None:
    if required_string(summary, "run_id", "summary") != binding.run_id:
        raise ValueError("summary run_id does not match the pre-run measurement binding")
    if workload_digest(binding.manifest_path) != binding.manifest_digest:
        raise ValueError("manifest or trace files changed after the measurement binding was captured")
    config = required_object(summary, "config", "summary")
    endpoint_model = required_string(config, "model", "summary.config")
    if sha256_bytes(endpoint_model.encode("utf-8")) != binding.endpoint_model_digest:
        raise ValueError("summary model does not match the pre-run measurement binding")
    artifacts = required_object(summary, "artifacts", "summary")
    expected_names = {
        "turns": TURNS_FILENAME,
        "tasks": TASKS_FILENAME,
        "tools": TOOLS_FILENAME,
        "failures": FAILURES_FILENAME,
    }
    if any(artifacts.get(key) != value for key, value in expected_names.items()):
        raise ValueError("summary artifact names do not match the run artifact contract")


def _validate_observer_policy(summary: JsonObject) -> float:
    """Check the observer bookkeeping and return the observer time the aggregate will state."""
    wall_duration_ms = required_number(summary, "wall_duration_ms", "summary")
    measured_duration_ms = required_number(summary, "measured_duration_ms", "summary")
    observer = required_object(summary, "observer", "summary")
    enabled = required_boolean(observer, "enabled", "summary.observer")
    duration_ms = required_number(observer, "duration_ms", "summary.observer")
    legacy_ranked_eligible = required_boolean(observer, "ranked_eligible", "summary.observer")
    if enabled and legacy_ranked_eligible:
        raise ValueError("an observer-enabled summary cannot claim ranked eligibility")
    if measured_duration_ms < 0 or duration_ms < 0:
        raise ValueError("summary observer and measured durations must be non-negative")
    if not enabled and duration_ms != 0:
        raise ValueError("a summary without an observer cannot report observer time")
    expected_wall_duration_ms = measured_duration_ms + duration_ms
    if not math.isclose(
        wall_duration_ms,
        expected_wall_duration_ms,
        rel_tol=0.0,
        abs_tol=DURATION_CONSISTENCY_TOLERANCE_MS,
    ):
        raise ValueError("summary wall, measured, and observer durations are inconsistent")
    return duration_ms


def build_public_submission(
    results_dir: Path,
) -> PublicSubmission:
    """Build the public aggregate from bound, allowlisted run fields."""
    binding = load_measurement_binding(results_dir / MEASUREMENT_BINDING_FILENAME)
    summary_path = results_dir / SUMMARY_FILENAME
    summary = _load_summary(summary_path)
    _validate_binding(binding, summary)
    observer_duration_ms = _validate_observer_policy(summary)
    run = PublicRunResult(
        success=required_boolean(summary, "success", "summary"),
        wall_duration_ms=required_number(summary, "wall_duration_ms", "summary"),
        observer_duration_ms=observer_duration_ms,
        totals=PublicTotals.from_json(required_object(summary, "totals", "summary")),
        latency_distributions_ms=PublicLatencyDistributions.from_json(
            required_object(summary, "latency_distributions_ms", "summary")
        ),
        policy=PublicRunPolicy.from_json(required_object(summary, "config", "summary")),
    )
    if run.policy.context_requested_tokens != binding.benchmark.context_tokens:
        raise ValueError("summary context does not match the bound benchmark identity")
    if run.policy.context_observed_tokens != binding.observed_context_tokens:
        raise ValueError("summary observed context does not match the pre-run measurement binding")
    _validate_evidence(results_dir, binding.manifest_path, run.totals, run.latency_distributions_ms)
    artifacts = ArtifactDigests.from_results_dir(results_dir)
    hardware = None if binding.deployment_digest is None else public_hardware_profile(binding.hardware)
    draft = PublicSubmission(
        run_id=binding.run_id,
        context=binding.benchmark,
        hardware=hardware,
        producer=binding.producer,
        artifacts=artifacts,
        run=run,
        payload_digest="",
    )
    return PublicSubmission(
        run_id=binding.run_id,
        context=binding.benchmark,
        hardware=hardware,
        producer=binding.producer,
        artifacts=artifacts,
        run=run,
        payload_digest=_payload_digest(draft._payload_json()),
    )
