"""Build allowlisted per-turn evidence for aggregate derivation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import orjson

from agentperf_local.common.identity import sha256_bytes, validate_digest
from agentperf_local.common.json_fields import (
    optional_integer,
    require_allowed_keys,
    require_exact_keys,
    required_integer,
    required_number,
    required_object,
    required_string,
)
from agentperf_local.common.json_records import json_field_names, json_record
from agentperf_local.common.json_types import JsonObject, JsonValue
from agentperf_local.provenance.benchmark import MEASUREMENT_BINDING_FILENAME, load_measurement_binding
from agentperf_local.reports.reporting import (
    OUTPUT_LENGTH_NORMALIZATION_POLICY,
    TURNS_FILENAME,
    normalize_output_length,
)
from agentperf_local.submission.aggregate import (
    PublicDistribution,
    PublicLatencyDistributions,
    PublicSubmission,
    PublicTotals,
    TotalsTurn,
    ValidatedEvidenceTurn,
    build_public_submission,
    load_validated_evidence_turns,
)
from agentperf_local.workload.schema import load_manifest

SANITIZED_EVIDENCE_VERSION = 1
SANITIZED_EVIDENCE_PROFILE = "sanitized-turn-evidence-v1-experimental"
SANITIZED_EVIDENCE_KIND = "agentperf_local_sanitized_turn_evidence"
SANITIZED_EVIDENCE_STATUS = "experimental-local-preview-no-upload"
MAX_SANITIZED_TURNS = 100_000
MAX_TURN_DURATION_MS = 24 * 60 * 60 * 1000.0
MAX_TOKENS_PER_TURN = 10_000_000
MAX_ACTIONS_PER_TURN = 10_000
MAX_RESPONSE_CHUNKS_PER_TURN = 10_000_000
SANITIZED_EXCLUDED_CATEGORIES = (
    "source task, turn, and conversation identifiers",
    "prompts, model responses, and reasoning",
    "tool names, arguments, outputs, commands, and exception text",
    "endpoint URLs, credentials, paths, and environment values",
    "hostnames, serial numbers, UUIDs, and raw hardware output",
    "raw cache namespaces, challenge secrets, and process data",
)
_TOKEN_KEYS = frozenset(
    (
        "server_prompt_tokens",
        "server_output_tokens",
        "local_output_tokens",
        "server_cached_prompt_tokens",
        "server_uncached_prompt_tokens",
        "observed_output_tokens",
        "target_output_tokens",
    )
)
_REQUIRED_TOKEN_KEYS = _TOKEN_KEYS - frozenset(("server_cached_prompt_tokens", "server_uncached_prompt_tokens"))
_EVIDENCE_KEYS = frozenset(
    (
        "version",
        "kind",
        "privacy_profile",
        "normalization_policy_id",
        "aggregate_payload_digest",
        "source_turns_digest",
        "rows_digest",
        "turns",
        "excluded_private_categories",
        "status",
    )
)


@dataclass(frozen=True, slots=True, kw_only=True)
class SanitizedTiming:
    """Store one turn's allowlisted timing evidence."""

    e2e_latency_ms: float
    time_to_first_byte_ms: float
    time_to_first_token_ms: float
    generation_ms: float

    def __post_init__(self) -> None:
        """Reject impossible or abusive timing values."""
        values = (
            self.e2e_latency_ms,
            self.time_to_first_byte_ms,
            self.time_to_first_token_ms,
            self.generation_ms,
        )
        if any(not math.isfinite(value) or value < 0 or value > MAX_TURN_DURATION_MS for value in values):
            raise ValueError("sanitized timing values must be finite and within the per-turn limit")
        if self.time_to_first_byte_ms > self.e2e_latency_ms:
            raise ValueError("time to first byte must not exceed end-to-end latency")
        if self.time_to_first_token_ms > self.e2e_latency_ms:
            raise ValueError("time to first token must not exceed end-to-end latency")
        if self.generation_ms > self.e2e_latency_ms:
            raise ValueError("generation time must not exceed end-to-end latency")

    def to_json(self) -> JsonObject:
        """Return safe timing evidence."""
        return json_record(self)

    @classmethod
    def from_json(cls, data: JsonObject, source: str) -> SanitizedTiming:
        """Parse one strict timing object."""
        require_exact_keys(data, json_field_names(cls), source)
        return cls(
            e2e_latency_ms=required_number(data, "e2e_latency_ms", source),
            time_to_first_byte_ms=required_number(data, "time_to_first_byte_ms", source),
            time_to_first_token_ms=required_number(data, "time_to_first_token_ms", source),
            generation_ms=required_number(data, "generation_ms", source),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SanitizedTokens:
    """Store one turn's allowlisted token evidence."""

    server_prompt_tokens: int
    server_output_tokens: int
    local_output_tokens: int
    server_cached_prompt_tokens: int | None
    server_uncached_prompt_tokens: int | None
    observed_output_tokens: int
    target_output_tokens: int
    cache_accounting_fields_present: bool = True

    def __post_init__(self) -> None:
        """Reject impossible or abusive token values."""
        values = (
            self.server_prompt_tokens,
            self.server_output_tokens,
            self.local_output_tokens,
            self.observed_output_tokens,
            self.target_output_tokens,
        )
        if any(value < 0 or value > MAX_TOKENS_PER_TURN for value in values):
            raise ValueError("sanitized token counts must be within the per-turn limit")
        if self.observed_output_tokens <= 0 or self.target_output_tokens <= 0:
            raise ValueError("normalization observed and target token counts must be positive")
        if self.observed_output_tokens != self.server_output_tokens:
            raise ValueError("normalization observed tokens must match server output tokens")
        cache_counts = (self.server_cached_prompt_tokens, self.server_uncached_prompt_tokens)
        if (cache_counts[0] is None) != (cache_counts[1] is None):
            raise ValueError("cached and uncached prompt token counts must be reported together")
        if any(value is not None and (value < 0 or value > MAX_TOKENS_PER_TURN) for value in cache_counts):
            raise ValueError("sanitized cache token counts must be within the per-turn limit")
        if (
            self.server_cached_prompt_tokens is not None
            and self.server_uncached_prompt_tokens is not None
            and self.server_cached_prompt_tokens + self.server_uncached_prompt_tokens != self.server_prompt_tokens
        ):
            raise ValueError("cached and uncached prompt tokens must add up to prompt tokens")
        if not self.cache_accounting_fields_present and any(value is not None for value in cache_counts):
            raise ValueError("absent cache accounting fields cannot carry token counts")

    def to_json(self) -> JsonObject:
        """Return safe token evidence."""
        data: JsonObject = {
            "server_prompt_tokens": self.server_prompt_tokens,
            "server_output_tokens": self.server_output_tokens,
            "local_output_tokens": self.local_output_tokens,
            "observed_output_tokens": self.observed_output_tokens,
            "target_output_tokens": self.target_output_tokens,
        }
        if self.cache_accounting_fields_present:
            data["server_cached_prompt_tokens"] = self.server_cached_prompt_tokens
            data["server_uncached_prompt_tokens"] = self.server_uncached_prompt_tokens
        return data

    @classmethod
    def from_json(cls, data: JsonObject, source: str) -> SanitizedTokens:
        """Parse one strict token object."""
        require_allowed_keys(data, _REQUIRED_TOKEN_KEYS, _TOKEN_KEYS, source)
        cached_field_present = "server_cached_prompt_tokens" in data
        uncached_field_present = "server_uncached_prompt_tokens" in data
        if cached_field_present != uncached_field_present:
            raise ValueError(f"{source} must contain both cache accounting fields or neither")
        return cls(
            server_prompt_tokens=required_integer(data, "server_prompt_tokens", source),
            server_output_tokens=required_integer(data, "server_output_tokens", source),
            local_output_tokens=required_integer(data, "local_output_tokens", source),
            server_cached_prompt_tokens=optional_integer(data, "server_cached_prompt_tokens", source),
            server_uncached_prompt_tokens=optional_integer(data, "server_uncached_prompt_tokens", source),
            observed_output_tokens=required_integer(data, "observed_output_tokens", source),
            target_output_tokens=required_integer(data, "target_output_tokens", source),
            cache_accounting_fields_present=cached_field_present,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SanitizedTurn:
    """Store one ordinal-only turn row for server recomputation."""

    turn_ordinal: int
    task_ordinal: int
    turn_in_task: int
    finish_reason: str
    response_chunks: int
    response_action_count: int
    recorded_action_count: int
    recorded_pacing_ms: float
    replayed_pacing_ms: float
    timing: SanitizedTiming
    tokens: SanitizedTokens

    def __post_init__(self) -> None:
        """Validate ordinal, action, and pacing evidence."""
        if self.turn_ordinal < 0 or self.task_ordinal < 0 or self.turn_in_task < 0:
            raise ValueError("sanitized ordinals must be non-negative")
        if not self.finish_reason:
            raise ValueError("sanitized finish reason must not be empty")
        if self.response_chunks <= 0 or self.response_chunks > MAX_RESPONSE_CHUNKS_PER_TURN:
            raise ValueError("response chunk count must be within the per-turn limit")
        action_counts = (self.response_action_count, self.recorded_action_count)
        if any(value < 0 or value > MAX_ACTIONS_PER_TURN for value in action_counts):
            raise ValueError("action counts must be within the per-turn limit")
        pacing = (self.recorded_pacing_ms, self.replayed_pacing_ms)
        if any(not math.isfinite(value) or value < 0 or value > MAX_TURN_DURATION_MS for value in pacing):
            raise ValueError("pacing values must be finite and within the per-turn limit")

    def to_json(self) -> JsonObject:
        """Return one sanitized turn row."""
        return json_record(self)

    @classmethod
    def from_json(cls, data: JsonObject, source: str) -> SanitizedTurn:
        """Parse one strict sanitized turn."""
        require_exact_keys(data, json_field_names(cls), source)
        return cls(
            turn_ordinal=required_integer(data, "turn_ordinal", source),
            task_ordinal=required_integer(data, "task_ordinal", source),
            turn_in_task=required_integer(data, "turn_in_task", source),
            finish_reason=required_string(data, "finish_reason", source),
            response_chunks=required_integer(data, "response_chunks", source),
            response_action_count=required_integer(data, "response_action_count", source),
            recorded_action_count=required_integer(data, "recorded_action_count", source),
            recorded_pacing_ms=required_number(data, "recorded_pacing_ms", source),
            replayed_pacing_ms=required_number(data, "replayed_pacing_ms", source),
            timing=SanitizedTiming.from_json(required_object(data, "timing", source), f"{source}.timing"),
            tokens=SanitizedTokens.from_json(required_object(data, "tokens", source), f"{source}.tokens"),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SanitizedEvidence:
    """Store one experimental sanitized evidence envelope."""

    aggregate_payload_digest: str
    source_turns_digest: str
    rows_digest: str
    turns: tuple[SanitizedTurn, ...]
    normalization_policy_id: str = OUTPUT_LENGTH_NORMALIZATION_POLICY
    version: int = SANITIZED_EVIDENCE_VERSION

    def __post_init__(self) -> None:
        """Validate bindings and ordinal completeness."""
        validate_digest(self.aggregate_payload_digest, "aggregate_payload_digest")
        validate_digest(self.source_turns_digest, "source_turns_digest")
        validate_digest(self.rows_digest, "rows_digest")
        if self.version != SANITIZED_EVIDENCE_VERSION:
            raise ValueError("sanitized evidence version is not supported")
        if self.normalization_policy_id != OUTPUT_LENGTH_NORMALIZATION_POLICY:
            raise ValueError("sanitized normalization policy is not supported")
        if not self.turns:
            raise ValueError("sanitized evidence must contain turns")
        if len(self.turns) > MAX_SANITIZED_TURNS:
            raise ValueError("sanitized evidence exceeds the turn limit")
        if tuple(turn.turn_ordinal for turn in self.turns) != tuple(range(len(self.turns))):
            raise ValueError("sanitized turn ordinals must be complete and ordered")
        _validate_task_layout(self.turns)
        if self.rows_digest != _rows_digest(self.turns):
            raise ValueError("sanitized rows digest does not match the turn rows")

    def to_json(self) -> JsonObject:
        """Return the sanitized evidence envelope."""
        turn_values: list[JsonValue] = [turn.to_json() for turn in self.turns]
        return {
            "version": self.version,
            "kind": SANITIZED_EVIDENCE_KIND,
            "privacy_profile": SANITIZED_EVIDENCE_PROFILE,
            "normalization_policy_id": self.normalization_policy_id,
            "aggregate_payload_digest": self.aggregate_payload_digest,
            "source_turns_digest": self.source_turns_digest,
            "rows_digest": self.rows_digest,
            "turns": turn_values,
            "excluded_private_categories": list(SANITIZED_EXCLUDED_CATEGORIES),
            "status": SANITIZED_EVIDENCE_STATUS,
        }

    @classmethod
    def from_json(cls, data: JsonObject) -> SanitizedEvidence:
        """Parse and semantically validate one evidence envelope."""
        require_exact_keys(data, _EVIDENCE_KEYS, "evidence")
        if required_string(data, "kind", "evidence") != SANITIZED_EVIDENCE_KIND:
            raise ValueError("evidence.kind is not supported")
        if required_string(data, "privacy_profile", "evidence") != SANITIZED_EVIDENCE_PROFILE:
            raise ValueError("evidence.privacy_profile is not supported")
        if required_string(data, "status", "evidence") != SANITIZED_EVIDENCE_STATUS:
            raise ValueError("evidence.status is not supported")
        excluded = data.get("excluded_private_categories")
        if excluded != list(SANITIZED_EXCLUDED_CATEGORIES):
            raise ValueError("evidence.excluded_private_categories is not supported")
        values = data.get("turns")
        if not isinstance(values, list):
            raise ValueError("evidence.turns must be an array")
        turns: list[SanitizedTurn] = []
        for index, value in enumerate(values):
            if not isinstance(value, dict):
                raise ValueError(f"evidence.turns[{index}] must be an object")
            turns.append(SanitizedTurn.from_json(value, f"evidence.turns[{index}]"))
        return cls(
            version=required_integer(data, "version", "evidence"),
            aggregate_payload_digest=required_string(data, "aggregate_payload_digest", "evidence"),
            source_turns_digest=required_string(data, "source_turns_digest", "evidence"),
            rows_digest=required_string(data, "rows_digest", "evidence"),
            normalization_policy_id=required_string(data, "normalization_policy_id", "evidence"),
            turns=tuple(turns),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class RecomputedSanitizedMetrics:
    """Store aggregates derived only from sanitized rows."""

    totals: PublicTotals
    latency_distributions_ms: PublicLatencyDistributions


def _sanitized_turn(
    turn: ValidatedEvidenceTurn,
    *,
    turn_ordinal: int,
    task_ordinal: int,
    turn_in_task: int,
) -> SanitizedTurn:
    finish_reason = turn.finish_reason
    if finish_reason is None:
        raise ValueError("validated turn evidence has no finish reason")
    return SanitizedTurn(
        turn_ordinal=turn_ordinal,
        task_ordinal=task_ordinal,
        turn_in_task=turn_in_task,
        finish_reason=finish_reason,
        response_chunks=turn.response_chunks,
        response_action_count=turn.response_tool_calls,
        recorded_action_count=turn.tool_calls,
        recorded_pacing_ms=turn.recorded_tool_delay_ms,
        replayed_pacing_ms=turn.replayed_tool_delay_ms,
        timing=SanitizedTiming(
            e2e_latency_ms=turn.e2e_latency_ms,
            time_to_first_byte_ms=turn.time_to_first_byte_ms,
            time_to_first_token_ms=turn.time_to_first_token_ms,
            generation_ms=turn.generation_ms,
        ),
        tokens=SanitizedTokens(
            server_prompt_tokens=turn.server_prompt_tokens,
            server_output_tokens=turn.server_output_tokens,
            local_output_tokens=turn.local_output_tokens,
            server_cached_prompt_tokens=turn.server_cached_prompt_tokens,
            server_uncached_prompt_tokens=turn.server_uncached_prompt_tokens,
            observed_output_tokens=turn.observed_output_tokens,
            target_output_tokens=turn.target_output_tokens,
        ),
    )


def _rows_digest(turns: tuple[SanitizedTurn, ...]) -> str:
    values: list[JsonValue] = [turn.to_json() for turn in turns]
    return sha256_bytes(orjson.dumps(values, option=orjson.OPT_SORT_KEYS))


def _validate_task_layout(turns: tuple[SanitizedTurn, ...]) -> None:
    if turns[0].task_ordinal != 0 or turns[0].turn_in_task != 0:
        raise ValueError("sanitized task layout must start at task 0, turn 0")
    expected_task_ordinal = 0
    expected_turn_in_task = 0
    for turn in turns:
        if turn.task_ordinal == expected_task_ordinal:
            if turn.turn_in_task != expected_turn_in_task:
                raise ValueError("sanitized turn_in_task ordinals must be complete and ordered")
            expected_turn_in_task += 1
            continue
        if turn.task_ordinal != expected_task_ordinal + 1 or turn.turn_in_task != 0:
            raise ValueError("sanitized task ordinals must form contiguous ordered blocks")
        expected_task_ordinal = turn.task_ordinal
        expected_turn_in_task = 1


def _totals_turn(turn: SanitizedTurn) -> TotalsTurn:
    """Return the values one sanitized turn adds to public totals.

    Sanitized rows hold successful turns only, and they carry the raw inputs, so the
    normalized timings are recomputed here instead of trusted.
    """
    normalization = normalize_output_length(
        e2e_latency_ms=turn.timing.e2e_latency_ms,
        generation_ms=turn.timing.generation_ms,
        observed_output_tokens=turn.tokens.observed_output_tokens,
        target_output_tokens=turn.tokens.target_output_tokens,
    )
    normalized_generation_ms = normalization.normalized_generation_ms
    normalized_e2e_latency_ms = normalization.normalized_e2e_latency_ms
    if normalized_generation_ms is None or normalized_e2e_latency_ms is None:
        raise ValueError("sanitized normalization inputs are incomplete")
    return TotalsTurn(
        success=True,
        has_short_output_warning=normalization.warning is not None,
        tool_calls=turn.recorded_action_count,
        e2e_latency_ms=turn.timing.e2e_latency_ms,
        recorded_tool_delay_ms=turn.recorded_pacing_ms,
        replayed_tool_delay_ms=turn.replayed_pacing_ms,
        normalized_generation_ms=normalized_generation_ms,
        normalized_e2e_latency_ms=normalized_e2e_latency_ms,
        server_prompt_tokens=turn.tokens.server_prompt_tokens,
        server_output_tokens=turn.tokens.server_output_tokens,
        local_output_tokens=turn.tokens.local_output_tokens,
    )


def recompute_sanitized_metrics(evidence: SanitizedEvidence) -> RecomputedSanitizedMetrics:
    """Derive per-turn totals and distributions from sanitized rows."""
    turns = evidence.turns
    if not turns:
        raise ValueError("sanitized evidence must contain turns")
    totals_turns = tuple(_totals_turn(turn) for turn in turns)
    distributions = PublicLatencyDistributions(
        e2e=PublicDistribution.from_values(tuple(turn.timing.e2e_latency_ms for turn in turns)),
        time_to_first_token=PublicDistribution.from_values(tuple(turn.timing.time_to_first_token_ms for turn in turns)),
        normalized_e2e=PublicDistribution.from_values(tuple(turn.normalized_e2e_latency_ms for turn in totals_turns)),
    )
    return RecomputedSanitizedMetrics(
        totals=PublicTotals.from_turns(totals_turns), latency_distributions_ms=distributions
    )


def build_sanitized_evidence(results_dir: Path) -> tuple[PublicSubmission, SanitizedEvidence]:
    """Build the validated aggregate and its sanitized turn evidence together."""
    submission = build_public_submission(results_dir)
    binding = load_measurement_binding(results_dir / MEASUREMENT_BINDING_FILENAME)
    manifest = load_manifest(binding.manifest_path)
    task_ordinals = {task.task_id: index for index, task in enumerate(manifest.tasks)}
    task_turn_counts = {task.task_id: 0 for task in manifest.tasks}
    validated_turns = load_validated_evidence_turns(results_dir / TURNS_FILENAME)
    sanitized_turns: list[SanitizedTurn] = []
    for turn_ordinal, turn in enumerate(validated_turns):
        task_ordinal = task_ordinals.get(turn.task_id)
        turn_in_task = task_turn_counts.get(turn.task_id)
        if task_ordinal is None or turn_in_task is None:
            raise ValueError("turn evidence contains a task outside the bound manifest")
        sanitized_turns.append(
            _sanitized_turn(
                turn,
                turn_ordinal=turn_ordinal,
                task_ordinal=task_ordinal,
                turn_in_task=turn_in_task,
            )
        )
        task_turn_counts[turn.task_id] = turn_in_task + 1
    rows = tuple(sanitized_turns)
    evidence = SanitizedEvidence(
        aggregate_payload_digest=submission.payload_digest,
        source_turns_digest=submission.artifacts.turns,
        rows_digest=_rows_digest(rows),
        turns=rows,
    )
    recomputed = recompute_sanitized_metrics(evidence)
    if recomputed.totals != submission.run.totals:
        raise ValueError("sanitized rows do not reproduce public totals")
    if recomputed.latency_distributions_ms != submission.run.latency_distributions_ms:
        raise ValueError("sanitized rows do not reproduce public latency distributions")
    return submission, evidence


def validate_sanitized_evidence(data: JsonObject) -> SanitizedEvidence:
    """Apply closed-field and cross-field evidence validation."""
    return SanitizedEvidence.from_json(data)
