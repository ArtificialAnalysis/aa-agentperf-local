"""Evaluate tool-call transport and semantic fidelity after stream close."""

from __future__ import annotations

import math
from typing import Literal, Self

import orjson
from pydantic import BaseModel, model_validator

from agentperf_local.common.json_types import JsonObject, JsonValue, normalize_json_object
from agentperf_local.metrics.response import ToolCall

MAX_TOOL_NAME_CHARACTERS = 128
LENGTH_FINISH_REASON = "length"
TEXT_FINISH_REASONS = frozenset(("stop", LENGTH_FINISH_REASON))

type FidelityLevel = Literal["invalid", "transport-valid", "action-shape-match", "exact-argument-match"]
type FidelityIssue = Literal[
    "finish_reason_mismatch",
    "truncated_by_output_cap",
    "non_contiguous_indices",
    "invalid_tool_name",
    "invalid_arguments_json",
    "action_count_mismatch",
    "tool_name_mismatch",
    "tool_arguments_mismatch",
]


def generated_whole_budget(finish_reason: str | None, server_output_tokens: int | None, max_tokens: int) -> bool:
    """Report whether a response ran to the max_tokens it was given.

    A "length" finish is the server saying it hit the cap. A usage count that disagrees
    overrides it; a missing count cannot, because an honouring server that omits usage
    would otherwise read as one that stopped early.
    """
    if finish_reason != LENGTH_FINISH_REASON:
        return False
    return server_output_tokens is None or server_output_tokens >= max_tokens


def _valid_tool_name(value: str) -> bool:
    return (
        bool(value)
        and len(value) <= MAX_TOOL_NAME_CHARACTERS
        and value.isascii()
        and value.isprintable()
        and not any(character.isspace() for character in value)
    )


def _canonical_arguments(value: str) -> bytes | None:
    try:
        arguments = normalize_json_object(orjson.loads(value))
    except (orjson.JSONDecodeError, ValueError):
        return None
    return orjson.dumps(arguments, option=orjson.OPT_SORT_KEYS)


def _validate_finite_json(value: JsonValue) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("expected arguments must contain finite JSON numbers")
    if isinstance(value, list):
        for item in value:
            _validate_finite_json(item)
    elif isinstance(value, dict):
        for item in value.values():
            _validate_finite_json(item)


class ExpectedToolCall(BaseModel, frozen=True):
    """Describe one canonical action without provider-generated identifiers."""

    name: str
    arguments: JsonObject

    @model_validator(mode="after")
    def check_invariants(self) -> Self:
        """Validate one expected action."""
        if not _valid_tool_name(self.name):
            raise ValueError("expected tool name must be short printable ASCII without whitespace")
        _validate_finite_json(self.arguments)
        orjson.dumps(self.arguments, option=orjson.OPT_SORT_KEYS)
        return self

    @property
    def canonical_arguments(self) -> bytes:
        """Return deterministic JSON argument bytes."""
        return orjson.dumps(self.arguments, option=orjson.OPT_SORT_KEYS)


class ToolFidelityReport(BaseModel, frozen=True):
    """Store content-free action fidelity outcomes for one response."""

    expected_actions: int
    observed_actions: int
    transport_valid: bool
    action_shape_match: bool
    exact_argument_match: bool
    highest_level: FidelityLevel
    issues: tuple[FidelityIssue, ...]

    @model_validator(mode="after")
    def check_invariants(self) -> Self:
        """Require consistent monotonic fidelity levels."""
        if self.expected_actions < 0 or self.observed_actions < 0:
            raise ValueError("action counts must be non-negative")
        if self.exact_argument_match and not self.action_shape_match:
            raise ValueError("exact argument match requires an action shape match")
        if self.action_shape_match and not self.transport_valid:
            raise ValueError("action shape match requires valid transport")
        expected_level: FidelityLevel
        if self.exact_argument_match:
            expected_level = "exact-argument-match"
        elif self.action_shape_match:
            expected_level = "action-shape-match"
        elif self.transport_valid:
            expected_level = "transport-valid"
        else:
            expected_level = "invalid"
        if self.highest_level != expected_level:
            raise ValueError("highest_level does not match fidelity outcomes")
        if len(set(self.issues)) != len(self.issues):
            raise ValueError("fidelity issues must be unique")
        return self

    def to_json(self) -> JsonObject:
        """Return outcomes without names, arguments, or response text."""
        return {
            "expected_actions": self.expected_actions,
            "observed_actions": self.observed_actions,
            "transport_valid": self.transport_valid,
            "action_shape_match": self.action_shape_match,
            "exact_argument_match": self.exact_argument_match,
            "highest_level": self.highest_level,
            "issues": list(self.issues),
            "tool_names_included": False,
            "tool_arguments_included": False,
            "response_text_included": False,
        }


def _finish_reason_matches(finish_reason: str | None, observed_actions: int) -> bool:
    if observed_actions == 0:
        # A turn capped at max_tokens ends with "length" and still carries valid transport.
        return finish_reason in TEXT_FINISH_REASONS
    return finish_reason == "tool_calls"


def _truncated_by_output_cap(finish_reason: str | None, observed_actions: int, output_capped: bool) -> bool:
    """Report an action the run's own output cap cut short, rather than a broken response.

    The replay sets max_tokens itself, so a run that spends its whole budget mid-action
    is measuring the cap, not a transport defect. On the streamed path the server hands
    back exactly the argument text emitted before the cut, so the arguments of such an
    action never parse; that is the same event, not a second defect. A "length" finish
    that stops under the cap is a different thing and stays a mismatch.
    """
    return output_capped and observed_actions > 0 and finish_reason == LENGTH_FINISH_REASON


def evaluate_tool_fidelity(
    observed: tuple[ToolCall, ...],
    expected: tuple[ExpectedToolCall, ...],
    *,
    finish_reason: str | None,
    output_capped: bool = False,
) -> ToolFidelityReport:
    """Evaluate transport, ordered names, and exact JSON arguments.

    Pass output_capped when the response spent the whole max_tokens budget the run
    asked for, so an action the cap cut short is recorded as a divergence from the
    recording's length rather than counted against transport.
    """
    issues: list[FidelityIssue] = []
    expected_indices = tuple(range(len(observed)))
    observed_indices = tuple(call.index for call in observed)
    truncated = _truncated_by_output_cap(finish_reason, len(observed), output_capped)
    if truncated:
        issues.append("truncated_by_output_cap")
    elif not _finish_reason_matches(finish_reason, len(observed)):
        issues.append("finish_reason_mismatch")
    if observed_indices != expected_indices:
        issues.append("non_contiguous_indices")
    if any(not _valid_tool_name(call.name) for call in observed):
        issues.append("invalid_tool_name")
    canonical_observed = tuple(_canonical_arguments(call.arguments) for call in observed)
    # Arguments cut short by the cap are incomplete by construction; the truncation
    # issue already says so, and counting the parse failure too would fail the turn.
    if not truncated and any(arguments is None for arguments in canonical_observed):
        issues.append("invalid_arguments_json")

    transport_issues = {
        "finish_reason_mismatch",
        "non_contiguous_indices",
        "invalid_tool_name",
        "invalid_arguments_json",
    }
    transport_valid = not any(issue in transport_issues for issue in issues)
    count_matches = len(observed) == len(expected)
    if not count_matches:
        issues.append("action_count_mismatch")
    names_match = count_matches and all(
        call.name == target.name for call, target in zip(observed, expected, strict=True)
    )
    if count_matches and not names_match:
        issues.append("tool_name_mismatch")
    action_shape_match = transport_valid and count_matches and names_match
    arguments_match = action_shape_match and all(
        actual == target.canonical_arguments for actual, target in zip(canonical_observed, expected, strict=True)
    )
    if action_shape_match and not arguments_match:
        issues.append("tool_arguments_mismatch")
    exact_argument_match = action_shape_match and arguments_match
    if exact_argument_match:
        level: FidelityLevel = "exact-argument-match"
    elif action_shape_match:
        level = "action-shape-match"
    elif transport_valid:
        level = "transport-valid"
    else:
        level = "invalid"
    return ToolFidelityReport(
        expected_actions=len(expected),
        observed_actions=len(observed),
        transport_valid=transport_valid,
        action_shape_match=action_shape_match,
        exact_argument_match=exact_argument_match,
        highest_level=level,
        issues=tuple(issues),
    )
