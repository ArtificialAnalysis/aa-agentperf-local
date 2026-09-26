"""Exercise content-free post-close tool fidelity evaluation."""

from dataclasses import dataclass

import pytest

from agentperf_local.metrics.response import ToolCall
from agentperf_local.replay.fidelity import ExpectedToolCall, FidelityIssue, FidelityLevel, evaluate_tool_fidelity


@dataclass(frozen=True, slots=True, kw_only=True)
class FidelityCase:
    name: str
    observed: tuple[ToolCall, ...]
    expected: tuple[ExpectedToolCall, ...]
    finish_reason: str | None
    level: FidelityLevel
    issues: tuple[FidelityIssue, ...]
    output_capped: bool = False


def _observed(*, index: int = 0, name: str = "shell", arguments: str = '{"command":"rg needle"}') -> ToolCall:
    return ToolCall(index=index, identifier="provider-private-id", name=name, arguments=arguments)


def _expected(*, name: str = "shell", command: str = "rg needle") -> ExpectedToolCall:
    return ExpectedToolCall(name=name, arguments={"command": command})


CASES = (
    FidelityCase(
        name="exact",
        observed=(_observed(arguments='{"command": "rg needle"}'),),
        expected=(_expected(),),
        finish_reason="tool_calls",
        level="exact-argument-match",
        issues=(),
    ),
    FidelityCase(
        name="no-action",
        observed=(),
        expected=(),
        finish_reason="stop",
        level="exact-argument-match",
        issues=(),
    ),
    FidelityCase(
        name="no-action-length-capped",
        observed=(),
        expected=(),
        finish_reason="length",
        level="exact-argument-match",
        issues=(),
    ),
    FidelityCase(
        name="no-action-content-filter",
        observed=(),
        expected=(),
        finish_reason="content_filter",
        level="invalid",
        issues=("finish_reason_mismatch",),
    ),
    FidelityCase(
        # Stopping on "length" without spending the budget is the model's doing.
        name="action-length-under-budget",
        observed=(_observed(),),
        expected=(_expected(),),
        finish_reason="length",
        level="invalid",
        issues=("finish_reason_mismatch",),
    ),
    FidelityCase(
        # The run's own cap cut this action short, so transport still holds. The streamed
        # arguments stop where the cut fell, so they do not parse; that is the same event.
        name="action-length-spent-budget",
        observed=(_observed(arguments='{"command":"rg ne'),),
        expected=(_expected(),),
        finish_reason="length",
        output_capped=True,
        level="action-shape-match",
        issues=("truncated_by_output_cap", "tool_arguments_mismatch"),
    ),
    FidelityCase(
        # A spent budget excuses only "length"; every other finish stays a mismatch.
        name="action-stop-spent-budget",
        observed=(_observed(),),
        expected=(_expected(),),
        finish_reason="stop",
        output_capped=True,
        level="invalid",
        issues=("finish_reason_mismatch",),
    ),
    FidelityCase(
        # With no action there is nothing for the cap to cut short.
        name="no-action-spent-budget",
        observed=(),
        expected=(),
        finish_reason="length",
        output_capped=True,
        level="exact-argument-match",
        issues=(),
    ),
    FidelityCase(
        name="argument-mismatch",
        observed=(_observed(arguments='{"command":"rg other"}'),),
        expected=(_expected(),),
        finish_reason="tool_calls",
        level="action-shape-match",
        issues=("tool_arguments_mismatch",),
    ),
    FidelityCase(
        name="name-mismatch",
        observed=(_observed(name="python"),),
        expected=(_expected(),),
        finish_reason="tool_calls",
        level="transport-valid",
        issues=("tool_name_mismatch",),
    ),
    FidelityCase(
        name="count-mismatch",
        observed=(_observed(),),
        expected=(_expected(), _expected(command="rg other")),
        finish_reason="tool_calls",
        level="transport-valid",
        issues=("action_count_mismatch",),
    ),
    FidelityCase(
        name="invalid-json",
        observed=(_observed(arguments="{not-json"),),
        expected=(_expected(),),
        finish_reason="tool_calls",
        level="invalid",
        issues=("invalid_arguments_json",),
    ),
    FidelityCase(
        name="index-gap",
        observed=(_observed(index=1),),
        expected=(_expected(),),
        finish_reason="tool_calls",
        level="invalid",
        issues=("non_contiguous_indices",),
    ),
    FidelityCase(
        name="finish-mismatch",
        observed=(_observed(),),
        expected=(_expected(),),
        finish_reason="stop",
        level="invalid",
        issues=("finish_reason_mismatch",),
    ),
)


@pytest.mark.parametrize("case", CASES, ids=tuple(case.name for case in CASES))
def test_evaluates_monotonic_fidelity_without_exposing_content(case: FidelityCase) -> None:
    report = evaluate_tool_fidelity(
        case.observed, case.expected, finish_reason=case.finish_reason, output_capped=case.output_capped
    )
    encoded = report.to_json()

    assert report.highest_level == case.level
    assert report.issues == case.issues
    assert report.exact_argument_match is (case.level == "exact-argument-match")
    assert report.action_shape_match is (case.level in {"action-shape-match", "exact-argument-match"})
    assert report.transport_valid is (case.level != "invalid")
    assert encoded["tool_names_included"] is False
    assert encoded["tool_arguments_included"] is False
    assert encoded["response_text_included"] is False
    assert "provider-private-id" not in repr(encoded)
    assert "rg needle" not in repr(encoded)


@pytest.mark.parametrize("name", ["", "bad name", "bad\nname", "x" * 129])
def test_rejects_invalid_expected_tool_names(name: str) -> None:
    with pytest.raises(ValueError, match="expected tool name"):
        _expected(name=name)


def test_rejects_non_finite_expected_arguments() -> None:
    with pytest.raises(ValueError, match="finite JSON numbers"):
        ExpectedToolCall(name="shell", arguments={"temperature": float("nan")})
