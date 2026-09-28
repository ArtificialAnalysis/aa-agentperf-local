"""Reduce normalized telemetry after benchmark phases close."""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable
from typing import Annotated, Self

from pydantic import BaseModel, Field, PositiveInt, model_validator

from agentperf_local.common.units import NANOSECONDS_PER_MILLISECOND, NANOSECONDS_PER_SECOND
from agentperf_local.telemetry.nvidia import NvidiaSample

DEFAULT_MINIMUM_COVERAGE = 0.95
DEFAULT_MAX_SAMPLE_GAP_INTERVALS = 3


class PhaseInterval(BaseModel, frozen=True):
    """Describe one half-open benchmark phase interval."""

    phase_id: Annotated[str, Field(min_length=1)]
    start_ns: int
    end_ns: int

    @model_validator(mode="after")
    def check_invariants(self) -> Self:
        """Validate the phase interval."""
        if self.start_ns <= 0 or self.end_ns <= self.start_ns:
            raise ValueError("phase interval must be positive and non-empty")
        return self

    @property
    def duration_ns(self) -> int:
        """Return phase duration in nanoseconds."""
        return self.end_ns - self.start_ns


class CoveragePolicy(BaseModel, frozen=True):
    """Store explicit telemetry admission thresholds."""

    minimum_coverage: float = DEFAULT_MINIMUM_COVERAGE
    max_sample_gap_intervals: PositiveInt = DEFAULT_MAX_SAMPLE_GAP_INTERVALS

    @model_validator(mode="after")
    def check_invariants(self) -> Self:
        """Validate coverage thresholds."""
        if not math.isfinite(self.minimum_coverage) or not 0 < self.minimum_coverage <= 1:
            raise ValueError("minimum_coverage must be greater than zero and at most one")
        return self


class ScalarObservation(BaseModel, frozen=True):
    """Hold one optional scalar at a local monotonic time."""

    monotonic_ns: PositiveInt
    value: float | None

    @model_validator(mode="after")
    def check_invariants(self) -> Self:
        """Validate one scalar observation."""
        if self.value is not None and (not math.isfinite(self.value) or self.value < 0):
            raise ValueError("observation value must be finite and non-negative")
        return self


class ScalarTelemetrySummary(BaseModel, frozen=True):
    """Store coverage and aggregates for one field and phase."""

    field: str
    phase_id: str
    phase_duration_ms: float
    total_sample_count: int
    valid_sample_count: int
    missing_sample_count: int
    coverage: float
    largest_uncovered_gap_ms: float
    first_sample_slack_ms: float | None
    last_sample_slack_ms: float | None
    maximum: float | None
    median: float | None
    time_weighted_mean: float | None
    integral_value_seconds: float | None
    integration_coverage: float
    policy_passed: bool


class NvidiaPhaseSummary(BaseModel, frozen=True):
    """Store normalized NVIDIA aggregates for one phase."""

    phase: PhaseInterval
    interval_ms: int
    minimum_coverage: float
    gpu_utilization: ScalarTelemetrySummary
    memory_used_mib: ScalarTelemetrySummary
    temperature_c: ScalarTelemetrySummary
    power_w: ScalarTelemetrySummary
    graphics_clock_mhz: ScalarTelemetrySummary
    memory_clock_mhz: ScalarTelemetrySummary


def _ordered(observations: tuple[ScalarObservation, ...]) -> tuple[ScalarObservation, ...]:
    ordered = tuple(sorted(observations, key=lambda observation: observation.monotonic_ns))
    if len({observation.monotonic_ns for observation in ordered}) != len(ordered):
        raise ValueError("telemetry timestamps must be unique")
    return ordered


def _coverage_windows(
    valid: tuple[ScalarObservation, ...],
    phase: PhaseInterval,
    interval_ns: int,
) -> tuple[int, int]:
    half_interval = interval_ns // 2
    windows = sorted(
        (
            max(phase.start_ns, observation.monotonic_ns - half_interval),
            min(phase.end_ns, observation.monotonic_ns + half_interval),
        )
        for observation in valid
        if observation.monotonic_ns + half_interval > phase.start_ns
        and observation.monotonic_ns - half_interval < phase.end_ns
    )
    if not windows:
        return 0, phase.duration_ns
    covered_ns = 0
    largest_uncovered_ns = max(0, windows[0][0] - phase.start_ns)
    current_start, current_end = windows[0]
    for start, end in windows[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        covered_ns += current_end - current_start
        largest_uncovered_ns = max(largest_uncovered_ns, start - current_end)
        current_start, current_end = start, end
    covered_ns += current_end - current_start
    largest_uncovered_ns = max(largest_uncovered_ns, phase.end_ns - current_end)
    return covered_ns, largest_uncovered_ns


def _interpolate(
    left: ScalarObservation,
    right: ScalarObservation,
    timestamp_ns: int,
) -> float:
    if left.value is None or right.value is None:
        raise ValueError("cannot interpolate a missing observation")
    fraction = (timestamp_ns - left.monotonic_ns) / (right.monotonic_ns - left.monotonic_ns)
    return left.value + fraction * (right.value - left.value)


def _integrate(
    valid: tuple[ScalarObservation, ...],
    phase: PhaseInterval,
    maximum_gap_ns: int,
) -> tuple[float | None, float | None, float]:
    area_value_ns = 0.0
    integrated_ns = 0
    for left, right in zip(valid, valid[1:]):
        gap_ns = right.monotonic_ns - left.monotonic_ns
        if gap_ns > maximum_gap_ns:
            continue
        start_ns = max(phase.start_ns, left.monotonic_ns)
        end_ns = min(phase.end_ns, right.monotonic_ns)
        if end_ns <= start_ns:
            continue
        start_value = _interpolate(left, right, start_ns)
        end_value = _interpolate(left, right, end_ns)
        duration_ns = end_ns - start_ns
        area_value_ns += (start_value + end_value) * duration_ns / 2
        integrated_ns += duration_ns
    if integrated_ns == 0:
        return None, None, 0.0
    time_weighted_mean = area_value_ns / integrated_ns
    integral_value_seconds = area_value_ns / NANOSECONDS_PER_SECOND
    return time_weighted_mean, integral_value_seconds, integrated_ns / phase.duration_ns


def reduce_scalar_telemetry(
    field: str,
    observations: tuple[ScalarObservation, ...],
    phase: PhaseInterval,
    *,
    interval_ms: int,
    policy: CoveragePolicy = CoveragePolicy(),
) -> ScalarTelemetrySummary:
    """Reduce one optional scalar after its phase closes."""
    if not field:
        raise ValueError("field must not be empty")
    if interval_ms <= 0:
        raise ValueError("interval_ms must be positive")
    ordered = _ordered(observations)
    in_phase = tuple(
        observation for observation in ordered if phase.start_ns <= observation.monotonic_ns < phase.end_ns
    )
    valid = tuple(observation for observation in ordered if observation.value is not None)
    valid_in_phase = tuple(observation for observation in in_phase if observation.value is not None)
    interval_ns = interval_ms * NANOSECONDS_PER_MILLISECOND
    covered_ns, largest_uncovered_ns = _coverage_windows(valid, phase, interval_ns)
    coverage = covered_ns / phase.duration_ns
    maximum_gap_ns = interval_ns * policy.max_sample_gap_intervals
    time_weighted_mean, integral_value_seconds, integration_coverage = _integrate(valid, phase, maximum_gap_ns)
    values = tuple(observation.value for observation in valid_in_phase if observation.value is not None)
    first_slack = (
        (valid_in_phase[0].monotonic_ns - phase.start_ns) / NANOSECONDS_PER_MILLISECOND if valid_in_phase else None
    )
    last_slack = (
        (phase.end_ns - valid_in_phase[-1].monotonic_ns) / NANOSECONDS_PER_MILLISECOND if valid_in_phase else None
    )
    policy_passed = coverage >= policy.minimum_coverage and largest_uncovered_ns <= maximum_gap_ns
    return ScalarTelemetrySummary(
        field=field,
        phase_id=phase.phase_id,
        phase_duration_ms=phase.duration_ns / NANOSECONDS_PER_MILLISECOND,
        total_sample_count=len(in_phase),
        valid_sample_count=len(valid_in_phase),
        missing_sample_count=len(in_phase) - len(valid_in_phase),
        coverage=coverage,
        largest_uncovered_gap_ms=largest_uncovered_ns / NANOSECONDS_PER_MILLISECOND,
        first_sample_slack_ms=first_slack,
        last_sample_slack_ms=last_slack,
        maximum=max(values) if values else None,
        median=statistics.median(values) if values else None,
        time_weighted_mean=time_weighted_mean,
        integral_value_seconds=integral_value_seconds,
        integration_coverage=integration_coverage,
        policy_passed=policy_passed,
    )


# Each reduced NVIDIA field, and how to read it from one sample.
_NVIDIA_FIELD_READERS: dict[str, Callable[[NvidiaSample], float | None]] = {
    "gpu_utilization_percent": lambda sample: sample.gpu_utilization_percent,
    "memory_used_mib": lambda sample: sample.memory_used_mib,
    "temperature_c": lambda sample: sample.temperature_c,
    "power_w": lambda sample: sample.power_w,
    "graphics_clock_mhz": lambda sample: sample.graphics_clock_mhz,
    "memory_clock_mhz": lambda sample: sample.memory_clock_mhz,
}


def _observations(samples: tuple[NvidiaSample, ...], field: str) -> tuple[ScalarObservation, ...]:
    read = _NVIDIA_FIELD_READERS[field]
    # Each NvidiaSample already passed the same checks, so skip validation for every sample and field.
    return tuple(
        ScalarObservation.model_construct(monotonic_ns=sample.monotonic_ns, value=read(sample)) for sample in samples
    )


def reduce_nvidia_phase(
    samples: tuple[NvidiaSample, ...],
    phase: PhaseInterval,
    *,
    interval_ms: int,
    policy: CoveragePolicy = CoveragePolicy(),
) -> NvidiaPhaseSummary:
    """Reduce all normalized NVIDIA fields for one closed phase."""

    def reduce(field: str) -> ScalarTelemetrySummary:
        return reduce_scalar_telemetry(
            field,
            _observations(samples, field),
            phase,
            interval_ms=interval_ms,
            policy=policy,
        )

    return NvidiaPhaseSummary(
        phase=phase,
        interval_ms=interval_ms,
        minimum_coverage=policy.minimum_coverage,
        gpu_utilization=reduce("gpu_utilization_percent"),
        memory_used_mib=reduce("memory_used_mib"),
        temperature_c=reduce("temperature_c"),
        power_w=reduce("power_w"),
        graphics_clock_mhz=reduce("graphics_clock_mhz"),
        memory_clock_mhz=reduce("memory_clock_mhz"),
    )
