"""Exercise post-close telemetry coverage and energy reduction."""

import orjson
import pytest

from agentperf_local.common.units import NANOSECONDS_PER_SECOND
from agentperf_local.telemetry.nvidia import NvidiaSample
from agentperf_local.telemetry.power import PowerPhaseSummary
from agentperf_local.telemetry.reduction import (
    PhaseInterval,
    ScalarObservation,
    reduce_nvidia_phase,
    reduce_scalar_telemetry,
)

PHASE_START_NS = NANOSECONDS_PER_SECOND
PHASE_END_NS = 2 * NANOSECONDS_PER_SECOND
INTERVAL_MS = 200


def _observation(offset_ms: int, value: float | None) -> ScalarObservation:
    return ScalarObservation(monotonic_ns=PHASE_START_NS + offset_ms * 1_000_000, value=value)


def _sample(offset_ms: int, power_w: float = 10.0) -> NvidiaSample:
    return NvidiaSample(
        monotonic_ns=PHASE_START_NS + offset_ms * 1_000_000,
        gpu_utilization_percent=80.0,
        memory_used_mib=14_000.0 + offset_ms,
        temperature_c=65.0,
        power_w=power_w,
        graphics_clock_mhz=2_700.0,
        memory_clock_mhz=14_000.0,
    )


def _phase() -> PhaseInterval:
    return PhaseInterval(phase_id="measured-1", start_ns=PHASE_START_NS, end_ns=PHASE_END_NS)


def test_reduces_full_sample_support_without_extrapolating_energy() -> None:
    observations = tuple(
        _observation(offset_ms, value)
        for offset_ms, value in ((100, 10.0), (300, 20.0), (500, 30.0), (700, 40.0), (900, 50.0))
    )

    summary = reduce_scalar_telemetry("power_w", observations, _phase(), interval_ms=INTERVAL_MS)

    assert summary.coverage == 1.0
    assert summary.largest_uncovered_gap_ms == 0.0
    assert summary.first_sample_slack_ms == 100.0
    assert summary.last_sample_slack_ms == 100.0
    assert summary.maximum == 50.0
    assert summary.median == 30.0
    assert summary.time_weighted_mean == 30.0
    assert summary.integral_value_seconds == pytest.approx(24.0)
    assert summary.integration_coverage == pytest.approx(0.8)
    assert summary.policy_passed


def test_missing_samples_reduce_coverage_instead_of_becoming_zero() -> None:
    observations = tuple(
        _observation(offset_ms, value)
        for offset_ms, value in ((100, 10.0), (300, 20.0), (500, None), (700, 40.0), (900, 50.0))
    )

    summary = reduce_scalar_telemetry("power_w", observations, _phase(), interval_ms=INTERVAL_MS)

    assert summary.total_sample_count == 5
    assert summary.valid_sample_count == 4
    assert summary.missing_sample_count == 1
    assert summary.coverage == pytest.approx(0.8)
    assert summary.largest_uncovered_gap_ms == 200.0
    assert not summary.policy_passed
    assert summary.integral_value_seconds == pytest.approx(24.0)
    assert summary.integration_coverage == pytest.approx(0.8)


def test_reduces_all_nvidia_fields_and_omits_absolute_timestamps() -> None:
    samples = tuple(_sample(offset_ms) for offset_ms in (-100, 100, 300, 500, 700, 900, 1_100))

    summary = reduce_nvidia_phase(samples, _phase(), interval_ms=INTERVAL_MS)
    phase_record = PowerPhaseSummary.from_nvidia(summary).to_json()
    encoded = orjson.dumps(phase_record)

    assert summary.power_w.coverage == 1.0
    assert summary.power_w.integration_coverage == 1.0
    assert summary.power_w.integral_value_seconds == pytest.approx(10.0)
    assert summary.memory_used_mib.maximum == 14_900.0
    assert phase_record["sampled_power_energy_valid"] is True
    assert str(PHASE_START_NS).encode() not in encoded
    assert b"start_ns" not in encoded


def test_rejects_duplicate_observation_timestamps() -> None:
    observations = (_observation(100, 10.0), _observation(100, 20.0))

    with pytest.raises(ValueError, match="unique"):
        reduce_scalar_telemetry("power_w", observations, _phase(), interval_ms=INTERVAL_MS)
