"""Exercise the replay-scoped NVIDIA power collector through its public surface."""

import time
from pathlib import Path

import orjson
import pytest
from jsonschema import Draft202012Validator

from agentperf_local.common.json_types import normalize_json_object
from agentperf_local.common.models import read_object
from agentperf_local.replay.runner import RunFinishedBoundary, RunStartedBoundary
from agentperf_local.telemetry.power import (
    MEASURED_PHASE_ID,
    NvidiaPowerCollector,
    PhaseClockObserver,
    PowerSummary,
    load_power_summary,
    write_power_summary,
)
from tests.fake_nvidia_smi import FAKE_POWER_W, write_looping_nvidia_smi

RUN_ID = "8f5b2f2e-4c3a-4d6e-9b1a-2c3d4e5f6a7b"
# The fake sampler prints every 20 ms; a 200 ms interval keeps coverage full on a loaded machine,
# and the start-up allowance covers two interpreter launches under that load.
SAMPLE_INTERVAL_MS = 200
FIRST_SAMPLE_TIMEOUT_SECONDS = 20.0
MEASURED_SECONDS = 0.4
TELEMETRY_SCHEMA = Path(__file__).parents[1] / "docs" / "schemas" / "private-nvidia-telemetry-v2.schema.json"


def _clock_around_a_sleep(seconds: float) -> PhaseClockObserver:
    clock = PhaseClockObserver()
    clock.on_boundary(RunStartedBoundary(tasks=1, turns=1))
    time.sleep(seconds)
    clock.on_boundary(
        RunFinishedBoundary(completed_tasks=1, tasks=1, completed_turns=1, turns=1, elapsed_ms=0.0, success=True)
    )
    return clock


def test_collector_brackets_the_measured_phase_and_reduces_energy(tmp_path: Path) -> None:
    telemetry_path = tmp_path / "telemetry.jsonl"
    collector = NvidiaPowerCollector(
        output_path=telemetry_path,
        device_ordinal=0,
        run_id=RUN_ID,
        interval_ms=SAMPLE_INTERVAL_MS,
        executable=str(write_looping_nvidia_smi(tmp_path / "fake-nvidia-smi")),
    )

    collector.start()
    assert collector.wait_for_first_sample(timeout_seconds=FIRST_SAMPLE_TIMEOUT_SECONDS)
    clock = _clock_around_a_sleep(MEASURED_SECONDS)
    collector.stop()
    summary = collector.summarize(clock.measured_phase())
    summary_path = tmp_path / "power.json"
    write_power_summary(summary_path, summary)
    encoded = summary_path.read_bytes()
    loaded = load_power_summary(summary_path)
    telemetry_records = [normalize_json_object(orjson.loads(line)) for line in telemetry_path.read_bytes().splitlines()]

    validator = Draft202012Validator(orjson.loads(TELEMETRY_SCHEMA.read_bytes()))
    for record in telemetry_records:
        validator.validate(record)
    assert telemetry_records[0]["run_id"] == RUN_ID
    assert telemetry_records[-1]["graceful"] is True
    assert loaded == summary
    assert summary.first_sample_before_phase
    assert summary.collection.graceful
    assert summary.telemetry_digest is not None
    measured = summary.measured
    assert measured.phase_id == MEASURED_PHASE_ID
    assert measured.sampled_power_energy_valid
    assert measured.power_coverage >= 0.95
    assert measured.power_w_time_weighted_mean == pytest.approx(FAKE_POWER_W)
    assert measured.sampled_power_energy_joules == pytest.approx(
        FAKE_POWER_W * measured.phase_duration_ms / 1000, rel=0.1
    )
    assert b"monotonic_ns" not in encoded
    assert b"start_ns" not in encoded
    assert str(tmp_path).encode() not in encoded
    with pytest.raises(FileExistsError):
        write_power_summary(summary_path, summary)


def test_missing_sampler_still_yields_an_invalid_power_summary(tmp_path: Path) -> None:
    collector = NvidiaPowerCollector(
        output_path=tmp_path / "telemetry.jsonl",
        device_ordinal=0,
        run_id=RUN_ID,
        interval_ms=SAMPLE_INTERVAL_MS,
        executable=str(tmp_path / "no-such-nvidia-smi"),
    )

    collector.start()
    assert not collector.wait_for_first_sample(timeout_seconds=2.0)
    clock = _clock_around_a_sleep(0.05)
    collector.stop()
    summary = collector.summarize(clock.measured_phase())

    assert not summary.first_sample_before_phase
    assert not summary.collection.graceful
    assert summary.collection.failure_code == "collector_start_failed"
    assert not summary.measured.sampled_power_energy_valid
    assert summary.measured.sampled_power_energy_joules is None
    assert summary.measured.power_coverage == 0.0


def test_summarize_refuses_a_running_collector_and_the_clock_needs_both_edges(tmp_path: Path) -> None:
    collector = NvidiaPowerCollector(
        output_path=tmp_path / "telemetry.jsonl",
        device_ordinal=0,
        run_id=RUN_ID,
        executable=str(write_looping_nvidia_smi(tmp_path / "fake-nvidia-smi")),
    )
    clock = PhaseClockObserver()
    clock.on_boundary(RunStartedBoundary(tasks=1, turns=1))

    collector.start()
    try:
        with pytest.raises(ValueError, match="start and finish"):
            clock.measured_phase()
        clock.on_boundary(
            RunFinishedBoundary(completed_tasks=1, tasks=1, completed_turns=1, turns=1, elapsed_ms=0.0, success=True)
        )
        with pytest.raises(RuntimeError, match="stop the power collector"):
            collector.summarize(clock.measured_phase())
    finally:
        collector.stop()


def test_power_summary_rejects_relabeled_or_incomplete_records(tmp_path: Path) -> None:
    collector = NvidiaPowerCollector(
        output_path=tmp_path / "telemetry.jsonl",
        device_ordinal=0,
        run_id=RUN_ID,
        executable=str(tmp_path / "no-such-nvidia-smi"),
    )
    collector.start()
    collector.stop()
    summary = collector.summarize(_clock_around_a_sleep(0.01).measured_phase())
    data = summary.to_json()

    with pytest.raises(ValueError, match="^power: operator: Extra inputs are not permitted$"):
        read_object(PowerSummary, {**data, "operator": "alice"}, "power")
    without_kind = {key: value for key, value in data.items() if key != "kind"}
    with pytest.raises(ValueError, match="^power: kind: Field required$"):
        read_object(PowerSummary, without_kind, "power")
    phases = data["phases"]
    assert isinstance(phases, list)
    phase = normalize_json_object(phases[0])
    with pytest.raises(ValueError, match="^power: power summary phases must be unique and include the measured phase$"):
        read_object(PowerSummary, {**data, "phases": [{**phase, "phase_id": "warmup"}]}, "power")
    with pytest.raises(ValueError, match="^power: a valid energy phase must carry its energy value$"):
        read_object(PowerSummary, {**data, "phases": [{**phase, "sampled_power_energy_valid": True}]}, "power")
