"""Collect NVIDIA power around one replay and reduce it after the last close.

Public surface: PhaseClockObserver, NvidiaPowerCollector, nvidia_power_collector,
PowerSummary, PowerPhaseSummary, write_power_summary, load_power_summary.
"""

from __future__ import annotations

import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, Field, FiniteFloat, ValidationInfo, model_validator

from agentperf_local.common.durable_files import NewFile, read_bounded_file, write_new_file
from agentperf_local.common.identity import sha256_bytes, validate_digest, validate_run_id
from agentperf_local.common.json_records import json_record
from agentperf_local.common.json_types import JsonObject, JsonValue, pretty_json_bytes
from agentperf_local.common.models import read_record, require_json_keys
from agentperf_local.common.units import NANOSECONDS_PER_MILLISECOND
from agentperf_local.provenance.hardware import AcceleratorPlatform
from agentperf_local.replay.runner import RunBoundaryEvent, RunFinishedBoundary, RunStartedBoundary
from agentperf_local.telemetry.nvidia import (
    COLLECTOR_ID,
    DEFAULT_INTERVAL_MS,
    NvidiaSample,
    TelemetryFooter,
    parse_nvidia_telemetry,
)
from agentperf_local.telemetry.reduction import NvidiaPhaseSummary, PhaseInterval, reduce_nvidia_phase

POWER_SUMMARY_VERSION = 1
POWER_SUMMARY_KIND = "nvidia_power_summary"
POWER_SUMMARY_FILENAME = "power.json"
TELEMETRY_FILENAME = "telemetry.jsonl"
MEASURED_PHASE_ID = "measured"
COLLECTOR_MODULE = "agentperf_local.telemetry.nvidia"
NVIDIA_SMI_EXECUTABLE = "nvidia-smi"
# nvidia-smi needs a moment to print its first row; the replay waits this long for it.
FIRST_SAMPLE_WAIT_SECONDS = 5.0
FIRST_SAMPLE_POLL_SECONDS = 0.05
COLLECTOR_STOP_TIMEOUT_SECONDS = 10.0
MAX_POWER_SUMMARY_BYTES = 64 * 1024
SAMPLE_MARKER = b'"kind":"telemetry_sample"'
# A power summary file must name these, although Python callers may leave them to their defaults.
_SUMMARY_CONSTANT_KEYS = ("version", "kind", "collector_id", "telemetry_file")


@dataclass(slots=True)
class PhaseClockObserver:
    """Record the monotonic clock at the run's first-request and last-close boundaries.

    The collector child stamps its samples with the same system-wide monotonic clock,
    so these two readings bound the measured phase without any cross-process handshake.
    """

    started_ns: int | None = None
    finished_ns: int | None = None

    def on_boundary(self, event: RunBoundaryEvent) -> None:
        """Stamp the start and finish boundaries; per-turn events cost nothing here."""
        if isinstance(event, RunStartedBoundary):
            self.started_ns = time.monotonic_ns()
        elif isinstance(event, RunFinishedBoundary):
            self.finished_ns = time.monotonic_ns()

    def measured_phase(self) -> PhaseInterval:
        """Return the measured phase, from the first request to the last close."""
        if self.started_ns is None or self.finished_ns is None:
            raise ValueError("the run did not report both its start and finish boundaries")
        return PhaseInterval(
            phase_id=MEASURED_PHASE_ID,
            start_ns=self.started_ns,
            end_ns=max(self.finished_ns, self.started_ns + 1),
        )


class PowerPhaseSummary(BaseModel, frozen=True):
    """Store the phase aggregates a private audit carries; no samples, no timestamps."""

    phase_id: Annotated[str, Field(min_length=1)]
    phase_duration_ms: FiniteFloat
    requested_interval_ms: int
    minimum_coverage: FiniteFloat
    total_sample_count: int
    valid_power_sample_count: int
    power_coverage: FiniteFloat
    power_integration_coverage: FiniteFloat
    largest_uncovered_gap_ms: FiniteFloat
    power_w_maximum: FiniteFloat | None
    power_w_median: FiniteFloat | None
    power_w_time_weighted_mean: FiniteFloat | None
    sampled_power_energy_joules: FiniteFloat | None
    sampled_power_energy_valid: bool
    gpu_utilization_percent_time_weighted_mean: FiniteFloat | None
    memory_used_mib_maximum: FiniteFloat | None
    temperature_c_maximum: FiniteFloat | None
    graphics_clock_mhz_median: FiniteFloat | None
    memory_clock_mhz_median: FiniteFloat | None

    @model_validator(mode="after")
    def check_invariants(self) -> Self:
        """Reject a phase that claims valid energy without an energy value."""
        if self.sampled_power_energy_valid and self.sampled_power_energy_joules is None:
            raise ValueError("a valid energy phase must carry its energy value")
        measures = (
            self.phase_duration_ms,
            self.minimum_coverage,
            self.power_coverage,
            self.power_integration_coverage,
            self.largest_uncovered_gap_ms,
            self.power_w_maximum,
            self.power_w_median,
            self.power_w_time_weighted_mean,
            self.sampled_power_energy_joules,
            self.gpu_utilization_percent_time_weighted_mean,
            self.memory_used_mib_maximum,
            self.temperature_c_maximum,
            self.graphics_clock_mhz_median,
            self.memory_clock_mhz_median,
        )
        if any(value is not None and value < 0 for value in measures):
            raise ValueError("phase measures must be non-negative")
        if self.total_sample_count < 0 or self.valid_power_sample_count < 0:
            raise ValueError("phase sample counts must be non-negative")
        return self

    @classmethod
    def from_nvidia(cls, summary: NvidiaPhaseSummary) -> PowerPhaseSummary:
        """Select the audit fields from one reduced NVIDIA phase."""
        power = summary.power_w
        return cls(
            phase_id=summary.phase.phase_id,
            phase_duration_ms=summary.phase.duration_ns / NANOSECONDS_PER_MILLISECOND,
            requested_interval_ms=summary.interval_ms,
            minimum_coverage=summary.minimum_coverage,
            total_sample_count=power.total_sample_count,
            valid_power_sample_count=power.valid_sample_count,
            power_coverage=power.coverage,
            power_integration_coverage=power.integration_coverage,
            largest_uncovered_gap_ms=power.largest_uncovered_gap_ms,
            power_w_maximum=power.maximum,
            power_w_median=power.median,
            power_w_time_weighted_mean=power.time_weighted_mean,
            sampled_power_energy_joules=power.integral_value_seconds,
            sampled_power_energy_valid=power.policy_passed and power.integration_coverage >= summary.minimum_coverage,
            gpu_utilization_percent_time_weighted_mean=summary.gpu_utilization.time_weighted_mean,
            memory_used_mib_maximum=summary.memory_used_mib.maximum,
            temperature_c_maximum=summary.temperature_c.maximum,
            graphics_clock_mhz_median=summary.graphics_clock_mhz.median,
            memory_clock_mhz_median=summary.memory_clock_mhz.median,
        )

    def to_json(self) -> JsonObject:
        """Return the closed phase record."""
        return json_record(self)


class PowerSummary(BaseModel, frozen=True):
    """Store one run's reduced power evidence and how it was collected."""

    run_id: str
    device_ordinal: int
    requested_interval_ms: int
    telemetry_digest: str | None
    first_sample_before_phase: bool
    collection: TelemetryFooter
    phases: tuple[PowerPhaseSummary, ...]
    version: Literal[1] = POWER_SUMMARY_VERSION
    # The constant keys below are fields so a strict read checks them; the writer emits them as constants.
    # They have defaults so Python callers can leave them out, but a file must spell each one out.
    kind: Literal["nvidia_power_summary"] = POWER_SUMMARY_KIND
    collector_id: Literal["aa-nvidia-smi-v1"] = COLLECTOR_ID
    telemetry_file: Literal["telemetry.jsonl"] = TELEMETRY_FILENAME

    @model_validator(mode="after")
    def check_invariants(self, info: ValidationInfo) -> Self:
        """Validate identifiers and require the measured phase.

        JSON input must carry every constant key, so a reader never assumes a format marker.
        """
        require_json_keys(self, info, _SUMMARY_CONSTANT_KEYS)
        validate_run_id(self.run_id, "run_id")
        if self.telemetry_digest is not None:
            validate_digest(self.telemetry_digest, "telemetry_digest")
        if self.device_ordinal < 0 or self.requested_interval_ms <= 0:
            raise ValueError("device ordinal must be non-negative and the interval positive")
        phase_ids = tuple(phase.phase_id for phase in self.phases)
        if MEASURED_PHASE_ID not in phase_ids or len(set(phase_ids)) != len(phase_ids):
            raise ValueError("power summary phases must be unique and include the measured phase")
        return self

    @property
    def measured(self) -> PowerPhaseSummary:
        """Return the measured phase."""
        return next(phase for phase in self.phases if phase.phase_id == MEASURED_PHASE_ID)

    def to_json(self) -> JsonObject:
        """Return the closed private summary."""
        phases: list[JsonValue] = [phase.to_json() for phase in self.phases]
        return {
            "version": self.version,
            "kind": POWER_SUMMARY_KIND,
            "run_id": self.run_id,
            "collector_id": COLLECTOR_ID,
            "device_ordinal": self.device_ordinal,
            "requested_interval_ms": self.requested_interval_ms,
            "telemetry_file": TELEMETRY_FILENAME,
            "telemetry_digest": self.telemetry_digest,
            "first_sample_before_phase": self.first_sample_before_phase,
            "collection": self.collection.to_json(),
            "phases": phases,
        }


@dataclass(slots=True, kw_only=True)
class NvidiaPowerCollector:
    """Own one collector child for the life of a replay.

    The child is the standalone collector module, so nothing here runs inside the
    process that streams responses. Stopping it after the last close and reducing
    the file afterwards keeps the measured loop untouched.
    """

    output_path: Path
    device_ordinal: int
    run_id: str
    interval_ms: int = DEFAULT_INTERVAL_MS
    executable: str = "nvidia-smi"
    first_sample_seen: bool = field(default=False, init=False)
    _process: subprocess.Popen[bytes] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Validate the collector settings before the child starts."""
        validate_run_id(self.run_id, "run_id")
        if self.device_ordinal < 0 or self.interval_ms <= 0:
            raise ValueError("device ordinal must be non-negative and the interval positive")

    def start(self) -> None:
        """Start the collector child; it writes the telemetry file itself."""
        if self._process is not None:
            raise RuntimeError("the power collector was already started")
        command = (
            sys.executable,
            "-m",
            COLLECTOR_MODULE,
            str(self.output_path),
            f"--device-ordinal={self.device_ordinal}",
            f"--interval-ms={self.interval_ms}",
            f"--run-id={self.run_id}",
            f"--executable={self.executable}",
        )
        # No new session: a terminal interrupt must reach the child too, so it can
        # write its footer instead of outliving the benchmark. Windows can signal a child
        # gracefully only through its own process group; stop() covers the interrupt there.
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        )

    def wait_for_first_sample(self, timeout_seconds: float = FIRST_SAMPLE_WAIT_SECONDS) -> bool:
        """Block until the child wrote one sample, the child exited, or the wait ran out."""
        process = self._process
        if process is None:
            raise RuntimeError("the power collector was not started")
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            try:
                if SAMPLE_MARKER in self.output_path.read_bytes():
                    self.first_sample_seen = True
                    return True
            except OSError:
                pass
            try:
                # Waiting on the child wakes at once if it dies, instead of on the next tick.
                process.wait(timeout=FIRST_SAMPLE_POLL_SECONDS)
            except subprocess.TimeoutExpired:
                continue
            return False
        return False

    def stop(self) -> None:
        """Ask the child to finish and wait for its footer; force it only if it hangs."""
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            # Windows terminate() kills the child before it can write its footer.
            if sys.platform == "win32":
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                process.terminate()
            try:
                process.wait(timeout=COLLECTOR_STOP_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=COLLECTOR_STOP_TIMEOUT_SECONDS)

    def summarize(self, phase: PhaseInterval) -> PowerSummary:
        """Reduce the stopped collection over one closed phase.

        An unreadable file still yields a summary, marked not graceful and with no
        valid energy, so the run directory documents why power evidence is absent.
        """
        if self._process is not None and self._process.poll() is None:
            raise RuntimeError("stop the power collector before summarizing it")
        samples: tuple[NvidiaSample, ...] = ()
        telemetry_digest: str | None = None
        collection = TelemetryFooter(
            sample_count=0,
            missing_value_count=0,
            unparseable_line_count=0,
            graceful=False,
            failure_code="telemetry_unreadable",
        )
        try:
            encoded = self.output_path.read_bytes()
            records = parse_nvidia_telemetry(encoded)
        except (OSError, ValueError):
            records = None
        else:
            telemetry_digest = sha256_bytes(encoded)
            samples = records.samples
            if records.footer is not None:
                collection = records.footer
            else:
                collection = TelemetryFooter(
                    sample_count=len(samples),
                    missing_value_count=sum(sample.missing_value_count for sample in samples),
                    unparseable_line_count=0,
                    graceful=False,
                    failure_code="footer_missing",
                )
        reduced = reduce_nvidia_phase(samples, phase, interval_ms=self.interval_ms)
        return PowerSummary(
            run_id=self.run_id,
            device_ordinal=self.device_ordinal,
            requested_interval_ms=self.interval_ms,
            telemetry_digest=telemetry_digest,
            first_sample_before_phase=self.first_sample_seen,
            collection=collection,
            phases=(PowerPhaseSummary.from_nvidia(reduced),),
        )


def write_power_summary(path: Path, summary: PowerSummary) -> None:
    """Write one private power summary without replacing an existing file."""
    write_new_file(NewFile(path=path, data=pretty_json_bytes(summary.to_json())))


def load_power_summary(path: Path) -> PowerSummary:
    """Read and validate one private power summary."""
    encoded = read_bounded_file(path, MAX_POWER_SUMMARY_BYTES, label="power summary")
    return read_record(PowerSummary, encoded, str(path))


def nvidia_power_collector(
    accelerator_platform: AcceleratorPlatform,
    device_index: int,
    run_id: str,
    telemetry_path: Path,
) -> NvidiaPowerCollector | None:
    """Build the collector when this host can sample GPU power: NVIDIA with nvidia-smi present."""
    if accelerator_platform != "nvidia-cuda":
        return None
    executable = shutil.which(NVIDIA_SMI_EXECUTABLE)
    if executable is None:
        return None
    return NvidiaPowerCollector(
        output_path=telemetry_path,
        device_ordinal=device_index,
        run_id=run_id,
        executable=executable,
    )
