"""Collect, parse, and load normalized NVIDIA telemetry."""

from __future__ import annotations

import argparse
import csv
import math
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from types import FrameType
from typing import IO, BinaryIO, Self

import orjson
from pydantic import BaseModel, model_validator

from agentperf_local.common.argparse_fields import (
    read_integer,
    read_optional_integer,
    read_optional_string,
    read_path,
    read_string,
)
from agentperf_local.common.durable_files import NEW_FILE_OPEN_FLAGS
from agentperf_local.common.identity import validate_run_id
from agentperf_local.common.json_fields import (
    decode_json_object,
    optional_number,
    optional_string,
    require_exact_keys,
    required_boolean,
    required_integer,
)
from agentperf_local.common.json_records import json_field_names, json_record
from agentperf_local.common.json_types import JsonObject
from agentperf_local.common.models import error_text

# Version 2 added the run identifier to the header, the unparseable-line count to the
# footer, and records a glitched sampler line as an all-missing sample.
TELEMETRY_VERSION = 2
# A long run at the default interval stays far below this; the bound stops a runaway file.
COLLECTOR_ID = "aa-nvidia-smi-v1"
TELEMETRY_BOUNDARY = "nvidia-device"
DEFAULT_INTERVAL_MS = 200
PROCESS_SHUTDOWN_TIMEOUT_SECONDS = 5.0
TELEMETRY_FILE_PERMISSIONS = 0o600
# A quiet collector must still let the loop notice a stop request this often.
STOP_CHECK_SECONDS = 0.25
READ_CHUNK_BYTES = 64 * 1024
# Raw driver text never reaches an artifact, so only enough is kept to explain a failed query.
MAX_STDERR_BYTES = 4096
POWER_FIELD = "power.draw.average"
LEGACY_POWER_FIELD = "power.draw"
INVALID_FIELD_MESSAGE = "is not a valid field"
NVIDIA_FIELDS = (
    "utilization.gpu",
    "memory.used",
    "temperature.gpu",
    POWER_FIELD,
    "clocks.current.graphics",
    "clocks.current.memory",
)
# Drivers older than the average-power sensor reject the whole query, so the retry asks for plain power.
LEGACY_NVIDIA_FIELDS = tuple(LEGACY_POWER_FIELD if field == POWER_FIELD else field for field in NVIDIA_FIELDS)
NORMALIZED_FIELDS = (
    "gpu_utilization_percent",
    "memory_used_mib",
    "temperature_c",
    "power_w",
    "graphics_clock_mhz",
    "memory_clock_mhz",
)
MISSING_VALUES = (
    "",
    "n/a",
    "[n/a]",
    "not supported",
    "[not supported]",
    "[unknown error]",
    "[insufficient permissions]",
)


class CollectorConfig(BaseModel, frozen=True):
    """Store one NVIDIA collector policy."""

    output_path: Path
    device_ordinal: int = 0
    interval_ms: int = DEFAULT_INTERVAL_MS
    sample_limit: int | None = None
    executable: str = "nvidia-smi"
    # Set when a benchmark run owns this collection; the standalone command leaves it unset.
    run_id: str | None = None

    @model_validator(mode="after")
    def check_invariants(self) -> Self:
        """Validate collector settings."""
        if self.run_id is not None:
            validate_run_id(self.run_id, "run_id")
        if self.device_ordinal < 0:
            raise ValueError("device_ordinal must be non-negative")
        if self.interval_ms <= 0:
            raise ValueError("interval_ms must be positive")
        if self.sample_limit is not None and self.sample_limit <= 0:
            raise ValueError("sample_limit must be positive")
        if not self.executable:
            raise ValueError("executable must not be empty")
        return self

    def command(self, fields: tuple[str, ...]) -> tuple[str, ...]:
        """Return the exact allowlisted nvidia-smi command."""
        return (
            self.executable,
            f"--id={self.device_ordinal}",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
            f"--loop-ms={self.interval_ms}",
        )


class NvidiaSample(BaseModel, frozen=True):
    """Store one normalized sensor sample."""

    monotonic_ns: int
    gpu_utilization_percent: float | None
    memory_used_mib: float | None
    temperature_c: float | None
    power_w: float | None
    graphics_clock_mhz: float | None
    memory_clock_mhz: float | None

    @model_validator(mode="after")
    def check_invariants(self) -> Self:
        """Reject invalid normalized values."""
        if self.monotonic_ns <= 0:
            raise ValueError("monotonic_ns must be positive")
        values = (
            self.gpu_utilization_percent,
            self.memory_used_mib,
            self.temperature_c,
            self.power_w,
            self.graphics_clock_mhz,
            self.memory_clock_mhz,
        )
        if any(value is not None and (not math.isfinite(value) or value < 0) for value in values):
            raise ValueError("telemetry values must be finite and non-negative")
        if self.gpu_utilization_percent is not None and self.gpu_utilization_percent > 100:
            raise ValueError("GPU utilization must not exceed 100 percent")
        return self

    @classmethod
    def all_missing(cls, monotonic_ns: int) -> NvidiaSample:
        """Return the record written for a sampler line that could not be parsed."""
        return cls(
            monotonic_ns=monotonic_ns,
            gpu_utilization_percent=None,
            memory_used_mib=None,
            temperature_c=None,
            power_w=None,
            graphics_clock_mhz=None,
            memory_clock_mhz=None,
        )

    @classmethod
    def from_json(cls, data: JsonObject, source: str) -> NvidiaSample:
        """Read one private sample record."""
        values = {field: optional_number(data, field, source) for field in NORMALIZED_FIELDS}
        return cls(
            monotonic_ns=required_integer(data, "monotonic_ns", source),
            gpu_utilization_percent=values["gpu_utilization_percent"],
            memory_used_mib=values["memory_used_mib"],
            temperature_c=values["temperature_c"],
            power_w=values["power_w"],
            graphics_clock_mhz=values["graphics_clock_mhz"],
            memory_clock_mhz=values["memory_clock_mhz"],
        )

    @property
    def missing_value_count(self) -> int:
        """Return the number of unsupported fields in this sample."""
        values = (
            self.gpu_utilization_percent,
            self.memory_used_mib,
            self.temperature_c,
            self.power_w,
            self.graphics_clock_mhz,
            self.memory_clock_mhz,
        )
        return sum(value is None for value in values)

    def to_json(self) -> JsonObject:
        """Return one private normalized sample."""
        return {
            "kind": "telemetry_sample",
            "monotonic_ns": self.monotonic_ns,
            "gpu_utilization_percent": self.gpu_utilization_percent,
            "memory_used_mib": self.memory_used_mib,
            "temperature_c": self.temperature_c,
            "power_w": self.power_w,
            "graphics_clock_mhz": self.graphics_clock_mhz,
            "memory_clock_mhz": self.memory_clock_mhz,
        }


class CollectionResult(BaseModel, frozen=True):
    """Describe collector completion without raw error text."""

    output_path: Path
    sample_count: int
    missing_value_count: int
    unparseable_line_count: int
    graceful: bool
    failure_code: str | None
    process_exit_code: int | None

    def to_json(self) -> JsonObject:
        """Return local collector status."""
        return {
            "output": str(self.output_path),
            "sample_count": self.sample_count,
            "missing_value_count": self.missing_value_count,
            "unparseable_line_count": self.unparseable_line_count,
            "graceful": self.graceful,
            "failure_code": self.failure_code,
            "process_exit_code": self.process_exit_code,
        }


def _optional_number(value: str) -> float | None:
    normalized = value.strip()
    if normalized.lower() in MISSING_VALUES:
        return None
    try:
        parsed = float(normalized)
    except ValueError as error:
        raise ValueError("nvidia-smi returned a non-numeric field") from error
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError("nvidia-smi returned an invalid numeric field")
    return parsed


def parse_nvidia_csv_line(line: str, *, monotonic_ns: int) -> NvidiaSample:
    """Normalize one exact nvidia-smi CSV row."""
    rows = list(csv.reader((line,)))
    if len(rows) != 1 or len(rows[0]) != len(NVIDIA_FIELDS):
        raise ValueError(f"nvidia-smi sample must contain {len(NVIDIA_FIELDS)} fields")
    values = tuple(_optional_number(value) for value in rows[0])
    return NvidiaSample(
        monotonic_ns=monotonic_ns,
        gpu_utilization_percent=values[0],
        memory_used_mib=values[1],
        temperature_c=values[2],
        power_w=values[3],
        graphics_clock_mhz=values[4],
        memory_clock_mhz=values[5],
    )


def _write_line(output: BinaryIO, value: JsonObject) -> None:
    encoded = orjson.dumps(value, option=orjson.OPT_APPEND_NEWLINE)
    output.write(encoded)


def _header(config: CollectorConfig) -> JsonObject:
    return {
        "kind": "telemetry_header",
        "version": TELEMETRY_VERSION,
        "collector_id": COLLECTOR_ID,
        "boundary": TELEMETRY_BOUNDARY,
        "run_id": config.run_id,
        "device_ordinal": config.device_ordinal,
        "requested_interval_ms": config.interval_ms,
        "fields": list(NORMALIZED_FIELDS),
    }


def _footer(
    *,
    sample_count: int,
    missing_value_count: int,
    unparseable_line_count: int,
    graceful: bool,
    failure_code: str | None,
) -> JsonObject:
    return {
        "kind": "telemetry_footer",
        "sample_count": sample_count,
        "missing_value_count": missing_value_count,
        "unparseable_line_count": unparseable_line_count,
        "graceful": graceful,
        "failure_code": failure_code,
    }


class TelemetryFooter(BaseModel, frozen=True):
    """Store how one collection ended, without any raw driver text."""

    sample_count: int
    missing_value_count: int
    unparseable_line_count: int
    graceful: bool
    failure_code: str | None

    def to_json(self) -> JsonObject:
        """Return the closed completion record."""
        return json_record(self)

    @classmethod
    def from_json(cls, data: JsonObject, source: str) -> TelemetryFooter:
        """Parse one strict completion record."""
        require_exact_keys(data, json_field_names(cls), source)
        return cls(
            sample_count=required_integer(data, "sample_count", source),
            missing_value_count=required_integer(data, "missing_value_count", source),
            unparseable_line_count=required_integer(data, "unparseable_line_count", source),
            graceful=required_boolean(data, "graceful", source),
            failure_code=optional_string(data, "failure_code", source),
        )


class TelemetryRecords(BaseModel, frozen=True):
    """Hold one telemetry file: its samples, and the footer when the collector finished."""

    samples: tuple[NvidiaSample, ...]
    footer: TelemetryFooter | None


def parse_nvidia_telemetry(encoded: bytes) -> TelemetryRecords:
    """Parse one private telemetry file written by this collector.

    A missing footer is reported as None rather than an error: a collector that was
    killed hard leaves samples worth reducing, and the caller records the gap.
    """
    lines = [line for line in encoded.splitlines() if line]
    if not lines:
        raise ValueError("telemetry file is empty")
    header_data = decode_json_object(lines[0], "telemetry line 0 is not a JSON object")
    if header_data.get("kind") != "telemetry_header" or header_data.get("version") != TELEMETRY_VERSION:
        raise ValueError("telemetry header is missing or its version is not supported")
    if header_data.get("collector_id") != COLLECTOR_ID or header_data.get("fields") != list(NORMALIZED_FIELDS):
        raise ValueError("telemetry header does not describe this collector")
    samples: list[NvidiaSample] = []
    footer: TelemetryFooter | None = None
    for index, line in enumerate(lines):
        if index == 0:
            continue
        data = decode_json_object(line, f"telemetry line {index} is not a JSON object")
        kind = data.get("kind")
        if kind == "telemetry_sample" and footer is None:
            samples.append(NvidiaSample.from_json(data, f"telemetry line {index}"))
            continue
        if kind == "telemetry_footer" and footer is None and index == len(lines) - 1:
            fields = {key: value for key, value in data.items() if key != "kind"}
            footer = TelemetryFooter.from_json(fields, "telemetry_footer")
            continue
        raise ValueError(f"telemetry line {index} is out of place")
    if footer is not None and footer.sample_count != len(samples):
        raise ValueError("telemetry footer sample count does not match the sample records")
    return TelemetryRecords(samples=tuple(samples), footer=footer)


def _finish_process(process: subprocess.Popen[bytes], intentional_stop: bool) -> int:
    if process.poll() is None and intentional_stop:
        process.terminate()
    try:
        return process.wait(timeout=PROCESS_SHUTDOWN_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.wait(timeout=PROCESS_SHUTDOWN_TIMEOUT_SECONDS)


def _stopped(stop_event: threading.Event | None) -> bool:
    return stop_event is not None and stop_event.is_set()


def _read_until_closed(stream: IO[bytes], deliver: Callable[[bytes], None]) -> None:
    """Hand every chunk of one pipe to deliver, then an empty chunk once the pipe closes.

    The reader closes its own pipe. A grandchild of a stopped collector can hold the pipe
    open, and closing it from another thread while this read is blocked is not safe.
    """
    with stream:
        try:
            while chunk := stream.read(READ_CHUNK_BYTES):
                deliver(chunk)
        finally:
            deliver(b"")


def _start_pipe_readers(
    *,
    stdout: IO[bytes],
    stderr: IO[bytes],
    stdout_chunks: queue.Queue[bytes],
    stderr_buffer: bytearray,
) -> tuple[threading.Thread, ...]:
    """Drain both collector pipes on threads.

    Windows cannot wait on a pipe with select, so blocking reads run on threads on every
    platform. Stderr is drained too, so a chatty driver cannot fill its pipe and stall.
    """

    def keep_stderr(chunk: bytes) -> None:
        stderr_buffer.extend(chunk[: MAX_STDERR_BYTES - len(stderr_buffer)])

    readers = (
        threading.Thread(target=_read_until_closed, args=(stdout, stdout_chunks.put), daemon=True),
        threading.Thread(target=_read_until_closed, args=(stderr, keep_stderr), daemon=True),
    )
    for reader in readers:
        reader.start()
    return readers


def _stream_lines(stdout_chunks: queue.Queue[bytes], stop_event: threading.Event | None) -> Iterator[str]:
    """Yield collector output lines, waking often enough to honor a stop request."""
    pending = bytearray()
    while True:
        if _stopped(stop_event):
            return
        try:
            chunk = stdout_chunks.get(timeout=STOP_CHECK_SECONDS)
        except queue.Empty:
            continue
        if not chunk:
            break
        pending.extend(chunk)
        while (break_index := pending.find(b"\n")) >= 0:
            line = bytes(pending[:break_index])
            del pending[: break_index + 1]
            yield line.decode("utf-8", errors="replace")
    if pending:
        yield bytes(pending).decode("utf-8", errors="replace")


def _rejected_query(stderr_text: str) -> bool:
    """Report whether the driver refused the queried fields outright."""
    lowered = stderr_text.lower()
    return POWER_FIELD in lowered or INVALID_FIELD_MESSAGE in lowered


class _AttemptOutcome(BaseModel, frozen=True):
    """Store what one collector process produced."""

    sample_count: int = 0
    missing_value_count: int = 0
    unparseable_line_count: int = 0
    failure_code: str | None = None
    process_exit_code: int | None = None
    rejected_query: bool = False


def _collect_once(
    *,
    config: CollectorConfig,
    fields: tuple[str, ...],
    output: BinaryIO,
    stop_event: threading.Event | None,
) -> _AttemptOutcome:
    """Run one collector process and write every parsed sample."""
    try:
        # On Windows the owner stops this collector with a break event sent to its whole process
        # group. The sampler gets its own group, so it keeps sampling until the collector stops it,
        # as on POSIX. That last sample after the phase ends is what closes the energy integral.
        process = subprocess.Popen(
            config.command(fields),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        )
    except OSError:
        return _AttemptOutcome(failure_code="collector_start_failed")
    stdout = process.stdout
    stderr = process.stderr
    if stdout is None or stderr is None:
        return _AttemptOutcome(
            failure_code="collector_pipe_failed",
            process_exit_code=_finish_process(process, intentional_stop=True),
        )
    sample_count = 0
    missing_value_count = 0
    unparseable_line_count = 0
    last_monotonic_ns = 0
    reached_limit = False
    stderr_buffer = bytearray()
    stdout_chunks: queue.Queue[bytes] = queue.Queue()
    readers = _start_pipe_readers(
        stdout=stdout,
        stderr=stderr,
        stdout_chunks=stdout_chunks,
        stderr_buffer=stderr_buffer,
    )
    try:
        for line in _stream_lines(stdout_chunks, stop_event):
            if not line or line.isspace():
                # A quiet loop iteration is not a reading, so it must never become a sample.
                continue
            monotonic_ns = max(time.monotonic_ns(), last_monotonic_ns + 1)
            try:
                sample = parse_nvidia_csv_line(line, monotonic_ns=monotonic_ns)
            except ValueError:
                # A glitched line becomes an all-missing sample: it lowers coverage for
                # its own interval and nothing else. The raw text never reaches the file.
                unparseable_line_count += 1
                sample = NvidiaSample.all_missing(monotonic_ns)
            last_monotonic_ns = monotonic_ns
            _write_line(output, sample.to_json())
            sample_count += 1
            missing_value_count += sample.missing_value_count
            if config.sample_limit is not None and sample_count >= config.sample_limit:
                reached_limit = True
                break
        intentional_stop = reached_limit or _stopped(stop_event)
        process_exit_code = _finish_process(process, intentional_stop)
    except BaseException:
        # A failure here must not leave the collector process running behind the benchmark.
        _finish_process(process, intentional_stop=True)
        raise
    finally:
        # The child has exited, so its pipes reach end of file at once unless a grandchild
        # still holds them. A reader left waiting on such a grandchild must not hold up the run.
        for reader in readers:
            reader.join(timeout=STOP_CHECK_SECONDS)
    failure_code = None if intentional_stop or process_exit_code == 0 else "collector_process_failed"
    stderr_text = bytes(stderr_buffer).decode("utf-8", errors="replace")
    return _AttemptOutcome(
        sample_count=sample_count,
        missing_value_count=missing_value_count,
        unparseable_line_count=unparseable_line_count,
        failure_code=failure_code,
        process_exit_code=process_exit_code,
        rejected_query=sample_count == 0 and process_exit_code != 0 and _rejected_query(stderr_text),
    )


def collect_nvidia_telemetry(
    config: CollectorConfig,
    *,
    stop_event: threading.Event | None = None,
) -> CollectionResult:
    """Collect normalized rows outside the model response process."""
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(config.output_path, NEW_FILE_OPEN_FLAGS, TELEMETRY_FILE_PERMISSIONS)
    with os.fdopen(descriptor, "wb", buffering=0) as output:
        _write_line(output, _header(config))
        outcome = _collect_once(config=config, fields=NVIDIA_FIELDS, output=output, stop_event=stop_event)
        if outcome.rejected_query and not _stopped(stop_event):
            outcome = _collect_once(config=config, fields=LEGACY_NVIDIA_FIELDS, output=output, stop_event=stop_event)
        failure_code = outcome.failure_code
        # The footer codes are a closed schema enum, so a collection without one parsed
        # sample reports as the no-sample failure even when glitch records were written.
        parsed_sample_count = outcome.sample_count - outcome.unparseable_line_count
        if parsed_sample_count == 0 and failure_code is None:
            failure_code = "no_samples"
        graceful = failure_code is None and parsed_sample_count > 0
        _write_line(
            output,
            _footer(
                sample_count=outcome.sample_count,
                missing_value_count=outcome.missing_value_count,
                unparseable_line_count=outcome.unparseable_line_count,
                graceful=graceful,
                failure_code=failure_code,
            ),
        )
    return CollectionResult(
        output_path=config.output_path,
        sample_count=outcome.sample_count,
        missing_value_count=outcome.missing_value_count,
        unparseable_line_count=outcome.unparseable_line_count,
        graceful=graceful,
        failure_code=failure_code,
        process_exit_code=outcome.process_exit_code,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone collector command."""
    parser = argparse.ArgumentParser(description="Collect private normalized NVIDIA telemetry JSONL.")
    parser.add_argument("output", type=Path)
    parser.add_argument("--device-ordinal", type=int, default=0)
    parser.add_argument("--interval-ms", type=int, default=DEFAULT_INTERVAL_MS)
    parser.add_argument("--sample-count", type=int)
    parser.add_argument("--executable", default="nvidia-smi")
    parser.add_argument("--run-id", default=None, help="benchmark run that owns this collection")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the independent NVIDIA collector."""
    namespace = build_parser().parse_args(argv)
    stop_event = threading.Event()

    def request_stop(signum: int, frame: FrameType | None) -> None:
        del signum, frame
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    # Windows cannot deliver SIGTERM to a handler; the owner sends CTRL_BREAK_EVENT instead.
    if sys.platform == "win32":
        signal.signal(signal.SIGBREAK, request_stop)
    try:
        result = collect_nvidia_telemetry(
            CollectorConfig(
                output_path=read_path(namespace, "output"),
                device_ordinal=read_integer(namespace, "device_ordinal"),
                interval_ms=read_integer(namespace, "interval_ms"),
                sample_limit=read_optional_integer(namespace, "sample_count"),
                executable=read_string(namespace, "executable"),
                run_id=read_optional_string(namespace, "run_id"),
            ),
            stop_event=stop_event,
        )
    except Exception as error:
        print(f"error: {error_text(error)}", file=sys.stderr)
        return 1
    print(orjson.dumps(result.to_json(), option=orjson.OPT_INDENT_2).decode("utf-8"))
    return 0 if result.graceful else 1


if __name__ == "__main__":
    raise SystemExit(main())
