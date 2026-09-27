"""Exercise the independent normalized NVIDIA collector."""

import threading
import time
from pathlib import Path

import orjson
import pytest
from jsonschema import Draft202012Validator

from agentperf_local.common.json_types import JsonObject, normalize_json_object
from agentperf_local.telemetry.nvidia import (
    NORMALIZED_FIELDS,
    CollectorConfig,
    collect_nvidia_telemetry,
    parse_nvidia_csv_line,
)
from tests.fake_executable import write_python_executable
from tests.file_modes import has_mode

SAMPLE_TIME_NS = 1_000_000_000
SCHEMA = Path(__file__).parents[1] / "docs" / "schemas" / "private-nvidia-telemetry-v2.schema.json"
RUN_ID = "8f5b2f2e-4c3a-4d6e-9b1a-2c3d4e5f6a7b"


def _write_executable(path: Path, lines: tuple[str, ...]) -> Path:
    return write_python_executable(path, "".join(f"print({line!r})\n" for line in lines))


def _records(path: Path) -> list[JsonObject]:
    return [normalize_json_object(orjson.loads(line)) for line in path.read_bytes().splitlines()]


def _integer(record: JsonObject, key: str) -> int:
    value = record.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise AssertionError(f"{key} is not an integer")
    return value


def test_parses_supported_and_missing_nvidia_fields() -> None:
    complete = parse_nvidia_csv_line("91, 14322, 67, 418.2, 2745, 14001", monotonic_ns=SAMPLE_TIME_NS)
    partial = parse_nvidia_csv_line("N/A, 14323, 68, [Not Supported], 2700, 13990", monotonic_ns=SAMPLE_TIME_NS)

    assert complete.gpu_utilization_percent == 91.0
    assert complete.power_w == 418.2
    assert complete.missing_value_count == 0
    assert partial.gpu_utilization_percent is None
    assert partial.power_w is None
    assert partial.missing_value_count == 2

    with pytest.raises(ValueError, match="6 fields"):
        parse_nvidia_csv_line("1, 2", monotonic_ns=SAMPLE_TIME_NS)
    with pytest.raises(ValueError, match="exceed"):
        parse_nvidia_csv_line("101, 2, 3, 4, 5, 6", monotonic_ns=SAMPLE_TIME_NS)


def test_collects_normalized_jsonl_without_raw_device_identifiers(tmp_path: Path) -> None:
    executable = _write_executable(
        tmp_path / "fake-nvidia-smi",
        (
            "91, 14322, 67, 418.2, 2745, 14001",
            "N/A, 14323, 68, [Not Supported], 2700, 13990",
        ),
    )
    output_path = tmp_path / "telemetry.jsonl"
    config = CollectorConfig(
        output_path=output_path,
        device_ordinal=0,
        interval_ms=250,
        executable=str(executable),
        run_id=RUN_ID,
    )

    result = collect_nvidia_telemetry(config)
    records = _records(output_path)
    encoded = output_path.read_bytes()
    schema = orjson.loads(SCHEMA.read_bytes())

    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    for record in records:
        validator.validate(record)
    assert result.graceful
    assert result.sample_count == 2
    assert result.missing_value_count == 2
    assert result.process_exit_code == 0
    assert [record["kind"] for record in records] == [
        "telemetry_header",
        "telemetry_sample",
        "telemetry_sample",
        "telemetry_footer",
    ]
    assert records[0]["boundary"] == "nvidia-device"
    assert records[0]["requested_interval_ms"] == 250
    assert records[0]["run_id"] == RUN_ID
    assert _integer(records[1], "monotonic_ns") < _integer(records[2], "monotonic_ns")
    assert records[2]["gpu_utilization_percent"] is None
    assert records[3]["graceful"] is True
    assert str(executable).encode() not in encoded
    assert b"Not Supported" not in encoded
    assert has_mode(output_path, 0o600)

    with pytest.raises(FileExistsError):
        collect_nvidia_telemetry(config)


def test_glitched_lines_become_all_missing_samples_and_collection_continues(tmp_path: Path) -> None:
    executable = _write_executable(
        tmp_path / "glitching-nvidia-smi",
        (
            "91, 14322, 67, 418.2, 2745, 14001",
            "",
            "private raw driver error",
            "[Unknown Error], 14323, 68, [Insufficient Permissions], 2700, 13990",
            "88, 14324, 69, 402.1, 2740, 13995",
        ),
    )
    output_path = tmp_path / "glitching-telemetry.jsonl"

    result = collect_nvidia_telemetry(CollectorConfig(output_path=output_path, executable=str(executable)))
    records = _records(output_path)
    encoded = output_path.read_bytes()

    assert result.graceful
    assert result.failure_code is None
    assert result.sample_count == 4
    assert result.unparseable_line_count == 1
    assert result.missing_value_count == 2 + len(NORMALIZED_FIELDS)
    assert [record["kind"] for record in records] == [
        "telemetry_header",
        "telemetry_sample",
        "telemetry_sample",
        "telemetry_sample",
        "telemetry_sample",
        "telemetry_footer",
    ]
    # The glitched line is an all-missing sample: it lowers coverage without keeping any text.
    assert all(records[2][field] is None for field in NORMALIZED_FIELDS)
    assert records[3]["gpu_utilization_percent"] is None
    assert records[3]["power_w"] is None
    assert records[3]["memory_used_mib"] == 14323.0
    assert records[4]["gpu_utilization_percent"] == 88.0
    assert records[-1]["sample_count"] == 4
    assert records[-1]["unparseable_line_count"] == 1
    assert records[-1]["graceful"] is True
    assert b"private raw driver error" not in encoded
    assert b"Unknown Error" not in encoded


def test_output_without_one_parseable_sample_is_not_graceful(tmp_path: Path) -> None:
    executable = _write_executable(
        tmp_path / "unparseable-nvidia-smi", ("", "private raw driver error", "", "GPU is lost")
    )
    output_path = tmp_path / "unparseable-telemetry.jsonl"

    result = collect_nvidia_telemetry(CollectorConfig(output_path=output_path, executable=str(executable)))

    assert not result.graceful
    assert result.sample_count == 2
    assert result.unparseable_line_count == 2
    assert result.failure_code == "no_samples"
    assert [record["kind"] for record in _records(output_path)] == [
        "telemetry_header",
        "telemetry_sample",
        "telemetry_sample",
        "telemetry_footer",
    ]


def test_a_silent_collector_stops_when_the_stop_event_is_set(tmp_path: Path) -> None:
    executable = write_python_executable(tmp_path / "stalled-nvidia-smi", "import time\ntime.sleep(30)\n")
    output_path = tmp_path / "stalled-telemetry.jsonl"
    stop_event = threading.Event()
    timer = threading.Timer(0.2, stop_event.set)
    timer.start()

    started = time.monotonic()
    result = collect_nvidia_telemetry(
        CollectorConfig(output_path=output_path, executable=str(executable)),
        stop_event=stop_event,
    )
    elapsed = time.monotonic() - started
    timer.cancel()

    assert elapsed < 2.0
    assert not result.graceful
    assert result.sample_count == 0
    assert _records(output_path)[-1]["kind"] == "telemetry_footer"


def test_a_rejected_power_field_is_retried_with_the_legacy_field(tmp_path: Path) -> None:
    executable = write_python_executable(
        tmp_path / "old-driver-nvidia-smi",
        "import sys\n"
        "if any('power.draw.average' in argument for argument in sys.argv):\n"
        "    print('Field \"power.draw.average\" is not a valid field to query.', file=sys.stderr)\n"
        "    sys.exit(6)\n"
        "print('91, 14322, 67, 418.2, 2745, 14001')\n",
    )
    output_path = tmp_path / "legacy-telemetry.jsonl"

    result = collect_nvidia_telemetry(CollectorConfig(output_path=output_path, executable=str(executable)))
    records = _records(output_path)

    assert result.graceful
    assert result.sample_count == 1
    assert result.failure_code is None
    assert records[0]["fields"] == list(NORMALIZED_FIELDS)
    assert records[1]["power_w"] == 418.2
    assert b"not a valid field" not in output_path.read_bytes()


def test_writes_a_closed_failure_footer_when_the_collector_process_fails(tmp_path: Path) -> None:
    executable = write_python_executable(tmp_path / "failing-nvidia-smi", "import sys\nsys.exit(3)\n")
    output_path = tmp_path / "failed-telemetry.jsonl"

    result = collect_nvidia_telemetry(CollectorConfig(output_path=output_path, executable=str(executable)))

    assert not result.graceful
    assert result.failure_code == "collector_process_failed"
    assert result.sample_count == 0
    assert result.process_exit_code == 3
    assert _records(output_path)[-1] == {
        "kind": "telemetry_footer",
        "sample_count": 0,
        "missing_value_count": 0,
        "unparseable_line_count": 0,
        "graceful": False,
        "failure_code": "collector_process_failed",
    }
