"""Exercise recording conversion through its public file and object surfaces."""

from __future__ import annotations

import gzip
from pathlib import Path

import orjson
import pytest

from agentperf_local.common.json_types import JsonObject
from agentperf_local.workload.recording import (
    convert_manifest_to_dir,
    convert_one_recording_to_dir,
    convert_recording,
    read_recording_json,
)
from agentperf_local.workload.schema import FORMAT_VERSION, MessageSource, load_manifest, load_trace, parse_json_object
from tests.file_modes import has_mode

FIXTURES = Path(__file__).parent / "fixtures" / "recording"
RECORDING = FIXTURES / "recordings" / "demo.json"


@pytest.mark.parametrize(
    ("message_source", "expected_content"),
    [("provider-request", "wire prompt"), ("request-messages", "logical prompt")],
)
def test_convert_recording_preserves_order_tools_usage_and_metadata(
    message_source: MessageSource,
    expected_content: str,
) -> None:
    recording = read_recording_json(RECORDING)

    rows = convert_recording(
        recording,
        recording_path=Path("recordings/demo.json"),
        family="pinchbench-sanitized",
        adapter="pinchbench",
        message_source=message_source,
        include_tool_outputs=True,
    )

    assert [row.conversation_idx for row in rows] == [0, 1]
    assert [row.turn_id for row in rows] == ["recording_task:0000", "recording_task:0001"]
    assert rows[0].messages[0].content == expected_content
    assert rows[0].messages[0].to_dict().get("provider_internal") is None
    if message_source == "provider-request":
        assert [message.role for message in rows[1].messages] == ["assistant", "tool", "user"]
        assert "content" not in rows[1].messages[0].to_dict()
    expected_function: JsonObject = {
        "name": "bash",
        "description": "Run a shell command",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    }
    assert rows[0].tools[0].definition["function"] == expected_function
    assert (
        rows[0].target_output_tokens,
        rows[0].recorded_prompt_tokens,
        rows[0].recorded_completion_tokens,
        rows[0].recorded_total_tokens,
    ) == (5, 10, 5, 15)
    assert rows[0].recorded_model_duration_ms == 100.0
    assert rows[0].simulated_tool_delay_ms_after == 25.5
    assert rows[1].simulated_tool_delay_ms_after == 0.0
    assert len(rows[0].recorded_tool_calls_after) == 1
    tool_call = rows[0].recorded_tool_calls_after[0]
    assert (tool_call.tool_name, tool_call.tool_call_id, tool_call.command) == ("bash", "call_1", "pwd")
    assert (tool_call.action_index, tool_call.step, tool_call.recorded_returncode) == (0, 1, 0)
    assert tool_call.output == {"output": "/workspace\n", "returncode": 0}
    assert rows[0].source is not None
    assert rows[0].source.model_call_index == 0
    assert rows[0].source.message_source == message_source
    if message_source == "provider-request":
        expected_rows = [
            parse_json_object(line, "expected_trace.jsonl")
            for line in (FIXTURES / "expected_trace.jsonl").read_bytes().splitlines()
        ]
        assert [row.to_dict() for row in rows] == expected_rows


def test_convert_manifest_writes_versioned_typed_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "converted"

    converted = convert_manifest_to_dir(FIXTURES / "manifest.json", output_dir)
    loaded = load_manifest(output_dir / "manifest.json")
    rows = load_trace(output_dir / loaded.tasks[0].trace)

    assert loaded == converted
    assert loaded.version == FORMAT_VERSION
    assert loaded.name == "sanitized-demo-single-user"
    assert [task.task_id for task in loaded.tasks] == ["manifest_task"]
    task = loaded.tasks[0]
    assert (task.model_calls, task.tool_calls, task.total_recorded_tool_delay_ms) == (2, 1, 25.5)
    # The larger recorded prompt (20) plus its target output (7) is the replay's context demand.
    assert task.required_context_tokens == 27
    assert loaded.required_context_tokens == 27
    assert task.tool_environment is not None
    assert task.tool_environment.to_dict() == {
        "type": "docker",
        "image": "agentperf-pinchbench:latest",
        "cwd": "/workspace",
        "network": "none",
        "interpreter": ["bash", "-c"],
        "workspace_mount": True,
    }
    assert [row.version for row in rows] == [FORMAT_VERSION, FORMAT_VERSION]
    assert [row.task_id for row in rows] == ["manifest_task", "manifest_task"]
    assert [row.turn_id for row in rows] == ["manifest_task:0000", "manifest_task:0001"]
    assert rows[0].recorded_tool_calls_after[0].output is None
    assert has_mode(output_dir / "manifest.json", 0o600)
    assert has_mode(output_dir / task.trace, 0o600)


def test_convert_gzipped_recording_with_outputs(tmp_path: Path) -> None:
    recording_path = tmp_path / "demo.json.gz"
    with gzip.open(recording_path, "wb") as handle:
        handle.write(RECORDING.read_bytes())

    manifest, trace_path = convert_one_recording_to_dir(
        recording_path,
        tmp_path / "converted",
        family="swe-sanitized",
        adapter="swebench",
        include_tool_outputs=True,
    )
    rows = load_trace(trace_path)

    assert manifest.tasks[0].source_recording == recording_path
    assert manifest.tasks[0].tool_environment is not None
    assert manifest.tasks[0].tool_environment.to_dict() == {
        "type": "docker",
        "image": "example.invalid/swe/sanitized-task:latest",
        "cwd": "/testbed",
        "network": "none",
        "interpreter": ["bash", "-c"],
        "workspace_mount": False,
    }
    assert rows[0].recorded_tool_calls_after[0].output == {"output": "/workspace\n", "returncode": 0}


@pytest.mark.parametrize(
    "document",
    [
        {"name": "bad", "source": "test", "mode": "single_user_agentic_replay", "version": 2, "tasks": []},
        {"name": "bad", "source": "test", "mode": "other", "version": FORMAT_VERSION, "tasks": []},
    ],
)
def test_manifest_loader_rejects_incompatible_formats(tmp_path: Path, document: JsonObject) -> None:
    path = tmp_path / "manifest.json"
    path.write_bytes(orjson.dumps(document))

    with pytest.raises(ValueError):
        load_manifest(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("conversation_idx", -1),
        ("target_output_tokens", -1),
        ("max_output_tokens", -1),
        ("max_output_tokens", 0),
        ("recorded_model_duration_ms", -0.1),
        ("simulated_tool_delay_ms_after", -0.1),
    ],
)
def test_trace_loader_rejects_invalid_quantities(tmp_path: Path, field: str, value: int | float) -> None:
    row: JsonObject = {
        "version": FORMAT_VERSION,
        "turn_id": "task:0000",
        "task_id": "task",
        "conversation_id": "task",
        "conversation_idx": 0,
        "messages": [{"role": "user", "content": "hello"}],
        "simulated_tool_delay_ms_after": 0.0,
    }
    row[field] = value
    path = tmp_path / "trace.jsonl"
    path.write_bytes(orjson.dumps(row, option=orjson.OPT_APPEND_NEWLINE))

    with pytest.raises(ValueError, match=field):
        load_trace(path)


@pytest.mark.parametrize(
    ("environment", "errors"),
    [
        (
            {},
            [
                "tasks.0.tool_environment.image: Field required",
                "tasks.0.tool_environment.cwd: Field required",
                "tasks.0.tool_environment.interpreter: Field required",
                "tasks.0.tool_environment.workspace_mount: Field required",
            ],
        ),
        (
            {"image": "image", "cwd": "/w", "network": "none", "interpreter": ["bash"], "workspace_mount": False},
            ["type: Field required"],
        ),
    ],
)
def test_manifest_loader_rejects_incomplete_tool_environment(
    tmp_path: Path, environment: JsonObject, errors: list[str]
) -> None:
    document: JsonObject = {
        "name": "bad-environment",
        "source": "test",
        "mode": "single_user_agentic_replay",
        "version": FORMAT_VERSION,
        "tasks": [
            {
                "task_id": "task",
                "trace": "traces/task.jsonl",
                "source_recording": "task.json",
                "model_calls": 1,
                "tool_calls": 0,
                "total_recorded_tool_delay_ms": 0.0,
                "tool_environment": environment,
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_bytes(orjson.dumps(document))

    with pytest.raises(ValueError) as caught:
        load_manifest(path)
    assert str(caught.value) == "\n".join(f"{path}: {error}" for error in errors)


@pytest.mark.parametrize("task_id", ["../escape", "nested/task", r"nested\\task"])
def test_conversion_rejects_task_ids_that_are_not_filenames(tmp_path: Path, task_id: str) -> None:
    manifest_path = tmp_path / "unsafe.json"
    manifest_path.write_bytes(
        orjson.dumps(
            {
                "name": "unsafe",
                "tasks": [{"task_id": task_id, "recording": str(RECORDING)}],
            }
        )
    )

    with pytest.raises(ValueError):
        convert_manifest_to_dir(manifest_path, tmp_path / "converted")
