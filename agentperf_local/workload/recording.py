"""Convert agent recordings into replay manifests and traces."""

from __future__ import annotations

import gzip
from pathlib import Path

from agentperf_local.common.json_fields import (
    optional_integer,
    optional_non_negative_number,
    optional_object,
    optional_text,
    required_objects,
)
from agentperf_local.common.json_types import JsonObject, JsonValue
from agentperf_local.workload.schema import (
    DockerToolEnvironmentSpec,
    ManifestTask,
    MessageSource,
    RecordedToolCall,
    ReplayManifest,
    RequestMessage,
    ToolDefinition,
    TraceRow,
    TraceSource,
    parse_json_object,
    write_manifest,
    write_trace,
)

DEFAULT_PINCHBENCH_IMAGE = "agentperf-pinchbench:latest"
DEFAULT_SWEBENCH_CWD = "/testbed"
PINCHBENCH_CWD = "/workspace"


def read_recording_json(path: Path) -> JsonObject:
    """Read a recording JSON or gzipped JSON object."""
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as handle:
            data = handle.read()
    else:
        data = path.read_bytes()
    return parse_json_object(data, path)


def _duration_ms(data: JsonObject, key: str, source: str) -> float:
    """Read a recorded duration in milliseconds, treating an absent value as zero."""
    return optional_non_negative_number(data, key, source) or 0.0


def _events(recording: JsonObject) -> list[JsonObject]:
    return required_objects(recording, "events", "recording")


def _model_call_events(recording: JsonObject) -> list[JsonObject]:
    events = [event for event in _events(recording) if event.get("type") == "model_call"]
    if not events:
        raise ValueError("recording contains no model_call events")
    for index, event in enumerate(events, start=1):
        if not isinstance(event.get("response_message"), dict):
            raise ValueError(f"model_call {index} has no response_message object")
        request_messages = event.get("request_messages")
        if not isinstance(request_messages, list) or not request_messages:
            raise ValueError(f"model_call {index} has no request_messages")
    return events


def _sanitize_message(message: JsonObject, source: str) -> RequestMessage:
    role = message.get("role")
    if not isinstance(role, str) or not role:
        raise ValueError(f"{source}.role must be a non-empty string")
    raw_tool_calls = message.get("tool_calls")
    tool_calls = None if raw_tool_calls is None else required_objects(message, "tool_calls", source)
    return RequestMessage(
        role=role,
        content=message.get("content"),
        content_present="content" in message,
        name=optional_text(message, "name", source),
        tool_calls=tool_calls,
        tool_call_id=optional_text(message, "tool_call_id", source),
    )


def _message_field(event: JsonObject, source: MessageSource) -> tuple[JsonObject, str]:
    """Return the object and key holding the request messages for one model call."""
    if source == "provider-request":
        provider_request = optional_object(event, "provider_request", "model_call")
        if provider_request is not None and provider_request.get("messages"):
            return provider_request, "messages"
    return event, "request_messages"


def _messages(event: JsonObject, source: MessageSource) -> list[RequestMessage]:
    holder, key = _message_field(event, source)
    messages = required_objects(holder, key, "model_call")
    if not messages:
        raise ValueError("model_call messages must not be empty")
    return [_sanitize_message(message, f"model_call messages[{index}]") for index, message in enumerate(messages)]


def _tools(event: JsonObject) -> list[ToolDefinition]:
    provider_request = optional_object(event, "provider_request", "model_call")
    if provider_request is None or provider_request.get("tools") is None:
        return []
    return [
        ToolDefinition(definition=definition)
        for definition in required_objects(provider_request, "tools", "model_call.provider_request")
    ]


def _nested_object(data: JsonObject, keys: tuple[str, ...]) -> JsonObject | None:
    current = data
    for key in keys:
        value = current.get(key)
        if value is None:
            return None
        if not isinstance(value, dict):
            return None
        current = value
    return current


def _usage(event: JsonObject) -> tuple[int | None, int | None, int | None]:
    usage = _nested_object(event, ("response_message", "extra", "response", "usage"))
    if usage is None:
        return None, None, None
    return (
        optional_integer(usage, "prompt_tokens", "usage"),
        optional_integer(usage, "completion_tokens", "usage"),
        optional_integer(usage, "total_tokens", "usage"),
    )


def _wrapped_object(value: JsonValue | None) -> JsonObject | None:
    if value is None:
        return None
    return value if isinstance(value, dict) else {"value": value}


def _tool_name(action: JsonObject | None) -> str | None:
    if action is None:
        return None
    if "command" in action:
        return "bash"
    name = action.get("name")
    return name if isinstance(name, str) else None


def _tool_call_id(action: JsonObject | None) -> str | None:
    if action is None:
        return None
    value = action.get("tool_call_id")
    if value is None:
        return None
    if isinstance(value, str | int | float | bool):
        return str(value)
    raise ValueError("tool_call.action.tool_call_id must be a scalar")


def _tool_call(event: JsonObject, include_outputs: bool) -> RecordedToolCall:
    action = _wrapped_object(event.get("action"))
    raw_output = event.get("output")
    event_output = raw_output if isinstance(raw_output, dict) else None
    recorded_returncode = (
        optional_integer(event_output, "returncode", "tool_call.output") if event_output is not None else None
    )
    return RecordedToolCall(
        duration_ms=_duration_ms(event, "duration_ms", "tool_call"),
        action_index=optional_integer(event, "action_index", "tool_call"),
        step=optional_integer(event, "step", "tool_call"),
        recorded_returncode=recorded_returncode,
        tool_call_id=_tool_call_id(action),
        tool_name=_tool_name(action),
        action=action,
        output=_wrapped_object(raw_output) if include_outputs else None,
    )


def _tool_calls_after_model_calls(recording: JsonObject, include_outputs: bool) -> list[list[RecordedToolCall]]:
    grouped: list[list[RecordedToolCall]] = []
    current: list[RecordedToolCall] | None = None
    for event in _events(recording):
        event_type = event.get("type")
        if event_type == "model_call":
            current = []
            grouped.append(current)
        elif event_type == "tool_call" and current is not None:
            current.append(_tool_call(event, include_outputs))
    return grouped


def _task_id(recording: JsonObject, recording_path: Path) -> str:
    metadata = optional_object(recording, "metadata", "recording") or {}
    value = metadata.get("task_id")
    if value is None:
        instance = optional_object(metadata, "instance", "recording.metadata") or {}
        value = instance.get("instance_id")
    if value is not None:
        if not isinstance(value, str | int | float | bool):
            raise ValueError("recording task ID must be a scalar")
        return str(value)
    return recording_path.name.removesuffix(".json.gz").removesuffix(".json")


def _resolve_recording_path(manifest_path: Path, recording: str) -> Path:
    recording_path = Path(recording)
    if recording_path.is_absolute():
        return recording_path
    candidates = [manifest_path.parent / recording_path, *(parent / recording_path for parent in manifest_path.parents)]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"recording listed in {manifest_path} was not found: {recording}")


def _trace_path(output_dir: Path, task_id: str) -> Path:
    if not task_id or "/" in task_id or "\\" in task_id or task_id in {".", ".."}:
        raise ValueError(f"task_id cannot be used as a trace filename: {task_id!r}")
    return output_dir / "traces" / f"{task_id}.jsonl"


def convert_recording(
    recording: JsonObject,
    *,
    recording_path: Path,
    task_id: str | None = None,
    family: str | None = None,
    adapter: str | None = None,
    message_source: MessageSource = "provider-request",
    include_tool_outputs: bool = False,
) -> list[TraceRow]:
    """Convert one parsed agent recording into ordered trace rows."""
    if message_source not in {"provider-request", "request-messages"}:
        raise ValueError(f"unsupported message source: {message_source}")
    resolved_task_id = task_id or _task_id(recording, recording_path)
    model_events = _model_call_events(recording)
    grouped_tools = _tool_calls_after_model_calls(recording, include_tool_outputs)
    if len(grouped_tools) != len(model_events):
        raise ValueError(f"tool grouping mismatch: {len(grouped_tools)} groups for {len(model_events)} model calls")

    rows: list[TraceRow] = []
    for index, (event, tool_calls) in enumerate(zip(model_events, grouped_tools, strict=True)):
        prompt_tokens, completion_tokens, total_tokens = _usage(event)
        rows.append(
            TraceRow(
                turn_id=f"{resolved_task_id}:{index:04d}",
                task_id=resolved_task_id,
                conversation_id=resolved_task_id,
                conversation_idx=index,
                messages=_messages(event, message_source),
                tools=_tools(event),
                target_output_tokens=completion_tokens,
                recorded_prompt_tokens=prompt_tokens,
                recorded_completion_tokens=completion_tokens,
                recorded_total_tokens=total_tokens,
                recorded_model_duration_ms=optional_non_negative_number(event, "duration_ms", "model_call"),
                simulated_tool_delay_ms_after=sum(call.duration_ms for call in tool_calls),
                recorded_tool_calls_after=tool_calls,
                source=TraceSource(
                    recording=recording_path,
                    family=family,
                    adapter=adapter,
                    model_call_index=index,
                    message_source=message_source,
                ),
            )
        )
    return rows


def _docker_image(recording: JsonObject) -> str | None:
    metadata = optional_object(recording, "metadata", "recording") or {}
    return optional_text(metadata, "docker_image", "recording.metadata")


def _tool_environment(
    family: str | None, adapter: str | None, recording: JsonObject
) -> DockerToolEnvironmentSpec | None:
    if adapter == "pinchbench" or (family or "").startswith("pinchbench"):
        return DockerToolEnvironmentSpec(
            image=DEFAULT_PINCHBENCH_IMAGE,
            cwd=PINCHBENCH_CWD,
            network="none",
            interpreter=("bash", "-c"),
            workspace_mount=True,
        )
    if adapter == "swebench" or (family or "").startswith("swe"):
        docker_image = _docker_image(recording)
        if docker_image:
            return DockerToolEnvironmentSpec(
                image=docker_image,
                cwd=DEFAULT_SWEBENCH_CWD,
                network="none",
                interpreter=("bash", "-c"),
                workspace_mount=False,
            )
    return None


def _required_context_tokens(rows: list[TraceRow]) -> int | None:
    """Return the largest recorded prompt-plus-target demand over one task's rows."""
    demands = tuple(
        row.recorded_prompt_tokens + (row.target_output_tokens or 0)
        for row in rows
        if row.recorded_prompt_tokens is not None
    )
    return max(demands) if demands else None


def _manifest_task(
    *,
    task_id: str,
    trace_path: Path,
    output_dir: Path,
    family: str | None,
    adapter: str | None,
    source_recording: Path,
    rows: list[TraceRow],
    recording: JsonObject,
) -> ManifestTask:
    return ManifestTask(
        task_id=task_id,
        trace=trace_path.relative_to(output_dir),
        family=family,
        adapter=adapter,
        tool_environment=_tool_environment(family, adapter, recording),
        source_recording=source_recording,
        model_calls=len(rows),
        tool_calls=sum(len(row.recorded_tool_calls_after) for row in rows),
        total_recorded_tool_delay_ms=sum(row.simulated_tool_delay_ms_after for row in rows),
        required_context_tokens=_required_context_tokens(rows),
    )


def convert_one_recording_to_dir(
    recording_path: Path,
    output_dir: Path,
    *,
    family: str | None = None,
    adapter: str | None = None,
    message_source: MessageSource = "provider-request",
    include_tool_outputs: bool = False,
    manifest_name: str | None = None,
) -> tuple[ReplayManifest, Path]:
    """Convert one recording and write its manifest and trace."""
    recording = read_recording_json(recording_path)
    rows = convert_recording(
        recording,
        recording_path=recording_path,
        family=family,
        adapter=adapter,
        message_source=message_source,
        include_tool_outputs=include_tool_outputs,
    )
    task_id = rows[0].task_id
    trace_path = _trace_path(output_dir, task_id)
    write_trace(trace_path, rows)
    manifest = ReplayManifest(
        name=manifest_name or f"{task_id}-single-user",
        source="agent-recording",
        tasks=[
            _manifest_task(
                task_id=task_id,
                trace_path=trace_path,
                output_dir=output_dir,
                family=family,
                adapter=adapter,
                source_recording=recording_path,
                rows=rows,
                recording=recording,
            )
        ],
    )
    write_manifest(output_dir / "manifest.json", manifest)
    return manifest, trace_path


def convert_manifest_to_dir(
    manifest_path: Path,
    output_dir: Path,
    *,
    message_source: MessageSource = "provider-request",
    include_tool_outputs: bool = False,
    manifest_name: str | None = None,
) -> ReplayManifest:
    """Convert every task in one recording manifest."""
    recording_manifest = read_recording_json(manifest_path)
    tasks = required_objects(recording_manifest, "tasks", "recording manifest")
    if not tasks:
        raise ValueError(f"recording manifest contains no tasks: {manifest_path}")

    converted_tasks: list[ManifestTask] = []
    task_ids: set[str] = set()
    for index, task in enumerate(tasks):
        source = f"recording manifest.tasks[{index}]"
        task_id = optional_text(task, "task_id", source)
        recording_name = optional_text(task, "recording", source)
        if not task_id:
            raise ValueError(f"{source}.task_id must be a non-empty string")
        if task_id in task_ids:
            raise ValueError(f"recording manifest contains duplicate task_id: {task_id}")
        task_ids.add(task_id)
        if not recording_name:
            raise ValueError(f"{source}.recording must be a non-empty string")
        family = optional_text(task, "family", source)
        adapter = optional_text(task, "adapter", source)

        recording_path = _resolve_recording_path(manifest_path, recording_name)
        recording = read_recording_json(recording_path)
        rows = convert_recording(
            recording,
            recording_path=recording_path,
            task_id=task_id,
            family=family,
            adapter=adapter,
            message_source=message_source,
            include_tool_outputs=include_tool_outputs,
        )
        trace_path = _trace_path(output_dir, task_id)
        write_trace(trace_path, rows)
        converted_tasks.append(
            _manifest_task(
                task_id=task_id,
                trace_path=trace_path,
                output_dir=output_dir,
                family=family,
                adapter=adapter,
                source_recording=recording_path,
                rows=rows,
                recording=recording,
            )
        )

    manifest_source = recording_manifest.get("name")
    source_name = manifest_source if isinstance(manifest_source, str) and manifest_source else manifest_path.stem
    manifest = ReplayManifest(
        name=manifest_name or f"{source_name}-single-user",
        source=str(manifest_path),
        tasks=converted_tasks,
    )
    write_manifest(output_dir / "manifest.json", manifest)
    return manifest
