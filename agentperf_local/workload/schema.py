"""Define and read the replay manifest and trace format."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import orjson

from agentperf_local.common.durable_files import NewFile, write_new_file
from agentperf_local.common.json_fields import (
    optional_integer,
    optional_non_negative_integer,
    optional_non_negative_number,
    optional_object,
    optional_string,
    optional_text,
    required_boolean,
    required_integer,
    required_list,
    required_non_negative_integer,
    required_non_negative_number,
    required_string,
)
from agentperf_local.common.json_types import JsonObject, JsonValue, normalize_json_object

FORMAT_VERSION = 1
REPLAY_MODE = "single_user_agentic_replay"

type ReplayMode = Literal["single_user_agentic_replay"]
type MessageSource = Literal["provider-request", "request-messages"]


def parse_json_object(data: bytes | str, source: Path | str) -> JsonObject:
    """Parse one JSON object and validate its recursive value types."""
    source_name = str(source)
    try:
        value: object = orjson.loads(data)
        return normalize_json_object(value)
    except (orjson.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid JSON object in {source_name}: {error}") from error


def _relative_path(value: str, source: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{source} must stay within the manifest directory")
    return path


def _required_relative_path(data: JsonObject, key: str, source: str) -> Path:
    return _relative_path(required_string(data, key, source), f"{source}.{key}")


def _optional_relative_path(data: JsonObject, key: str, source: str) -> Path | None:
    value = optional_string(data, key, source)
    if value is None:
        return None
    return _relative_path(value, f"{source}.{key}")


def _check_version(data: JsonObject, source: str) -> None:
    version = required_integer(data, "version", source)
    if version != FORMAT_VERSION:
        raise ValueError(f"{source}.version must be {FORMAT_VERSION}, got {version}")


@dataclass(frozen=True, kw_only=True)
class RequestMessage:
    """Store one sanitized message sent to the model provider."""

    role: str
    content: JsonValue = None
    content_present: bool = True
    name: str | None = None
    tool_calls: list[JsonObject] | None = None
    tool_call_id: str | None = None

    def to_dict(self) -> JsonObject:
        """Return the message in OpenAI wire format."""
        data: JsonObject = {"role": self.role}
        if self.content_present:
            data["content"] = self.content
        if self.name is not None:
            data["name"] = self.name
        if self.tool_calls is not None:
            tool_calls: list[JsonValue] = []
            tool_calls.extend(self.tool_calls)
            data["tool_calls"] = tool_calls
        if self.tool_call_id is not None:
            data["tool_call_id"] = self.tool_call_id
        return data

    @classmethod
    def from_dict(cls, data: JsonObject, source: str) -> RequestMessage:
        """Validate and build one request message."""
        raw_tool_calls = data.get("tool_calls")
        tool_calls: list[JsonObject] | None = None
        if raw_tool_calls is not None:
            if not isinstance(raw_tool_calls, list) or not all(isinstance(call, dict) for call in raw_tool_calls):
                raise ValueError(f"{source}.tool_calls must be an array of objects")
            tool_calls = [call for call in raw_tool_calls if isinstance(call, dict)]
        return cls(
            role=required_string(data, "role", source),
            content=data.get("content"),
            content_present="content" in data,
            name=optional_text(data, "name", source),
            tool_calls=tool_calls,
            tool_call_id=optional_text(data, "tool_call_id", source),
        )


@dataclass(frozen=True, kw_only=True)
class ToolDefinition:
    """Wrap one provider tool definition without changing its schema."""

    definition: JsonObject

    def to_dict(self) -> JsonObject:
        """Return the provider tool definition."""
        return self.definition


@dataclass(frozen=True, kw_only=True)
class RecordedToolCall:
    """Describe one recorded tool call after a model turn."""

    duration_ms: float
    action_index: int | None = None
    step: int | None = None
    recorded_returncode: int | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    action: JsonObject | None = None
    output: JsonObject | None = None

    @property
    def command(self) -> str | None:
        """Return a recorded shell command when one exists."""
        if self.action is None:
            return None
        command = self.action.get("command")
        return command if isinstance(command, str) else None

    def to_dict(self) -> JsonObject:
        """Return JSON data and omit unavailable metadata."""
        data: JsonObject = {"duration_ms": self.duration_ms}
        optional_values: tuple[tuple[str, JsonValue], ...] = (
            ("action_index", self.action_index),
            ("step", self.step),
            ("recorded_returncode", self.recorded_returncode),
            ("tool_call_id", self.tool_call_id),
            ("tool_name", self.tool_name),
            ("action", self.action),
            ("output", self.output),
        )
        for key, value in optional_values:
            if value is not None:
                data[key] = value
        return data

    @classmethod
    def from_dict(cls, data: JsonObject, source: str) -> RecordedToolCall:
        """Validate and build one recorded tool call."""
        return cls(
            duration_ms=required_non_negative_number(data, "duration_ms", source),
            action_index=optional_non_negative_integer(data, "action_index", source),
            step=optional_non_negative_integer(data, "step", source),
            recorded_returncode=optional_integer(data, "recorded_returncode", source),
            tool_call_id=optional_text(data, "tool_call_id", source),
            tool_name=optional_text(data, "tool_name", source),
            action=optional_object(data, "action", source),
            output=optional_object(data, "output", source),
        )


@dataclass(frozen=True, kw_only=True)
class TraceSource:
    """Identify the recorded model call that produced one replay turn."""

    recording: Path
    model_call_index: int
    message_source: MessageSource
    family: str | None = None
    adapter: str | None = None
    format: str = "agent-recording"

    def to_dict(self) -> JsonObject:
        """Return trace provenance as JSON data."""
        return {
            "format": self.format,
            "recording": self.recording.as_posix(),
            "family": self.family,
            "adapter": self.adapter,
            "model_call_index": self.model_call_index,
            "message_source": self.message_source,
        }

    @classmethod
    def from_dict(cls, data: JsonObject, source: str) -> TraceSource:
        """Validate and build trace provenance."""
        raw_message_source = required_string(data, "message_source", source)
        if raw_message_source not in {"provider-request", "request-messages"}:
            raise ValueError(f"{source}.message_source is not supported: {raw_message_source}")
        return cls(
            format=required_string(data, "format", source),
            recording=Path(required_string(data, "recording", source)),
            family=optional_text(data, "family", source),
            adapter=optional_text(data, "adapter", source),
            model_call_index=required_non_negative_integer(data, "model_call_index", source),
            message_source=raw_message_source,
        )


@dataclass(frozen=True, kw_only=True)
class TraceRow:
    """Describe one model turn in a replay task."""

    turn_id: str
    task_id: str
    conversation_id: str
    conversation_idx: int
    messages: list[RequestMessage]
    tools: list[ToolDefinition] = field(default_factory=list)
    target_output_tokens: int | None = None
    max_output_tokens: int | None = None
    recorded_prompt_tokens: int | None = None
    recorded_completion_tokens: int | None = None
    recorded_total_tokens: int | None = None
    recorded_model_duration_ms: float | None = None
    simulated_tool_delay_ms_after: float = 0.0
    recorded_tool_calls_after: list[RecordedToolCall] = field(default_factory=list)
    source: TraceSource | None = None
    version: int = FORMAT_VERSION

    def __post_init__(self) -> None:
        """Require a positive request cap when the trace supplies one."""
        if self.max_output_tokens is not None and self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be greater than zero")

    def to_dict(self) -> JsonObject:
        """Return one versioned JSONL trace row."""
        data: JsonObject = {
            "version": self.version,
            "turn_id": self.turn_id,
            "task_id": self.task_id,
            "conversation_id": self.conversation_id,
            "conversation_idx": self.conversation_idx,
            "messages": [message.to_dict() for message in self.messages],
            "tools": [tool.to_dict() for tool in self.tools],
            "simulated_tool_delay_ms_after": self.simulated_tool_delay_ms_after,
            "recorded_tool_calls_after": [call.to_dict() for call in self.recorded_tool_calls_after],
        }
        optional_values: tuple[tuple[str, JsonValue], ...] = (
            ("target_output_tokens", self.target_output_tokens),
            ("max_output_tokens", self.max_output_tokens),
            ("recorded_prompt_tokens", self.recorded_prompt_tokens),
            ("recorded_completion_tokens", self.recorded_completion_tokens),
            ("recorded_total_tokens", self.recorded_total_tokens),
            ("recorded_model_duration_ms", self.recorded_model_duration_ms),
            ("source", self.source.to_dict() if self.source is not None else None),
        )
        for key, value in optional_values:
            if value is not None:
                data[key] = value
        return data

    @classmethod
    def from_dict(cls, data: JsonObject, source: str) -> TraceRow:
        """Validate and build one trace row."""
        _check_version(data, source)
        raw_messages = required_list(data, "messages", source)
        if not all(isinstance(message, dict) for message in raw_messages):
            raise ValueError(f"{source}.messages must contain objects")
        messages = [
            RequestMessage.from_dict(message, f"{source}.messages[{index}]")
            for index, message in enumerate(raw_messages)
            if isinstance(message, dict)
        ]
        raw_tools = data.get("tools", [])
        if not isinstance(raw_tools, list) or not all(isinstance(tool, dict) for tool in raw_tools):
            raise ValueError(f"{source}.tools must be an array of objects")
        tools = [ToolDefinition(definition=tool) for tool in raw_tools if isinstance(tool, dict)]
        raw_tool_calls = data.get("recorded_tool_calls_after", [])
        if not isinstance(raw_tool_calls, list) or not all(isinstance(call, dict) for call in raw_tool_calls):
            raise ValueError(f"{source}.recorded_tool_calls_after must be an array of objects")
        tool_calls = [
            RecordedToolCall.from_dict(call, f"{source}.recorded_tool_calls_after[{index}]")
            for index, call in enumerate(raw_tool_calls)
            if isinstance(call, dict)
        ]
        raw_source = optional_object(data, "source", source)
        return cls(
            version=FORMAT_VERSION,
            turn_id=required_string(data, "turn_id", source),
            task_id=required_string(data, "task_id", source),
            conversation_id=required_string(data, "conversation_id", source),
            conversation_idx=required_non_negative_integer(data, "conversation_idx", source),
            messages=messages,
            tools=tools,
            target_output_tokens=optional_non_negative_integer(data, "target_output_tokens", source),
            max_output_tokens=optional_non_negative_integer(data, "max_output_tokens", source),
            recorded_prompt_tokens=optional_non_negative_integer(data, "recorded_prompt_tokens", source),
            recorded_completion_tokens=optional_non_negative_integer(data, "recorded_completion_tokens", source),
            recorded_total_tokens=optional_non_negative_integer(data, "recorded_total_tokens", source),
            recorded_model_duration_ms=optional_non_negative_number(data, "recorded_model_duration_ms", source),
            simulated_tool_delay_ms_after=required_non_negative_number(data, "simulated_tool_delay_ms_after", source),
            recorded_tool_calls_after=tool_calls,
            source=TraceSource.from_dict(raw_source, f"{source}.source") if raw_source is not None else None,
        )


@dataclass(frozen=True, kw_only=True)
class DockerToolEnvironmentSpec:
    """Describe an isolated Docker environment for live tool replay."""

    image: str
    cwd: str
    interpreter: tuple[str, ...]
    workspace_mount: bool
    workspace_path: Path | None = None
    network: str = "none"
    type: Literal["docker"] = "docker"

    def to_dict(self) -> JsonObject:
        """Return the Docker settings as JSON data."""
        data: JsonObject = {
            "type": self.type,
            "image": self.image,
            "cwd": self.cwd,
            "network": self.network,
            "interpreter": list(self.interpreter),
            "workspace_mount": self.workspace_mount,
        }
        if self.workspace_path is not None:
            data["workspace_path"] = self.workspace_path.as_posix()
        return data

    @classmethod
    def from_dict(cls, data: JsonObject, source: str) -> DockerToolEnvironmentSpec:
        """Validate and build one Docker environment."""
        environment_type = required_string(data, "type", source)
        if environment_type != "docker":
            raise ValueError(f"{source}.type must be docker, got {environment_type}")
        raw_interpreter = required_list(data, "interpreter", source)
        if not raw_interpreter or not all(isinstance(item, str) for item in raw_interpreter):
            raise ValueError(f"{source}.interpreter must be a non-empty array of strings")
        return cls(
            type="docker",
            image=required_string(data, "image", source),
            cwd=required_string(data, "cwd", source),
            network=required_string(data, "network", source),
            interpreter=tuple(item for item in raw_interpreter if isinstance(item, str)),
            workspace_mount=required_boolean(data, "workspace_mount", source),
            workspace_path=_optional_relative_path(data, "workspace_path", source),
        )


@dataclass(frozen=True, kw_only=True)
class ManifestTask:
    """Point to one task trace and its recorded totals."""

    task_id: str
    trace: Path
    source_recording: Path
    model_calls: int
    tool_calls: int
    total_recorded_tool_delay_ms: float
    family: str | None = None
    adapter: str | None = None
    tool_environment: DockerToolEnvironmentSpec | None = None
    # The smallest context that replays the task: at least the largest
    # recorded_prompt_tokens + target_output_tokens over its rows. Bundled manifests
    # round it up to a context rung for headroom. Older manifests lack it; missing means unknown.
    required_context_tokens: int | None = None

    def to_dict(self) -> JsonObject:
        """Return the manifest task as JSON data."""
        data: JsonObject = {
            "task_id": self.task_id,
            "trace": self.trace.as_posix(),
            "source_recording": self.source_recording.as_posix(),
            "model_calls": self.model_calls,
            "tool_calls": self.tool_calls,
            "total_recorded_tool_delay_ms": self.total_recorded_tool_delay_ms,
        }
        if self.tool_environment is not None:
            data["tool_environment"] = self.tool_environment.to_dict()
        if self.family is not None:
            data["family"] = self.family
        if self.adapter is not None:
            data["adapter"] = self.adapter
        if self.required_context_tokens is not None:
            data["required_context_tokens"] = self.required_context_tokens
        return data

    @classmethod
    def from_dict(cls, data: JsonObject, source: str) -> ManifestTask:
        """Validate and build one manifest task."""
        raw_environment = optional_object(data, "tool_environment", source)
        environment = None
        if raw_environment is not None:
            environment = DockerToolEnvironmentSpec.from_dict(raw_environment, f"{source}.tool_environment")
        model_calls = required_non_negative_integer(data, "model_calls", source)
        tool_calls = required_non_negative_integer(data, "tool_calls", source)
        return cls(
            task_id=required_string(data, "task_id", source),
            trace=_required_relative_path(data, "trace", source),
            family=optional_text(data, "family", source),
            adapter=optional_text(data, "adapter", source),
            tool_environment=environment,
            source_recording=Path(required_string(data, "source_recording", source)),
            model_calls=model_calls,
            tool_calls=tool_calls,
            total_recorded_tool_delay_ms=required_non_negative_number(data, "total_recorded_tool_delay_ms", source),
            required_context_tokens=optional_non_negative_integer(data, "required_context_tokens", source),
        )


@dataclass(frozen=True, kw_only=True)
class ReplayManifest:
    """List the tasks in one replay workload."""

    name: str
    source: str
    tasks: list[ManifestTask] = field(default_factory=list)
    mode: ReplayMode = REPLAY_MODE
    version: int = FORMAT_VERSION

    @property
    def required_context_tokens(self) -> int | None:
        """Return the largest declared per-task context demand.

        Tasks without the field weaken this into a lower bound; None means no task
        declares its demand, so the replay's context requirement is unknown.
        """
        declared = tuple(
            task.required_context_tokens for task in self.tasks if task.required_context_tokens is not None
        )
        return max(declared) if declared else None

    def to_dict(self) -> JsonObject:
        """Return the versioned replay manifest as JSON data."""
        return {
            "name": self.name,
            "source": self.source,
            "mode": self.mode,
            "version": self.version,
            "tasks": [task.to_dict() for task in self.tasks],
        }

    @classmethod
    def from_dict(cls, data: JsonObject, source: str) -> ReplayManifest:
        """Validate and build one replay manifest."""
        _check_version(data, source)
        mode = required_string(data, "mode", source)
        if mode != REPLAY_MODE:
            raise ValueError(f"{source}.mode must be {REPLAY_MODE}, got {mode}")
        raw_tasks = required_list(data, "tasks", source)
        if not all(isinstance(task, dict) for task in raw_tasks):
            raise ValueError(f"{source}.tasks must contain objects")
        tasks = [
            ManifestTask.from_dict(task, f"{source}.tasks[{index}]")
            for index, task in enumerate(raw_tasks)
            if isinstance(task, dict)
        ]
        task_ids = [task.task_id for task in tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError(f"{source}.tasks contains duplicate task IDs")
        return cls(
            name=required_string(data, "name", source),
            source=required_string(data, "source", source),
            tasks=tasks,
        )


def write_manifest(path: Path, manifest: ReplayManifest) -> None:
    """Write one private replay manifest without replacement."""
    options = orjson.OPT_APPEND_NEWLINE | orjson.OPT_INDENT_2
    write_new_file(NewFile(path=path, data=orjson.dumps(manifest.to_dict(), option=options)))


def load_manifest(path: Path) -> ReplayManifest:
    """Read and validate one replay manifest."""
    return ReplayManifest.from_dict(parse_json_object(path.read_bytes(), path), str(path))


def write_trace(path: Path, rows: list[TraceRow]) -> None:
    """Write private versioned replay turns without replacement."""
    encoded = b"".join(orjson.dumps(row.to_dict(), option=orjson.OPT_APPEND_NEWLINE) for row in rows)
    write_new_file(NewFile(path=path, data=encoded))


def load_trace(path: Path) -> list[TraceRow]:
    """Read and validate all non-empty rows in one replay trace."""
    rows: list[TraceRow] = []
    for line_number, line in enumerate(path.read_bytes().splitlines(), start=1):
        if not line.strip():
            continue
        source = f"{path}:{line_number}"
        rows.append(TraceRow.from_dict(parse_json_object(line, source), source))
    return rows
