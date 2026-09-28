"""Define and read the replay manifest and trace format.

- `load_manifest` / `write_manifest`: read or write one `ReplayManifest`.
- `load_trace` / `write_trace`: read or write the `TraceRow` lines of one task.
- `parse_json_object`: decode one JSON object without a model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Self

import orjson
from pydantic import (
    BaseModel,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    ValidationInfo,
    field_validator,
    model_validator,
)

from agentperf_local.common.durable_files import NewFile, write_new_file
from agentperf_local.common.json_types import JsonObject, JsonValue, normalize_json_object
from agentperf_local.common.models import read_record, require_json_keys

FORMAT_VERSION = 1
REPLAY_MODE = "single_user_agentic_replay"

type ReplayMode = Literal["single_user_agentic_replay"]
type MessageSource = Literal["provider-request", "request-messages"]
type NonEmptyText = Annotated[str, Field(min_length=1)]


def parse_json_object(data: bytes | str, source: Path | str) -> JsonObject:
    """Parse one JSON object and validate its recursive value types."""
    source_name = str(source)
    try:
        value: object = orjson.loads(data)
        return normalize_json_object(value)
    except (orjson.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid JSON object in {source_name}: {error}") from error


def _non_empty_path_text(value: object, info: ValidationInfo) -> object:
    """Reject empty path text, which `Path` would read as the current directory."""
    if value == "":
        raise ValueError(f"{info.field_name} must be non-empty text")
    return value


def _relative_path(path: Path | None, info: ValidationInfo) -> Path | None:
    """Reject a path that leaves the manifest directory."""
    if path is not None and (path.is_absolute() or ".." in path.parts):
        raise ValueError(f"{info.field_name} must stay within the manifest directory")
    return path


def _check_version(version: int) -> None:
    if version != FORMAT_VERSION:
        raise ValueError(f"version must be {FORMAT_VERSION}, got {version}")


class RequestMessage(BaseModel, frozen=True, allow_inf_nan=False):
    """Store one sanitized message sent to the model provider.

    In JSON, `content_present` is not a key. It records whether the "content"
    key was there, so a message without content is written back without it.
    """

    role: NonEmptyText
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

    @model_validator(mode="before")
    @classmethod
    def read_content_present(cls, data: object, info: ValidationInfo) -> object:
        """Set `content_present` from the JSON keys, ignoring any key of that name."""
        if info.mode != "json" or not isinstance(data, dict):
            return data
        return {**data, "content_present": "content" in data}


class ToolDefinition(BaseModel, frozen=True, allow_inf_nan=False):
    """Wrap one provider tool definition without changing its schema.

    In JSON the definition is the object itself, with no wrapper key.
    """

    definition: JsonObject

    @model_validator(mode="before")
    @classmethod
    def wrap_definition(cls, data: object, info: ValidationInfo) -> object:
        """Read a bare JSON value as the definition."""
        if info.mode != "json":
            return data
        return {"definition": data}

    def to_dict(self) -> JsonObject:
        """Return the provider tool definition."""
        return self.definition


class RecordedToolCall(BaseModel, frozen=True, allow_inf_nan=False):
    """Describe one recorded tool call after a model turn."""

    duration_ms: NonNegativeFloat
    action_index: NonNegativeInt | None = None
    step: NonNegativeInt | None = None
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


class TraceSource(BaseModel, frozen=True, allow_inf_nan=False):
    """Identify the recorded model call that produced one replay turn."""

    recording: Path
    model_call_index: NonNegativeInt
    message_source: MessageSource
    family: str | None = None
    adapter: str | None = None
    format: NonEmptyText = "agent-recording"

    _recording_text = field_validator("recording", mode="before")(_non_empty_path_text)

    @model_validator(mode="after")
    def check_invariants(self, info: ValidationInfo) -> Self:
        """Require the format marker in JSON."""
        require_json_keys(self, info, ("format",))
        return self

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


class TraceRow(BaseModel, frozen=True, allow_inf_nan=False):
    """Describe one model turn in a replay task."""

    turn_id: NonEmptyText
    task_id: NonEmptyText
    conversation_id: NonEmptyText
    conversation_idx: NonNegativeInt
    messages: list[RequestMessage]
    tools: list[ToolDefinition] = []
    target_output_tokens: NonNegativeInt | None = None
    max_output_tokens: PositiveInt | None = None
    recorded_prompt_tokens: NonNegativeInt | None = None
    recorded_completion_tokens: NonNegativeInt | None = None
    recorded_total_tokens: NonNegativeInt | None = None
    recorded_model_duration_ms: NonNegativeFloat | None = None
    simulated_tool_delay_ms_after: NonNegativeFloat = 0.0
    recorded_tool_calls_after: list[RecordedToolCall] = []
    source: TraceSource | None = None
    version: int = FORMAT_VERSION

    @model_validator(mode="after")
    def check_invariants(self, info: ValidationInfo) -> Self:
        """Require the current format, and the tool delay in JSON."""
        _check_version(self.version)
        require_json_keys(self, info, ("version", "simulated_tool_delay_ms_after"))
        return self

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


class DockerToolEnvironmentSpec(BaseModel, frozen=True, allow_inf_nan=False):
    """Describe an isolated Docker environment for live tool replay."""

    image: NonEmptyText
    cwd: NonEmptyText
    interpreter: Annotated[tuple[str, ...], Field(min_length=1)]
    workspace_mount: bool
    workspace_path: Path | None = None
    network: NonEmptyText = "none"
    type: Literal["docker"] = "docker"

    _workspace_path_text = field_validator("workspace_path", mode="before")(_non_empty_path_text)
    _workspace_path_inside = field_validator("workspace_path")(_relative_path)

    @model_validator(mode="after")
    def check_invariants(self, info: ValidationInfo) -> Self:
        """Require the type and network in JSON."""
        require_json_keys(self, info, ("type", "network"))
        return self

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


class ManifestTask(BaseModel, frozen=True, allow_inf_nan=False):
    """Point to one task trace and its recorded totals."""

    task_id: NonEmptyText
    trace: Path
    source_recording: Path
    model_calls: NonNegativeInt
    tool_calls: NonNegativeInt
    total_recorded_tool_delay_ms: NonNegativeFloat
    family: str | None = None
    adapter: str | None = None
    tool_environment: DockerToolEnvironmentSpec | None = None
    # The smallest context that replays the task: at least the largest
    # recorded_prompt_tokens + target_output_tokens over its rows. Bundled manifests
    # round it up to a context rung for headroom. Older manifests lack it; missing means unknown.
    required_context_tokens: NonNegativeInt | None = None

    _path_text = field_validator("trace", "source_recording", mode="before")(_non_empty_path_text)
    _trace_inside = field_validator("trace")(_relative_path)

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


class ReplayManifest(BaseModel, frozen=True, allow_inf_nan=False):
    """List the tasks in one replay workload."""

    name: NonEmptyText
    source: NonEmptyText
    tasks: list[ManifestTask] = []
    mode: ReplayMode = REPLAY_MODE
    version: int = FORMAT_VERSION

    @model_validator(mode="after")
    def check_invariants(self, info: ValidationInfo) -> Self:
        """Require the current format, its markers in JSON, and unique task IDs."""
        _check_version(self.version)
        require_json_keys(self, info, ("version", "mode", "tasks"))
        task_ids = [task.task_id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("tasks contains duplicate task IDs")
        return self

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


def write_manifest(path: Path, manifest: ReplayManifest) -> None:
    """Write one private replay manifest without replacement."""
    options = orjson.OPT_APPEND_NEWLINE | orjson.OPT_INDENT_2
    write_new_file(NewFile(path=path, data=orjson.dumps(manifest.to_dict(), option=options)))


def load_manifest(path: Path) -> ReplayManifest:
    """Read and validate one replay manifest."""
    return read_record(ReplayManifest, path.read_bytes(), str(path), unknown_keys="skip")


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
        rows.append(read_record(TraceRow, line, f"{path}:{line_number}", unknown_keys="skip"))
    return rows
