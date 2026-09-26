"""Bind benchmark identity and hardware to one measured run."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import orjson

from agentperf_local import __version__
from agentperf_local.common.durable_files import NewFile, write_new_file
from agentperf_local.common.identity import (
    mint_run_id,
    sha256_bytes,
    sha256_file,
    validate_digest,
    validate_identifier,
    validate_run_id,
)
from agentperf_local.common.json_fields import decode_json_object, optional_string, required_object, required_string
from agentperf_local.common.json_records import json_record
from agentperf_local.common.json_types import JsonObject, JsonValue, pretty_json_bytes
from agentperf_local.common.package_paths import PACKAGE_ROOT
from agentperf_local.provenance.hardware import HardwareSnapshot, hardware_snapshot_from_json
from agentperf_local.workload.schema import load_manifest

# A full AgentPerf run serves one 65,536-token sequence; every smaller context is exploratory.
BENCHMARK_CONTEXT_TOKENS = 65_536
# Version 2 added the required observed_context_tokens record. Version 3 added the
# run identifier and the managed deployment digest. Older bindings predate these
# records and are refused by load_measurement_binding.
MEASUREMENT_BINDING_VERSION = 3
MEASUREMENT_BINDING_FILENAME = "measurement.json"
PROVENANCE_COMMAND_TIMEOUT_SECONDS = 5.0
PRODUCER_CLIENT_NAME = "agentperf-local"

type SourceState = Literal["clean", "dirty", "not_git", "unavailable"]


@dataclass(frozen=True, slots=True, kw_only=True)
class SubmissionContext:
    """Identify the immutable suite, model semantics, runtime recipe, and served context."""

    suite_id: str
    suite_epoch: str
    suite_digest: str
    model_semantics_id: str
    model_artifact_digest: str
    runtime_id: str
    # The served context joins the identity so a reduced run can never collide with a
    # full run. The workload digest stays untouched: bundled replays pin that digest.
    context_tokens: int = BENCHMARK_CONTEXT_TOKENS

    def __post_init__(self) -> None:
        """Reject ambiguous or privacy-sensitive identifiers."""
        for field, value in (
            ("suite_id", self.suite_id),
            ("suite_epoch", self.suite_epoch),
            ("model_semantics_id", self.model_semantics_id),
            ("runtime_id", self.runtime_id),
        ):
            validate_identifier(value, field)
        validate_digest(self.suite_digest, "suite_digest")
        validate_digest(self.model_artifact_digest, "model_artifact_digest")
        if self.context_tokens <= 0 or self.context_tokens > BENCHMARK_CONTEXT_TOKENS:
            raise ValueError(f"context_tokens must be between 1 and {BENCHMARK_CONTEXT_TOKENS}")

    @classmethod
    def from_json(cls, data: JsonObject, source: str = "benchmark") -> SubmissionContext:
        """Read one benchmark identity, refusing artifacts without an explicit context.

        Absence never defaults to the full benchmark: a lenient default would let a
        deleted field upgrade a reduced run, and no consumer of this loader is
        display-only — every one feeds a packaging or validation gate.
        """
        raw_context = data.get("context_tokens")
        if raw_context is None:
            raise ValueError(
                f"{source}.context_tokens is missing; this artifact predates the "
                f"{BENCHMARK_CONTEXT_TOKENS:,}-token benchmark context — re-run with a current client"
            )
        if not isinstance(raw_context, int) or isinstance(raw_context, bool):
            raise ValueError(f"{source}.context_tokens must be an integer")
        return cls(
            suite_id=required_string(data, "suite_id", source),
            suite_epoch=required_string(data, "suite_epoch", source),
            suite_digest=required_string(data, "suite_digest", source),
            model_semantics_id=required_string(data, "model_semantics_id", source),
            model_artifact_digest=required_string(data, "model_artifact_digest", source),
            runtime_id=required_string(data, "runtime_id", source),
            context_tokens=raw_context,
        )

    def to_json(self) -> JsonObject:
        """Return public benchmark identity fields."""
        return json_record(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceProvenance:
    """Identify the benchmark client source used for a run."""

    client_version: str
    source_revision: str | None
    source_state: SourceState

    def __post_init__(self) -> None:
        """Validate public source provenance."""
        validate_identifier(self.client_version, "client_version")
        if self.source_state not in {"clean", "dirty", "not_git", "unavailable"}:
            raise ValueError("source_state is not supported")
        if self.source_revision is not None and (
            len(self.source_revision) != 40
            or any(character not in "0123456789abcdef" for character in self.source_revision)
        ):
            raise ValueError("source_revision must be 40 lowercase hex digits or null")
        if self.source_state in {"clean", "dirty"} and self.source_revision is None:
            raise ValueError("clean or dirty source state requires a source revision")

    @classmethod
    def from_json(cls, data: JsonObject) -> SourceProvenance:
        """Read one source provenance block."""
        if required_string(data, "client_name", "producer") != PRODUCER_CLIENT_NAME:
            raise ValueError(f"producer.client_name must be {PRODUCER_CLIENT_NAME}")
        state = required_string(data, "source_state", "producer")
        if state == "clean":
            source_state: SourceState = "clean"
        elif state == "dirty":
            source_state = "dirty"
        elif state == "not_git":
            source_state = "not_git"
        elif state == "unavailable":
            source_state = "unavailable"
        else:
            raise ValueError("producer.source_state is not supported")
        return cls(
            client_version=required_string(data, "client_version", "producer"),
            source_revision=optional_string(data, "source_revision", "producer"),
            source_state=source_state,
        )

    def to_json(self) -> JsonObject:
        """Return source provenance as JSON data."""
        return {
            "client_name": PRODUCER_CLIENT_NAME,
            "client_version": self.client_version,
            "source_revision": self.source_revision,
            "source_state": self.source_state,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class MeasurementBinding:
    """Bind one run to its inputs before inference starts."""

    # Minted before inference so every artifact of one attempt shares one identifier.
    run_id: str
    benchmark: SubmissionContext
    hardware: HardwareSnapshot
    producer: SourceProvenance
    manifest_path: Path
    manifest_digest: str
    endpoint_model_digest: str
    # The served context this run observed before inference: a probe result for
    # attached endpoints, the readiness report for managed servers, and None when the
    # server would not say. Binding it here lets the packaging gates cross-check the
    # editable summary against a record written before any measurement existed.
    observed_context_tokens: int | None
    # The exact bytes of the deployment record when the app launched the server, and
    # None for a user-supplied endpoint. Binding it here chains the public aggregate
    # to the deployment through this record's own digest.
    deployment_digest: str | None = None
    version: int = MEASUREMENT_BINDING_VERSION

    def __post_init__(self) -> None:
        """Validate measurement binding digests and the context observation."""
        validate_run_id(self.run_id, "run_id")
        validate_digest(self.manifest_digest, "manifest_digest")
        validate_digest(self.endpoint_model_digest, "endpoint_model_digest")
        if self.deployment_digest is not None:
            validate_digest(self.deployment_digest, "deployment_digest")
        if self.manifest_digest != self.benchmark.suite_digest:
            raise ValueError("manifest_digest must match benchmark.suite_digest")
        if self.observed_context_tokens is not None and self.observed_context_tokens <= 0:
            raise ValueError("observed_context_tokens must be positive or null")

    def to_json(self) -> JsonObject:
        """Return private run-binding data."""
        return {
            "version": self.version,
            "kind": "measurement_binding",
            "run_id": self.run_id,
            "benchmark": self.benchmark.to_json(),
            "observed_context_tokens": self.observed_context_tokens,
            "deployment_digest": self.deployment_digest,
            "hardware": self.hardware.to_json(),
            "producer": self.producer.to_json(),
            "private": {
                "manifest_path": str(self.manifest_path),
                "manifest_digest": self.manifest_digest,
                "endpoint_model_digest": self.endpoint_model_digest,
            },
        }


def workload_digest(manifest_path: Path) -> str:
    """Hash the manifest and every referenced trace as one workload."""
    resolved_manifest = manifest_path.resolve()
    manifest = load_manifest(resolved_manifest)
    root = resolved_manifest.parent
    member_paths = {resolved_manifest}
    member_paths.update((root / task.trace).resolve() for task in manifest.tasks)
    members: list[JsonValue] = []
    for path in sorted(member_paths):
        relative_path = path.relative_to(root).as_posix()
        members.append([relative_path, sha256_file(path)])
    canonical = orjson.dumps(members)
    return sha256_bytes(canonical)


def _git_output(root: Path, arguments: tuple[str, ...]) -> bytes | None:
    if shutil.which("git") is None:
        return None
    try:
        completed = subprocess.run(
            ("git", "-C", str(root), *arguments),
            check=False,
            capture_output=True,
            timeout=PROVENANCE_COMMAND_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def collect_source_provenance(root: Path | None = None) -> SourceProvenance:
    """Collect a revision and an explicit source state."""
    selected_root = root if root is not None else PACKAGE_ROOT.parent
    revision_bytes = _git_output(selected_root, ("rev-parse", "HEAD"))
    if revision_bytes is None:
        state: SourceState = "not_git" if not (selected_root / ".git").exists() else "unavailable"
        return SourceProvenance(client_version=__version__, source_revision=None, source_state=state)
    try:
        revision = revision_bytes.decode("ascii")
    except UnicodeDecodeError:
        return SourceProvenance(client_version=__version__, source_revision=None, source_state="unavailable")
    status = _git_output(selected_root, ("status", "--porcelain", "--untracked-files=normal"))
    state = "unavailable" if status is None else ("dirty" if status else "clean")
    return SourceProvenance(client_version=__version__, source_revision=revision, source_state=state)


def create_measurement_binding(
    context: SubmissionContext,
    manifest_path: Path,
    endpoint_model: str,
    hardware: HardwareSnapshot,
    producer: SourceProvenance | None = None,
    *,
    observed_context_tokens: int | None = None,
    run_id: str | None = None,
    deployment_digest: str | None = None,
) -> MeasurementBinding:
    """Capture benchmark identity and hardware before a run.

    Pass the served-context observation once it exists. The None default is the
    conservative record: an unobserved context can never support a comparable
    submission, so a caller that skips it produces only exploratory evidence.
    A caller that already minted the run identifier passes it so the summary and
    every companion file can carry the same one.
    """
    measured_digest = workload_digest(manifest_path)
    if measured_digest != context.suite_digest:
        raise ValueError("benchmark context suite_digest does not match the manifest and trace files")
    return MeasurementBinding(
        run_id=run_id if run_id is not None else mint_run_id(),
        benchmark=context,
        hardware=hardware,
        producer=producer if producer is not None else collect_source_provenance(),
        manifest_path=manifest_path.resolve(),
        manifest_digest=measured_digest,
        endpoint_model_digest=sha256_bytes(endpoint_model.encode("utf-8")),
        observed_context_tokens=observed_context_tokens,
        deployment_digest=deployment_digest,
    )


def create_attached_submission_context(
    manifest_path: Path,
    endpoint_model: str,
    *,
    model_semantics_id: str | None = None,
) -> SubmissionContext:
    """Create a self-reported benchmark identity for a user-supplied endpoint."""
    return SubmissionContext(
        suite_id="agentperf-local-attached",
        suite_epoch="community-local-v1",
        suite_digest=workload_digest(manifest_path),
        model_semantics_id=model_semantics_id or "user-supplied",
        model_artifact_digest=sha256_bytes(endpoint_model.encode("utf-8")),
        runtime_id="attached-openai-compatible",
    )


def write_measurement_binding(path: Path, binding: MeasurementBinding) -> None:
    """Write the private run binding before inference starts."""
    write_new_file(NewFile(path=path, data=pretty_json_bytes(binding.to_json())))


def load_measurement_binding(path: Path) -> MeasurementBinding:
    """Read and validate one private run binding."""
    try:
        encoded = path.read_bytes()
    except FileNotFoundError as error:
        raise ValueError(
            f"{path.parent} has no {MEASUREMENT_BINDING_FILENAME}; "
            "choose a results directory written by run, managed-run, or tui"
        ) from error
    data = decode_json_object(encoded, f"invalid measurement binding JSON: {path}")
    if data.get("version") != MEASUREMENT_BINDING_VERSION:
        raise ValueError(
            "measurement binding version is not supported; bindings from older clients predate "
            "the served-context, run identifier, or deployment records — re-run the benchmark with a current client"
        )
    if data.get("kind") != "measurement_binding":
        raise ValueError("measurement binding kind must be measurement_binding")
    if "observed_context_tokens" not in data:
        raise ValueError("measurement.observed_context_tokens is missing — re-run the benchmark")
    raw_observed = data.get("observed_context_tokens")
    if raw_observed is not None and (not isinstance(raw_observed, int) or isinstance(raw_observed, bool)):
        raise ValueError("measurement.observed_context_tokens must be an integer or null")
    if "deployment_digest" not in data:
        raise ValueError("measurement.deployment_digest is missing — re-run the benchmark")
    private = required_object(data, "private", "measurement")
    return MeasurementBinding(
        run_id=required_string(data, "run_id", "measurement"),
        benchmark=SubmissionContext.from_json(required_object(data, "benchmark", "measurement")),
        hardware=hardware_snapshot_from_json(required_object(data, "hardware", "measurement")),
        producer=SourceProvenance.from_json(required_object(data, "producer", "measurement")),
        manifest_path=Path(required_string(private, "manifest_path", "measurement.private")),
        manifest_digest=required_string(private, "manifest_digest", "measurement.private"),
        endpoint_model_digest=required_string(private, "endpoint_model_digest", "measurement.private"),
        observed_context_tokens=raw_observed,
        deployment_digest=optional_string(data, "deployment_digest", "measurement"),
    )
