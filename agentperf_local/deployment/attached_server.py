"""Describe a server the user started on this machine, so an attached run can be submitted.

Public surface: AttachedServer, ATTACHED_SERVER_FILENAME, read_attached_server_description,
write_attached_server, read_attached_server, MAX_ATTACHED_SERVER_BYTES, and require_backend_matches.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, Field, field_validator

from agentperf_local.common.durable_files import WrittenFile, read_bounded_file, write_digest_file
from agentperf_local.common.identity import GIT_COMMIT_PATTERN, SHORT_GIT_COMMIT_PATTERN
from agentperf_local.common.json_records import json_record
from agentperf_local.common.json_types import normalize_json_object, pretty_json_bytes
from agentperf_local.common.models import read_object, read_record
from agentperf_local.deployment.catalog import REPOSITORY_PATTERN
from agentperf_local.deployment.launch_command import redact_launch_command
from agentperf_local.submission.contract import (
    CONTAINER_REFERENCE_PATTERN,
    MODEL_RELEASE_SLUG_PATTERN,
    AcceleratorBackend,
    AcceleratorVendor,
    Framework,
)

ATTACHED_SERVER_FILENAME = "attached-server.json"
MAX_ATTACHED_SERVER_BYTES = 64 * 1024

# The backends each vendor's hardware can serve on.
_VENDOR_BACKENDS: dict[AcceleratorVendor, tuple[AcceleratorBackend, ...]] = {
    "nvidia": ("cuda", "vulkan"),
    "amd": ("rocm", "vulkan"),
    "apple": ("metal",),
    "intel": ("sycl", "xpu", "vulkan"),
}


class AttachedServer(BaseModel, frozen=True, extra="forbid"):
    """Describe the model and framework build of a server the user started on this machine.

    `run --attached-server FILE` reads a YAML description, redacts its launch command, and
    records it in the results folder before the replay starts. The measurement binding
    carries the record's digest, as it does for a managed deployment record.
    """

    model_release_slug: Annotated[str, Field(pattern=MODEL_RELEASE_SLUG_PATTERN)]
    hf_repository: Annotated[str, Field(pattern=REPOSITORY_PATTERN.pattern)]
    hf_revision: Annotated[str, Field(pattern=GIT_COMMIT_PATTERN)]
    framework: Framework
    framework_version: Annotated[str, Field(min_length=1)]
    # A short commit is enough; the submission resolves it to all 40 characters.
    framework_commit: Annotated[str, Field(pattern=SHORT_GIT_COMMIT_PATTERN)] | None = None
    framework_container_reference: Annotated[str, Field(pattern=CONTAINER_REFERENCE_PATTERN)] | None = None
    accelerator_backend: AcceleratorBackend
    server_launch_command: Annotated[str, Field(min_length=1)]

    @field_validator("server_launch_command")
    @classmethod
    def _redacted(cls, command: str) -> str:
        """Keep the launch command only with its local paths and secrets replaced."""
        return redact_launch_command(command)


def require_backend_matches(server: AttachedServer, vendor: AcceleratorVendor) -> None:
    """Refuse a backend that the detected accelerator's vendor cannot serve on."""
    allowed = _VENDOR_BACKENDS[vendor]
    if server.accelerator_backend not in allowed:
        raise ValueError(
            f"accelerator_backend {server.accelerator_backend} does not run on this {vendor} accelerator; "
            f"use one of {', '.join(allowed)}"
        )


def read_attached_server_description(path: Path) -> AttachedServer:
    """Read one YAML description a person wrote, and redact its launch command."""
    encoded = read_bounded_file(path, MAX_ATTACHED_SERVER_BYTES, label="attached server description")
    try:
        data = normalize_json_object(yaml.safe_load(encoded))
    except (yaml.YAMLError, ValueError) as error:
        raise ValueError(f"{path} must be a YAML mapping of the attached server's fields") from error
    return read_object(AttachedServer, data, str(path))


def write_attached_server(path: Path, server: AttachedServer) -> WrittenFile:
    """Write the redacted description into the results folder, without replacing a file."""
    return write_digest_file(path, pretty_json_bytes(json_record(server)))


def read_attached_server(encoded: bytes, source: str) -> AttachedServer:
    """Parse the exact bytes of one recorded description."""
    return read_record(AttachedServer, encoded, source)
