"""Hold every published schema to the model that produces it.

A schema under `docs/schemas` is a contract the submission service reads. The
map below names, for each schema object, the class whose fields it describes.
A field added to one side and not the other fails here rather than at intake.
"""

from __future__ import annotations

import pathlib

import orjson
import pytest
from pydantic import BaseModel

from agentperf_local.common.json_records import json_field_names, required_json_field_names
from agentperf_local.common.json_types import JsonObject
from agentperf_local.deployment.catalog import (
    DeploymentArtifact,
    DeploymentMemory,
    LlamaCppLaunch,
    ModelCandidate,
    ModelDeployment,
    VllmLaunch,
)
from agentperf_local.deployment.qualification import ProbeOutcome
from agentperf_local.submission import contract
from agentperf_local.submission.spec import SUBMISSION_SPEC_PATH

SCHEMA_ROOT = pathlib.Path(__file__).parents[1] / "docs" / "schemas"

# (schema file, pointer to the object, the class whose fields it describes).
SCHEMA_OBJECTS: tuple[tuple[str, str, type[BaseModel]], ...] = (
    ("recipe-v2.schema.json", "$defs/artifact", DeploymentArtifact),
    ("recipe-v2.schema.json", "$defs/memory", DeploymentMemory),
    ("recipe-v2.schema.json", "$defs/llama_cpp", LlamaCppLaunch),
    ("recipe-v2.schema.json", "$defs/vllm", VllmLaunch),
    ("recipe-v2.schema.json", "$defs/deployment", ModelDeployment),
    ("recipe-v2.schema.json", "$defs/model", ModelCandidate),
    ("runtime-qualification-v1.schema.json", "$defs/outcome", ProbeOutcome),
    ("runtime-qualification-v2.schema.json", "$defs/outcome", ProbeOutcome),
)
# Each request component of the service's pinned spec, and the client model that writes it.
SPEC_COMPONENTS: tuple[tuple[str, type[BaseModel]], ...] = (
    ("SubmissionRequest", contract.SubmissionRequest),
    ("Client", contract.Client),
    ("Benchmark", contract.Benchmark),
    ("Hardware", contract.Hardware),
    ("Accelerator", contract.Accelerator),
    ("ManagedDeployment", contract.ManagedDeployment),
    ("AttachedDeployment", contract.AttachedDeployment),
    ("CappedOutputPolicy", contract.CappedOutputPolicy),
    ("FreeOutputPolicy", contract.FreeOutputPolicy),
    ("Run", contract.Run),
    ("Turn", contract.Turn),
    ("Qualification", contract.Qualification),
    ("QualificationOutcome", contract.QualificationOutcome),
    ("Power", contract.Power),
)


def _resolve(schema_file: str, pointer: str) -> JsonObject:
    node = orjson.loads((SCHEMA_ROOT / schema_file).read_bytes())
    for segment in pointer.split("/"):
        assert isinstance(node, dict), f"{schema_file}#{pointer} does not name an object"
        node = node[segment]
    assert isinstance(node, dict), f"{schema_file}#{pointer} does not name an object"
    return node


@pytest.mark.parametrize(
    ("schema_file", "pointer", "record"),
    SCHEMA_OBJECTS,
    ids=[f"{name}#{pointer}" for name, pointer, _ in SCHEMA_OBJECTS],
)
def test_schema_object_describes_exactly_its_record(schema_file: str, pointer: str, record: type[BaseModel]) -> None:
    node = _resolve(schema_file, pointer)
    properties = node.get("properties")
    required = node.get("required")

    assert isinstance(properties, dict)
    assert frozenset(properties) == json_field_names(record)
    # A closed contract: intake must reject a key this client would never write.
    assert node.get("additionalProperties") is False
    assert isinstance(required, list)
    assert frozenset(required) <= frozenset(properties)


def test_every_schema_is_a_valid_draft_and_is_covered_or_named() -> None:
    """No schema file is forgotten: each is parseable and either mapped or listed as unmapped."""
    unmapped = {
        # Envelopes and records whose JSON shape is not one model's field list.
        "private-nvidia-telemetry-v2.schema.json",
    }
    present = {path.name for path in SCHEMA_ROOT.glob("*.json")}
    mapped = {name for name, _, _ in SCHEMA_OBJECTS}

    assert mapped | unmapped == present


@pytest.mark.parametrize(("component", "record"), SPEC_COMPONENTS, ids=[name for name, _ in SPEC_COMPONENTS])
def test_pinned_spec_component_describes_exactly_its_request_model(component: str, record: type[BaseModel]) -> None:
    """A field the spec gains or loses fails here, before the service refuses a body."""
    spec = orjson.loads(SUBMISSION_SPEC_PATH.read_bytes())
    node = spec["components"]["schemas"][component]

    assert frozenset(node["properties"]) == json_field_names(record)
    assert frozenset(node.get("required", ())) == required_json_field_names(record)
    assert node.get("additionalProperties") is False
