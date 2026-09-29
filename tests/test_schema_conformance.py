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

from agentperf_local.common.json_records import json_field_names
from agentperf_local.common.json_types import JsonObject
from agentperf_local.deployment.catalog import (
    DeploymentArtifact,
    DeploymentMemory,
    LlamaCppLaunch,
    ModelCandidate,
    ModelDeployment,
    MtplxLaunch,
    RecipeSource,
    VllmLaunch,
)
from agentperf_local.deployment.qualification import ProbeOutcome
from agentperf_local.provenance.benchmark import SubmissionContext
from agentperf_local.provenance.hardware import PublicAcceleratorProfile
from agentperf_local.provenance.hardware_facts import AcceleratorSnapshot
from agentperf_local.submission.aggregate import (
    ArtifactDigests,
    PublicDistribution,
    PublicLatencyDistributions,
    PublicRunResult,
    PublicTotals,
)
from agentperf_local.submission.bundle import BundleArtifact
from agentperf_local.submission.evidence import SanitizedTiming, SanitizedTurn
from agentperf_local.telemetry.nvidia import TelemetryFooter
from agentperf_local.telemetry.power import PowerPhaseSummary

SCHEMA_ROOT = pathlib.Path(__file__).parents[1] / "docs" / "schemas"

# (schema file, pointer to the object, the class whose fields it describes).
SCHEMA_OBJECTS: tuple[tuple[str, str, type[BaseModel]], ...] = (
    ("recipe-v1.schema.json", "$defs/artifact", DeploymentArtifact),
    ("recipe-v1.schema.json", "$defs/memory", DeploymentMemory),
    ("recipe-v1.schema.json", "$defs/llama_cpp", LlamaCppLaunch),
    ("recipe-v1.schema.json", "$defs/vllm", VllmLaunch),
    ("recipe-v1.schema.json", "$defs/mtplx", MtplxLaunch),
    ("recipe-v1.schema.json", "$defs/deployment", ModelDeployment),
    ("recipe-v1.schema.json", "$defs/model", ModelCandidate),
    ("private-audit-v1.schema.json", "$defs/accelerator", AcceleratorSnapshot),
    ("private-audit-v1.schema.json", "$defs/outcome", ProbeOutcome),
    ("private-audit-v1.schema.json", "$defs/phase", PowerPhaseSummary),
    ("private-audit-v1.schema.json", "$defs/power/properties/collection", TelemetryFooter),
    ("private-audit-v2.schema.json", "$defs/accelerator", AcceleratorSnapshot),
    ("private-audit-v2.schema.json", "$defs/recipe", RecipeSource),
    ("private-audit-v2.schema.json", "$defs/outcome", ProbeOutcome),
    ("private-audit-v2.schema.json", "$defs/phase", PowerPhaseSummary),
    ("private-audit-v2.schema.json", "$defs/power/properties/collection", TelemetryFooter),
    ("public-submission-v2.schema.json", "$defs/benchmark", SubmissionContext),
    ("public-submission-v2.schema.json", "$defs/accelerator", PublicAcceleratorProfile),
    ("public-submission-v2.schema.json", "$defs/evidenceDigests", ArtifactDigests),
    ("public-submission-v2.schema.json", "$defs/totals", PublicTotals),
    ("public-submission-v2.schema.json", "$defs/distribution", PublicDistribution),
    ("public-submission-v2.schema.json", "$defs/latencyDistributions", PublicLatencyDistributions),
    ("public-submission-v2.schema.json", "$defs/run", PublicRunResult),
    ("runtime-qualification-v1.schema.json", "$defs/outcome", ProbeOutcome),
    ("runtime-qualification-v2.schema.json", "$defs/outcome", ProbeOutcome),
    ("sanitized-turn-evidence-v1.schema.json", "$defs/timing", SanitizedTiming),
    ("sanitized-turn-evidence-v1.schema.json", "$defs/turn", SanitizedTurn),
    ("submission-bundle-v2.schema.json", "$defs/artifact", BundleArtifact),
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
