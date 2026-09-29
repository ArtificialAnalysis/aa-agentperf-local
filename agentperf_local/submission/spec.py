"""Check a submission body against the service's pinned OpenAPI spec.

Public surface: SUBMISSION_SPEC_PATH and validate_against_spec.
"""

from __future__ import annotations

from functools import cache

import orjson
from jsonschema import Draft202012Validator
from jsonschema.protocols import Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

from agentperf_local.common.json_types import JsonValue, normalize_json, normalize_json_object
from agentperf_local.common.package_paths import PACKAGE_DATA_ROOT

# A copy of https://submit.artificialanalysis.ai/v1/openapi.json. CI fails when the live spec differs.
SUBMISSION_SPEC_PATH = PACKAGE_DATA_ROOT / "submission-openapi.json"
# The spec is registered under this name so its "#/components/..." references resolve.
_SPEC_URI = "urn:agentperf-local:submission-openapi"
_REQUEST_SCHEMA_POINTER = "#/components/schemas/SubmissionRequest"
# A body that fails usually fails the same way on every turn, so the message lists only the first few.
MAX_REPORTED_SPEC_ERRORS = 5


@cache
def _request_validator() -> Validator:
    """Build one validator for the request schema, with the whole spec available to its references."""
    spec = normalize_json_object(orjson.loads(SUBMISSION_SPEC_PATH.read_bytes()))
    registry = Registry().with_resource(_SPEC_URI, Resource.from_contents(spec, default_specification=DRAFT202012))
    return Draft202012Validator({"$ref": f"{_SPEC_URI}{_REQUEST_SCHEMA_POINTER}"}, registry=registry)


def _location(path: tuple[str | int, ...]) -> str:
    return ".".join(str(part) for part in path) or "request"


def validate_against_spec(encoded: bytes) -> None:
    """Raise ValueError naming each field of `encoded` that the pinned spec refuses."""
    body: JsonValue = normalize_json(orjson.loads(encoded))
    errors = sorted(_request_validator().iter_errors(body), key=lambda error: tuple(map(str, error.absolute_path)))
    if not errors:
        return
    lines = [f"{_location(tuple(error.absolute_path))}: {error.message}" for error in errors[:MAX_REPORTED_SPEC_ERRORS]]
    if len(errors) > MAX_REPORTED_SPEC_ERRORS:
        lines.append(f"and {len(errors) - MAX_REPORTED_SPEC_ERRORS} more")
    raise ValueError("the submission does not match the service's spec:\n" + "\n".join(lines))
