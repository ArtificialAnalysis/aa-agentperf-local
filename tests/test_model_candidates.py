"""Check the bundled model candidates against their published schema."""

from pathlib import Path

import orjson
from jsonschema import Draft202012Validator

from agentperf_local.deployment.catalog import BUNDLED_MODEL_CATALOG_PATH

SCHEMA_PATH = Path(__file__).parents[1] / "docs" / "schemas" / "model-candidates-v2.schema.json"


def test_bundled_model_candidates_are_schema_valid() -> None:
    schema = orjson.loads(SCHEMA_PATH.read_bytes())
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(orjson.loads(BUNDLED_MODEL_CATALOG_PATH.read_bytes()))
