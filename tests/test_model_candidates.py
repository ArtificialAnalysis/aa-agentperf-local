"""Check every bundled recipe against its published schema."""

from pathlib import Path

import orjson
import yaml
from jsonschema import Draft202012Validator

from agentperf_local.deployment.catalog import BUNDLED_RECIPES_ROOT

SCHEMA_PATH = Path(__file__).parents[1] / "docs" / "schemas" / "recipe-v1.schema.json"


def test_bundled_recipes_are_schema_valid() -> None:
    schema = orjson.loads(SCHEMA_PATH.read_bytes())
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    recipes = sorted(BUNDLED_RECIPES_ROOT.glob("*/*/*.yaml"))
    assert recipes
    for recipe in recipes:
        validator.validate(yaml.safe_load(recipe.read_bytes()))
