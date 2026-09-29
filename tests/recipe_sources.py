"""Build the recipe file a test candidate stands for."""

from __future__ import annotations

import orjson

from agentperf_local.common.identity import sha256_bytes
from agentperf_local.deployment.catalog import ModelCandidate, RecipeSource


def recipe_source(candidate: ModelCandidate) -> RecipeSource:
    """Return a recipe file named after the candidate, whose text is the candidate's fields."""
    text = orjson.dumps(candidate.model_dump(mode="json"), option=orjson.OPT_INDENT_2).decode()
    return RecipeSource(
        path=f"test-model/test-hardware/{candidate.profile_id}.yaml",
        sha256=sha256_bytes(text.encode()),
        text=text,
    )
