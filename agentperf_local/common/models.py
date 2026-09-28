"""Helpers for the package's Pydantic models.

- `replace_fields`: copy a model with some fields changed, and validate the copy.
- `raised_error`: return the exception a validator raised inside a `ValidationError`.
- `raising_validator_errors`: re-raise that exception in place of its `ValidationError`.
- `error_text`: show an error to a person, without Pydantic's wrapper text.
- `read_record`: parse one JSON document into a model, strictly, naming its source in errors.
- `read_object`: the same for a JSON object that is already decoded.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import orjson
from pydantic import BaseModel, ValidationError

from agentperf_local.common.json_types import JsonObject


def replace_fields[Model: BaseModel](model: Model, **changes: object) -> Model:
    """Return a copy of `model` with `changes` applied.

    The copy is built through the constructor, so field and model validators
    run again. `model_copy(update=...)` skips both. An unknown field name is an
    error, as it is for `dataclasses.replace`.
    """
    if unknown := sorted(changes.keys() - type(model).model_fields.keys()):
        raise TypeError(f"{type(model).__name__} has no field {', '.join(unknown)}")
    return type(model)(**(dict(model) | changes))


def raised_error(error: ValidationError) -> Exception | None:
    """Return the exception a validator raised, or None when a field failed its type check.

    Pydantic keeps a validator's `ValueError` in the error context.
    """
    for detail in error.errors(include_url=False):
        raised = detail.get("ctx", {}).get("error")
        if isinstance(raised, Exception):
            return raised
    return None


@contextmanager
def raising_validator_errors() -> Iterator[None]:
    """Re-raise the exception a validator raised, so a caller can catch its own type."""
    try:
        yield
    except ValidationError as error:
        raised = raised_error(error)
        if raised is None:
            raise
        raise raised from error


def error_text(error: BaseException) -> str:
    """Return the text to show a person for one error.

    A validation error becomes its messages, one per line. A message from a
    field type check names the field. Other errors keep their own text.
    """
    if not isinstance(error, ValidationError):
        return str(error)
    lines: list[str] = []
    for detail in error.errors(include_url=False):
        raised = detail.get("ctx", {}).get("error")
        if isinstance(raised, Exception):
            lines.append(str(raised))
            continue
        field = ".".join(str(part) for part in detail["loc"])
        lines.append(f"{field}: {detail['msg']}" if field else detail["msg"])
    return "\n".join(lines)


def read_record[Model: BaseModel](model: type[Model], encoded: bytes | str, source: str) -> Model:
    """Parse one JSON document into `model`, and name `source` in any error.

    Strict JSON mode accepts only the declared JSON type for each field: "5" is not
    an integer and true is not a number. An array fills a tuple field. An unknown
    key is an error, so every reader keeps a closed key set.
    """
    try:
        return model.model_validate_json(encoded, strict=True, extra="forbid")
    except ValidationError as error:
        raise ValueError("\n".join(f"{source}: {line}" for line in error_text(error).splitlines())) from error


def read_object[Model: BaseModel](model: type[Model], data: JsonObject, source: str) -> Model:
    """Parse one decoded JSON object into `model`, with the rules of `read_record`.

    Strict Python mode would reject a list for a tuple field, so the object is
    encoded again and read in JSON mode.
    """
    return read_record(model, orjson.dumps(data), source)
