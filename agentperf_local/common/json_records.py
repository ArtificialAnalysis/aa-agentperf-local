"""Describe one dataclass as a closed JSON record.

A record's field list is written once, in the dataclass. Its closed key set and
its encoding are derived from that list, so a new field cannot reach the reader
without reaching the writer. Fields keep declaration order, which is the order
the hand-written encoders emitted and the order a digest covers.

A class whose JSON shape is not its field list — one that adds an envelope,
renames a field, or nests a sub-object — writes its own `to_json` instead.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol, runtime_checkable

from agentperf_local.common.json_types import JsonObject, JsonValue


@runtime_checkable
class JsonRecord(Protocol):
    """Encode one value as JSON data."""

    def to_json(self) -> JsonObject:
        """Return the value as a JSON object."""
        ...


def json_field_names(record: type) -> frozenset[str]:
    """Return the closed key set of one dataclass, for `require_exact_keys`."""
    if not is_dataclass(record):
        raise TypeError(f"{record.__name__} is not a dataclass")
    return frozenset(field.name for field in fields(record))


def json_value(value: object) -> JsonValue:
    """Encode one field value as JSON data.

    A string enum becomes its value, a path its text, a nested record its own
    object, and a sequence a list of the same. Everything else is already a
    JSON value and passes through.
    """
    if isinstance(value, Enum):
        member = value.value
        if not isinstance(member, bool | float | int | str | None):
            raise TypeError(f"enum {type(value).__name__} does not hold a JSON scalar")
        return member
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, JsonRecord) and is_dataclass(value):
        return value.to_json()
    if isinstance(value, tuple | list | frozenset | set):
        return [json_value(item) for item in value]
    if isinstance(value, bool | float | int | str | None):
        return value
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    raise TypeError(f"{type(value).__name__} has no JSON encoding")


def json_record(record: object) -> JsonObject:
    """Encode one dataclass as a JSON object of exactly its fields, in order."""
    if not is_dataclass(record) or isinstance(record, type):
        raise TypeError("a JSON record must be a dataclass instance")
    return {field.name: json_value(getattr(record, field.name)) for field in fields(record)}
