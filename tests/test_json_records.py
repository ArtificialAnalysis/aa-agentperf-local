"""Exercise the derived closed key set and record encoder."""

from __future__ import annotations

import ast
import pathlib
from enum import StrEnum
from pathlib import Path

import pytest
from pydantic import BaseModel

from agentperf_local.common.json_records import json_field_names, json_record, json_value
from agentperf_local.common.json_types import JsonObject

PACKAGE_ROOT = pathlib.Path(__file__).parents[1] / "agentperf_local"


class Colour(StrEnum):
    RED = "red"


class Inner(BaseModel, frozen=True):
    depth: int

    def to_json(self) -> JsonObject:
        """Return the inner record."""
        return json_record(self)


class Outer(BaseModel, frozen=True):
    name: str
    colour: Colour
    where: Path
    inner: Inner
    tags: tuple[str, ...]
    missing: int | None

    def to_json(self) -> JsonObject:
        """Return the outer record."""
        return json_record(self)


def test_record_encodes_every_field_kind_in_declaration_order() -> None:
    record = Outer(
        name="one",
        colour=Colour.RED,
        where=Path("results/x"),
        inner=Inner(depth=2),
        tags=("a", "b"),
        missing=None,
    )

    encoded = record.to_json()

    assert list(encoded) == ["name", "colour", "where", "inner", "tags", "missing"]
    assert encoded == {
        "name": "one",
        "colour": "red",
        "where": "results/x",
        "inner": {"depth": 2},
        "tags": ["a", "b"],
        "missing": None,
    }
    assert json_field_names(Outer) == frozenset(encoded)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (Colour.RED, "red"),
        (Path("a/b"), "a/b"),
        ((1, 2), [1, 2]),
        (frozenset({"only"}), ["only"]),
        ({"k": Colour.RED}, {"k": "red"}),
        (None, None),
    ],
)
def test_json_value_encodes_each_supported_kind(value: object, expected: object) -> None:
    assert json_value(value) == expected


def test_unencodable_values_are_refused() -> None:
    with pytest.raises(TypeError, match="has no JSON encoding"):
        json_value(object())


def _classes_with_derived_keys() -> list[tuple[str, str, bool, bool]]:
    """Return every class whose reader derives its key set, and how its writer is spelled."""
    found: list[tuple[str, str, bool, bool]] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.ClassDef):
                continue
            body = ast.unparse(node)
            if "json_field_names(cls)" not in body:
                continue
            writer = next((m for m in node.body if isinstance(m, ast.FunctionDef) and m.name == "to_json"), None)
            derived_writer = writer is not None and "json_record(self)" in ast.unparse(writer)
            found.append((path.name, node.name, writer is not None, derived_writer))
    return found


def test_a_derived_key_set_is_never_paired_with_a_hand_written_writer() -> None:
    """One field list drives both directions, so a new field cannot reach one side only."""
    classes = _classes_with_derived_keys()

    assert classes, "no class derives its closed key set; this rule has nothing to protect"
    drifted = [(module, name) for module, name, has_writer, derived in classes if has_writer and not derived]
    assert drifted == []
