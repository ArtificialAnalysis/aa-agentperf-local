"""Read parsed argparse values with their expected types."""

from __future__ import annotations

import argparse
from pathlib import Path


def _argument(namespace: argparse.Namespace, name: str) -> object:
    return getattr(namespace, name)


def read_string(namespace: argparse.Namespace, name: str) -> str:
    value = _argument(namespace, name)
    if not isinstance(value, str):
        raise RuntimeError(f"argument {name} was not parsed as text")
    return value


def read_optional_string(namespace: argparse.Namespace, name: str) -> str | None:
    return None if _argument(namespace, name) is None else read_string(namespace, name)


def read_path(namespace: argparse.Namespace, name: str) -> Path:
    value = _argument(namespace, name)
    if not isinstance(value, Path):
        raise RuntimeError(f"argument {name} was not parsed as a path")
    return value


def read_optional_path(namespace: argparse.Namespace, name: str) -> Path | None:
    return None if _argument(namespace, name) is None else read_path(namespace, name)


def read_integer(namespace: argparse.Namespace, name: str) -> int:
    value = _argument(namespace, name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise RuntimeError(f"argument {name} was not parsed as an integer")
    return value


def read_optional_integer(namespace: argparse.Namespace, name: str) -> int | None:
    return None if _argument(namespace, name) is None else read_integer(namespace, name)


def read_number(namespace: argparse.Namespace, name: str) -> float:
    value = _argument(namespace, name)
    if not isinstance(value, float):
        raise RuntimeError(f"argument {name} was not parsed as a number")
    return value


def read_optional_number(namespace: argparse.Namespace, name: str) -> float | None:
    return None if _argument(namespace, name) is None else read_number(namespace, name)


def read_boolean(namespace: argparse.Namespace, name: str) -> bool:
    value = _argument(namespace, name)
    if not isinstance(value, bool):
        raise RuntimeError(f"argument {name} was not parsed as a boolean")
    return value
