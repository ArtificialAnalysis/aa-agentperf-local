"""Read typed fields from decoded JSON objects with one error message per shape."""

from __future__ import annotations

import math

import orjson

from agentperf_local.common.json_types import JsonObject, JsonValue, normalize_json_object


def decode_json_object(encoded: bytes, invalid_message: str) -> JsonObject:
    """Decode one JSON object and report the caller's message when the bytes are not JSON."""
    try:
        return normalize_json_object(orjson.loads(encoded))
    except orjson.JSONDecodeError as error:
        raise ValueError(invalid_message) from error


def require_exact_keys(data: JsonObject, expected: frozenset[str], source: str) -> None:
    """Reject an object whose key set differs from the closed contract, naming each difference."""
    actual = frozenset(data)
    if actual == expected:
        return
    details: list[str] = []
    if missing := sorted(expected - actual):
        details.append(f"missing {', '.join(missing)}")
    if unexpected := sorted(actual - expected):
        details.append(f"unexpected {', '.join(unexpected)}")
    raise ValueError(f"{source} has invalid fields: {'; '.join(details)}")


def require_allowed_keys(data: JsonObject, required: frozenset[str], allowed: frozenset[str], source: str) -> None:
    """Reject an object that omits a required key or carries a key outside the allowed set."""
    actual = frozenset(data)
    if required.issubset(actual) and actual.issubset(allowed):
        return
    details: list[str] = []
    if missing := sorted(required - actual):
        details.append(f"missing {', '.join(missing)}")
    if unexpected := sorted(actual - allowed):
        details.append(f"unexpected {', '.join(unexpected)}")
    raise ValueError(f"{source} has invalid fields: {'; '.join(details)}")


def required_object(data: JsonObject, key: str, source: str) -> JsonObject:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{source}.{key} must be an object")
    return value


def optional_object(data: JsonObject, key: str, source: str) -> JsonObject | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{source}.{key} must be an object or null")
    return value


def required_objects(data: JsonObject, key: str, source: str) -> list[JsonObject]:
    value = data.get(key)
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"{source}.{key} must be an array of objects")
    return [item for item in value if isinstance(item, dict)]


def required_list(data: JsonObject, key: str, source: str) -> list[JsonValue]:
    value = data.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{source}.{key} must be an array")
    return value


def required_string(data: JsonObject, key: str, source: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source}.{key} must be non-empty text")
    return value


def optional_string(data: JsonObject, key: str, source: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source}.{key} must be non-empty text or null")
    return value


def optional_text(data: JsonObject, key: str, source: str) -> str | None:
    """Read text that is allowed to be empty or absent."""
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{source}.{key} must be a string or null")
    return value


def required_boolean(data: JsonObject, key: str, source: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{source}.{key} must be a boolean")
    return value


def required_integer(data: JsonObject, key: str, source: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{source}.{key} must be an integer")
    return value


def optional_integer(data: JsonObject, key: str, source: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{source}.{key} must be an integer or null")
    return value


def required_number(data: JsonObject, key: str, source: str) -> float:
    value = data.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{source}.{key} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{source}.{key} must be finite")
    return number


def optional_number(data: JsonObject, key: str, source: str) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{source}.{key} must be a number or null")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{source}.{key} must be finite or null")
    return number


def lenient_string(data: JsonObject, key: str) -> str | None:
    """Return non-empty text from a document this package does not control, else None.

    Untrusted documents — command output and service responses — carry fields
    this package reads opportunistically. A wrong shape means absent, not invalid.
    """
    value = data.get(key)
    return value if isinstance(value, str) and value else None


def lenient_integer(data: JsonObject, key: str) -> int | None:
    """Return an integer from a document this package does not control, else None."""
    value = data.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def required_non_negative_integer(data: JsonObject, key: str, source: str) -> int:
    value = required_integer(data, key, source)
    if value < 0:
        raise ValueError(f"{source}.{key} must not be negative")
    return value


def optional_non_negative_integer(data: JsonObject, key: str, source: str) -> int | None:
    value = optional_integer(data, key, source)
    if value is not None and value < 0:
        raise ValueError(f"{source}.{key} must not be negative")
    return value


def required_non_negative_number(data: JsonObject, key: str, source: str) -> float:
    value = required_number(data, key, source)
    if value < 0:
        raise ValueError(f"{source}.{key} must not be negative")
    return value


def optional_non_negative_number(data: JsonObject, key: str, source: str) -> float | None:
    value = optional_number(data, key, source)
    if value is not None and value < 0:
        raise ValueError(f"{source}.{key} must not be negative")
    return value


def one_of[Allowed: str](value: str, allowed: tuple[Allowed, ...], field: str) -> Allowed:
    """Return the allowed literal equal to the value, or reject the value."""
    for candidate in allowed:
        if value == candidate:
            return candidate
    raise ValueError(f"{field} is not supported")


def required_strings(data: JsonObject, key: str, source: str) -> tuple[str, ...]:
    value = data.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{source}.{key} must be an array of text")
    return tuple(item for item in value if isinstance(item, str))
