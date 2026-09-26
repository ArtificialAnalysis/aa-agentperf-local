"""Define JSON values used at file and wire boundaries."""

import orjson

type JsonScalar = bool | float | int | str | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]

_PRETTY_JSON_OPTIONS = orjson.OPT_APPEND_NEWLINE | orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS


def pretty_json_bytes(value: JsonValue) -> bytes:
    """Encode stable, readable JSON with one trailing newline."""
    return orjson.dumps(value, option=_PRETTY_JSON_OPTIONS)


def normalize_json(value: object) -> JsonValue:
    """Validate a decoded value and return a typed JSON value."""
    if value is None or isinstance(value, (bool, float, int, str)):
        return value
    if isinstance(value, list):
        return [normalize_json(item) for item in value]
    if isinstance(value, dict):
        normalized: JsonObject = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("JSON object keys must be strings")
            normalized[key] = normalize_json(item)
        return normalized
    raise ValueError(f"unsupported JSON value: {type(value).__name__}")


def normalize_json_object(value: object) -> JsonObject:
    """Validate a decoded value and return a JSON object."""
    normalized = normalize_json(value)
    if not isinstance(normalized, dict):
        raise ValueError("expected a JSON object")
    return normalized


def json_object_or_none(value: JsonValue | None) -> JsonObject | None:
    """Return the value when it is a JSON object, else None."""
    return value if isinstance(value, dict) else None
