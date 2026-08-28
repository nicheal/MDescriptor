"""Shared strict-JSON value handling for public contract boundaries."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

JSON_UNHANDLED = object()


def freeze_json(value: Any) -> Any:
    """Convert a JSON value to an immutable mapping/sequence tree."""

    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("JSON object keys must be strings")
        return MappingProxyType({key: freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item) for item in value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, float):
        raise TypeError("JSON values cannot contain non-finite numbers")
    raise TypeError(f"unsupported value type {type(value).__name__}")


def thaw_json(value: Any) -> Any:
    """Return a mutable JSON copy of a value frozen by :func:`freeze_json`."""

    if isinstance(value, MappingProxyType):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


def json_safe_value(
    value: Any,
    *,
    context: str,
    scalar_keys: bool = False,
    nonfinite_message: str | None = None,
    converter: Callable[[Any], Any] | None = None,
) -> Any:
    """Validate and copy a JSON tree, optionally normalizing live values.

    ``converter`` may return :data:`JSON_UNHANDLED` for ordinary JSON values
    or a JSON-compatible replacement for supported live values such as NumPy
    scalars.  Keeping recursion here gives metadata, error details, and
    configuration boundaries one set of JSON safety rules.
    """

    if converter is not None:
        converted = converter(value)
        if converted is not JSON_UNHANDLED:
            return json_safe_value(
                converted,
                context=context,
                scalar_keys=scalar_keys,
                nonfinite_message=nonfinite_message,
                converter=converter,
            )
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(key, str):
                json_key = key
            elif scalar_keys and isinstance(key, (bool, int, float)):
                json_key = str(key)
            else:
                if scalar_keys:
                    raise TypeError(
                        "JSON object keys must be strings or JSON scalar keys"
                    )
                raise TypeError(f"{context} keys must be strings")
            result[json_key] = json_safe_value(
                item,
                context=context,
                scalar_keys=scalar_keys,
                nonfinite_message=nonfinite_message,
                converter=converter,
            )
        return result
    if isinstance(value, (list, tuple)):
        return [
            json_safe_value(
                item,
                context=context,
                scalar_keys=scalar_keys,
                nonfinite_message=nonfinite_message,
                converter=converter,
            )
            for item in value
        ]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            if nonfinite_message is not None:
                raise TypeError(nonfinite_message)
            raise TypeError(
                f"{context} value of type float is not JSON-safe"
            )
        return value
    raise TypeError(f"{context} value of type {type(value).__name__} is not JSON-safe")


__all__ = ["JSON_UNHANDLED", "freeze_json", "json_safe_value", "thaw_json"]
