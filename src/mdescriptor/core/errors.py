"""Errors exposed by the descriptor contracts."""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from typing import Any


class MDescriptorError(Exception):
    """Base class for errors raised by the public descriptor API."""

    default_code = "mdescriptor_error"

    def __init__(
        self,
        message: str = "",
        *,
        code: str | None = None,
        path: Any = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.message = str(message)
        self.code = str(code or self.default_code)
        self.path = _error_path(path)
        self.details = None if details is None else _json_safe_details(details)
        Exception.__init__(self, self.message)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe structured representation of the error."""

        result: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.path is not None:
            result["path"] = list(self.path)
        if self.details is not None:
            result["details"] = self.details
        return result


class DescriptorConfigError(MDescriptorError, ValueError):
    """A descriptor option or capability declaration is invalid."""

    default_code = "invalid_configuration"

class DescriptorInputError(MDescriptorError, ValueError):
    """An input batch does not satisfy the descriptor input contract."""

    default_code = "invalid_input"

class ModelLoadError(MDescriptorError, RuntimeError):
    """A model resource could not be resolved, validated, or loaded."""

    default_code = "model_load_error"

class ClosedDescriptorError(MDescriptorError, RuntimeError):
    """An operation was attempted after a descriptor was closed."""

    default_code = "closed_descriptor"

class CancelledError(MDescriptorError, RuntimeError):
    """A cooperative descriptor computation was cancelled."""

    default_code = "cancelled"


def is_native_cancelled_error(value: BaseException) -> bool:
    """Return whether ``value`` is the native cancellation exception.

    Native is deliberately resolved only while handling an exception. Importing
    the public package and querying static metadata must remain independent of
    the optional compiled runtime.
    """

    try:
        native_module = importlib.import_module("mdescriptor._native")
    except ImportError:
        return False
    native_type = getattr(native_module, "CancelledError", None)
    return isinstance(native_type, type) and isinstance(value, native_type)


def _error_path(value: Any) -> tuple[str | int, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, int)) and not isinstance(value, bool):
        return (value,)
    try:
        path = tuple(value)
    except TypeError as exc:
        raise TypeError("error path must be a string, integer, or sequence") from exc
    if any(
        not isinstance(item, (str, int)) or isinstance(item, bool)
        for item in path
    ):
        raise TypeError("error path entries must be strings or integers")
    return path


def _json_safe_details(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("error details must be a mapping")
    return _json_safe_value(value)


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("error detail keys must be strings")
        return {key: _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError(f"error detail value of type {type(value).__name__} is not JSON-safe")
