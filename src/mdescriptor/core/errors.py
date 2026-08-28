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


class DescriptorConfigError(ValueError, MDescriptorError):
    """A descriptor option or capability declaration is invalid."""

    default_code = "invalid_configuration"

    def __init__(
        self,
        message: str = "",
        *,
        code: str | None = None,
        path: Any = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        MDescriptorError.__init__(self, message, code=code, path=path, details=details)


class DescriptorInputError(ValueError, MDescriptorError):
    """An input batch does not satisfy the descriptor input contract."""

    default_code = "invalid_input"

    def __init__(
        self,
        message: str = "",
        *,
        code: str | None = None,
        path: Any = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        MDescriptorError.__init__(self, message, code=code, path=path, details=details)


class ModelLoadError(RuntimeError, MDescriptorError):
    """A model resource could not be resolved, validated, or loaded."""


class ClosedDescriptorError(RuntimeError, MDescriptorError):
    """An operation was attempted after a descriptor was closed."""


class CancelledError(RuntimeError, MDescriptorError):
    """A cooperative descriptor computation was cancelled."""


NativeCancelledError: type[Exception]
try:  # Native kernels raise their own registered exception type.
    _native_module = importlib.import_module("mdescriptor._native")
except ImportError:  # pragma: no cover - before native build
    NativeCancelledError = Exception
else:
    NativeCancelledError = getattr(_native_module, "CancelledError", Exception)


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
