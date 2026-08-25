"""Errors exposed by the descriptor contracts."""

from __future__ import annotations

import importlib


class MDescriptorError(Exception):
    """Base class for errors raised by the public descriptor API."""


class DescriptorConfigError(ValueError, MDescriptorError):
    """A descriptor option or capability declaration is invalid."""


class DescriptorInputError(ValueError, MDescriptorError):
    """An input batch does not satisfy the descriptor input contract."""


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
