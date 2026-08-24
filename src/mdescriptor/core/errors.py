"""Errors exposed by the descriptor contracts."""

from __future__ import annotations


class DescriptorError(Exception):
    """Base class for errors raised by the public descriptor API."""


class DescriptorConfigError(ValueError, DescriptorError):
    """A descriptor option or capability declaration is invalid."""


class DescriptorInputError(ValueError, DescriptorError):
    """An input batch does not satisfy the descriptor input contract."""


class ModelLoadError(RuntimeError, DescriptorError):
    """A model resource could not be resolved, validated, or loaded."""


class ClosedDescriptorError(RuntimeError, DescriptorError):
    """An operation was attempted after a descriptor was closed."""


class CancelledError(RuntimeError, DescriptorError):
    """A cooperative descriptor computation was cancelled."""


try:  # Native kernels raise their own registered exception type.
    from .._native import CancelledError as NativeCancelledError
except ImportError:  # pragma: no cover - available after the native build
    class NativeCancelledError(Exception):
        """Fallback marker used before the optional native extension is built."""
