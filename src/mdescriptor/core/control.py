"""Cooperative computation control exposed by the core contract."""

from __future__ import annotations

from typing import Any


class _FallbackComputeControl:
    """Fallback shape used only when the native extension is unavailable."""

    def __init__(self) -> None:
        self._cancelled = False
        self._completed = 0
        self._total = 0

    def reset(self, total: int) -> None:
        if isinstance(total, bool) or not isinstance(total, int) or total < 0:
            raise ValueError("control total must be a non-negative integer")
        self._cancelled = False
        self._completed = 0
        self._total = total

    def cancel(self) -> None:
        self._cancelled = True

    def cancelled(self) -> bool:
        return self._cancelled

    def completed(self) -> int:
        return self._completed

    def total(self) -> int:
        return self._total

    def mark_completed(self) -> None:
        self._completed += 1


class ComputeControl:
    """Stable wrapper for native or pure-Python cooperative control.

    The wrapper keeps the public type stable.  Native kernels receive the
    underlying pybind object through :func:`_unwrap_native_control`, so the wrapper
    never crosses the C++ ABI boundary.
    """

    def __init__(self) -> None:
        try:
            from .. import _native
        except ImportError:
            self._implementation: Any = _FallbackComputeControl()
        else:
            self._implementation = _native.ComputeControl()

    def reset(self, total: int) -> None:
        self._implementation.reset(total)

    def cancel(self) -> None:
        self._implementation.cancel()

    def cancelled(self) -> bool:
        return bool(self._implementation.cancelled())

    def completed(self) -> int:
        return int(self._implementation.completed())

    def total(self) -> int:
        return int(self._implementation.total())

    def mark_completed(self) -> None:
        self._implementation.mark_completed()

    @property
    def _native_control(self) -> Any:
        """Return the implementation object expected by native kernels."""

        return self._implementation


def _unwrap_native_control(value: Any) -> Any:
    """Unwrap the public control while accepting legacy native instances."""

    if isinstance(value, ComputeControl):
        return value._native_control
    return value


__all__ = ["ComputeControl"]
