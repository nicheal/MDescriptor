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
    """Lazily construct the native control without importing native at package import."""

    def __new__(cls) -> Any:
        try:
            from .. import _native
        except ImportError:
            return _FallbackComputeControl()
        return _native.ComputeControl()


__all__ = ["ComputeControl"]
