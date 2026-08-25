"""Cooperative computation control exposed by the core contract."""

from __future__ import annotations

try:
    from .._native import ComputeControl
except ImportError:  # pragma: no cover - source tree before native build
    class ComputeControl:  # type: ignore[no-redef]
        """Fallback shape used only when the native extension is unavailable."""

        def __init__(self) -> None:
            self._cancelled = False

        def reset(self, total: int) -> None:
            del total
            self._cancelled = False

        def cancel(self) -> None:
            self._cancelled = True

        def cancelled(self) -> bool:
            return self._cancelled

        def completed(self) -> int:
            return 0

        def total(self) -> int:
            return 0

        def mark_completed(self) -> None:
            return None


__all__ = ["ComputeControl"]
