"""Base contract for descriptor implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .errors import ClosedDescriptorError, DescriptorInputError
from .input import StructureBatch, StructureInput
from .result import DescriptorResult


class Descriptor(ABC):
    """Small template that centralizes lifecycle and input handling.

    Concrete descriptors implement only ``_compute_batch``.  Legacy adapters
    can be wrapped incrementally without duplicating the public checks.
    """

    name: str

    def __init__(self) -> None:
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def feature_count(self) -> int | None:
        return None

    def compute(self, value: StructureInput, control: Any = None) -> DescriptorResult:
        self._ensure_open()
        try:
            batch = self._as_batch(value)
        except DescriptorInputError:
            raise
        except (TypeError, ValueError) as exc:
            raise DescriptorInputError(str(exc)) from exc
        return self._compute_batch(batch, control=control)

    @abstractmethod
    def _compute_batch(self, batch: StructureBatch, *, control: Any = None) -> DescriptorResult:
        raise NotImplementedError

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> "Descriptor":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def __getstate__(self) -> Any:
        raise TypeError("active descriptors are not pickleable; store a JSON configuration and rebuild")

    def _ensure_open(self) -> None:
        if self._closed:
            raise ClosedDescriptorError(f"descriptor {self.name!r} is closed")

    @staticmethod
    def _as_batch(value: StructureInput) -> StructureBatch:
        if isinstance(value, StructureBatch):
            return value
        return StructureBatch.from_ase(value)
