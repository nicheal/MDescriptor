"""Base contract for descriptor implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, ClassVar

from .control import ComputeControl
from .errors import (
    CancelledError,
    ClosedDescriptorError,
    DescriptorInputError,
    MDescriptorError,
    is_native_cancelled_error,
)
from .input import StructureBatch, StructureInput, coerce_batch
from .options import CONFIGURATION_SCHEMA_VERSION, DescriptorConfiguration
from .result import DescriptorResult


class Descriptor(ABC):
    """Small template that centralizes lifecycle and input handling.

    Concrete descriptors implement only ``_compute_batch``.  Adapters cannot
    bypass the shared input, lifecycle, cancellation, and result checks.
    """

    name: ClassVar[str]

    def __init__(self) -> None:
        self._closed = False
        self._configuration = DescriptorConfiguration(
            CONFIGURATION_SCHEMA_VERSION,
            self.name,
            {},
        )
        self._metadata_snapshot: dict[str, Any] = {}

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def feature_count(self) -> int | None:
        return None

    @property
    def configuration(self) -> DescriptorConfiguration:
        """Immutable construction snapshot retained after ``close()``."""

        return self._configuration

    @property
    def metadata(self) -> Mapping[str, Any]:
        """JSON-safe descriptor metadata retained after ``close()``.

        A descriptor result carries the same metadata, but keeping the latest
        snapshot on the descriptor makes diagnostics available after its
        native/model runtime has been released.
        """

        return deepcopy(self._metadata_snapshot)

    def compute(
        self,
        value: StructureInput,
        *,
        control: ComputeControl | None = None,
    ) -> DescriptorResult:
        """Compute one batch through the single public execution boundary.

        Implementations only provide ``_compute_batch``.  Keeping conversion,
        lifecycle, cancellation, and result validation here prevents each
        descriptor family from quietly growing a different contract.
        """

        self._ensure_open()
        if _is_cancelled(control):
            raise CancelledError("descriptor computation was cancelled")
        try:
            batch = self._as_batch(value)
            self._validate_batch(batch)
        except DescriptorInputError:
            raise
        except (ImportError, TypeError, ValueError) as exc:
            raise DescriptorInputError(str(exc)) from exc
        try:
            result = self._compute_batch(batch, control=control)
        except Exception as exc:
            if is_native_cancelled_error(exc):
                raise CancelledError("descriptor computation was cancelled") from exc
            raise
        if _is_cancelled(control):
            raise CancelledError("descriptor computation was cancelled")
        if not isinstance(result, DescriptorResult):
            raise MDescriptorError(
                f"{self.name} returned {type(result).__name__}, expected DescriptorResult"
            )
        return result

    @abstractmethod
    def _compute_batch(
        self,
        batch: StructureBatch,
        *,
        control: ComputeControl | None = None,
    ) -> DescriptorResult:
        raise NotImplementedError

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> Descriptor:
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
        return coerce_batch(value)

    def _validate_batch(self, batch: StructureBatch) -> None:
        """Validate descriptor-specific input capabilities at the public seam."""

        del batch


def _is_cancelled(control: Any) -> bool:
    if control is None:
        return False
    checker = getattr(control, "cancelled", None)
    if checker is None:
        return False
    return bool(checker() if callable(checker) else checker)
