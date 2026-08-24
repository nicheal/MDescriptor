"""Shared immutable model artifact and per-descriptor execution session types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.errors import ClosedDescriptorError


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """Validated model state that is safe to share between descriptor instances."""

    path: Path
    config: Any
    weights: Any


@dataclass(slots=True)
class ModelSession:
    """Device-bound execution state owned by one descriptor instance."""

    model: LoadedModel
    device: str = "cpu"
    closed: bool = False

    def ensure_open(self) -> None:
        if self.closed:
            raise ClosedDescriptorError("model session is closed")

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> "ModelSession":
        self.ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()
