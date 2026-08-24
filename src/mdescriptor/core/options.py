"""Typed execution and output options shared by descriptor adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class OutputOptions:
    """How a descriptor result is represented."""

    dtype: Literal["float32", "float64"] = "float64"
    sparse: bool = False

    def __post_init__(self) -> None:
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("output dtype must be 'float32' or 'float64'")
        if not isinstance(self.sparse, bool):
            raise ValueError("output sparse must be a boolean")


@dataclass(frozen=True, slots=True)
class ExecutionOptions:
    """Where and how a descriptor is evaluated."""

    device: str = "cpu"
    num_threads: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("execution device cannot be empty")
        if isinstance(self.num_threads, bool):
            raise ValueError("num_threads must be a positive integer or None")
        if self.num_threads is not None and (
            not isinstance(self.num_threads, int) or self.num_threads <= 0
        ):
            raise ValueError("num_threads must be positive or None")
