"""Shared CUDA plugin setup for tests that opt into the host GPU."""

from __future__ import annotations

from pathlib import Path

import pytest

from mdescriptor._cuda_loader import CudaPluginUnavailable, load_cuda_plugin


def load_cuda_for_tests() -> None:
    """Load the repository CUDA build or skip when this runner has none."""

    try:
        load_cuda_plugin(Path(__file__).parents[1] / "build-cuda")
    except CudaPluginUnavailable as error:
        pytest.skip(str(error))


__all__ = ["load_cuda_for_tests"]
