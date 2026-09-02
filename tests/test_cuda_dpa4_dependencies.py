"""Release-time dependency checks for the custom DPA4 CUDA path."""

from __future__ import annotations

from pathlib import Path

from scripts.verify_cuda_wheel import RUNTIME_LIBRARY_NAMES


ROOT = Path(__file__).parents[1]


def test_dpa4_cuda_does_not_require_blas_runtime() -> None:
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    source = (ROOT / "cpp/cuda/src/dpa4.cu").read_text(encoding="utf-8")

    assert RUNTIME_LIBRARY_NAMES == ("libcudart",)
    assert "CUDA::cublas" not in cmake
    assert "CUDA::cublasLt" not in cmake
    assert "cublas" not in source.lower()
