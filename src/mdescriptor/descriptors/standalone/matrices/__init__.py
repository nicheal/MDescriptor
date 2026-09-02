"""Structure-level matrix descriptor family."""

from ....core.adapter import adapter_class
from ..._kernels.matrix import (
    CoulombMatrixKernel,
    EwaldSumMatrixKernel,
    SineMatrixKernel,
)

CoulombMatrix = adapter_class("CoulombMatrix", CoulombMatrixKernel, __name__)
EwaldSumMatrix = adapter_class("EwaldSumMatrix", EwaldSumMatrixKernel, __name__)
SineMatrix = adapter_class("SineMatrix", SineMatrixKernel, __name__)

__all__ = ["CoulombMatrix", "EwaldSumMatrix", "SineMatrix"]
