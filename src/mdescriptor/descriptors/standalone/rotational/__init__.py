"""Rotationally invariant descriptor family."""

from ....core.adapter import adapter_class
from ..._kernels.rotational import (
    EadKernel,
    LbispectrumKernel,
    SnapKernel,
    So3Kernel,
    So4Kernel,
)

EAD = adapter_class("EAD", EadKernel, __name__)
LBispectrum = adapter_class("LBispectrum", LbispectrumKernel, __name__)
SNAP = adapter_class("SNAP", SnapKernel, __name__)
SO3 = adapter_class("SO3", So3Kernel, __name__)
SO4 = adapter_class("SO4", So4Kernel, __name__)

__all__ = ["EAD", "LBispectrum", "SNAP", "SO3", "SO4"]
