"""Structure-level matrix and MBTR descriptor names."""

from ..._legacy.extra import (
    CoulombMatrixCalculator,
    EwaldSumMatrixCalculator,
    LMBTRCalculator,
    MBTRCalculator,
    SineMatrixCalculator,
    ValleOganovCalculator,
)
from ...core.legacy_adapter import adapter_class

CoulombMatrix = adapter_class("CoulombMatrix", CoulombMatrixCalculator, __name__)
EwaldSumMatrix = adapter_class("EwaldSumMatrix", EwaldSumMatrixCalculator, __name__)
LMBTR = adapter_class("LMBTR", LMBTRCalculator, __name__)
MBTR = adapter_class("MBTR", MBTRCalculator, __name__)
SineMatrix = adapter_class("SineMatrix", SineMatrixCalculator, __name__)
ValleOganov = adapter_class("ValleOganov", ValleOganovCalculator, __name__)

__all__ = ["CoulombMatrix", "EwaldSumMatrix", "LMBTR", "MBTR", "SineMatrix", "ValleOganov"]
