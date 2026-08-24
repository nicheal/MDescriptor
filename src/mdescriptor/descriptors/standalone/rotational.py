"""Rotational descriptor names."""

from ..._legacy.rotational import (
    EadCalculator,
    LbispectrumCalculator,
    SnapCalculator,
    So3Calculator,
    So4Calculator,
)
from ...core.legacy_adapter import adapter_class

EAD = adapter_class("EAD", EadCalculator, __name__)
LBispectrum = adapter_class("LBispectrum", LbispectrumCalculator, __name__)
SNAP = adapter_class("SNAP", SnapCalculator, __name__)
SO3 = adapter_class("SO3", So3Calculator, __name__)
SO4 = adapter_class("SO4", So4Calculator, __name__)

__all__ = ["EAD", "LBispectrum", "SNAP", "SO3", "SO4"]
