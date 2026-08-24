"""Local descriptor names."""

from ..._legacy.local import (
    AtomicCompositionCalculator,
    LodeSphericalExpansionCalculator,
    NeighborListCalculator,
    SoapPowerSpectrumCalculator,
    SoapRadialSpectrumCalculator,
    SortedDistancesCalculator,
    SphericalExpansionByPairCalculator,
    SphericalExpansionCalculator,
)
from ...core.legacy_adapter import adapter_class

AtomicComposition = adapter_class("AtomicComposition", AtomicCompositionCalculator, __name__)
LodeSphericalExpansion = adapter_class("LodeSphericalExpansion", LodeSphericalExpansionCalculator, __name__)
NeighborList = adapter_class("NeighborList", NeighborListCalculator, __name__)
SoapPowerSpectrum = adapter_class("SoapPowerSpectrum", SoapPowerSpectrumCalculator, __name__)
SoapRadialSpectrum = adapter_class("SoapRadialSpectrum", SoapRadialSpectrumCalculator, __name__)
SortedDistances = adapter_class("SortedDistances", SortedDistancesCalculator, __name__)
SphericalExpansionByPair = adapter_class("SphericalExpansionByPair", SphericalExpansionByPairCalculator, __name__)
SphericalExpansion = adapter_class("SphericalExpansion", SphericalExpansionCalculator, __name__)

__all__ = [
    "AtomicComposition",
    "LodeSphericalExpansion",
    "NeighborList",
    "SoapPowerSpectrum",
    "SoapRadialSpectrum",
    "SortedDistances",
    "SphericalExpansion",
    "SphericalExpansionByPair",
]
