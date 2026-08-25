"""Local and neighbour descriptor family."""

from ....core.adapter import adapter_class
from ..._kernels.local import (
    AtomicCompositionKernel,
    LodeSphericalExpansionKernel,
    NeighborListKernel,
    SoapPowerSpectrumKernel,
    SoapRadialSpectrumKernel,
    SortedDistancesKernel,
    SphericalExpansionByPairKernel,
    SphericalExpansionKernel,
)

AtomicComposition = adapter_class("AtomicComposition", AtomicCompositionKernel, __name__)
LodeSphericalExpansion = adapter_class("LodeSphericalExpansion", LodeSphericalExpansionKernel, __name__)
NeighborList = adapter_class("NeighborList", NeighborListKernel, __name__)
SoapPowerSpectrum = adapter_class("SoapPowerSpectrum", SoapPowerSpectrumKernel, __name__)
SoapRadialSpectrum = adapter_class("SoapRadialSpectrum", SoapRadialSpectrumKernel, __name__)
SortedDistances = adapter_class("SortedDistances", SortedDistancesKernel, __name__)
SphericalExpansionByPair = adapter_class("SphericalExpansionByPair", SphericalExpansionByPairKernel, __name__)
SphericalExpansion = adapter_class("SphericalExpansion", SphericalExpansionKernel, __name__)

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
