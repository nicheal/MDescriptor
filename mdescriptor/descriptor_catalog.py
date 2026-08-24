"""Canonical inventory of public descriptor implementations."""

from __future__ import annotations

from .descriptors import AcsfCalculator, SoapCalculator
from .descriptors_soap_turbo import SoapTurboCalculator
from .descriptors_extra import (
    CoulombMatrixCalculator,
    EwaldSumMatrixCalculator,
    LMBTRCalculator,
    MBTRCalculator,
    SineMatrixCalculator,
    ValleOganovCalculator,
)
from .descriptors_featomic import (
    AtomicCompositionCalculator,
    LodeSphericalExpansionCalculator,
    NeighborListCalculator,
    SoapPowerSpectrumCalculator,
    SoapRadialSpectrumCalculator,
    SortedDistancesCalculator,
    SphericalExpansionByPairCalculator,
    SphericalExpansionCalculator,
)
from .descriptors_pyxtal import EadCalculator, LbispectrumCalculator, SnapCalculator, So3Calculator, So4Calculator
from .descriptors_mtp import MtpCalculator
from .descriptors_dpa4c import Dpa4cCalculator
from .descriptors_c00ps_mlff import C00PSMlffCalculator


DESCRIPTOR_CATALOG = {
    "SOAP": SoapCalculator,
    "SOAPTurbo": SoapTurboCalculator,
    "ACSF": AcsfCalculator,
    "CoulombMatrix": CoulombMatrixCalculator,
    "SineMatrix": SineMatrixCalculator,
    "EwaldSumMatrix": EwaldSumMatrixCalculator,
    "MBTR": MBTRCalculator,
    "LMBTR": LMBTRCalculator,
    "ValleOganov": ValleOganovCalculator,
    "AtomicComposition": AtomicCompositionCalculator,
    "NeighborList": NeighborListCalculator,
    "SortedDistances": SortedDistancesCalculator,
    "SphericalExpansion": SphericalExpansionCalculator,
    "SphericalExpansionByPair": SphericalExpansionByPairCalculator,
    "SoapRadialSpectrum": SoapRadialSpectrumCalculator,
    "SoapPowerSpectrum": SoapPowerSpectrumCalculator,
    "LodeSphericalExpansion": LodeSphericalExpansionCalculator,
    "EAD": EadCalculator,
    "SO3": So3Calculator,
    "SO4": So4Calculator,
    "SNAP": SnapCalculator,
    "LBispectrum": LbispectrumCalculator,
    "MTP": MtpCalculator,
    "C00PSMLFF": C00PSMlffCalculator,
    "DPA4C": Dpa4cCalculator,
}


def descriptor_inventory() -> tuple[str, ...]:
    return tuple(DESCRIPTOR_CATALOG)


__all__ = ["DESCRIPTOR_CATALOG", "descriptor_inventory"]
