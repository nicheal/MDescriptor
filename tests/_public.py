"""Canonical imports shared by the contract and numerical tests."""

from mdescriptor import (
    AssetPolicy,
    BUILTIN_REGISTRY,
    CancelledError,
    ComputeControl,
    ModelLoadError,
    StructureBatch,
)
from mdescriptor.descriptors import (
    ACSF,
    C00PSMLFF,
    DPA4,
    DPA4C,
    EAD,
    LBispectrum,
    MTP,
    NEP,
    SNAP,
    SOAP,
    SOAPTurbo,
    SO3,
    SO4,
    AtomicComposition,
    CoulombMatrix,
    EwaldSumMatrix,
    LMBTR,
    MBTR,
    NeighborList,
    SineMatrix,
    SortedDistances,
    SoapPowerSpectrum,
    SoapRadialSpectrum,
    SphericalExpansion,
    SphericalExpansionByPair,
    LodeSphericalExpansion,
    ValleOganov,
)


DESCRIPTOR_CATALOG = {
    spec.name: spec.load_class()
    for spec in BUILTIN_REGISTRY
    if spec.asset_policy is not AssetPolicy.REQUIRED
}
MODEL_DESCRIPTOR_CATALOG = {
    spec.name: spec.load_class()
    for spec in BUILTIN_REGISTRY
    if spec.asset_policy is AssetPolicy.REQUIRED
}

__all__ = [
    "ACSF",
    "AssetPolicy",
    "AtomicComposition",
    "BUILTIN_REGISTRY",
    "C00PSMLFF",
    "CancelledError",
    "CoulombMatrix",
    "ComputeControl",
    "DPA4",
    "DPA4C",
    "DESCRIPTOR_CATALOG",
    "EAD",
    "EwaldSumMatrix",
    "LBispectrum",
    "LMBTR",
    "LodeSphericalExpansion",
    "MBTR",
    "MODEL_DESCRIPTOR_CATALOG",
    "MTP",
    "ModelLoadError",
    "NEP",
    "NeighborList",
    "SNAP",
    "SOAP",
    "SOAPTurbo",
    "SO3",
    "SO4",
    "SineMatrix",
    "SortedDistances",
    "SoapPowerSpectrum",
    "SoapRadialSpectrum",
    "SphericalExpansion",
    "SphericalExpansionByPair",
    "StructureBatch",
    "ValleOganov",
]
