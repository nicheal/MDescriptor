"""The single built-in descriptor specification list."""

from __future__ import annotations

from .registry import DescriptorRegistry
from .spec import AssetPolicy, DescriptorSpec


_STANDALONE = "mdescriptor.descriptors.standalone:"
_MODEL = "mdescriptor.descriptors.model_backed."
_COMMON_CAPABILITIES = frozenset({"sparse"})


BUILTIN_SPECS = (
    DescriptorSpec("SOAP", _STANDALONE + "SOAP", AssetPolicy.NONE, "cpp", "structure", capabilities=_COMMON_CAPABILITIES),
    DescriptorSpec("SOAPTurbo", _STANDALONE + "SOAPTurbo", AssetPolicy.NONE, "cpp", "atom", capabilities=_COMMON_CAPABILITIES),
    DescriptorSpec("ACSF", _STANDALONE + "ACSF", AssetPolicy.NONE, "cpp", "atom", capabilities=_COMMON_CAPABILITIES),
    DescriptorSpec("CoulombMatrix", _STANDALONE + "CoulombMatrix", AssetPolicy.NONE, "cpp", "structure", capabilities=_COMMON_CAPABILITIES),
    DescriptorSpec("SineMatrix", _STANDALONE + "SineMatrix", AssetPolicy.NONE, "cpp", "structure", capabilities=_COMMON_CAPABILITIES),
    DescriptorSpec("EwaldSumMatrix", _STANDALONE + "EwaldSumMatrix", AssetPolicy.NONE, "cpp", "structure", capabilities=_COMMON_CAPABILITIES),
    DescriptorSpec("MBTR", _STANDALONE + "MBTR", AssetPolicy.NONE, "cpp", "structure", capabilities=_COMMON_CAPABILITIES),
    DescriptorSpec("LMBTR", _STANDALONE + "LMBTR", AssetPolicy.NONE, "cpp", "atom", capabilities=_COMMON_CAPABILITIES),
    DescriptorSpec("ValleOganov", _STANDALONE + "ValleOganov", AssetPolicy.NONE, "cpp", "structure", capabilities=_COMMON_CAPABILITIES),
    DescriptorSpec("AtomicComposition", _STANDALONE + "AtomicComposition", AssetPolicy.NONE, "cpp", "structure", capabilities=_COMMON_CAPABILITIES),
    DescriptorSpec("NeighborList", _STANDALONE + "NeighborList", AssetPolicy.NONE, "cpp", "pair", capabilities=_COMMON_CAPABILITIES),
    DescriptorSpec("SortedDistances", _STANDALONE + "SortedDistances", AssetPolicy.NONE, "cpp", "atom", capabilities=_COMMON_CAPABILITIES),
    DescriptorSpec("SphericalExpansion", _STANDALONE + "SphericalExpansion", AssetPolicy.NONE, "cpp", "atom", capabilities=_COMMON_CAPABILITIES),
    DescriptorSpec("SphericalExpansionByPair", _STANDALONE + "SphericalExpansionByPair", AssetPolicy.NONE, "cpp", "pair", capabilities=_COMMON_CAPABILITIES),
    DescriptorSpec("SoapRadialSpectrum", _STANDALONE + "SoapRadialSpectrum", AssetPolicy.NONE, "cpp", "atom", capabilities=_COMMON_CAPABILITIES),
    DescriptorSpec("SoapPowerSpectrum", _STANDALONE + "SoapPowerSpectrum", AssetPolicy.NONE, "cpp", "atom", capabilities=_COMMON_CAPABILITIES),
    DescriptorSpec("LodeSphericalExpansion", _STANDALONE + "LodeSphericalExpansion", AssetPolicy.NONE, "cpp", "atom", capabilities=_COMMON_CAPABILITIES),
    DescriptorSpec("EAD", _STANDALONE + "EAD", AssetPolicy.NONE, "cpp", "atom", capabilities=_COMMON_CAPABILITIES),
    DescriptorSpec("SO3", _STANDALONE + "SO3", AssetPolicy.NONE, "cpp", "atom", capabilities=_COMMON_CAPABILITIES),
    DescriptorSpec("SO4", _STANDALONE + "SO4", AssetPolicy.NONE, "cpp", "atom", capabilities=_COMMON_CAPABILITIES),
    DescriptorSpec("SNAP", _STANDALONE + "SNAP", AssetPolicy.NONE, "cpp", "atom", capabilities=_COMMON_CAPABILITIES),
    DescriptorSpec("LBispectrum", _STANDALONE + "LBispectrum", AssetPolicy.NONE, "cpp", "atom", capabilities=_COMMON_CAPABILITIES),
    DescriptorSpec("MTP", _STANDALONE + "MTP", AssetPolicy.OPTIONAL, "cpp", "atom", capabilities=_COMMON_CAPABILITIES),
    DescriptorSpec("C00PSMLFF", _STANDALONE + "C00PSMLFF", AssetPolicy.NONE, "cpp", "atom", capabilities=_COMMON_CAPABILITIES),
    DescriptorSpec("NEP", _MODEL + "nep:NEP", AssetPolicy.REQUIRED, "cpp", "atom", capabilities=_COMMON_CAPABILITIES),
    DescriptorSpec("DPA4", _MODEL + "dpa4:DPA4", AssetPolicy.REQUIRED, "torch", "atom", capabilities=_COMMON_CAPABILITIES, optional_extra="model"),
    DescriptorSpec("DPA4C", _MODEL + "dpa4c:DPA4C", AssetPolicy.REQUIRED, "torch", "atom", capabilities=_COMMON_CAPABILITIES, optional_extra="model"),
)

BUILTIN_REGISTRY = DescriptorRegistry(BUILTIN_SPECS, frozen=True)
