"""The single built-in descriptor specification list."""

from __future__ import annotations

from .registry import DescriptorRegistry
from .spec import AssetPolicy, DescriptorSpec

_STANDALONE = "mdescriptor.descriptors.standalone."
_LOCAL = _STANDALONE + "local:"
_MATRICES = _STANDALONE + "matrices:"
_MANY_BODY = _STANDALONE + "many_body:"
_ROTATIONAL = _STANDALONE + "rotational:"
_MODEL = "mdescriptor.descriptors.model_backed."
_COMMON_CAPABILITIES = frozenset({"sparse"})
_CPP_CAPABILITIES = _COMMON_CAPABILITIES | {"cooperative_cancel"}
_CPP_THREAD_CAPABILITIES = _CPP_CAPABILITIES | {"num_threads"}


_BUILTIN_SPECS = (
    DescriptorSpec("SOAP", _STANDALONE + "soap:SOAP", AssetPolicy.NONE, "cpp", "structure", capabilities=_CPP_THREAD_CAPABILITIES),
    DescriptorSpec("SOAPTurbo", _STANDALONE + "soap_turbo:SOAPTurbo", AssetPolicy.NONE, "cpp", "atom", capabilities=_CPP_THREAD_CAPABILITIES),
    DescriptorSpec("ACSF", _STANDALONE + "acsf:ACSF", AssetPolicy.NONE, "cpp", "atom", capabilities=_CPP_THREAD_CAPABILITIES),
    DescriptorSpec("CoulombMatrix", _MATRICES + "CoulombMatrix", AssetPolicy.NONE, "cpp", "structure", capabilities=_CPP_CAPABILITIES),
    DescriptorSpec("SineMatrix", _MATRICES + "SineMatrix", AssetPolicy.NONE, "cpp", "structure", capabilities=_CPP_CAPABILITIES),
    DescriptorSpec("EwaldSumMatrix", _MATRICES + "EwaldSumMatrix", AssetPolicy.NONE, "cpp", "structure", capabilities=_CPP_CAPABILITIES),
    DescriptorSpec("MBTR", _MANY_BODY + "MBTR", AssetPolicy.NONE, "cpp", "structure", capabilities=_CPP_CAPABILITIES),
    DescriptorSpec("LMBTR", _MANY_BODY + "LMBTR", AssetPolicy.NONE, "cpp", "atom", capabilities=_CPP_CAPABILITIES),
    DescriptorSpec("ValleOganov", _MANY_BODY + "ValleOganov", AssetPolicy.NONE, "cpp", "structure", capabilities=_CPP_CAPABILITIES),
    DescriptorSpec("AtomicComposition", _LOCAL + "AtomicComposition", AssetPolicy.NONE, "cpp", "structure", capabilities=_CPP_CAPABILITIES),
    DescriptorSpec("NeighborList", _LOCAL + "NeighborList", AssetPolicy.NONE, "cpp", "pair", capabilities=_CPP_CAPABILITIES),
    DescriptorSpec("SortedDistances", _LOCAL + "SortedDistances", AssetPolicy.NONE, "cpp", "atom", capabilities=_CPP_THREAD_CAPABILITIES),
    DescriptorSpec("SphericalExpansion", _LOCAL + "SphericalExpansion", AssetPolicy.NONE, "cpp", "atom", capabilities=_CPP_THREAD_CAPABILITIES),
    DescriptorSpec("SphericalExpansionByPair", _LOCAL + "SphericalExpansionByPair", AssetPolicy.NONE, "cpp", "pair", capabilities=_CPP_THREAD_CAPABILITIES),
    DescriptorSpec("SoapRadialSpectrum", _LOCAL + "SoapRadialSpectrum", AssetPolicy.NONE, "cpp", "atom", capabilities=_CPP_THREAD_CAPABILITIES),
    DescriptorSpec("SoapPowerSpectrum", _LOCAL + "SoapPowerSpectrum", AssetPolicy.NONE, "cpp", "atom", capabilities=_CPP_THREAD_CAPABILITIES),
    DescriptorSpec("LodeSphericalExpansion", _LOCAL + "LodeSphericalExpansion", AssetPolicy.NONE, "cpp", "atom", capabilities=_CPP_THREAD_CAPABILITIES),
    DescriptorSpec("EAD", _ROTATIONAL + "EAD", AssetPolicy.NONE, "cpp", "atom", capabilities=_CPP_CAPABILITIES),
    DescriptorSpec("SO3", _ROTATIONAL + "SO3", AssetPolicy.NONE, "cpp", "atom", capabilities=_CPP_CAPABILITIES),
    DescriptorSpec("SO4", _ROTATIONAL + "SO4", AssetPolicy.NONE, "cpp", "atom", capabilities=_CPP_CAPABILITIES),
    DescriptorSpec("SNAP", _ROTATIONAL + "SNAP", AssetPolicy.NONE, "cpp", "atom", capabilities=_CPP_CAPABILITIES),
    DescriptorSpec("LBispectrum", _ROTATIONAL + "LBispectrum", AssetPolicy.NONE, "cpp", "atom", capabilities=_CPP_CAPABILITIES),
    DescriptorSpec("MTP", _STANDALONE + "mtp:MTP", AssetPolicy.OPTIONAL, "cpp", "atom", capabilities=_CPP_THREAD_CAPABILITIES | {"model"}),
    DescriptorSpec("C00PSMLFF", _STANDALONE + "c00ps_mlff:C00PSMLFF", AssetPolicy.NONE, "cpp", "atom", capabilities=_CPP_THREAD_CAPABILITIES),
    DescriptorSpec("NEP", _MODEL + "nep.descriptor:NEP", AssetPolicy.REQUIRED, "cpp", "atom", capabilities=_CPP_THREAD_CAPABILITIES | {"model"}),
    DescriptorSpec("DPA4", _MODEL + "dpa4.descriptor:DPA4", AssetPolicy.REQUIRED, "torch", "atom", capabilities=_COMMON_CAPABILITIES | {"model", "cuda", "spin", "charge_spin"}, optional_extra="model"),
    DescriptorSpec("DPA4C", _MODEL + "dpa4c.descriptor:DPA4C", AssetPolicy.REQUIRED, "torch", "atom", capabilities=_COMMON_CAPABILITIES | {"model", "cuda", "spin", "charge_spin"}, optional_extra="model"),
)

builtin_registry = DescriptorRegistry(_BUILTIN_SPECS, frozen=True)
