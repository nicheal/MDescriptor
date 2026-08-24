"""Public algorithm namespace grouped by asset policy.

Implementations are imported lazily. Shared contracts and errors live in
``mdescriptor.core``; the private migration adapters under ``mdescriptor._legacy``
are never re-exported here.
"""

__all__ = [
    "ACSF",
    "C00PSMLFF",
    "AtomicComposition",
    "CoulombMatrix",
    "DPA4",
    "DPA4C",
    "EAD",
    "EwaldSumMatrix",
    "LBispectrum",
    "LMBTR",
    "MBTR",
    "MTP",
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
    "LodeSphericalExpansion",
    "ValleOganov",
]


_PUBLIC_MODULES = {
    "ACSF": ".standalone.acsf",
    "C00PSMLFF": ".standalone.c00ps_mlff",
    "AtomicComposition": ".standalone.local",
    "CoulombMatrix": ".standalone.matrices",
    "EAD": ".standalone.rotational",
    "EwaldSumMatrix": ".standalone.matrices",
    "LBispectrum": ".standalone.rotational",
    "LMBTR": ".standalone.matrices",
    "MBTR": ".standalone.matrices",
    "MTP": ".standalone.mtp",
    "NEP": ".model_backed.nep.descriptor",
    "NeighborList": ".standalone.local",
    "SNAP": ".standalone.rotational",
    "SOAP": ".standalone.soap",
    "SOAPTurbo": ".standalone.soap_turbo",
    "SO3": ".standalone.rotational",
    "SO4": ".standalone.rotational",
    "SineMatrix": ".standalone.matrices",
    "SortedDistances": ".standalone.local",
    "SoapPowerSpectrum": ".standalone.local",
    "SoapRadialSpectrum": ".standalone.local",
    "SphericalExpansion": ".standalone.local",
    "SphericalExpansionByPair": ".standalone.local",
    "LodeSphericalExpansion": ".standalone.local",
    "ValleOganov": ".standalone.matrices",
    "DPA4": ".model_backed.dpa4.descriptor",
    "DPA4C": ".model_backed.dpa4c.descriptor",
}


def __getattr__(name: str):
    module_name = _PUBLIC_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    from importlib import import_module

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
