"""Single source of truth for the descriptor numerical-baseline inventory.

The committed ``tests/golden`` files and the external-reference jobs serve
different purposes.  A committed file is a cheap regression check; an
external provider is an independent numerical oracle.  The 21 provider
entries use an ``external_manifest.json`` sidecar for the static oracle
while retaining the original project snapshot as a secondary contract
fixture.
"""

from __future__ import annotations

from typing import Final

# The values in this table are deliberately small and JSON-safe.  The paths
# are repository-relative so the inventory is portable to a wheel checkout or
# an isolated CI working directory.
BASELINES: Final[dict[str, dict[str, str]]] = {
    "SOAP": {
        "kind": "external_static",
        "provider": "DScribe",
        "package": "dscribe",
        "version": "2.1.2",
        "test": "tests/external_reference/test_dscribe.py",
        "marker": "dscribe",
    },
    "SOAPTurbo": {
        "kind": "external_static",
        "provider": "soap_turbo-master",
        "package": "soap_turbo",
        "version": "pinned source archive",
        "test": "benchmarks/_legacy_oracles/soapturbo/generate_golden.py",
    },
    "ACSF": {
        "kind": "external_static",
        "provider": "DScribe",
        "package": "dscribe",
        "version": "2.1.2",
        "test": "tests/external_reference/test_dscribe.py",
        "marker": "dscribe",
    },
    "ACE": {
        "kind": "external_static",
        "provider": "ACE1.jl",
        "package": "ACE1.jl",
        "version": "0.12.5",
        "test": "scripts/generate_ace1_reference.py",
    },
    "CoulombMatrix": {
        "kind": "external_static",
        "provider": "DScribe",
        "package": "dscribe",
        "version": "2.1.2",
        "test": "tests/external_reference/test_dscribe.py",
        "marker": "dscribe",
    },
    "SineMatrix": {
        "kind": "external_static",
        "provider": "DScribe",
        "package": "dscribe",
        "version": "2.1.2",
        "test": "tests/external_reference/test_dscribe.py",
        "marker": "dscribe",
    },
    "EwaldSumMatrix": {
        "kind": "external_static",
        "provider": "DScribe",
        "package": "dscribe",
        "version": "2.1.2",
        "test": "tests/external_reference/test_dscribe.py",
        "marker": "dscribe",
    },
    "MBTR": {
        "kind": "external_static",
        "provider": "DScribe",
        "package": "dscribe",
        "version": "2.1.2",
        "test": "tests/external_reference/test_dscribe.py",
        "marker": "dscribe",
    },
    "LMBTR": {
        "kind": "external_static",
        "provider": "DScribe",
        "package": "dscribe",
        "version": "2.1.2",
        "test": "tests/external_reference/test_dscribe.py",
        "marker": "dscribe",
    },
    "ValleOganov": {
        "kind": "external_static",
        "provider": "DScribe",
        "package": "dscribe",
        "version": "2.1.2",
        "test": "tests/external_reference/test_dscribe.py",
        "marker": "dscribe",
    },
    "AtomicComposition": {
        "kind": "external_static",
        "provider": "Featomic",
        "package": "featomic",
        "version": "0.6.6",
        "test": "tests/external_reference/test_featomic.py",
        "marker": "featomic",
    },
    "NeighborList": {
        "kind": "external_static",
        "provider": "Featomic",
        "package": "featomic",
        "version": "0.6.6",
        "test": "tests/external_reference/test_featomic.py",
        "marker": "featomic",
    },
    "SortedDistances": {
        "kind": "external_static",
        "provider": "Featomic",
        "package": "featomic",
        "version": "0.6.6",
        "test": "tests/external_reference/test_featomic.py",
        "marker": "featomic",
    },
    "SphericalExpansion": {
        "kind": "external_static",
        "provider": "Featomic",
        "package": "featomic",
        "version": "0.6.6",
        "test": "tests/external_reference/test_featomic.py",
        "marker": "featomic",
    },
    "SphericalExpansionByPair": {
        "kind": "external_static",
        "provider": "Featomic",
        "package": "featomic",
        "version": "0.6.6",
        "test": "tests/external_reference/test_featomic.py",
        "marker": "featomic",
    },
    "SoapRadialSpectrum": {
        "kind": "external_static",
        "provider": "Featomic",
        "package": "featomic",
        "version": "0.6.6",
        "test": "tests/external_reference/test_featomic.py",
        "marker": "featomic",
    },
    "SoapPowerSpectrum": {
        "kind": "external_static",
        "provider": "Featomic",
        "package": "featomic",
        "version": "0.6.6",
        "test": "tests/external_reference/test_featomic.py",
        "marker": "featomic",
    },
    "LodeSphericalExpansion": {
        "kind": "external_static",
        "provider": "Featomic",
        "package": "featomic",
        "version": "0.6.6",
        "test": "tests/external_reference/test_featomic.py",
        "marker": "featomic",
    },
    "EAD": {
        "kind": "external_static",
        "provider": "PyXtal_FF",
        "package": "pyxtal_ff",
        "version": "0.2.3",
        "test": "tests/external_reference/test_pyxtalff.py",
        "marker": "pyxtalff",
    },
    "SO3": {
        "kind": "external_static",
        "provider": "PyXtal_FF",
        "package": "pyxtal_ff",
        "version": "0.2.3",
        "test": "tests/external_reference/test_pyxtalff.py",
        "marker": "pyxtalff",
    },
    "SO4": {
        "kind": "external_static",
        "provider": "PyXtal_FF",
        "package": "pyxtal_ff",
        "version": "0.2.3",
        "test": "tests/external_reference/test_pyxtalff.py",
        "marker": "pyxtalff",
    },
    "SNAP": {
        "kind": "external_static",
        "provider": "PyXtal_FF",
        "package": "pyxtal_ff",
        "version": "0.2.3",
        "test": "tests/external_reference/test_pyxtalff.py",
        "marker": "pyxtalff",
    },
    "LBispectrum": {
        "kind": "external_static",
        "provider": "LAMMPS/PyXtal_FF",
        "package": "lammps + pyxtal_ff",
        "version": "pinned source archive + 0.2.3",
        "test": "benchmarks/_legacy_oracles/lbispectrum/generate_golden.py",
    },
    "MTP": {
        "kind": "external_static",
        "provider": "MLIP-4",
        "package": "MLIP-4",
        "version": "pinned source archive",
        "test": "benchmarks/_legacy_oracles/mtp/generate_golden.py",
    },
    "C00PSMLFF": {
        "kind": "external_static",
        "provider": "licensed external MLFF",
        "package": "local-only input",
        "version": "user-supplied",
        "test": "scripts/generate_external_c00ps_reference.py",
    },
    "NEP": {
        "kind": "external_static",
        "provider": "nep-adapters",
        "package": "nep-adapters",
        "version": "1.0.2",
        "test": "tests/external_reference/test_nep_adapters.py",
        "marker": "nepadapters",
    },
    "DPA4": {
        "kind": "external_static",
        "provider": "deepmd-kit",
        "package": "deepmd-kit",
        "version": "3.2.0",
        "test": "tests/external_reference/test_deepmd.py",
        "marker": "deepmd",
    },
    "DPA4C": {
        "kind": "external_static",
        "provider": "deepmd-kit",
        "package": "deepmd-kit",
        "version": "3.2.0",
        "test": "tests/external_reference/test_deepmd.py",
        "marker": "deepmd",
    },
}


STATIC_GOLDEN_KINDS: Final[frozenset[str]] = frozenset(
    {"ace1_julia_source", "licensed_external_mlff_source", "deepmd_kit", "external_upstream"}
)


def external_runtime_baselines() -> dict[str, dict[str, str]]:
    """Return descriptors with a pinned provider smoke test."""

    return {name: dict(data) for name, data in BASELINES.items() if "marker" in data}


def external_static_baselines() -> dict[str, dict[str, str]]:
    """Return the committed external-golden subset."""

    return {name: dict(data) for name, data in BASELINES.items() if data["kind"] == "external_static"}
