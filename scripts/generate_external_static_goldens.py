"""Generate static goldens from the pinned third-party descriptor providers.

This is an explicit, opt-in generation command.  It writes a small periodic
water fixture beside each legacy project snapshot under ``tests/golden`` and
uses only the independent provider for the values.  MDescriptor is called
afterwards solely to obtain the public result contract (labels, samples,
metadata and row offsets) and to verify the provider output before accepting
it.

The provider packages are intentionally not imported by the normal test job.
Run this command in an environment containing the exact reference extras, for
example with ``PYTHONPATH`` pointed at an isolated ``featomic==0.6.6`` wheel.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from mdescriptor import DescriptorConfiguration, StructureBatch, create_descriptor

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_ROOT = ROOT / "tests" / "golden"
GENERATOR = Path(__file__).resolve()

WATER_NUMBERS = np.asarray([8, 1, 1], dtype=np.int32)
WATER_POSITIONS = np.asarray(
    [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
    dtype=np.float64,
)
WATER_CELL = np.diag([8.0, 8.0, 8.0]).astype(np.float64)

TARGETS = (
    "SOAP",
    "ACSF",
    "CoulombMatrix",
    "SineMatrix",
    "EwaldSumMatrix",
    "MBTR",
    "LMBTR",
    "ValleOganov",
    "AtomicComposition",
    "NeighborList",
    "SortedDistances",
    "SphericalExpansion",
    "SphericalExpansionByPair",
    "SoapRadialSpectrum",
    "SoapPowerSpectrum",
    "LodeSphericalExpansion",
    "EAD",
    "SO3",
    "SO4",
    "SNAP",
    "NEP",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable(value: Any) -> Any:
    """Replace checkout-specific paths in the generated manifest."""

    package_root = Path(importlib.import_module("mdescriptor").__file__).resolve().parent
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        package = str(package_root)
        root = str(ROOT)
        if value == package:
            return "${PACKAGE_ROOT}"
        if value.startswith(package + "/"):
            return "${PACKAGE_ROOT}/" + value[len(package) + 1 :]
        if value == root:
            return "${PROJECT_ROOT}"
        if value.startswith(root + "/"):
            return "${PROJECT_ROOT}/" + value[len(root) + 1 :]
        return value
    if isinstance(value, dict):
        return {str(key): _portable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _batch() -> StructureBatch:
    return StructureBatch(
        WATER_NUMBERS,
        WATER_POSITIONS,
        WATER_CELL[None],
        np.ones((1, 3), dtype=np.int32),
        np.asarray([0, 3], dtype=np.int64),
        ("external-water-periodic",),
    )


def _atoms() -> Atoms:
    return Atoms(
        numbers=WATER_NUMBERS,
        positions=WATER_POSITIONS,
        cell=WATER_CELL,
        pbc=True,
    )


def _configuration(name: str) -> dict[str, Any]:
    species = [1, 8]
    configurations: dict[str, dict[str, Any]] = {
        "SOAP": {
            "species": species,
            "r_cut": 3.5,
            "n_max": 3,
            "l_max": 2,
            "sigma": 0.5,
            "average": "off",
            "rbf": "gto",
            "compression": {"mode": "off", "species_weighting": None},
        },
        "ACSF": {
            "species": species,
            "r_cut": 3.5,
            "g2_params": {"eta": [0.4, 1.0], "Rs": [0.0]},
            "g3_params": [0.7, 1.3],
            "g4_params": {"eta": [0.2], "zeta": [1.0, 2.0], "lambda": [1.0]},
            "g5_params": {"eta": [0.3], "zeta": [1.0, 2.0], "lambda": [1.0]},
        },
        "CoulombMatrix": {"n_atoms_max": 3, "permutation": "none"},
        "SineMatrix": {"n_atoms_max": 3, "permutation": "none"},
        "EwaldSumMatrix": {
            "n_atoms_max": 3,
            "permutation": "none",
            "accuracy": 1e-5,
            "w": 1.0,
            "r_cut": 6.0,
            "g_cut": 3.0,
            "a": 0.3,
        },
        "MBTR": {
            "species": species,
            "geometry": {"function": "distance"},
            "grid": {"min": 0.0, "max": 4.0, "n": 20, "sigma": 0.1},
            "weighting": {"function": "exp", "scale": 0.3, "threshold": 1e-3},
            "periodic": True,
        },
        "LMBTR": {
            "species": species,
            "geometry": {"function": "distance"},
            "grid": {"min": 0.0, "max": 4.0, "n": 20, "sigma": 0.1},
            "weighting": {"function": "exp", "scale": 0.3, "threshold": 1e-3},
            "periodic": True,
        },
        "ValleOganov": {
            "species": species,
            "function": "distance",
            "n": 20,
            "sigma": 0.1,
            "r_cut": 3.5,
        },
        "AtomicComposition": {"species": species, "per_system": False},
        "NeighborList": {"cutoff": 3.5, "full_neighbor_list": True, "self_pairs": False},
        "SortedDistances": {
            "species": species,
            "cutoff": 3.5,
            "max_neighbors": 4,
            "separate_neighbor_types": True,
        },
        "SphericalExpansion": {
            "species": species,
            "cutoff": 3.5,
            "density_width": 0.6,
            "max_radial": 2,
            "max_angular": 2,
        },
        "SphericalExpansionByPair": {
            "species": species,
            "cutoff": 3.5,
            "density_width": 0.6,
            "max_radial": 2,
            "max_angular": 2,
        },
        "SoapRadialSpectrum": {
            "species": species,
            "cutoff": 3.5,
            "density_width": 0.6,
            "max_radial": 2,
            "max_angular": 2,
        },
        "SoapPowerSpectrum": {
            "species": species,
            "cutoff": 3.5,
            "density_width": 0.6,
            "max_radial": 2,
            "max_angular": 2,
        },
        "LodeSphericalExpansion": {
            "species": species,
            "cutoff": 3.5,
            "density_width": 0.5,
            "max_radial": 2,
            "max_angular": 2,
            "k_cutoff": 2.5,
            "exponent": 1,
            "radial_radius": 3.5,
        },
        "EAD": {
            "parameters": {"L": 2, "eta": [0.05, 0.1], "Rs": [0.0]},
            "Rc": 3.5,
            "cutoff": "cosine",
        },
        "SO3": {"nmax": 2, "lmax": 2, "rcut": 3.5, "alpha": 2.0, "weight_on": False},
        "SO4": {"lmax": 2, "rcut": 3.5, "normalize_U": False},
        "SNAP": {
            "weights": {"H": 1.0, "O": 2.0},
            "lmax": 2,
            "rcut": 3.5,
            "normalize_U": False,
        },
    }
    if name == "NEP":
        parent = json.loads(
            (GOLDEN_ROOT / "nep" / "manifest.json").read_text(encoding="utf-8")
        )
        return copy.deepcopy(parent["configuration"]["parameters"])
    try:
        return copy.deepcopy(configurations[name])
    except KeyError as exc:
        raise ValueError(f"no external static configuration for {name!r}") from exc


def _rows(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 1:
        return result.reshape(1, -1)
    if result.ndim != 2:
        raise ValueError(f"provider returned an unsupported shape: {result.shape}")
    return result


def _require_version(distribution: str, expected: str) -> None:
    actual = importlib.metadata.version(distribution)
    if actual != expected:
        raise RuntimeError(f"expected {distribution}=={expected}, got {actual}")


def _dscribe_values(name: str, system: Atoms) -> np.ndarray:
    _require_version("dscribe", "2.1.2")
    from dscribe.descriptors import ACSF as DscribeACSF
    from dscribe.descriptors import LMBTR as DscribeLMBTR
    from dscribe.descriptors import MBTR as DscribeMBTR
    from dscribe.descriptors import SOAP as DscribeSOAP
    from dscribe.descriptors import CoulombMatrix as DscribeCoulombMatrix
    from dscribe.descriptors import EwaldSumMatrix as DscribeEwaldSumMatrix
    from dscribe.descriptors import SineMatrix as DscribeSineMatrix
    from dscribe.descriptors import ValleOganov as DscribeValleOganov

    parameters = _configuration(name)
    if name == "SOAP":
        return _rows(DscribeSOAP(periodic=True, **parameters).create(system))
    if name == "ACSF":
        provider_parameters = dict(parameters)
        for key, fields in {
            "g2_params": ("eta", "Rs"),
            "g4_params": ("eta", "zeta", "lambda"),
            "g5_params": ("eta", "zeta", "lambda"),
        }.items():
            values = provider_parameters[key]
            if key == "g2_params":
                provider_parameters[key] = [
                    [float(eta), float(rs)]
                    for eta in values[fields[0]]
                    for rs in values[fields[1]]
                ]
            else:
                provider_parameters[key] = [
                    [float(eta), float(zeta), float(lam)]
                    for eta in values[fields[0]]
                    for zeta in values[fields[1]]
                    for lam in values[fields[2]]
                ]
        return _rows(DscribeACSF(periodic=True, **provider_parameters).create(system))
    if name == "CoulombMatrix":
        return _rows(DscribeCoulombMatrix(**parameters).create(system))
    if name == "SineMatrix":
        return _rows(DscribeSineMatrix(**parameters).create(system))
    if name == "EwaldSumMatrix":
        evaluation = {
            key: parameters[key] for key in ("accuracy", "w", "r_cut", "g_cut", "a")
        }
        constructor = {
            key: parameters[key] for key in ("n_atoms_max", "permutation")
        }
        return _rows(DscribeEwaldSumMatrix(**constructor).create(system, **evaluation))
    if name == "MBTR":
        return _rows(DscribeMBTR(**parameters).create(system))
    if name == "LMBTR":
        return _rows(DscribeLMBTR(**parameters).create(system))
    if name == "ValleOganov":
        return _rows(DscribeValleOganov(**parameters).create(system))
    raise ValueError(f"{name} is not a DScribe descriptor")


def _block_data(block: Any) -> tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(block.samples.values, dtype=np.int64)
    raw_values = np.asarray(block.values, dtype=np.float64)
    width = int(np.prod(raw_values.shape[1:], dtype=np.int64))
    return samples, raw_values.reshape(samples.shape[0], width)


def _keys_and_blocks(tensor_map: Any) -> dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]]:
    return {
        tuple(int(value) for value in key): _block_data(tensor_map[key])
        for key in tensor_map.keys
    }


def _values_for_atom(data: tuple[np.ndarray, np.ndarray], atom: int, width: int) -> np.ndarray:
    samples, values = data
    matches = np.flatnonzero((samples[:, 0] == 0) & (samples[:, 1] == atom))
    if len(matches) == 0:
        return np.zeros(width, dtype=np.float64)
    if len(matches) != 1:
        raise ValueError(f"expected one provider row for atom {atom}, got {len(matches)}")
    return values[matches[0]]


def _flatten_atomic_composition(tensor_map: Any) -> np.ndarray:
    blocks = _keys_and_blocks(tensor_map)
    result = np.zeros((3, 2), dtype=np.float64)
    for column, species in enumerate((1, 8)):
        samples, values = blocks[(species,)]
        for sample, value in zip(samples, values, strict=True):
            result[int(sample[1]), column] = value[0]
    return result


def _flatten_sorted_distances(tensor_map: Any) -> np.ndarray:
    blocks = _keys_and_blocks(tensor_map)
    result = np.zeros((3, 8), dtype=np.float64)
    for atom, center in enumerate(WATER_NUMBERS):
        offset = 0
        for neighbor in (1, 8):
            data = blocks.get(
                (int(center), neighbor),
                (np.empty((0, 2)), np.empty((0, 4))),
            )
            result[atom, offset : offset + 4] = _values_for_atom(data, atom, 4)
            offset += 4
    return result


def _flatten_spherical(tensor_map: Any) -> np.ndarray:
    blocks = _keys_and_blocks(tensor_map)
    radial_count = 3
    max_angular = 2
    group_width = sum((2 * angular + 1) * radial_count for angular in range(max_angular + 1))
    result = np.zeros((3, 2 * 2 * group_width), dtype=np.float64)
    offset = 0
    for center in (1, 8):
        for neighbor in (1, 8):
            for angular in range(max_angular + 1):
                width = (2 * angular + 1) * radial_count
                candidates = [
                    (key, data)
                    for key, data in blocks.items()
                    if key[0] == angular and key[2:] == (center, neighbor)
                ]
                data = candidates[0][1] if candidates else (np.empty((0, 2)), np.empty((0, width)))
                for atom in range(3):
                    result[atom, offset : offset + width] = _values_for_atom(data, atom, width)
                offset += width
    return result


def _flatten_power(tensor_map: Any) -> np.ndarray:
    blocks = _keys_and_blocks(tensor_map)
    # The power-spectrum block keeps the angular channel as the leading
    # feature axis: (max_angular + 1) * radial * radial.
    group_width = 3 * 3 * 3
    group_count = 3
    result = np.zeros((3, 2 * group_count * group_width), dtype=np.float64)
    offset = 0
    for center in (1, 8):
        for first_index, first in enumerate((1, 8)):
            for second in (1, 8)[first_index:]:
                data = blocks.get(
                    (center, first, second),
                    (np.empty((0, 2)), np.empty((0, group_width))),
                )
                for atom in range(3):
                    result[atom, offset : offset + group_width] = _values_for_atom(
                        data, atom, group_width
                    )
                offset += group_width
    return result


def _flatten_radial(tensor_map: Any) -> np.ndarray:
    blocks = _keys_and_blocks(tensor_map)
    result = np.zeros((3, 2 * 2 * 3), dtype=np.float64)
    offset = 0
    for center in (1, 8):
        for neighbor in (1, 8):
            data = blocks.get(
                (center, neighbor),
                (np.empty((0, 2)), np.empty((0, 3))),
            )
            for atom in range(3):
                result[atom, offset : offset + 3] = _values_for_atom(data, atom, 3)
            offset += 3
    return result


def _flatten_neighbor_values(tensor_map: Any, native_samples: np.ndarray) -> np.ndarray:
    rows: dict[tuple[int, ...], np.ndarray] = {}
    for key in tensor_map.keys:
        samples, vectors = _block_data(tensor_map[key])
        vectors = vectors.reshape(samples.shape[0], 3)
        for sample, vector in zip(samples, vectors, strict=True):
            rows[tuple(int(value) for value in sample)] = np.concatenate(
                (vector, [np.linalg.norm(vector)])
            )
    try:
        return np.asarray([rows[tuple(int(value) for value in sample)] for sample in native_samples])
    except KeyError as exc:
        raise ValueError(f"provider did not return neighbor sample {exc.args[0]}") from exc


def _featomic_values(name: str, system: Atoms, native_samples: np.ndarray) -> np.ndarray:
    _require_version("featomic", "0.6.6")
    import featomic
    from featomic.basis import Gto, TensorProduct
    from featomic.cutoff import Cutoff, ShiftedCosine
    from featomic.density import Gaussian, SmearedPowerLaw

    cutoff = 3.5
    radial = Gto(max_radial=2, radius=cutoff)
    basis = TensorProduct(max_angular=2, radial=radial)
    shifted = Cutoff(cutoff, ShiftedCosine(width=0.5))
    density = Gaussian(width=0.6)
    if name == "AtomicComposition":
        return _flatten_atomic_composition(featomic.AtomicComposition(per_system=False).compute(system))
    if name == "SortedDistances":
        return _flatten_sorted_distances(
            featomic.SortedDistances(
                cutoff=cutoff, max_neighbors=4, separate_neighbor_types=True
            ).compute(system)
        )
    if name == "NeighborList":
        tensor_map = featomic.NeighborList(cutoff=cutoff, full_neighbor_list=True).compute(system)
        return _flatten_neighbor_values(tensor_map, native_samples)
    if name == "SphericalExpansion":
        return _flatten_spherical(
            featomic.SphericalExpansion(cutoff=shifted, density=density, basis=basis).compute(system)
        )
    if name == "SphericalExpansionByPair":
        tensor_map = featomic.SphericalExpansionByPair(
            cutoff=shifted, density=density, basis=basis
        ).compute(system)
        blocks = _keys_and_blocks(tensor_map)
        group_width = sum((2 * angular + 1) * 3 for angular in range(3))
        result = np.zeros((9, group_width), dtype=np.float64)
        for first in range(3):
            for second in range(3):
                offset = 0
                for angular in range(3):
                    width = (2 * angular + 1) * 3
                    data = blocks.get(
                        (angular, 1, int(WATER_NUMBERS[first]), int(WATER_NUMBERS[second])),
                        (np.empty((0, 6)), np.empty((0, width))),
                    )
                    samples, values = data
                    matches = np.flatnonzero(
                        (samples[:, 0] == 0)
                        & (samples[:, 1] == first)
                        & (samples[:, 2] == second)
                    )
                    if len(matches) == 1:
                        result[first * 3 + second, offset : offset + width] = values[matches[0]]
                    elif len(matches) > 1:
                        raise ValueError("provider returned duplicate pair samples")
                    offset += width
        return result
    if name == "SoapPowerSpectrum":
        return _flatten_power(
            featomic.SoapPowerSpectrum(cutoff=shifted, density=density, basis=basis).compute(system)
        )
    if name == "SoapRadialSpectrum":
        return _flatten_radial(
            featomic.SoapRadialSpectrum(
                cutoff=shifted,
                density=density,
                basis={"radial": Gto(max_radial=2, radius=cutoff)},
            ).compute(system)
        )
    if name == "LodeSphericalExpansion":
        lode = featomic.LodeSphericalExpansion(
            density=SmearedPowerLaw(smearing=0.5, exponent=1),
            basis=basis,
            k_cutoff=2.5,
        ).compute(system)
        return _flatten_spherical(lode)
    raise ValueError(f"{name} is not a Featomic descriptor")


def _pyxtal_module(name: str) -> Any:
    _require_version("pyxtal-ff", "0.2.3")
    package_spec = importlib.util.find_spec("pyxtal_ff")
    if package_spec is None or package_spec.submodule_search_locations is None:
        raise RuntimeError("pyxtal_ff==0.2.3 is not installed")
    package_path = Path(next(iter(package_spec.submodule_search_locations)))
    package = sys.modules.get("pyxtal_ff")
    if package is None:
        package = importlib.util.module_from_spec(package_spec)
        sys.modules["pyxtal_ff"] = package
    descriptor_package = sys.modules.get("pyxtal_ff.descriptors")
    if descriptor_package is None:
        descriptor_package = type(sys)("pyxtal_ff.descriptors")
        descriptor_package.__path__ = [str(package_path / "descriptors")]
        sys.modules["pyxtal_ff.descriptors"] = descriptor_package
    return importlib.import_module(f"pyxtal_ff.descriptors.{name}")


def _pyxtal_values(name: str, system: Atoms) -> np.ndarray:
    if name == "SO3":
        import scipy.special as scipy_special

        if not hasattr(scipy_special, "sph_harm") and hasattr(scipy_special, "sph_harm_y"):
            scipy_special.sph_harm = lambda m, n, theta, phi: scipy_special.sph_harm_y(
                n, m, theta, phi
            )
    module = _pyxtal_module({"EAD": "EAD", "SO3": "SO3", "SO4": "SO4", "SNAP": "SNAP"}[name])
    if name == "EAD":
        parameters = _configuration(name)
        return np.asarray(
            module.EAD(parameters=parameters["parameters"], Rc=parameters["Rc"], derivative=False, stress=False).calculate(system)["x"],
            dtype=np.float64,
        )
    if name == "SO3":
        parameters = _configuration(name)
        return np.asarray(
            module.SO3(derivative=False, stress=False, **parameters).calculate(system)["x"],
            dtype=np.float64,
        )
    parameters = _configuration(name)
    kwargs = {
        "lmax": parameters["lmax"],
        "rcut": parameters["rcut"],
        "derivative": False,
        "stress": False,
        "normalize_U": parameters["normalize_U"],
    }
    if name == "SNAP":
        kwargs["weights"] = parameters["weights"]
    return np.asarray(module.SO4_Bispectrum(**kwargs).calculate(system)["x"], dtype=np.float64)


def _nep_values(system: Atoms) -> np.ndarray:
    _require_version("nep-adapters", "1.0.2")
    from nep_adapters import NEPCalculator

    from mdescriptor.models import NEP_MODEL

    reference = NEPCalculator(str(NEP_MODEL))
    try:
        return np.asarray(reference.get_descriptor(system), dtype=np.float64)
    finally:
        reference.close()


def _provider_values(name: str, system: Atoms, native_samples: np.ndarray) -> np.ndarray:
    if name in {
        "SOAP",
        "ACSF",
        "CoulombMatrix",
        "SineMatrix",
        "EwaldSumMatrix",
        "MBTR",
        "LMBTR",
        "ValleOganov",
    }:
        return _dscribe_values(name, system)
    if name in {
        "AtomicComposition",
        "NeighborList",
        "SortedDistances",
        "SphericalExpansion",
        "SphericalExpansionByPair",
        "SoapRadialSpectrum",
        "SoapPowerSpectrum",
        "LodeSphericalExpansion",
    }:
        return _featomic_values(name, system, native_samples)
    if name in {"EAD", "SO3", "SO4", "SNAP"}:
        return _pyxtal_values(name, system)
    if name == "NEP":
        return _nep_values(system)
    raise ValueError(f"no provider for {name!r}")


def _tolerance(name: str) -> dict[str, float]:
    if name in {
        "SphericalExpansion",
        "SphericalExpansionByPair",
        "SoapRadialSpectrum",
        "SoapPowerSpectrum",
        "LodeSphericalExpansion",
    }:
        return {"rtol": 1e-7, "atol": 5e-8}
    if name == "NEP":
        return {"rtol": 1e-6, "atol": 1e-7}
    if name in {"EAD", "SO3", "SO4", "SNAP"}:
        return {"rtol": 1e-9, "atol": 1e-10}
    return {"rtol": 1e-9, "atol": 1e-11}


def _write_one(name: str, *, accept: bool) -> None:
    fixture_dir = GOLDEN_ROOT / name.lower()
    parent_manifest_path = fixture_dir / "manifest.json"
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    config = {
        "schema_version": 1,
        "descriptor": name,
        "parameters": _configuration(name),
    }
    batch = _batch()
    system = _atoms()
    # Obtain only sample/schema information from the project implementation.
    contract_descriptor = create_descriptor(DescriptorConfiguration.from_dict(config))
    try:
        contract = contract_descriptor.compute(batch)
        native_samples = np.asarray(contract.samples, dtype=np.int64)
        provider = _provider_values(name, system, native_samples)
        provider = np.asarray(provider, dtype=np.float64)
        tolerance = _tolerance(name)
        if provider.shape != contract.values.shape:
            raise RuntimeError(
                f"{name}: provider/project shape mismatch: {provider.shape} != {contract.values.shape}"
            )
        delta = np.abs(provider - np.asarray(contract.values, dtype=np.float64))
        np.testing.assert_allclose(
            provider,
            contract.values,
            rtol=tolerance["rtol"],
            atol=tolerance["atol"],
        )
        result = {
            "level": contract.level.value,
            "feature_count": int(contract.feature_count),
            "labels": list(contract.labels),
            "structure_ids": list(contract.structure_ids),
            "row_offsets": None
            if contract.row_offsets is None
            else contract.row_offsets.tolist(),
            "metadata": _portable(contract.metadata),
        }
    finally:
        contract_descriptor.close()

    input_path = fixture_dir / "external_input.npz"
    output_path = fixture_dir / "external_expected_output.npz"
    np.savez_compressed(
        input_path,
        numbers=WATER_NUMBERS,
        positions=WATER_POSITIONS,
        cells=WATER_CELL[None],
        pbc=np.ones((1, 3), dtype=np.int32),
        offsets=np.asarray([0, 3], dtype=np.int64),
    )
    np.savez_compressed(output_path, values=provider, samples=native_samples)
    provider_name = {
        "SOAP": "DScribe",
        "ACSF": "DScribe",
        "CoulombMatrix": "DScribe",
        "SineMatrix": "DScribe",
        "EwaldSumMatrix": "DScribe",
        "MBTR": "DScribe",
        "LMBTR": "DScribe",
        "ValleOganov": "DScribe",
        "AtomicComposition": "Featomic",
        "NeighborList": "Featomic",
        "SortedDistances": "Featomic",
        "SphericalExpansion": "Featomic",
        "SphericalExpansionByPair": "Featomic",
        "SoapRadialSpectrum": "Featomic",
        "SoapPowerSpectrum": "Featomic",
        "LodeSphericalExpansion": "Featomic",
        "EAD": "PyXtal_FF",
        "SO3": "PyXtal_FF",
        "SO4": "PyXtal_FF",
        "SNAP": "PyXtal_FF",
        "NEP": "nep-adapters",
    }[name]
    distribution = {
        "DScribe": ("dscribe", "2.1.2"),
        "Featomic": ("featomic", "0.6.6"),
        "PyXtal_FF": ("pyxtal_ff", "0.2.3"),
        "nep-adapters": ("nep-adapters", "1.0.2"),
    }[provider_name]
    reference = {
        "kind": "external_static",
        "provider": provider_name,
        "package": distribution[0],
        "version": distribution[1],
        "generator": str(GENERATOR.relative_to(ROOT)),
        "generator_sha256": _sha256(GENERATOR),
        "source": f"{distribution[0]} runtime evaluation",
        "verification": {
            **tolerance,
            "max_abs": float(delta.max(initial=0.0)),
            "mae": float(delta.mean()) if delta.size else 0.0,
        },
    }
    external_manifest = {
        "schema_version": 1,
        "descriptor": name,
        "configuration": config,
        "dataset": {
            "name": "external-water-v1",
            "sha256": _sha256(input_path),
            "source": "pinned-external-provider",
        },
        "input": input_path.name,
        "input_ids": ["external-water-periodic"],
        "expected_output": output_path.name,
        "nonperiodic": {"mode": "output"},
        "reference": reference,
        "result": result,
        "tolerance": tolerance,
    }
    external_manifest_path = fixture_dir / "external_manifest.json"
    old_project_hash = _sha256(fixture_dir / parent_manifest["expected_output"])
    parent_baseline = dict(parent_manifest.get("numeric_baseline", {}))
    parent_baseline.update(
        {
            "kind": "external_static",
            "committed_golden": "external_static",
            "fixture": external_manifest_path.name,
            "generator": str(GENERATOR.relative_to(ROOT)),
            "generator_sha256": _sha256(GENERATOR),
            "expected_output_sha256": _sha256(output_path),
            "project_snapshot_sha256": old_project_hash,
        }
    )
    parent_manifest["numeric_baseline"] = parent_baseline
    if accept:
        external_manifest_path.write_text(
            json.dumps(external_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        parent_manifest_path.write_text(
            json.dumps(parent_manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"accepted {name} external static golden: {provider.shape}")
    else:
        print(f"verified {name} external static golden: {provider.shape}; pass --accept to write")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor", choices=TARGETS)
    parser.add_argument("--accept", action="store_true", help="write the generated fixtures")
    args = parser.parse_args(argv)
    if not args.accept:
        raise SystemExit("refusing to write external golden without explicit --accept")
    names = (args.descriptor,) if args.descriptor else TARGETS
    for name in names:
        _write_one(name, accept=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
