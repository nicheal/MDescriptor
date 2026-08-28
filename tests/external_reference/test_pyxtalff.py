"""PyXtal_FF 0.2.3 comparisons for overlapping descriptor implementations.

PyXtal_FF's top-level package imports its training and model stack eagerly.  These
tests intentionally load only its descriptor submodules so the optional reference
job exercises descriptor numerics without making MDescriptor depend on that stack.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from mdescriptor import StructureBatch
from mdescriptor.descriptors import EAD, SNAP, SO3, SO4

pytestmark = [pytest.mark.reference, pytest.mark.pyxtalff]


def _water() -> Atoms:
    return Atoms(
        "OHH",
        positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
        cell=np.diag([8.0, 8.0, 8.0]),
        pbc=True,
    )


def _descriptor_module(name: str):
    """Load a PyXtal_FF descriptor without importing its unrelated training stack."""

    package_spec = importlib.util.find_spec("pyxtal_ff")
    if package_spec is None or package_spec.submodule_search_locations is None:
        pytest.skip("pyxtal_ff is not installed")

    package_path = Path(next(iter(package_spec.submodule_search_locations)))
    package = sys.modules.get("pyxtal_ff")
    if package is None:
        package = types.ModuleType("pyxtal_ff")
        package.__path__ = [str(package_path)]
        package.__package__ = "pyxtal_ff"
        package.__spec__ = package_spec
        sys.modules["pyxtal_ff"] = package

    descriptor_package = sys.modules.get("pyxtal_ff.descriptors")
    if descriptor_package is None:
        descriptor_package = types.ModuleType("pyxtal_ff.descriptors")
        descriptor_package.__path__ = [str(package_path / "descriptors")]
        descriptor_package.__package__ = "pyxtal_ff.descriptors"
        sys.modules["pyxtal_ff.descriptors"] = descriptor_package

    try:
        return importlib.import_module(f"pyxtal_ff.descriptors.{name}")
    except ImportError as exc:
        pytest.skip(f"PyXtal_FF descriptor dependencies are unavailable: {exc}")


def _assert_matches(expected: np.ndarray, actual: np.ndarray) -> None:
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_ead_matches_pyxtalff():
    module = _descriptor_module("EAD")
    system = _water()
    parameters = {"L": 2, "eta": [0.05, 0.1], "Rs": [0.0]}

    expected = np.asarray(
        module.EAD(parameters=parameters, Rc=3.5, derivative=False, stress=False).calculate(system)["x"]
    )
    actual = EAD(parameters=parameters, Rc=3.5).compute(StructureBatch.from_ase(system)).values
    _assert_matches(expected, actual)


def test_so4_matches_pyxtalff():
    module = _descriptor_module("SO4")
    system = _water()

    expected = np.asarray(
        module.SO4_Bispectrum(
            lmax=2,
            rcut=3.5,
            derivative=False,
            stress=False,
            normalize_U=False,
        ).calculate(system)["x"]
    )
    actual = SO4(lmax=2, rcut=3.5, normalize_U=False).compute(
        StructureBatch.from_ase(system)
    ).values
    _assert_matches(expected, actual)


def test_so3_matches_pyxtalff(monkeypatch):
    scipy_special = pytest.importorskip("scipy.special")
    if not hasattr(scipy_special, "sph_harm") and hasattr(scipy_special, "sph_harm_y"):
        monkeypatch.setattr(
            scipy_special,
            "sph_harm",
            lambda m, n, theta, phi: scipy_special.sph_harm_y(n, m, theta, phi),
            raising=False,
        )

    module = _descriptor_module("SO3")
    system = _water()
    parameters = {
        "nmax": 2,
        "lmax": 2,
        "rcut": 3.5,
        "alpha": 2.0,
        "weight_on": False,
    }
    expected = np.asarray(
        module.SO3(derivative=False, stress=False, **parameters).calculate(system)["x"]
    )
    actual = SO3(**parameters).compute(StructureBatch.from_ase(system)).values
    _assert_matches(expected, actual)


def test_snap_matches_pyxtalff():
    module = _descriptor_module("SNAP")
    system = _water()
    weights = {"H": 1.0, "O": 2.0}

    expected = np.asarray(
        module.SO4_Bispectrum(
            weights=weights,
            lmax=2,
            rcut=3.5,
            derivative=False,
            stress=False,
            normalize_U=False,
        ).calculate(system)["x"]
    )
    actual = SNAP(weights=weights, lmax=2, rcut=3.5, normalize_U=False).compute(
        StructureBatch.from_ase(system)
    ).values
    _assert_matches(expected, actual)
