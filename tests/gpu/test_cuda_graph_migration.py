"""CUDA regression tests for the device graph and pair-order migration."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

import mdescriptor
from mdescriptor import ExecutionOptions, MDescriptorError, StructureBatch
from mdescriptor.descriptors import (
    NeighborList,
    SoapPowerSpectrum,
    SoapRadialSpectrum,
    SphericalExpansion,
    SphericalExpansionByPair,
)


def _load_cuda_plugin() -> None:
    """Make a source-tree CUDA build visible to the editable package."""

    try:
        importlib.import_module("mdescriptor._cuda")
        return
    except (ImportError, OSError):
        pass

    configured = os.environ.get("MDESCRIPTOR_CUDA_PLUGIN_DIR")
    candidates = [
        Path(configured) if configured else None,
        Path(__file__).parents[2] / "build-cuda",
    ]
    for candidate in candidates:
        if candidate is None or not any(candidate.glob("_cuda*.so")):
            continue
        mdescriptor.__path__.insert(0, str(candidate))
        try:
            importlib.import_module("mdescriptor._cuda")
        except (ImportError, OSError):
            continue
        return
    pytest.skip("CUDA plugin is not installed in this test environment")


def _batch() -> StructureBatch:
    return StructureBatch.from_ase(
        [
            Atoms(
                numbers=[1, 8, 6],
                positions=[
                    [0.15, 0.25, 0.35],
                    [1.35, 0.35, 0.25],
                    [3.7, 2.1, 1.4],
                ],
                cell=[
                    [5.0, 0.0, 0.0],
                    [0.35, 4.8, 0.0],
                    [0.2, 0.25, 5.4],
                ],
                pbc=True,
            ),
            Atoms(
                numbers=[1, 1, 8],
                positions=[
                    [-0.3, 0.0, 0.1],
                    [1.15, 0.2, -0.15],
                    [0.4, 1.3, 0.0],
                ],
            ),
        ],
        ids=["periodic", "isolated"],
    )


def _compute_or_skip(descriptor: object, batch: StructureBatch):
    try:
        return descriptor.compute(batch)
    except MDescriptorError as error:
        if error.code == "device_unavailable":
            pytest.skip(str(error))
        raise


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("full_neighbor_list", "self_pairs"),
    [(True, False), (False, True)],
)
def test_cuda_neighbor_list_matches_cpu_order_and_filtering(
    full_neighbor_list: bool,
    self_pairs: bool,
) -> None:
    """Device CSR filtering preserves public pair order and exact-self rules."""

    _load_cuda_plugin()
    batch = _batch()
    parameters = {
        "cutoff": 3.5,
        "full_neighbor_list": full_neighbor_list,
        "self_pairs": self_pairs,
    }
    cpu = NeighborList(**parameters, execution=ExecutionOptions(device="cpu", num_threads=1))
    gpu = NeighborList(**parameters, execution=ExecutionOptions(device="cuda"))
    try:
        expected = cpu.compute(batch)
        actual = _compute_or_skip(gpu, batch)
        np.testing.assert_array_equal(actual.samples, expected.samples)
        np.testing.assert_array_equal(actual.row_offsets, expected.row_offsets)
        np.testing.assert_allclose(actual.values, expected.values, rtol=1e-12, atol=1e-12)
    finally:
        cpu.close()
        gpu.close()


@pytest.mark.gpu
def test_cuda_spherical_expansion_by_pair_matches_cpu_order() -> None:
    """Pair feature rows are ordered on CUDA, without host-side reordering."""

    _load_cuda_plugin()
    batch = _batch()
    parameters = {
        "species": [1, 6, 8],
        "cutoff": 3.5,
        "density_width": 0.6,
        "max_radial": 2,
        "max_angular": 2,
    }
    cpu = SphericalExpansionByPair(
        **parameters, execution=ExecutionOptions(device="cpu", num_threads=1)
    )
    gpu = SphericalExpansionByPair(**parameters, execution=ExecutionOptions(device="cuda"))
    try:
        expected = cpu.compute(batch)
        actual = _compute_or_skip(gpu, batch)
        np.testing.assert_array_equal(actual.samples, expected.samples)
        np.testing.assert_array_equal(actual.row_offsets, expected.row_offsets)
        np.testing.assert_allclose(actual.values, expected.values, rtol=1e-10, atol=1e-10)
    finally:
        cpu.close()
        gpu.close()


@pytest.mark.gpu
@pytest.mark.parametrize(
    "descriptor_type",
    [SphericalExpansion, SoapRadialSpectrum, SoapPowerSpectrum],
)
def test_cuda_local_descriptors_use_device_graph_and_match_cpu(descriptor_type: type[object]) -> None:
    """The local descriptor family remains numerically equivalent after graph migration."""

    _load_cuda_plugin()
    batch = _batch()
    parameters = {
        "species": [1, 6, 8],
        "cutoff": 3.5,
        "density_width": 0.6,
        "max_radial": 2,
        "max_angular": 2,
    }
    cpu = descriptor_type(
        **parameters, execution=ExecutionOptions(device="cpu", num_threads=1)
    )
    gpu = descriptor_type(**parameters, execution=ExecutionOptions(device="cuda"))
    try:
        expected = cpu.compute(batch)
        actual = _compute_or_skip(gpu, batch)
        np.testing.assert_array_equal(actual.samples, expected.samples)
        np.testing.assert_array_equal(actual.row_offsets, expected.row_offsets)
        np.testing.assert_allclose(actual.values, expected.values, rtol=1e-10, atol=1e-10)
    finally:
        cpu.close()
        gpu.close()
