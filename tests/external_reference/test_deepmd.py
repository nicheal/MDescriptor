"""Pinned deepmd-kit comparisons for the bundled DPA4 descriptors.

deepmd-kit exposes DPA4 through ``eval_descriptor``.  DPA4C is graph-native in
deepmd-kit 3.2, so its official graph descriptor path is used here instead of
the dense sentinel-capacity path.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from ase import Atoms
from ase.data import atomic_numbers

from mdescriptor import StructureBatch
from mdescriptor.descriptors import DPA4, DPA4C
from mdescriptor.models import DPA4_MODEL, DPA4C_MODEL

pytestmark = [pytest.mark.reference, pytest.mark.deepmd]


def _water() -> Atoms:
    return Atoms(
        "OHH",
        positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
        cell=np.diag([8.0, 8.0, 8.0]),
        pbc=True,
    )


def _torch_graph(graph, torch):
    """Convert deepmd-kit's NumPy graph to the torch graph ABI."""

    values = {}
    for field in dataclasses.fields(graph):
        value = getattr(graph, field.name)
        if value is None or field.name == "destination_sorted":
            values[field.name] = value
        elif field.name == "edge_vec":
            values[field.name] = torch.as_tensor(value, dtype=torch.float64)
        elif field.name == "edge_mask":
            values[field.name] = torch.as_tensor(value, dtype=torch.bool)
        else:
            values[field.name] = torch.as_tensor(value, dtype=torch.int64)
    return dataclasses.replace(graph, **values)


def _reference_values(name: str, system: Atoms) -> np.ndarray:
    torch = pytest.importorskip("torch")
    pytest.importorskip("deepmd")

    from deepmd.infer import DeepPot
    from deepmd.pt_expt.utils.env import DEVICE

    model = DPA4_MODEL if name == "DPA4" else DPA4C_MODEL
    kwargs = {"neighbor_graph_method": "ase"} if name == "DPA4C" else {}
    deep_pot = DeepPot(str(model), **kwargs)
    try:
        type_indices = {
            int(atomic_numbers[symbol]): index
            for index, symbol in enumerate(deep_pot.get_type_map())
        }
        atom_types = np.asarray(
            [[type_indices[int(number)] for number in system.numbers]],
            dtype=np.int32,
        )
        positions = np.asarray(system.positions, dtype=np.float64)[None, :, :]
        cells = np.asarray(system.cell.array, dtype=np.float64).reshape(1, 9)
        deep_eval = deep_pot.deep_eval

        if name == "DPA4":
            value = deep_eval.eval_descriptor(positions, cells, atom_types)
            return np.asarray(value, dtype=np.float64).reshape(len(system), -1)

        deep_eval._dpmodel.eval()
        graph_descriptor = deep_eval._dpmodel.get_dp_atomic_model().descriptor
        type_embedding = graph_descriptor.type_embedding.call()
        graph = deep_eval._build_eval_graph(positions, atom_types, cells, DEVICE)
        graph = _torch_graph(graph, torch)
        with torch.no_grad():
            output, _ = graph_descriptor.call_graph(
                graph,
                torch.as_tensor(atom_types.reshape(-1), dtype=torch.int64),
                type_embedding=type_embedding,
            )
        return output.detach().cpu().numpy().astype(np.float64).reshape(len(system), -1)
    finally:
        close = getattr(deep_pot, "close", None)
        if close is not None:
            close()


@pytest.mark.parametrize(
    ("name", "descriptor"),
    [("DPA4", DPA4), ("DPA4C", DPA4C)],
)
def test_dpa_descriptors_match_deepmd_kit(name, descriptor):
    system = _water()
    expected = _reference_values(name, system)

    native = descriptor(model=DPA4_MODEL if name == "DPA4" else DPA4C_MODEL)
    try:
        actual = np.asarray(
            native.compute(StructureBatch.from_ase(system)).values,
            dtype=np.float64,
        )
    finally:
        native.close()

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-5)
