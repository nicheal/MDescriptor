import numpy as np
from ase import Atoms

from tests._public import ACSF, SOAP, OutputOptions, StructureBatch


def _system():
    return Atoms(
        "OHH",
        positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
        cell=np.diag([8.0, 8.0, 8.0]),
        pbc=True,
    )


def test_descriptor_dtype_is_preserved():
    system = _system()
    batch = StructureBatch.from_ase(system)
    acsf = ACSF(species=[1, 8], output=OutputOptions(dtype="float32")).compute(batch)
    soap = SOAP(species=[1, 8], r_cut=3.5, n_max=2, l_max=1, output=OutputOptions(dtype="float32")).compute(batch)
    assert acsf.values.dtype == np.float32
    assert soap.values.dtype == np.float32


def test_sparse_output_matches_dense_values():
    from scipy.sparse import issparse
    system = _system()
    batch = StructureBatch.from_ase(system)
    for sparse_calculator, dense_calculator in (
        (ACSF(species=[1, 8], output=OutputOptions(sparse=True)), ACSF(species=[1, 8])),
        (
            SOAP(species=[1, 8], r_cut=3.5, n_max=2, l_max=1, output=OutputOptions(sparse=True)),
            SOAP(species=[1, 8], r_cut=3.5, n_max=2, l_max=1),
        ),
    ):
        result = sparse_calculator.compute(batch)
        assert issparse(result.values)
        assert result.values.__class__.__name__ == "csr_matrix"
        dense = dense_calculator.compute(batch).values
        np.testing.assert_allclose(result.values.toarray(), dense)
