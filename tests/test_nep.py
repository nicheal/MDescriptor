import numpy as np
from ase import Atoms

from tests._public import NEP, StructureBatch


def _minimal_model(path):
    # nep3, n_max=0 and l_max=0 leave one radial descriptor column.  The
    # four ANN values are skipped by the model parser; the following values
    # are radial coefficient, angular coefficient and q_scaler.
    path.write_text(
        "\n".join(
            [
                "nep3 1 C",
                "cutoff 3.0 3.0 8 8",
                "n_max 0 0",
                "basis_size 0 0",
                "l_max 0 0 0",
                "ANN 1 0",
                "0",
                "0",
                "0",
                "0",
                "1",
                "0",
                "1",
            ]
        )
    )


def test_nep_model_descriptor_is_native_and_scaled(tmp_path):
    model_path = tmp_path / "nep.txt"
    _minimal_model(model_path)
    atoms = Atoms("C2", positions=[[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]], cell=np.diag([10.0] * 3), pbc=True)

    calculator = NEP(model_path)
    result = calculator.compute(StructureBatch.from_ase([atoms]))

    # The NEP cosine cutoff at r=1, rc=3 is 0.5*cos(pi/3)+0.5=0.75.
    assert result.values.shape == (2, 1)
    np.testing.assert_allclose(result.values, 0.75)
    assert result.metadata["backend"] == "mdescriptor-cpp"
    assert result.labels == ("nep:q1",)

