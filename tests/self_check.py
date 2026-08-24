"""Small no-framework smoke check for the public descriptor contract."""

import numpy as np
from ase import Atoms
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mdescriptor import AcsfCalculator, SoapCalculator, StructureBatch


def main() -> None:
    systems = [
        Atoms("Si2", positions=[[0, 0, 0], [2.35, 0, 0]], cell=np.diag([8.0, 8.0, 8.0]), pbc=True),
        Atoms("Si", positions=[[0.2, 0.4, 0.6]], cell=np.diag([8.0, 8.0, 8.0]), pbc=True),
    ]
    batch = StructureBatch.from_ase(systems, ids=["a#0", "b#0"])
    soap = SoapCalculator([14], {"r_cut": 3.0, "n_max": 2, "l_max": 2, "sigma": 0.5, "average": "inner"})
    result = soap.compute(batch)
    assert result.level == "structure"
    assert result.values.shape == (2, (1 * 2) * (1 * 2 + 1) // 2 * 3)
    assert np.isfinite(result.values).all()

    acsf = AcsfCalculator([14], {"r_cut": 3.0, "g2_params": [[1.0, 0.0]], "g4_params": [[0.5, 1.0, 1.0]]})
    acsf_result = acsf.compute(batch)
    assert acsf_result.values.shape == (3, 1 + 1 + 1)
    assert np.isfinite(acsf_result.values).all()


if __name__ == "__main__":
    main()
