import numpy as np
from ase import Atoms

from mdescriptor import AcsfCalculator, SoapCalculator, StructureBatch


def test_native_core_is_translation_invariant_for_periodic_batch():
    base = Atoms("NaCl2", positions=[[0.1, 0.2, 0.3], [1.3, 1.1, 1.0], [2.1, 0.4, 2.3]], cell=np.diag([8.0, 8.0, 8.0]), pbc=True)
    shifted = base.copy()
    shifted.positions += [0.37, -0.21, 0.19]
    first = StructureBatch.from_ase([base])
    second = StructureBatch.from_ase([shifted])
    soap = SoapCalculator([11, 17], {"r_cut": 2.4, "n_max": 3, "l_max": 3, "sigma": 0.7, "average": "off"})
    acsf = AcsfCalculator([11, 17], {"r_cut": 2.4, "g2_params": [[1.0, 0.0]], "g4_params": [[0.2, 1.0, 1.0]]})
    np.testing.assert_allclose(soap.compute(first).values, soap.compute(second).values, rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(acsf.compute(first).values, acsf.compute(second).values, rtol=1e-10, atol=1e-12)
