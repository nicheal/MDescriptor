# Descriptor inventory / 描述符清单

This page is generated from the immutable built-in registry.  It is a
controlled artifact: run `python scripts/check_descriptor_inventory.py
--write` when a registry spec changes, and keep the `--check` gate in CI.

<!-- registry-names: SOAP, SOAPTurbo, ACSF, ACE, CoulombMatrix, SineMatrix, EwaldSumMatrix, MBTR, LMBTR, ValleOganov, AtomicComposition, NeighborList, SortedDistances, SphericalExpansion, SphericalExpansionByPair, SoapRadialSpectrum, SoapPowerSpectrum, LodeSphericalExpansion, EAD, SO3, SO4, SNAP, LBispectrum, MTP, C00PSMLFF, NEP, DPA4, DPA4C -->

| Name | Directory group | Asset policy | Backend | Level | Capabilities | Extra |
|---|---|---|---|---|---|---|
| SOAP | `standalone` | `NONE` | `cpp` | `structure` | cooperative_cancel, num_threads, sparse | `—` |
| SOAPTurbo | `standalone` | `NONE` | `cpp` | `atom` | cooperative_cancel, num_threads, sparse | `—` |
| ACSF | `standalone` | `NONE` | `cpp` | `atom` | cooperative_cancel, num_threads, sparse | `—` |
| ACE | `standalone` | `NONE` | `cpp` | `atom` | cooperative_cancel, num_threads, sparse | `—` |
| CoulombMatrix | `standalone` | `NONE` | `cpp` | `structure` | cooperative_cancel, num_threads, sparse | `—` |
| SineMatrix | `standalone` | `NONE` | `cpp` | `structure` | cooperative_cancel, num_threads, sparse | `—` |
| EwaldSumMatrix | `standalone` | `NONE` | `cpp` | `structure` | cooperative_cancel, num_threads, sparse | `—` |
| MBTR | `standalone` | `NONE` | `cpp` | `structure` | cooperative_cancel, num_threads, sparse | `—` |
| LMBTR | `standalone` | `NONE` | `cpp` | `atom` | cooperative_cancel, num_threads, sparse | `—` |
| ValleOganov | `standalone` | `NONE` | `cpp` | `structure` | cooperative_cancel, num_threads, sparse | `—` |
| AtomicComposition | `standalone` | `NONE` | `cpp` | `structure` | cooperative_cancel, num_threads, sparse | `—` |
| NeighborList | `standalone` | `NONE` | `cpp` | `pair` | cooperative_cancel, num_threads, sparse | `—` |
| SortedDistances | `standalone` | `NONE` | `cpp` | `atom` | cooperative_cancel, num_threads, sparse | `—` |
| SphericalExpansion | `standalone` | `NONE` | `cpp` | `atom` | cooperative_cancel, num_threads, sparse | `—` |
| SphericalExpansionByPair | `standalone` | `NONE` | `cpp` | `pair` | cooperative_cancel, num_threads, sparse | `—` |
| SoapRadialSpectrum | `standalone` | `NONE` | `cpp` | `atom` | cooperative_cancel, num_threads, sparse | `—` |
| SoapPowerSpectrum | `standalone` | `NONE` | `cpp` | `atom` | cooperative_cancel, num_threads, sparse | `—` |
| LodeSphericalExpansion | `standalone` | `NONE` | `cpp` | `atom` | cooperative_cancel, num_threads, sparse | `—` |
| EAD | `standalone` | `NONE` | `cpp` | `atom` | cooperative_cancel, num_threads, sparse | `—` |
| SO3 | `standalone` | `NONE` | `cpp` | `atom` | cooperative_cancel, num_threads, sparse | `—` |
| SO4 | `standalone` | `NONE` | `cpp` | `atom` | cooperative_cancel, num_threads, sparse | `—` |
| SNAP | `standalone` | `NONE` | `cpp` | `atom` | cooperative_cancel, num_threads, sparse | `—` |
| LBispectrum | `standalone` | `NONE` | `cpp` | `atom` | cooperative_cancel, num_threads, sparse | `—` |
| MTP | `standalone` | `OPTIONAL` | `cpp` | `atom` | cooperative_cancel, model, num_threads, sparse | `—` |
| C00PSMLFF | `standalone` | `NONE` | `cpp` | `atom` | cooperative_cancel, num_threads, sparse | `—` |
| NEP | `model_backed` | `REQUIRED` | `cpp` | `atom` | cooperative_cancel, model, num_threads, sparse | `—` |
| DPA4 | `model_backed` | `REQUIRED` | `numpy` | `atom` | charge_spin, model, num_threads, sparse, spin | `—` |
| DPA4C | `model_backed` | `REQUIRED` | `numpy` | `atom` | charge_spin, model, num_threads, sparse, spin | `—` |

The canonical algorithm imports are:

```python
from mdescriptor.descriptors import SOAP, ACSF, ACE, MTP, NEP, DPA4, DPA4C
```

The root package exposes contracts, registry queries and the configuration
factory.  Algorithm classes are deliberately not re-exported from the root.

## Backend boundaries

- C++17/OpenMP kernels implement the standalone numerical formulas.
- `mdescriptor._native` is the private pybind11 extension name.
- Python adapters own input packing, lifecycle, typed options and result
  normalization.
- DPA4/DPA4C use the bundled NumPy checkpoint loader and fallback inference
  core; supported default graphs are executed by C++17/OpenMP, and official
  ``.pt`` files are parsed without importing Torch.
- NEP, DPA4 and DPA4C model files are resolved locally and verified by
  streaming SHA-256 before loading.

## Result contract

Every public class returns `DescriptorResult` from `compute(...)`.  Values are
two-dimensional and `samples` is a contiguous `int64` array with the fixed
shape `[structure]`, `[structure, local_atom]`, or
`[structure, local_atom_1, local_atom_2, shift_a, shift_b, shift_c]` according
to the output level.  Metadata uses the versioned JSON-safe schema; descriptors
retain configuration and metadata after `close()`.

Common representation and execution settings are passed as
`OutputOptions`/`ExecutionOptions`.  Sparse output is SciPy CSR and fails at
construction when the optional dependency is unavailable.

## Optional dependencies

The base import and DPA4/DPA4C computation do not import Torch.  Install
`.[ase]` for ASE input conversion and `.[sparse]` for SciPy CSR output.
Official `.pt` loads use the package's restricted NumPy checkpoint reader;
network downloads are not implemented.

## Native source map

| Family | Source |
|---|---|
| SOAP / ACSF / ACE / SOAPTurbo | `cpp/src/standalone/soap.cpp`, `acsf.cpp`, `ace.cpp`, `soap_turbo.cpp` |
| matrices / MBTR | `cpp/src/standalone/*_matrix.cpp`, `matrix_dispatch.cpp`, `mbtr.cpp` |
| local descriptors | `cpp/src/standalone/atomic_composition.cpp`, `neighbor_list.cpp`, `sorted_distances.cpp`, `spherical_expansion*.cpp` |
| rotational / MTP / C00PS | `cpp/src/standalone/ead.cpp`, `rotational_descriptors.cpp`, `mtp*.cpp`, `c00ps_mlff.cpp` |
| NEP | `cpp/src/common/nep.cpp` |
| shared helpers | `cpp/include/mdescriptor/detail/`, `cpp/src/common/` |

Algorithm numerical behavior is frozen during this layout refactor.  Formula
changes require a separate reference/golden update.
