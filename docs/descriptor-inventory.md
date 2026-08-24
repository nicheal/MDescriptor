# Descriptor inventory / 描述符清单

The registry is the one source of truth for public descriptor identities. The
implementation is grouped by whether a model asset is required:

| Group | Public names | Asset policy |
|---|---|---|
| `standalone` | SOAP, SOAPTurbo, ACSF, CoulombMatrix, SineMatrix, EwaldSumMatrix, MBTR, LMBTR, ValleOganov, AtomicComposition, NeighborList, SortedDistances, SphericalExpansion, SphericalExpansionByPair, SoapRadialSpectrum, SoapPowerSpectrum, LodeSphericalExpansion, EAD, SO3, SO4, SNAP, LBispectrum, MTP, C00PSMLFF | `NONE` except MTP's optional potential |
| `model_backed` | NEP | `REQUIRED` local text model |
| `model_backed` | DPA4, DPA4C | `REQUIRED` official `.pt`; optional `[model]` runtime |

The canonical imports are:

```python
from mdescriptor.descriptors import SOAP, ACSF, MTP, NEP, DPA4, DPA4C
```

The root package deliberately does not export algorithm aliases or historical
catalog dictionaries. Use the immutable built-in registry instead:

```python
from mdescriptor import BUILTIN_REGISTRY, list_descriptors

assert tuple(BUILTIN_REGISTRY.names()) == list_descriptors()
```

## Backend boundaries

- C++17/OpenMP kernels implement all standalone numerical formulas.
- `mdescriptor._native` is the private pybind11 extension name.
- Python adapters perform configuration, input packing and result metadata only.
- DPA4/DPA4C are isolated Torch model adapters. Their neural-network cores are
  intentionally black-box in this phase and will receive a dedicated refactor
  later.
- NEP resolves a local model text file before constructing its native model;
  packaged NEP/DPA resources are checksum-verified before loading.

## Result contract

Every public class returns `DescriptorResult` from `compute(...)`. It includes
the output level, stable labels, structure IDs/offsets, JSON-safe metadata and
`feature_count`. Lifecycle is uniform: `close()` is idempotent, `closed` is
observable, and compute-after-close raises `ClosedDescriptorError`.

For atom- and pair-level results, `row_offsets` partitions rows by
`structure_ids`; structure-level results contain one row per structure and no
offsets. Use
`OutputOptions`/`ExecutionOptions` for common representation and execution
settings. The registry's `sparse` capability describes this common output
conversion (and requires the optional `sparse` package). Unsupported execution
settings fail with `DescriptorConfigError` instead of being silently forwarded
to a backend.

## Optional dependencies

The base import does not import Torch or any model module. Install
`.[model]` for DPA4/DPA4C, `.[ase]` for ASE input conversion and `.[sparse]`
for sparse output. Official `.pt` loads use `torch.load(..., weights_only=True)`;
unsafe pickle fallback is not supported.

## Native source map

| Family | Source |
|---|---|
| SOAP / ACSF / SOAPTurbo | `cpp/src/soap.cpp`, `acsf.cpp`, `soap_turbo.cpp` |
| matrices / MBTR | `cpp/src/*_matrix.cpp`, `matrix_dispatch.cpp`, `mbtr.cpp` |
| local descriptors | `cpp/src/atomic_composition.cpp`, `neighbor_list.cpp`, `sorted_distances.cpp`, `spherical_expansion*.cpp` |
| rotational / MTP / C00PS | `cpp/src/ead.cpp`, `rotational_descriptors.cpp`, `mtp*.cpp`, `c00ps_mlff.cpp` |
| NEP | `cpp/src/nep.cpp` |
| shared helpers | `cpp/include/mdescriptor/detail/`, `cpp/src/descriptor_common.hpp`, `extra_common.hpp`, `local_common.hpp` |

Algorithm numerical behavior is frozen during this layout refactor. Changes to
formulas require separate reference/golden updates.
