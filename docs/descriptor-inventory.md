# Descriptor inventory / 描述符清单

This page is generated from the immutable built-in registry.  It is a
controlled artifact: run `python scripts/check_descriptor_inventory.py
--write` when a registry spec changes, and keep the `--check` gate in CI.

<!-- registry-names: SOAP, SOAPTurbo, ACSF, ACE, CoulombMatrix, SineMatrix, EwaldSumMatrix, MBTR, LMBTR, ValleOganov, AtomicComposition, NeighborList, SortedDistances, SphericalExpansion, SphericalExpansionByPair, SoapRadialSpectrum, SoapPowerSpectrum, LodeSphericalExpansion, EAD, SO3, SO4, SNAP, LBispectrum, MTP, C00PSMLFF, NEP, DPA4, DPA4C -->

| Name | Directory group | Category | Asset policy | Backend | Level | Capabilities | Parameters | Extra |
|---|---|---|---|---|---|---|---|---|
| SOAP | `standalone` | `local` | `NONE` | `cpp` | `structure` | cooperative_cancel, num_threads, sparse | species, rbf, n_max, l_max, sigma, average, weighting, r_cut, compression | `—` |
| SOAPTurbo | `standalone` | `local` | `NONE` | `cpp` | `atom` | cooperative_cancel, num_threads, sparse | species, alpha_max, l_max, rcut_hard, rcut_soft, nf, radial_enhancement, basis, compression, atom_sigma_r, atom_sigma_r_scaling, atom_sigma_t, atom_sigma_t_scaling, amplitude_scaling, central_weight, central_species | `—` |
| ACSF | `standalone` | `local` | `NONE` | `cpp` | `atom` | cooperative_cancel, num_threads, sparse | species, r_cut, g2_params, g3_params, g4_params, g5_params | `—` |
| ACE | `standalone` | `local` | `NONE` | `cpp` | `atom` | cooperative_cancel, num_threads, sparse | species, N, r0, trans, wL, maxdeg, D, rcut, rin, pcut, pin, constants | `—` |
| CoulombMatrix | `standalone` | `matrix` | `NONE` | `cpp` | `structure` | cooperative_cancel, num_threads, sparse | n_atoms_max, permutation, exponent | `—` |
| SineMatrix | `standalone` | `matrix` | `NONE` | `cpp` | `structure` | cooperative_cancel, num_threads, sparse | n_atoms_max, permutation, exponent | `—` |
| EwaldSumMatrix | `standalone` | `matrix` | `NONE` | `cpp` | `structure` | cooperative_cancel, num_threads, sparse | n_atoms_max, permutation, accuracy, w, r_cut, g_cut, a | `—` |
| MBTR | `standalone` | `many_body` | `NONE` | `cpp` | `structure` | cooperative_cancel, num_threads, sparse | species, geometry, grid, weighting, periodic, normalize_gaussians, normalization | `—` |
| LMBTR | `standalone` | `many_body` | `NONE` | `cpp` | `atom` | cooperative_cancel, num_threads, sparse | species, geometry, grid, weighting, periodic, normalize_gaussians, normalization | `—` |
| ValleOganov | `standalone` | `many_body` | `NONE` | `cpp` | `structure` | cooperative_cancel, num_threads, sparse | species, function, n, sigma, r_cut, geometry, grid, weighting, periodic, normalize_gaussians, normalization | `—` |
| AtomicComposition | `standalone` | `local` | `NONE` | `cpp` | `structure` | cooperative_cancel, num_threads, sparse | species, per_system | `—` |
| NeighborList | `standalone` | `local` | `NONE` | `cpp` | `pair` | cooperative_cancel, num_threads, sparse | cutoff, full_neighbor_list, self_pairs | `—` |
| SortedDistances | `standalone` | `local` | `NONE` | `cpp` | `atom` | cooperative_cancel, num_threads, sparse | species, cutoff, max_neighbors, separate_neighbor_types | `—` |
| SphericalExpansion | `standalone` | `local` | `NONE` | `cpp` | `atom` | cooperative_cancel, num_threads, sparse | species, cutoff, density_width, max_radial, max_angular | `—` |
| SphericalExpansionByPair | `standalone` | `local` | `NONE` | `cpp` | `pair` | cooperative_cancel, num_threads, sparse | species, cutoff, density_width, max_radial, max_angular | `—` |
| SoapRadialSpectrum | `standalone` | `local` | `NONE` | `cpp` | `atom` | cooperative_cancel, num_threads, sparse | species, cutoff, density_width, max_radial, max_angular | `—` |
| SoapPowerSpectrum | `standalone` | `local` | `NONE` | `cpp` | `atom` | cooperative_cancel, num_threads, sparse | species, cutoff, density_width, max_radial, max_angular | `—` |
| LodeSphericalExpansion | `standalone` | `local` | `NONE` | `cpp` | `atom` | cooperative_cancel, num_threads, sparse | species, cutoff, density_width, max_radial, max_angular, k_cutoff, exponent, radial_radius | `—` |
| EAD | `standalone` | `rotational` | `NONE` | `cpp` | `atom` | cooperative_cancel, num_threads, sparse | parameters, Rc, cutoff | `—` |
| SO3 | `standalone` | `rotational` | `NONE` | `cpp` | `atom` | cooperative_cancel, num_threads, sparse | nmax, lmax, rcut, alpha, weight_on | `—` |
| SO4 | `standalone` | `rotational` | `NONE` | `cpp` | `atom` | cooperative_cancel, num_threads, sparse | lmax, rcut, normalize_U | `—` |
| SNAP | `standalone` | `rotational` | `NONE` | `cpp` | `atom` | cooperative_cancel, num_threads, sparse | weights, lmax, rcut, normalize_U | `—` |
| LBispectrum | `standalone` | `rotational` | `NONE` | `cpp` | `atom` | cooperative_cancel, num_threads, sparse | twojmax, diagonal, rfac0, rmin0, rcutfac, element_profile, element_radii, weights, rcut, normalize_U | `—` |
| MTP | `standalone` | `local` | `OPTIONAL` | `cpp` | `atom` | cooperative_cancel, model, num_threads, sparse | species, model, min_dist, max_dist, r_cut, radial_basis_size, radial_funcs_count, max_rank, radial_basis_type | `—` |
| C00PSMLFF | `standalone` | `local` | `NONE` | `cpp` | `atom` | cooperative_cancel, num_threads, sparse | species, r_cut, n_radial, l_max, cutoff_function, radial_sigma, include_radial, include_angular, normalize_radial, normalize_angular, super_vector, radial_weight, angular_weight, exclude_self_interaction | `—` |
| NEP | `model_backed` | `model_backed` | `REQUIRED` | `cpp` | `atom` | cooperative_cancel, model, num_threads, sparse | model | `—` |
| DPA4 | `model_backed` | `model_backed` | `REQUIRED` | `numpy` | `atom` | charge_spin, cooperative_cancel, model, num_threads, sparse, spin | model | `—` |
| DPA4C | `model_backed` | `model_backed` | `REQUIRED` | `numpy` | `atom` | charge_spin, cooperative_cancel, model, num_threads, sparse, spin | model, calibrate | `—` |

## Static descriptor metadata

Each built-in entry carries `DescriptorInfo` schema version
`3`.  Query it without importing an algorithm
implementation or resolving a model:

```python
import mdescriptor

metadata = mdescriptor.describe_descriptor("SOAP")
```

The returned object is JSON-safe and contains the canonical parameter schema,
execution devices, input periodicity, output representation, and model asset
policy.  Historical Python constructor aliases remain available to direct
callers but are intentionally omitted from the canonical GUI parameter list.
Each built-in parameter schema also exposes a GUI-facing ``display_name`` and
``description``.  The mapping key remains the canonical constructor name and
must be used when serializing values.

The canonical algorithm imports are:

```python
from mdescriptor.descriptors import SOAP, ACSF, ACE, MTP, NEP, DPA4, DPA4C
```

The root package exposes contracts, registry queries and the configuration
factory.  Algorithm classes are deliberately not re-exported from the root.

## Backend boundaries

- C++17/OpenMP kernels implement the standalone numerical formulas.
- `mdescriptor._native` is the private pybind11 extension name.
- The opt-in `MDESCRIPTOR_BUILD_CUDA` target installs `mdescriptor._cuda` as
  an independently loadable CUDA plugin for all 28 built-in descriptors. The
  registry is the canonical capability source; the plugin's extended dispatch
  table must cover the same names.
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
