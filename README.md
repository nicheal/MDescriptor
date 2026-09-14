# MDescriptor

MDescriptor is a Python library for computing atomic descriptors over batches
of isolated and fully periodic structures. It combines one input/result
contract with C++17/OpenMP kernels, an optional CUDA backend, and explicit
metadata for applications that need to discover and configure descriptors.

MDescriptor 是面向孤立结构和完整周期结构批量计算原子描述符的 Python 库。
它提供统一的输入/结果契约、C++17/OpenMP 数值核心、可选 CUDA 后端，以及适合
应用程序发现和配置描述符的显式元数据。

## Highlights / 特性

- 28 built-in descriptors covering local, matrix, many-body, rotational, and
  model-backed families.
- One validated `StructureBatch` input and one `DescriptorResult` output
  contract for every descriptor.
- Atom-, structure-, and pair-level results with stable samples, labels,
  metadata, and feature counts.
- Dense NumPy output by default, with optional read-only SciPy CSR output.
- C++17/OpenMP CPU kernels; published Linux and Windows wheels also include
  the CUDA backend in the same distribution.
- Explicit, versioned registry metadata for descriptor discovery, GUI forms,
  validation, and JSON configuration round-tripping.
- Bundled, checksum-verified NEP, DPA4, and DPA4C model resources. DPA
  checkpoints are read without importing or installing Torch, and models are
  never downloaded implicitly.
- Uniform lifecycle management and cooperative cancellation across the
  descriptor API.

## Installation / 安装

Install the published package with Python 3.10 or newer:

```bash
python -m pip install mdescriptor
```

Optional integrations:

```bash
python -m pip install "mdescriptor[ase]"     # ASE input conversion
python -m pip install "mdescriptor[sparse]"   # SciPy CSR output
```

The base package requires NumPy, `array-api-compat`, and `packaging`. ASE is
needed only for `StructureBatch.from_ase(...)` or direct ASE input. DPA4 and
DPA4C do not require Torch at runtime.

### Platform backends / 平台后端

The CPU backend works on every supported platform. The published wheel layout
is:

| Platform | CPU | CUDA |
|---|---:|---:|
| Linux x86_64 | Yes | Yes |
| Windows x86_64 | Yes | Yes |
| macOS arm64 | Yes | No |

Linux and Windows wheels bundle the CUDA user-space runtime; an NVIDIA driver
and a supported GPU are still required. CUDA is selected explicitly with
`ExecutionOptions(device="cuda")` and never silently falls back to CPU. If the
driver or GPU is unavailable, computation reports the structured
`device_unavailable` error.

When building from source, a CUDA toolkit is detected automatically. Disable
it explicitly when needed:

```bash
python -m pip install . --config-settings=cmake.define.MDESCRIPTOR_BUILD_CUDA=OFF
```

## Input contract / 输入契约

Every descriptor accepts a `StructureBatch`. It can also adapt a single ASE
`Atoms` object or a sequence of ASE objects when ASE is installed. GUI- or
application-owned frame records can be packed with `StructureBatch.from_frames`.

```python
import numpy as np
from mdescriptor import StructureBatch

batch = StructureBatch(
    numbers=np.array([1, 8], dtype=np.int32),
    positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    cells=np.eye(3, dtype=np.float64)[None] * 12.0,
    pbc=np.ones((1, 3), dtype=np.int32),
    offsets=np.array([0, 2], dtype=np.int64),
    ids=("water-0",),
)
```

For `S` structures and `N` total atoms, the core fields are:

| Field | Shape | Meaning |
|---|---|---|
| `numbers` | `(N,)` | Positive atomic numbers |
| `positions` | `(N, 3)` | Cartesian coordinates |
| `cells` | `(S, 3, 3)` | Unit-cell matrices; zero cells are allowed for isolated structures |
| `pbc` | `(S, 3)` | All zeros for isolated or all ones for fully periodic structures |
| `offsets` | `(S + 1,)` | Atom boundaries for each structure |
| `ids` | length `S` | Structure identifiers |

`StructureBatch` owns read-only snapshots of its arrays and validates shapes,
finite values, atomic numbers, offsets, cells, and periodicity. A batch may mix
isolated and fully periodic structures, but partial periodicity within one
structure is not supported. Optional `spins` and `charge_spin` fields are
available for model descriptors that declare those capabilities.

## Basic API / 基本 API

Algorithm classes live in the lazy `mdescriptor.descriptors` namespace. The
root package exposes the stable contracts, errors, and registry functions.

```python
from mdescriptor import ExecutionOptions, OutputOptions
from mdescriptor.descriptors import SOAP

with SOAP(
    species=[1, 8],
    r_cut=4.5,
    n_max=4,
    l_max=3,
    average="off",
    output=OutputOptions(dtype="float32"),
    execution=ExecutionOptions(num_threads=4),
) as descriptor:
    result = descriptor.compute(batch)

print(result.level)                 # DescriptorLevel.ATOM
print(result.values.shape)          # (number of atoms, feature_count)
print(result.labels[:2])
print(result.samples[:2])
```

All descriptors provide `compute(...)`, `close()`, `closed`, `configuration`,
`metadata`, and (when resolved) `feature_count`. Computing after `close()`
raises `ClosedDescriptorError`. Use `ComputeControl` as the `control=` argument
to cancel a long-running computation cooperatively.

## Results / 结果

`DescriptorResult.values` is a two-dimensional dense NumPy array by default.
With `OutputOptions(sparse=True)`, descriptors return a read-only SciPy CSR
matrix instead. The common result fields are:

| Field | Meaning |
|---|---|
| `values` | Feature matrix with shape `(rows, features)` |
| `level` | `atom`, `structure`, or `pair` |
| `structure_ids` | Identifiers copied from the input batch |
| `row_offsets` | Structure boundaries for atom- and pair-level rows; `None` for structure-level output |
| `samples` | Stable row identities for the selected output level |
| `labels` | One stable label per feature column |
| `metadata` | JSON-safe descriptor, execution, output, and model metadata |
| `feature_count` | Number of feature columns |

The sample layouts are:

- structure level: `[structure]`
- atom level: `[structure, local_atom]`
- pair level: `[structure, local_atom_1, local_atom_2, shift_a, shift_b, shift_c]`

## Built-in descriptors / 内置描述符

The built-in registry currently contains 28 descriptors:

| Family | Descriptors |
|---|---|
| Local | `SOAP`, `SOAPTurbo`, `ACSF`, `ACE`, `AtomicComposition`, `NeighborList`, `SortedDistances`, `SphericalExpansion`, `SphericalExpansionByPair`, `SoapRadialSpectrum`, `SoapPowerSpectrum`, `LodeSphericalExpansion`, `MTP`, `C00PSMLFF` |
| Matrix | `CoulombMatrix`, `SineMatrix`, `EwaldSumMatrix` |
| Many-body | `MBTR`, `LMBTR`, `ValleOganov` |
| Rotational | `EAD`, `SO3`, `SO4`, `SNAP`, `LBispectrum` |
| Model-backed | `NEP`, `DPA4`, `DPA4C` |

See the [descriptor inventory](docs/descriptor-inventory.md) for the
canonical parameters, output levels, periodicity, execution devices, model
policies, and GUI-facing descriptions. The inventory is generated from the
immutable built-in registry.

## Registry and application integration / 注册表与应用集成

The registry is the single discovery source for built-in descriptors. Static
metadata can be queried without constructing a descriptor or resolving a
model:

```python
import mdescriptor

names = mdescriptor.list_descriptors()
summaries = mdescriptor.list_descriptors(detailed=True)
soap_info = mdescriptor.describe_descriptor("SOAP")
runtime = mdescriptor.get_runtime_info()
```

`describe_descriptor(name)` returns JSON-safe parameter schemas, display names,
tooltips, execution devices, input periodicity, output options, and model asset
policy. `DescriptorConfiguration` is an immutable, versioned JSON form that
can be stored and reconstructed:

```python
from mdescriptor import DescriptorConfiguration, create_descriptor

saved = descriptor.configuration.to_dict()
restored = create_descriptor(DescriptorConfiguration.from_dict(saved))
restored.close()
```

`gui_baseline()` returns the packaged [GUI adaptation contract](docs/gui-adaptation-baseline.md),
including the error and metadata conventions expected by an application.

## Model resources / 模型资源

`NEP`, `DPA4`, and `DPA4C` use local model resources. Each has a bundled,
checksum-verified default model:

```python
from mdescriptor import ExecutionOptions
from mdescriptor.descriptors import DPA4, DPA4C, NEP

nep = NEP()
dpa4 = DPA4(execution=ExecutionOptions(device="cpu"))
dpa4c = DPA4C(calibrate=True)

for descriptor in (nep, dpa4, dpa4c):
    descriptor.close()
```

An explicit compatible local model can be passed with `model=/path/to/model`.
`MTP` accepts an optional local model for MLIP-2/MLIP-4-compatible features;
standalone descriptors do not require model files. No descriptor downloads a
model or searches the filesystem implicitly.

DPA4 and DPA4C official `.pt` checkpoints are parsed by the bundled NumPy
reader. The default inference graphs use the native execution backends when
available, while compatible specialized configurations retain the NumPy
fallback.

## Development / 开发

Install the project in editable mode and run the default test suite:

```bash
python -m pip install -e . --no-build-isolation
python -m pytest --import-mode=importlib tests -q
```

Quality and contract checks:

```bash
python -m ruff check src tests scripts
python -m mypy
python scripts/check_descriptor_inventory.py --check
python scripts/check_numerical_baselines.py --check
```

The direct C++ tests can be run independently:

```bash
cmake -S cpp/tests -B build/cpp-tests -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp-tests --config Release
ctest --test-dir build/cpp-tests -C Release --output-on-failure
```

The controlled CPU benchmark uses the committed golden fixtures:

```bash
python scripts/benchmarking/run_descriptor_benchmark.py \
  --output /tmp/mdescriptor-benchmark.json
```

Pushing a `v*` tag runs the release workflow. It builds CPython 3.10–3.14
wheels for Linux x86_64, Windows x86_64, and macOS arm64, builds an sdist, and
publishes the artifacts to PyPI through GitHub Trusted Publishing.

## License / 许可证

MDescriptor is distributed under the [GNU General Public License v3.0](LICENSE).
