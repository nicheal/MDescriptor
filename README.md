# MDescriptor

MDescriptor is a batch-oriented periodic atomic-descriptor library. The
numerical kernels are implemented in C++17 and all Python descriptors share
one input/result/lifecycle contract.

MDescriptor 是面向批量周期结构的原子描述符库。数值核心使用 C++17；所有
Python 描述符共享统一的输入、结果和生命周期契约。

## Layout / 布局

The public namespace is intentionally small:

```text
src/mdescriptor/             installable package
src/mdescriptor/core/         Descriptor, StructureBatch, DescriptorResult
src/mdescriptor/descriptors/
  standalone/                 no model required (ACE, matrices/, many_body/, local/, rotational/)
  model_backed/               graph seam plus NEP, DPA4, DPA4C
src/mdescriptor/models/assets/ packaged, hash-verified model resources
tests/golden/              descriptor-owned, benchmark-independent accuracy fixtures
docs/numerical-baselines.md  static-golden and external-runtime oracle inventory
scripts/benchmarking/      controlled local benchmark runners
benchmarks/                local benchmark snapshots (ignored except for tracked oracles)
benchmarks/_legacy_oracles/ pinned independent upstream oracle adapters
```

`standalone` descriptors never require a model file. `model_backed` descriptors
always resolve a model resource (a packaged default or an explicit `model=`
path). There are no network downloads or implicit model discovery.

`standalone` 中的描述符不需要模型文件；`model_backed` 中的 NEP、DPA4、DPA4C
始终解析模型资源（内置默认模型或显式 `model=` 路径）。不会联网下载，也不会
扫描目录自动发现模型。

## Installation / 安装

The base package requires Python 3.10+, NumPy and `array-api-compat`. ASE is
optional and is only needed for `StructureBatch.from_ase` or direct ASE input.
GUI/native frame records can use `StructureBatch.from_frames` without ASE.
Sparse output is provided as SciPy CSR by the optional `sparse` extra.

基础包需要 Python 3.10+、NumPy 和 `array-api-compat`。ASE 只在使用
`StructureBatch.from_ase` 或直接传入 ASE 对象时需要；稀疏输出由可选的
`sparse` extra 提供。GUI/原生帧记录可使用 `StructureBatch.from_frames`，不依赖 ASE。

```bash
python -m pip install .
python -m pip install ".[ase]"
python -m pip install ".[sparse]"
```

DPA4 and DPA4C are CPU-only implementations. Their official `.pt` checkpoints
are parsed by the bundled restricted NumPy reader without importing or
installing Torch; the supported default graphs run through the C++17/OpenMP
backend and specialised configurations retain the NumPy fallback. No network
model download is performed.

The base wheel keeps the CPU backend as its default. `NeighborList`,
`SphericalExpansion`, `SoapRadialSpectrum`, and `SoapPowerSpectrum` also
advertise `execution.device="cuda"`; the other built-ins remain CPU-only.
CUDA is an opt-in plugin build and never silently falls back to CPU:

```bash
cmake -S . -B build-cuda \
  -DMDESCRIPTOR_BUILD_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build-cuda
cmake --install build-cuda --prefix /path/to/your/environment
```

The CUDA target requires a CUDA toolkit and an explicit architecture list.
Without the plugin or a usable CUDA driver, a CUDA computation reports the
structured `device_unavailable` error.

基础 wheel 默认使用 CPU。`NeighborList`、`SphericalExpansion`、
`SoapRadialSpectrum` 和 `SoapPowerSpectrum` 声明支持 `device="cuda"`；其他
内置描述符仍为 CPU-only。CUDA 是显式启用的独立插件目标，没有插件或不可用
驱动时不会静默回退，而是返回结构化的 `device_unavailable` 错误。

可选的外部数值对照固定使用 `deepmd-kit==3.2.0`：

```bash
python -m pip install ".[reference-deepmd]"
python -m pytest -m deepmd tests/external_reference/test_deepmd.py
```

该对照同时覆盖周期和非周期输入；非周期帧按 DeepMD 的 `cells=None` 语义传入。
`tests/golden/dpa4*` 的 expected output（包括非周期行）也由该外部 evaluator 生成，
manifest 记录了 evaluator 脚本、模型 hash 和 `deepmd-kit` 版本；运行时仍只加载已
提交的 NPZ，不要求安装 DeepMD。

所有 28 个描述符都登记在[数值基线清单](docs/numerical-baselines.md)，并都有外部
静态 golden：7 个沿用独立上游/source NPZ，另外 21 个在对应目录的
`external_manifest.json` 中保存固定版本 provider 生成的数值。默认测试只依赖提交的
fixture；外部 provider 缺失或版本漂移会使 reference job 失败，而不会静默跳过。

性能说明：DPA4 native 路径现在使用固定大小分块的 SGEMM、可复用的计算工作区，
并只为每条边计算一次 attention logit；DPA4C 的类型对 MLP 采用每个 calculator
实例独立的惰性缓存。OpenBLAS 仅作为构建依赖，官方 wheel 会内置 prefixed 的
OpenBLAS 及其运行时闭包和许可文件，安装后不需要 `scipy-openblas32`、Torch
或其他新增运行时依赖。大批量独立结构按结构分块消费，以控制峰值内存。

本地可复现实测（脚本默认 2 次预热、5 次稳态；基线为提交 `334e159`）可用
以下命令生成完整 JSON 报告；报告同时记录构造、首次调用、p95、线程扫描、RSS
和 profiling 构建中的私有阶段计时：

```bash
python scripts/benchmark_dpa_native.py \
  --descriptor DPA4 --dataset carbon_dataset_pbc --limit-frames 50 \
  --threads 1,4,32 --output /tmp/dpa4-native.json
```

当前 Linux 主机的候选测量中，DPA4 的 41 原子小批 median 约为 0.47 s（1
线程）和 0.30 s（32 线程）；50 帧/3200 原子单次吞吐约为 57.8 s（1 线程）
和 21.6 s（32 线程）。这些数字用于同机 A/B 门禁，不代表跨机器性能保证。

## Input contract / 输入契约

```python
import numpy as np
from mdescriptor import StructureBatch

batch = StructureBatch(
    numbers=np.array([1, 8], dtype=np.int32),
    positions=np.array([[0., 0., 0.], [1., 0., 0.]]),
    cells=np.eye(3, dtype=np.float64)[None] * 12,
    pbc=np.ones((1, 3), dtype=np.int32),
    offsets=np.array([0, 2], dtype=np.int64),
    ids=("water-0",),
)
```

`StructureBatch` takes an owned, read-only snapshot of its contiguous arrays
and validates integer fields before narrowing, finite positions/cells,
nonsingular cells, positive atomic numbers, monotonic offsets and fully
periodic `pbc == (1, 1, 1)`. ASE conversion is available through
`StructureBatch.from_ase(...)`; GUI-style mappings or objects exposing
`numbers`, `positions`, `cell`, `pbc` and `id` can use
`StructureBatch.from_frames(...)`.

## Descriptor API / 描述符 API

Algorithm classes live under `mdescriptor.descriptors` and use canonical names:

```python
from mdescriptor.descriptors import SOAP

soap = SOAP(species=[1, 8], r_cut=4.5, n_max=4, l_max=3, average="off")
result = soap.compute(batch)

assert result.level == "atom"
assert result.values.shape == (batch.atoms, soap.feature_count)
assert len(result.labels) == result.values.shape[1]

with SOAP(species=[1, 8], r_cut=4.5, n_max=2, l_max=2) as descriptor:
    result = descriptor.compute(batch)
```

ACE (Atomic Cluster Expansion) is available as a standalone atom-level
descriptor.  Its public options mirror the standard ACE1.jl
`Utils.rpi_basis` path and accept atomic numbers or chemical symbols:

```python
from mdescriptor.descriptors import ACE

ace = ACE(species=["H", "O"], N=3, maxdeg=8, rcut=5.0)
result = ace.compute(batch)
```

Every descriptor has idempotent `close()`, a `closed` property and synchronous
context-manager support. Computing after close raises `ClosedDescriptorError`.
Instances are synchronous and not promised to be thread-safe.

每个描述符都提供幂等 `close()`、`closed` 属性和同步上下文管理器；关闭后计算
会抛出 `ClosedDescriptorError`。实例是同步的，不承诺线程安全。

`DescriptorResult` contains `values`, `level` (`atom`, `structure`, or `pair`),
`structure_ids`, row offsets, stable `labels`, JSON-safe `metadata`, per-row
`samples`, and `feature_count`. The descriptor itself also retains the latest
JSON-safe `metadata` and its `configuration` after `close()`. The standard shapes are `(N, F)`, `(S, F)`,
and `(P, F)` for atom, structure and pair outputs respectively. Result arrays
are owned snapshots and exposed read-only.

## Supported descriptors / 支持的描述符

The built-in registry currently provides 28 descriptors. `None` means no model
file is needed; `Optional` means a model can be supplied (as with MTP); and
`Required` means the descriptor always resolves a local model resource. See
the [full descriptor inventory](docs/descriptor-inventory.md) for parameters,
capabilities and backend details.

内置注册表目前提供 28 个描述符。`None` 表示不需要模型文件；`Optional` 表示
可以提供模型（例如 MTP）；`Required` 表示描述符始终解析本地模型资源。参数、
能力和后端详情请参阅[完整描述符清单](docs/descriptor-inventory.md)。

| Descriptor / 描述符 | Category / 类型 | Output / 输出 | Model / 模型 |
|---|---|---|---|
| `SOAP` | Local / 局部 | Structure / 结构 | `None` |
| `SOAPTurbo` | Local / 局部 | Atom / 原子 | `None` |
| `ACSF` | Local / 局部 | Atom / 原子 | `None` |
| `ACE` | Local / 局部 | Atom / 原子 | `None` |
| `CoulombMatrix` | Matrix / 矩阵 | Structure / 结构 | `None` |
| `SineMatrix` | Matrix / 矩阵 | Structure / 结构 | `None` |
| `EwaldSumMatrix` | Matrix / 矩阵 | Structure / 结构 | `None` |
| `MBTR` | Many-body / 多体 | Structure / 结构 | `None` |
| `LMBTR` | Many-body / 多体 | Atom / 原子 | `None` |
| `ValleOganov` | Many-body / 多体 | Structure / 结构 | `None` |
| `AtomicComposition` | Local / 局部 | Structure / 结构 | `None` |
| `NeighborList` | Local / 局部 | Pair / 原子对 | `None` |
| `SortedDistances` | Local / 局部 | Atom / 原子 | `None` |
| `SphericalExpansion` | Local / 局部 | Atom / 原子 | `None` |
| `SphericalExpansionByPair` | Local / 局部 | Pair / 原子对 | `None` |
| `SoapRadialSpectrum` | Local / 局部 | Atom / 原子 | `None` |
| `SoapPowerSpectrum` | Local / 局部 | Atom / 原子 | `None` |
| `LodeSphericalExpansion` | Local / 局部 | Atom / 原子 | `None` |
| `EAD` | Rotational / 旋转 | Atom / 原子 | `None` |
| `SO3` | Rotational / 旋转 | Atom / 原子 | `None` |
| `SO4` | Rotational / 旋转 | Atom / 原子 | `None` |
| `SNAP` | Rotational / 旋转 | Atom / 原子 | `None` |
| `LBispectrum` | Rotational / 旋转 | Atom / 原子 | `None` |
| `MTP` | Local / 局部 | Atom / 原子 | `Optional` |
| `C00PSMLFF` | Local / 局部 | Atom / 原子 | `None` |
| `NEP` | Model-backed / 模型 | Atom / 原子 | `Required` |
| `DPA4` | Model-backed / 模型 | Atom / 原子 | `Required` |
| `DPA4C` | Model-backed / 模型 | Atom / 原子 | `Required` |

## Registry / 注册表

Built-ins come from one explicit immutable specification list. Imports are
lazy and no decorator or filesystem scan is used.

```python
import mdescriptor
from mdescriptor.descriptors import SOAP

print(mdescriptor.list_descriptors())
summaries = mdescriptor.list_descriptors(detailed=True)
metadata = mdescriptor.describe_descriptor("SOAP")
runtime = mdescriptor.get_runtime_info()
baseline = mdescriptor.gui_baseline()
soap = SOAP(species=[1, 8], r_cut=4.5, n_max=2, l_max=2)
rebuilt = mdescriptor.create_descriptor(soap.configuration)

child = mdescriptor.DescriptorRegistry(parent=mdescriptor.builtin_registry)
child.register(my_spec)
```

The built-in list separates `AssetPolicy.NONE`, `OPTIONAL` (for example MTP
potentials), and `REQUIRED` (NEP/DPA4/DPA4C). The root package exposes stable
contracts, registry functions and errors only; algorithm implementations are
not re-exported from the root. `describe_descriptor(name)` reads static,
JSON-safe GUI metadata from the registry without constructing a descriptor or
resolving a model. `list_descriptors(detailed=True)` returns the name, version,
display name, category and level for every descriptor in one call.
`gui_baseline()` returns the packaged GUI contract document, and
`get_runtime_info()` reports its `baseline_version` alongside the package and
schema versions.

The keys in `metadata["parameters"]` are the unchanged constructor and
configuration names. Each parameter schema also contains a GUI-facing
`display_name` and `description`; use the former as the field label and the
latter as the tooltip, while submitting the value under the mapping key:

```python
metadata = mdescriptor.describe_descriptor("SOAP")
for parameter_name, schema in metadata["parameters"].items():
    label = schema["display_name"]
    tooltip = schema["description"]
    # Render `label` and `tooltip`, then serialize the value as `parameter_name`.
```

On Windows, import `mdescriptor` during single-threaded startup, before an
embedding host starts background stdin/stdio readers. The package preloads the
packaged native binary at that point; a host that controls startup explicitly
may also call `mdescriptor.preload_native()`.

## Model resources / 模型资源

`ModelResource`, `ModelResolver`, `LoadedModel` and `ModelSession` are the
shared model-resource seam. Resolution is explicit, local and checksum-aware:

```python
from pathlib import Path
from mdescriptor.descriptors import DPA4
from mdescriptor import ExecutionOptions

dpa4 = DPA4(
    model=Path("/path/to/official-checkpoint.pt"),
    execution=ExecutionOptions(device="cpu"),
)
result = dpa4.compute(batch)
```

For GUI configuration JSON, a model parameter may be an explicit path string
or the tagged object returned by `ModelResource.to_dict()`; the latter is used
for named/checksummed resources.

The DPA4/DPA4C checkpoint readers and NumPy fallback path remain isolated
vendor adapters. The default inference graphs are lowered into the private
`mdescriptor._native` C++17/OpenMP extension.

## Development / 开发

```bash
.venv/bin/python -m pip install -e . --no-build-isolation
.venv/bin/python -m pytest --import-mode=importlib tests -q
.venv/bin/python -m pytest --cov=mdescriptor --cov-report=term-missing tests

.venv/bin/cmake -S cpp/tests -B build/cpp-tests -DCMAKE_BUILD_TYPE=Release
.venv/bin/cmake --build build/cpp-tests --config Release
.venv/bin/ctest --test-dir build/cpp-tests -C Release --output-on-failure
```

Coverage excludes the isolated vendored DPA implementation and enforces 75%
branch coverage for project-owned Python code.  Tests marked `timing` record
repeatable timing samples but intentionally do not impose a runner-dependent
speedup threshold; controlled performance reports are produced by
`scripts/run_benchmark.py`.

The extension is private (`mdescriptor._native`). C++ shared math and batch
helpers live in named headers under `cpp/include/mdescriptor/detail/`.

MDescriptor is licensed under the GNU General Public License v3.0; see
[LICENSE](LICENSE).
