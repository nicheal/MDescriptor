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
  standalone/                 no model required (matrices/, many_body/, local/, rotational/)
  model_backed/               graph seam plus NEP, DPA4, DPA4C
src/mdescriptor/models/assets/ packaged, hash-verified model resources
tests/golden/              descriptor-owned, benchmark-independent accuracy fixtures
scripts/benchmarking/      controlled local benchmark runners
benchmarks/                local-only benchmark snapshots (ignored by Git)
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
Sparse output is provided as SciPy CSR by the optional `sparse` extra.

基础包需要 Python 3.10+、NumPy 和 `array-api-compat`。ASE 只在使用
`StructureBatch.from_ase` 或直接传入 ASE 对象时需要；稀疏输出由可选的
`sparse` extra 提供。

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

`StructureBatch` validates contiguous arrays, finite positions/cells,
nonsingular cells, positive atomic numbers, monotonic offsets and fully
periodic `pbc == (1, 1, 1)`. ASE conversion is available through
`StructureBatch.from_ase(...)`.

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

Every descriptor has idempotent `close()`, a `closed` property and synchronous
context-manager support. Computing after close raises `ClosedDescriptorError`.
Instances are synchronous and not promised to be thread-safe.

每个描述符都提供幂等 `close()`、`closed` 属性和同步上下文管理器；关闭后计算
会抛出 `ClosedDescriptorError`。实例是同步的，不承诺线程安全。

`DescriptorResult` contains `values`, `level` (`atom`, `structure`, or `pair`),
`structure_ids`, row offsets, stable `labels`, JSON-safe `metadata`, per-row
`samples`, and `feature_count`. The descriptor itself also retains the latest
JSON-safe `metadata` and its `configuration` after `close()`. The standard shapes are `(N, F)`, `(S, F)`,
and `(P, F)` for atom, structure and pair outputs respectively.

## Registry / 注册表

Built-ins come from one explicit immutable specification list. Imports are
lazy and no decorator or filesystem scan is used.

```python
import mdescriptor
from mdescriptor.descriptors import SOAP

print(mdescriptor.list_descriptors())
soap = SOAP(species=[1, 8], r_cut=4.5, n_max=2, l_max=2)
rebuilt = mdescriptor.create_descriptor(soap.configuration)

child = mdescriptor.DescriptorRegistry(parent=mdescriptor.builtin_registry)
child.register(my_spec)
```

The built-in list separates `AssetPolicy.NONE`, `OPTIONAL` (for example MTP
potentials), and `REQUIRED` (NEP/DPA4/DPA4C). The root package exposes stable
contracts, registry functions and errors only; algorithm implementations are
not re-exported from the root.

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

The DPA4/DPA4C checkpoint readers and NumPy reference path remain isolated
vendor adapters. The default inference graphs are lowered into the private
`mdescriptor._native` C++17/OpenMP extension.

## Development / 开发

```bash
.venv/bin/python -m pip install -e . --no-build-isolation
.venv/bin/python -m pytest --import-mode=importlib tests -q
```

The extension is private (`mdescriptor._native`). C++ shared math and batch
helpers live in named headers under `cpp/include/mdescriptor/detail/`.

MDescriptor is licensed under the GNU General Public License v3.0; see
[LICENSE](LICENSE).
