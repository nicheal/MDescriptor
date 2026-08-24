# MDescriptor

MDescriptor is a batch-oriented periodic atomic-descriptor library. Its numerical
kernels are implemented in C++17 and exposed through a small Python API. The
library accepts one or more fully periodic structures and returns labeled
descriptor arrays together with structure and atom boundary information.

MDescriptor 是一个面向批量周期性结构的原子描述符库。数值计算核心使用
C++17 实现，并通过简洁的 Python API 调用。库接收一个或多个完整周期结构，
返回带有标签的描述符数组，以及结构边界和原子边界信息。

## Features / 特性

- Native C++17 kernels with optional OpenMP parallelism.
- One input contract for all descriptors.
- Atom-level, structure-level, and pair-level outputs.
- Stable feature labels and metadata for every result.
- Optional ASE input conversion and optional sparse.COO output.
- 25 entries in DESCRIPTOR_CATALOG: native descriptors, the C00PS-MLFF
  C00/PS descriptor, and bundled model-backed DPA4/DPA4C/NEP descriptors.

- 原生 C++17 计算核心，可选 OpenMP 并行。
- 所有描述符统一使用同一种输入契约。
- 支持原子级、结构级和邻居对级输出。
- 每个结果都包含稳定的特征标签和元数据。
- 可选使用 ASE 构造输入，也可选输出 sparse.COO。
- DESCRIPTOR_CATALOG 中包含 25 个入口：原生描述符、C00PS-MLFF
  C00/PS 描述符，以及内置模型驱动的 DPA4/DPA4C/NEP 描述符。

## Installation / 安装

The core runtime requires Python 3.10 or newer, NumPy 1.23 or newer, and
array-api-compat 1.8 or newer. PyTorch is only required by the DPA4/DPA4C model
descriptors. Installing from source requires a C++17 compiler; the build
backend supplies the CMake and pybind11 build dependencies automatically.

核心运行环境需要 Python 3.10 及以上版本、NumPy 1.23 及以上版本和
array-api-compat 1.8 及以上版本。只有使用 DPA4/DPA4C 模型描述符时才需要
PyTorch。源码安装需要 C++17 编译器；CMake 和 pybind11 构建依赖由构建后端自动
提供。

~~~bash
# Development install: builds the C++ extension through CMake.
python -m pip install -e .

# Optional integrations.
python -m pip install ".[ase]"
python -m pip install ".[torch]"
python -m pip install ".[all]"
~~~

ASE is only required for StructureBatch.from_ase or when passing ASE Atoms
objects directly. Install it with pip install ase. Install sparse when using
sparse=True on a calculator that supports sparse output.

只有使用 StructureBatch.from_ase，或直接传入 ASE Atoms 对象时才需要 ASE，
可通过 pip install ase 安装。需要 sparse=True 时，请安装 sparse。

## Input contract / 输入结构

All calculators accept one of the following:

所有计算器都接受以下输入之一：

1. A StructureBatch instance.
2. A single ASE Atoms object.
3. A sequence of ASE Atoms objects.

1. StructureBatch 实例。
2. 单个 ASE Atoms 对象。
3. ASE Atoms 对象序列。

### StructureBatch

StructureBatch stores all structures in contiguous arrays:

StructureBatch 使用连续数组存储多个结构：

| Field / 字段 | Shape / 形状 | Dtype / 类型 | Meaning / 含义 |
|---|---:|---|---|
| numbers | (N,) | int32 | Atomic numbers for all atoms / 所有原子的原子序数 |
| positions | (N, 3) | float64 | Cartesian positions / 笛卡尔坐标 |
| cells | (S, 3, 3) | float64 | Unit-cell matrices / 各结构晶胞矩阵 |
| pbc | (S, 3) | int32 | Periodic flags; all values must be 1 / 周期标志，所有值必须为 1 |
| offsets | (S + 1,) | int64 | Atom ranges for each structure / 每个结构的原子范围 |
| ids | length S | tuple[str, ...] | Structure identifiers / 结构标识符 |

Here S is the number of structures and N is the total number of atoms. The
atoms belonging to structure s are in the half-open range
offsets[s]:offsets[s + 1]. Cells must be nonsingular, positions and cells must
be finite, and only fully periodic structures (pbc == (1, 1, 1)) are supported.

其中 S 是结构数量，N 是所有结构的原子总数。第 s 个结构的原子位于
offsets[s]:offsets[s + 1] 半开区间内。晶胞必须非奇异，位置和晶胞必须为有限值，
当前只支持完整周期结构（pbc == (1, 1, 1)）。

### Constructing an input / 构造输入

~~~python
from ase import Atoms
from mdescriptor import StructureBatch

systems = [
    Atoms(
        "Si2",
        positions=[[0.0, 0.0, 0.0], [1.4, 1.4, 1.4]],
        cell=[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
        pbc=True,
    ),
]
batch = StructureBatch.from_ase(systems, ids=["si-cell-0"])
~~~

The same object can be passed to every calculator. Atomic numbers in species
arguments must be positive, unique, and cover every atomic number in the batch.

同一个 batch 可以传给所有计算器。计算器的 species 参数必须是正数、无重复，
并且覆盖输入中出现的全部原子序数。

## Output contract / 输出结构

Every compute(...) call returns a DescriptorResult. The create(...) method is a
convenience wrapper that returns only result.values.

每次调用 compute(...) 都返回 DescriptorResult。create(...) 是便捷接口，
只返回 result.values。

| Field / 字段 | Description / 说明 |
|---|---|
| values | A two-dimensional dense numpy.ndarray by default. Selected calculators can return sparse.COO when sparse=True. / 默认是二维 numpy.ndarray；部分计算器支持 sparse=True 返回 sparse.COO。 |
| level | atom, structure, or pair. / 原子级、结构级或邻居对级。 |
| structure_ids | The input batch.ids. / 输入的 batch.ids。 |
| atom_offsets | batch.offsets for atom-level results; per-structure row offsets for pair-level results; None for structure-level results. / 原子级结果使用 batch.offsets；邻居对级结果使用每个结构的行偏移；结构级结果为 None。 |
| labels | One stable label per feature column; len(labels) == values.shape[1]. / 每个特征列一个稳定标签。 |
| metadata | Descriptor name, backend, configuration, and descriptor-specific information. / 描述符名称、后端、配置及描述符专属信息。 |

For a batch with N atoms, S structures, P neighbor-pair rows, and F features,
the standard shapes are:

对于包含 N 个原子、S 个结构、P 条邻居对记录和 F 个特征的批次，标准形状为：

| Level / 层级 | values shape / values 形状 | atom_offsets |
|---|---:|---:|
| Atom / 原子级 | (N, F) | (S + 1,) |
| Structure / 结构级 | (S, F) | None |
| Pair / 邻居对级 | (P, F) | (S + 1,) |

All native descriptor results report metadata["backend"] == "mdescriptor-cpp".
The default numeric type is float64; SOAP, SOAPTurbo, ACSF, MTP, and NEP also
accept dtype="float32".

所有原生描述符结果都满足 metadata["backend"] == "mdescriptor-cpp"。
默认数值类型为 float64；SOAP、SOAPTurbo、ACSF、MTP 和 NEP 也支持
dtype="float32"。

## Descriptor outputs / 所有描述符输出

K is the number of configured species, R = max_radial + 1, A = max_angular + 1,
Nmax is the matrix padding size, and F denotes a descriptor-dependent feature
count. The exact feature count is also available as calculator.feature_count
after species or a model has been resolved.

其中 K 是配置的元素种类数，R = max_radial + 1，A = max_angular + 1，
Nmax 是矩阵补齐尺寸，F 表示由描述符参数决定的特征数。元素种类或模型解析后，
也可以通过 calculator.feature_count 获取精确特征数。

| Descriptor / 描述符 | Public entry / Python 入口 | Level / 层级 | Output / 输出 |
|---|---|---|---|
| SOAP | SoapCalculator | Atom or structure / 原子级或结构级 | average="off": (N, F); average="inner" or "outer": (S, F). Without compression, F = (K·n_max)(K·n_max + 1)(l_max + 1)/2. / 不压缩时的特征数如左。 |
| SOAPTurbo | SoapTurboCalculator | Atom / 原子级 | (N, F). Without compression, F = C(C + 1)(l_max + 1)/2, where C = sum(alpha_max). / 不压缩时的特征数如左。 |
| ACSF | AcsfCalculator | Atom / 原子级 | (N, F), F = (1 + G2 + G3)K + (G4 + G5)K(K + 1)/2. G1 contributes one channel per species. / G1 为每种元素提供一列。 |
| Coulomb matrix | CoulombMatrixCalculator | Structure / 结构级 | (S, Nmax²) for none/sorted_l2; (S, Nmax) for eigenspectrum. / none/sorted_l2 为矩阵展平，eigenspectrum 为特征值。 |
| Sine matrix | SineMatrixCalculator | Structure / 结构级 | Same layout as Coulomb matrix: (S, Nmax²) or (S, Nmax). / 与 Coulomb matrix 相同。 |
| Ewald sum matrix | EwaldSumMatrixCalculator | Structure / 结构级 | Same layout as Coulomb matrix: (S, Nmax²) or (S, Nmax). / 与 Coulomb matrix 相同。 |
| MBTR | MBTRCalculator | Structure / 结构级 | (S, F), a flattened grid of geometry channels. / (S, F)，几何通道展平后的网格直方图。 |
| LMBTR | LMBTRCalculator | Atom / 原子级 | (N, F), a local flattened grid. / (N, F)，局部几何通道展平后的网格。 |
| Valle–Oganov | ValleOganovCalculator | Structure / 结构级 | (S, F), a normalized MBTR-style histogram. / (S, F)，归一化的 MBTR 风格直方图。 |
| Atomic composition | AtomicCompositionCalculator | Structure or atom / 结构级或原子级 | per_system=True: (S, K); per_system=False: (N, K). / 根据 per_system 选择结构级或原子级。 |
| Sorted distances | SortedDistancesCalculator | Atom / 原子级 | (N, F), padded and sorted neighbor distances; F is controlled by max_neighbors and separate_neighbor_types. / (N, F)，补齐并排序后的邻居距离。 |
| Neighbor list | NeighborListCalculator | Pair / 邻居对级 | (P, 9). Columns: first, second, cell_shift_a, cell_shift_b, cell_shift_c, dx, dy, dz, distance. / 九列依次为原子索引、晶胞平移、位移和距离。 |
| Spherical expansion | SphericalExpansionCalculator | Atom / 原子级 | (N, K²RA²). / (N, K²RA²)。 |
| Spherical expansion by pair | SphericalExpansionByPairCalculator | Pair / 邻居对级 | (P, RA²), plus metadata["pair_records"] with shape (P, 9). / (P, RA²)，并提供 (P, 9) 的邻居对记录。 |
| SOAP radial spectrum | SoapRadialSpectrumCalculator | Atom / 原子级 | (N, K²R). / (N, K²R)。 |
| SOAP power spectrum | SoapPowerSpectrumCalculator | Atom / 原子级 | (N, K · K(K + 1)/2 · A · R²). / 与物种对、角向和径向通道组合对应。 |
| LODE spherical expansion | LodeSphericalExpansionCalculator | Atom / 原子级 | (N, K²RA²), using reciprocal-space channels. / (N, K²RA²)，使用倒空间通道。 |
| EAD | EadCalculator | Atom / 原子级 | (N, (L + 1)|eta||Rs|). / (N, (L + 1)|eta||Rs|)。 |
| SO3 | So3Calculator | Atom / 原子级 | (N, (l_max + 1)n_max(n_max + 1)/2). / (N, (l_max + 1)n_max(n_max + 1)/2)。 |
| SO4 | So4Calculator | Atom / 原子级 | (N, F), bispectrum components determined by lmax. / (N, F)，双谱分量由 lmax 决定。 |
| SNAP | SnapCalculator | Atom / 原子级 | (N, F), determined by lmax, weights, and normalization. / (N, F)，由 lmax、权重和归一化参数决定。 |
| LAMMPS bispectrum | LbispectrumCalculator | Atom / 原子级 | (N, F), determined by twojmax and diagonal. / (N, F)，由 twojmax 和 diagonal 决定。 |
| MTP | MtpCalculator or MTP | Atom / 原子级 | (N, F). Generic mode emits traces and contractions; MLIP-2 emits mlip2:* columns; MLIP-4 JSON emits mlip4:basis=* columns. / 通用模式输出 trace 和 contraction，MLIP-2 输出 mlip2:*，MLIP-4 JSON 输出 mlip4:basis=*。 |
| C00PS-MLFF | C00PSMlffCalculator / C00PSMLFF | Atom / 原子级 | (N, F), local C00 radial channels plus PS angular power-spectrum channels. / (N, F)，局部 C00 径向通道和 PS 角向 power-spectrum 通道。 |
| NEP | NepCalculator, NEPCalculator, or NEP | Atom / 原子级 | (N, F), the model-defined per-atom q vector with labels nep:q1, nep:q2, ... / (N, F)，模型定义的逐原子 q 向量。 |

## Bundled model descriptors / 内置模型描述符

DPA4 and DPA4C are project-owned PyTorch model-backed descriptors. Install the
optional `torch` extra before constructing them. The DPA4 entry is a project-local port of the official
inference core: it accepts only official DPA4 `.pt` files and does not import
or require the upstream training package. The default models are bundled under
`mdescriptor/models/`; users may pass their own compatible model path.

DPA4 和 DPA4C 是项目内置的 PyTorch 模型驱动描述符。构造它们前请先安装可选的
`torch` extra。DPA4 入口使用项目内移植的官方推理核心，只接受官方 DPA4 `.pt`
文件，不导入也不依赖上游训练包。默认模型位于 `mdescriptor/models/`，用户也可以
传入自己的兼容模型路径。

DPA4 和 DPA4C 是项目自有的 PyTorch 模型驱动描述符。它们是可选功能，因为原生
描述符目录的运行不依赖 PyTorch。它们的特征维度和标签由输入检查点决定；独立的
DPA4 适配器只接受项目自有的 `mdescriptor.dpa4.v1` 检查点格式。

| Descriptor / 描述符 | Entry / 入口 | Level / 层级 | Output / 输出 |
|---|---|---|---|
| DPA4C | Dpa4cCalculator / DPA4C | Atom / 原子级 | (N, F), a checkpoint-defined calibrated invariant descriptor; metadata backend is `mdescriptor-torch`. / (N, F)，由检查点决定维度的校准不变量描述符；元数据后端为 `mdescriptor-torch`。 |
| DPA4 | Dpa4Calculator / DPA4 | Atom / 原子级 | (N, 64) for the bundled model, official checkpoint-defined scalar channels; metadata backend is `mdescriptor-dpa4-official-native`. / 内置模型输出 (N, 64)，通道由官方检查点定义；元数据后端为 `mdescriptor-dpa4-official-native`。 |

### Example: atom-level output / 原子级输出示例

~~~python
from mdescriptor import SoapCalculator

calculator = SoapCalculator(
    species=[1, 8],
    r_cut=4.5,
    n_max=4,
    l_max=3,
    average="off",
)
result = calculator.compute(batch)

assert result.level == "atom"
assert result.values.shape == (batch.atoms, calculator.feature_count)
assert result.atom_offsets.shape == (batch.structures + 1,)
assert len(result.labels) == result.values.shape[1]
~~~

### Example: structure-level output / 结构级输出示例

~~~python
from mdescriptor import CoulombMatrixCalculator

result = CoulombMatrixCalculator(
    n_atoms_max=16,
    permutation="eigenspectrum",
).compute(batch)

assert result.level == "structure"
assert result.values.shape == (batch.structures, 16)
assert result.atom_offsets is None
~~~

### Example: pair-level output / 邻居对级输出示例

~~~python
from mdescriptor import NeighborListCalculator

result = NeighborListCalculator(cutoff=5.0).compute(batch)

assert result.level == "pair"
assert result.values.shape[1] == 9
assert result.atom_offsets.shape == (batch.structures + 1,)
first_structure_pairs = result.values[
    result.atom_offsets[0] : result.atom_offsets[1]
]
~~~

## Public catalog / 公共目录

DESCRIPTOR_CATALOG contains 25 entries: the 24 native descriptors and the
DPA4C model-backed descriptor. DPA4 and NEP are intentionally separate because
their feature dimensions and parameters come from a model checkpoint.

DESCRIPTOR_CATALOG 包含 25 个入口：24 个原生描述符和 DPA4C 模型驱动描述符。
DPA4 和 NEP 没有加入目录，因为它们的特征维度和参数由模型检查点决定。

~~~python
from mdescriptor import DESCRIPTOR_CATALOG, descriptor_inventory

print(descriptor_inventory())
calculator_class = DESCRIPTOR_CATALOG["SOAP"]
~~~

## Scope / 范围

MDescriptor implements native descriptor kernels and compatibility-oriented
Python adapters. Reference package names in class names or metadata describe
the supported mathematical conventions; those reference packages are not
required at runtime for the native calculation path.

MDescriptor 包含原生描述符计算核心和面向兼容性的 Python 适配器。类名或元数据中
出现的参考包名称表示所支持的数学约定；原生计算路径运行时不要求安装这些参考包。

## License / 许可证

MDescriptor is licensed under the GNU General Public License v3.0. See
[LICENSE](LICENSE) for the full text.

MDescriptor 采用 GNU General Public License v3.0，完整协议文本请参见
[LICENSE](LICENSE)。

## Developer / 开发者

Developer email / 开发者邮箱：[nicheal@gmail.com](mailto:nicheal@gmail.com)
