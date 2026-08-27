# C++ descriptor architecture / C++ 描述符架构

状态：布局重构已实施，DPA4 和 DPA4C 默认图配置的核心已下沉到 C++；特殊
spin/charge/compression 配置保留 NumPy fallback。

## Boundaries

`mdescriptor._native` is the only pybind11 extension exposed to Python
internals. Its C++17 kernels own periodic neighbor construction, cancellation,
finite-value validation and numerical loops. Python owns configuration parsing,
input conversion and result metadata. DPA4 and DPA4C use the private NumPy
vendor package for checkpoint loading and fallback inference, and do not
import Torch. Their validated default graph configurations are handed to
C++17/OpenMP calculators; specialised spin/charge/compressed variants retain
the NumPy fallback.

The public Python layer is split into `descriptors/standalone` and
`descriptors/model_backed`. Standalone descriptors do not load models. NEP,
DPA4 and DPA4C resolve model resources explicitly; the DPA adapters are
CPU-only and Torch-free.

## Shared native helpers

Small value types and linear algebra live in the named header
`cpp/include/mdescriptor/detail/math3.hpp`. Descriptor-family headers retain
family-specific validation and dispatch (`descriptor_common.hpp`,
`extra_common.hpp`, `local_common.hpp`, `matrix_common.hpp`). New shared code
must be placed in a named domain header rather than a catch-all `utils` file.

## Build and release

The CMake source list is the single native build manifest. The Python package
uses the `src/` layout and installs the extension as `_native`.

Supported model-wheel target policy:

- CPython 3.10–3.14;
- Linux x86_64 and Windows x86_64;
- macOS arm64 (macOS 14+);
- Linux arm64 can be added later; Windows arm64 is not promised.

The release workflow builds sdist and wheels separately, tests the installed
artifact outside the repository, and verifies the bundled model checkpoints
without a model-runtime extra.

## 性能现状 / Performance status

DPA4 native 已采用固定大小分块的 SGEMM、compute-local 工作区和一次计算/复用
attention logit；大批量独立结构按结构分块构图，避免全批次 image-cell 中间量
造成内存放大。DPA4C 类型对 MLP 则改为每个 calculator 实例独立的惰性缓存，按
实际出现的有序类型对确定性生成。OpenBLAS 只在构建期由精确版本
`scipy-openblas32==0.3.34.106.0` 提供，并随 wheel 内置其运行时闭包和许可文件；
安装后的 DPA 计算不需要该 Python 包或 Torch。

性能报告由 `scripts/benchmark_dpa_native.py` 生成（默认 2 次预热、5 次稳态），
覆盖构造、首次调用、稳态 median/p95、1/4/32 线程、峰值 RSS、边数和
（profiling 构建启用时）私有阶段计时。在同一 Linux 主机以提交 `334e159` 为
基线，当前候选测量约为 0.47 s/0.30 s（41 原子小批，1/32 线程）和
57.8 s/21.6 s（50 帧、3200 原子单次吞吐，1/32 线程）。
这些是同机 A/B 门禁数据，不作为跨平台性能承诺；wheel 仍需通过三平台安装、
动态依赖及许可检查。

## Contract tests

Native contract tests cover periodic input validation, cancellation, finite
outputs and atom/structure/pair row layouts. Python contract tests cover:

- lazy registry imports and immutable built-ins;
- `DescriptorResult` JSON-safe metadata and `feature_count`;
- close/context-manager behavior;
- explicit model resolution and strict `.pt` loading.

Numerical behavior is frozen for this refactor. The DPA4 and DPA4C native paths
are covered by model goldens, NumPy reference comparisons and thread-stability
checks; unsupported specialised variants continue to use the NumPy path.
