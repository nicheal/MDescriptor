# GPU backend design consensus / GPU 后端设计共识

状态：设计已确认；第一阶段代码已落地。2026-09-01 在宿主机权限下的
WSL2/RTX 2080 SUPER runner 已完成普通 NEP 的正式 NEPAdapters 精度/速度 gate：
固定批次精度通过，稳态速度达到 5.65x。实现同时按 GPUMD 原始 CUDA NEP 路径校准
float32 cutoff、距离和累加顺序，并固定周期 cell-list 的 lane-major 邻居顺序，避免
atomic scatter 导致结果漂移。多数据集结果也已记录；小批量按预期受 launch/建图开销
影响，CPU 的 double canonical path 与 GPU 的 NEP float32 path 不应混用为严格 bitwise
基准。DPA4/DPA4C 的 Python payload sidecar 和自有 CUDA descriptor path 已接入，
CUDA contract 回归通过；完整发布矩阵仍需按目标平台执行。本次重测使用 `.venv` 的
`torch 2.13.0+cu126`（
`torch.cuda.is_available()=True`，设备为 NVIDIA GeForce RTX 2080 SUPER），CUDA
矩阵 smoke test 通过。

当前实现包含 backend seam、惰性 CUDA plugin loader、可复用的 CUDA context /
batch / CSR+SoA graph、NeighborList CUDA kernel，以及
SphericalExpansion / SOAP radial / SOAP power 的共享 device coefficient
pipeline，以及普通 NEP descriptor 的 device model/kernel。扩展 CUDA 模块
`cpp/cuda/src/extended_descriptors.cu` 已覆盖本文件 11 节列出的全部 21 个
原 CPU-only 描述符；其中 SOAPTurbo、ACE、MTP 和 C00PSMLFF 会把 CPU parser
生成的扁平系数/evaluator payload 上传到 device。普通 NEP 的全周期
路径使用 NEPAdapters-compatible 的 device cell-list、周期小盒展开和 float32
中间算术；混合周期/孤立 batch 已切换到统一 CUDA neighbor path。周期小胞的物理
replica atom array 由 CUDA kernel 生成，孤立结构使用非周期 cell-list/image path，
不再创建 host canonical graph 或在 host 上展开 NEP 原子。

本文记录 MDescriptor GPU 版本与现有 MDescriptor Studio GUI 内核之间的接口和内部实现共识。目标是在不改变当前公开接口、配置格式和结果契约的前提下，为部分描述符增加 CUDA 执行能力。

## 1. 已冻结的公开接口

以下内容保持不变：

- Python 描述符构造函数；
- `Descriptor.compute()` 的同步调用方式；
- `StructureBatch` 的扁平、只读输入语义；
- `DescriptorResult` 的 shape、level、samples、labels、row offsets 和只读 host 输出；
- `DescriptorConfiguration` 的序列化格式；
- GUI baseline 的字段和 schema 版本；
- `ComputeControl` 的公开方法和取消语义。

只启用已有的 `ExecutionOptions.device` 预留字段。第一版只接受精确的 `"cpu"` 和 `"cuda"`，不增加 `cuda:0`、GPU tensor、CUDA stream、显存指针或其他 GPU 专属公开参数。

公共调用继续同步返回 host-owned NumPy/CSR 结果。GPU 内部可以使用异步 stream，但 `compute()` 返回前必须完成必要同步和 D2H 拷贝。

## 2. Backend seam

GPU 不应以 `if device == "cuda"` 的形式散落在各个描述符实现中。唯一的 Python backend seam 放在 `DescriptorAdapter` 后面：

```text
DescriptorAdapter
  └── BackendKernel
       ├── CpuBackend  → mdescriptor._native
       └── CudaBackend → 独立 CUDA plugin / mdescriptor._cuda
```

建议的私有 backend 接口为：

```python
class BackendKernel(Protocol):
    @property
    def feature_count(self) -> int | None: ...

    def compute(self, batch: StructureBatch, control: Any = None) -> Any: ...

    def close(self) -> None: ...

    def metadata(self) -> Mapping[str, Any]: ...
```

公开 descriptor class 和 CPU kernel 保持现有形态。公开构造签名仍从 CPU kernel 生成；CUDA backend 接收经过 adapter 验证和规范化后的内部配置，不参与公开参数签名生成。

`device` 是 adapter 的执行选择，不再要求底层 CPU/CUDA kernel 构造函数显式暴露 `device` 参数。`ExecutionOptions.num_threads` 在 CUDA 下保留以兼容已有配置，但不控制 CUDA launch；CUDA 使用内部固定的 host orchestration 策略。

## 3. Python 层代码规划

### `src/mdescriptor/core/adapter.py`

增加私有 backend factory/dispatcher，由它完成：

1. 根据 registry 声明校验设备；
2. 根据 `ExecutionOptions.device` 选择 CPU 或 CUDA adapter；
3. 向 backend 传递 normalized options；
4. 统一调用 `compute()`、结果适配、output formatting 和 lifecycle。

CUDA 不应改变现有 `DescriptorResult` 组装逻辑。

### `src/mdescriptor/core/control.py`

公开 `ComputeControl` 保持不变，但不再假设所有 backend 都使用 `_native.ComputeControl`：

- CPU backend 继续使用 `_native` control 实现；
- CUDA backend 接收公开 `ComputeControl`；
- CUDA binding 在结构块之间调用 `cancelled()` 和 `mark_completed()`。

两个独立扩展不共享 C++ control 对象 ABI。

### `src/mdescriptor/_runtime.py`

增加私有、惰性 CUDA plugin loader。静态 registry 查询和 `describe_descriptor()` 不加载 CUDA，也不检查 driver。第一次 CUDA `compute()` 时才加载 plugin、检查 device 并创建 context。

### `src/mdescriptor/registry/builtins.py`

设备能力由 registry metadata 唯一声明。首批 CUDA 描述符声明：

```json
{"execution": {"devices": ["cpu", "cuda"]}}
```

其他描述符继续声明 `devices: ["cpu"]`。adapter 的 `supported_devices` 应从 registry metadata 派生，避免重复维护。

DPA4 和 DPA4C 同样声明 `devices: ["cpu", "cuda"]`。模型验证仍由现有
Torch-free Python loader 完成；在选择 CUDA 时，validation kernel 在关闭前通过
私有 `_cuda_payload` seam 把已展平的 NumPy 权重数组、`feature_count`、`labels`
和 `type_numbers` 交给 backend。该 payload 使用固定的 backend option key
`_cuda_payload`，不进入 `DescriptorConfiguration` 或运行时 metadata；没有 CUDA
可见性时构造仍不触碰 driver，首次 `compute()` 才报告设备/插件错误。

## 4. CUDA C++ 私有模块

CPU 的 `StructureBatchView` 和 `NeighborGraph` 语义保持不变，不向现有 CPU 头文件加入 CUDA 指针。

建议新增独立 CUDA 目录：

```text
cpp/cuda/
  include/mdescriptor/cuda/
    context.hpp
    batch.hpp
    neighbor_graph.hpp
    backend.hpp
  src/
    context.cu
    batch.cu
    neighbor_graph.cu
    spherical_expansion.cu
    soap_spectrum.cu
    bindings.cu
```

核心私有模块：

```text
CudaExecutionContext
DeviceBatch / DeviceBatchView
DeviceNeighborGraph / DeviceNeighborGraphView
DeviceOutputBuffer
```

`CudaExecutionContext` 由一个描述符实例拥有，包含 stream、device buffers 和可增长 workspace。一个描述符实例仍不承诺并发安全；`close()` 必须先同步 stream，再释放资源。

## 5. DeviceBatch 布局

第一版保持与 CPU 逻辑布局一致，并使用 float64 内部计算：

```text
numbers      int32[N]
positions    float64[3N]
cells        float64[9S]
pbc          int32[3S]
offsets      int64[S+1]
```

`OutputOptions(dtype="float32")` 仍只控制最终 host 输出格式。内部 FP32 kernel 作为后续独立优化，必须经过单独数值容差和回归验证。

NEP 是本任务的独立模型描述符扩展：为匹配 NEPAdapters 的 CUDA 数值路径，
NEP descriptor kernel 使用 float32 中间值、host double 输出，并通过独立的
NEPAdapters tolerance 回归；这不改变 standalone 描述符的 float64 约定。

## 6. GPU NeighborGraph

GPU v1 必须支持现有输入语义：

- isolated 结构；
- fully periodic 结构；
- 任意非奇异 triclinic cell；
- 当前 cutoff 边界规则；
- 周期镜像和 exact-self 判断。

内部采用 CSR + SoA：

```text
neighbor_offsets       int64[N+1]
neighbor_atoms         int32[P]
neighbor_shifts        int32[3P]
neighbor_displacements float64[3P]
neighbor_distance2     float64[P]
```

普通 GPU graph 的构建流程固定为：

```text
计算周期 image/cell 信息
→ 构建 cell list
→ 统计每个 center 的邻居数
→ exclusive scan
→ 按确定性顺序填充 CSR
→ 检查 cancellation
```

不能用 atomic append 的到达顺序作为通用 public pair 顺序。普通 NEP 的
NEPAdapters-compatible device path 只把该顺序用于内部 slot-major descriptor
输入，并由独立 tolerance gate 验证最终数值；NeighborList 等 public graph
结果仍使用确定性 CSR 路径。GPU 结果中的 `offsets`、`atoms`、`shifts`、
displacements 和 pair samples 必须与 CPU 语义一致。

普通 NEP 对周期性分为两个 CUDA 分支：

- 当周期 cell 需要物理 replica 时，host 只计算每个 structure 的紧凑
  replication/cell metadata；`expand_nep_batch_kernel` 在 device 生成 numbers、
  positions 和 expanded offsets 对应的原子布局，随后使用 CUDA cell-list；
- 当不需要展开时，`build_nep_image_neighbors_kernel` 在 device 按
  structure/atom/image 枚举周期邻居；全零 PBC 的 structure 只枚举一次、禁止
  周期 wrapping。两种结构可以在同一 batch 中混合。

展开后的 descriptor rows 在 `reduce_expanded_nep_kernel` 中按原子 replica 顺序
在 device 求平均，D2H 只取最终原子结果。这样保留 NEPAdapters 的展开布局和累加
顺序，同时把实际原子展开、邻居构建、descriptor 和 replica reduction 放在同一
CUDA stream。

DPA4/DPA4C 使用另一条与 DeepMD-kit `build_nlist_gpu` 思路一致的两遍 graph path：

1. `DeviceBatch::upload()` 将 numbers、positions、cell、PBC 和 offsets 放入
   device；host 只计算每个 structure 的逆 cell 和有限 image bound；
2. CUDA kernel 在 device 完成 atom-to-structure 映射、周期坐标归一化和每个
   center 的邻居扫描；一个 CTA 协作统计一行，随后用 device inclusive scan
   得到 CSR offsets；
3. host 只同步一个 pair-count 标量来复用 graph capacity，pair atoms、shifts、
   displacement 和 distance2 的填充及确定性排序继续在 CUDA stream 上完成；
4. DPA4/DPA4C 的 rotation、radial、attention/equivariant blocks、readout 和
   最终输出均消费该 device graph，不再调用 CPU `build_neighbor_graph`。公共
   同步边界只在返回 NumPy 输出前发生；每次计算仍会将原子类型索引等很小的
   调用元数据上传到 device。

因此这里的“端到端 GPU”不意味着取消公共 API 必需的最终 D2H 输出，而是 pair
enumeration、CSR materialization 和模型前向都不经过 host。DPA4 保留 native ABI
要求的 endpoint fp32 rounding；DPA4C 直接使用 graph 的 double displacement。

graph 每次 `compute()` 重建，但由 context 复用已分配的容量。第一版不暴露 reusable graph，也不使用调用者数组指针或 hash 作为缓存 key。

## 7. 描述符 kernel 分层

首批描述符共享同一套 GPU 中间层：

```text
DeviceNeighborGraph
        ↓
radial basis
        ↓
spherical harmonics
        ↓
spherical coefficients
        ├── SphericalExpansion
        ├── SoapRadialSpectrum
        └── SoapPowerSpectrum
```

`NeighborList` 直接把 canonical graph 转换为 pair table。

feature count、feature index、species channel 顺序、SOAP species-pair 顺序和 labels 顺序必须由共享 layout 定义驱动。建议新增中性 layout 模块，例如：

```text
cpp/src/common/local_layout.hpp
```

CPU 和 CUDA 都使用它；Python 现有 labels 生成逻辑可以暂时保留，但必须由 CPU/CUDA 对照测试防止漂移。

## 8. Control、错误和结果

CUDA 计算按结构或结构块分批 launch：

- `ComputeControl.total()` 仍为 structure 数；
- 每完成一个 structure 或结构块调用 `mark_completed()`；
- kernel 阶段和 chunk 之间检查取消；
- 取消后同步并抛出公共 `CancelledError`；
- 不返回部分结果。

CUDA 错误不作为稳定接口直接暴露 CUDA 原始错误字符串。建议映射为：

```text
device_unavailable
backend_error
backend_out_of_memory
cancelled
```

诊断信息可放入现有 `details`，例如 CUDA error name、device id 和 requested bytes。

CUDA binding 内部完成 kernel launch、同步和 D2H，直接返回普通 NumPy 数组或 pair table；Python 层继续使用现有 `DescriptorResult` 归一化和只读快照逻辑。

## 9. Metadata 语义

静态 registry metadata 与运行时 metadata 分工如下：

- `execution.devices`：该描述符支持的设备；
- `execution_engine`：默认 engine；
- `result.metadata.execution.device`：本次实际执行设备。

CUDA 结果至少包含：

```json
{
  "execution": {
    "device": "cuda",
    "num_threads": null
  }
}
```

不新增 metadata 字段，不让 GUI 通过 backend 字符串猜测设备。CUDA kernel 的硬件诊断信息放入 `details`。

## 10. 构建和发布

现有 CPU `_native` target 保持不变。CUDA 使用独立 target 和 plugin：

```text
MDescriptor       # CPU wheel
MDescriptor-CUDA  # CUDA backend plugin
```

CUDA plugin 依赖匹配版本的基础包，并提供 `_cuda` 扩展和 backend entry point；不覆盖基础包的 Python 实现文件。
独立 CUDA wheel 将 CUDA 用户态 Runtime 放在 `mdescriptor/.cuda_libs`，并
通过 `$ORIGIN/.cuda_libs` 加载；宿主机 NVIDIA 驱动提供 `libcuda`，不随 wheel
分发。CUDA wheel 的构建入口为 `packaging/cuda/pyproject.toml`，需要在构建时
显式传入 `CMAKE_CUDA_ARCHITECTURES`。

### GPU 描述符依赖审计

| 描述符 | CUDA 实现 | BLAS 依赖 | 发布时的动态库 |
|---|---|---|---|
| NEP | `cpp/cuda/src/nep.cu` 自有 descriptor/device kernels | 无 | `libcudart`；`libcuda` 由宿主驱动提供 |
| DPA4 | `cpp/cuda/src/dpa4.cu` 自有 FP32 tiled GEMM 和 descriptor kernels | 无（已移除 cuBLAS/cuBLASLt） | `libcudart`；`libcuda` 由宿主驱动提供 |
| DPA4C | `cpp/cuda/src/dpa4c.cu` 自有 descriptor kernels | 无 | `libcudart`；`libcuda` 由宿主驱动提供 |
| 21 个 standalone 描述符 | `cpp/cuda/src/extended_descriptors.cu` 自有 descriptor kernels | 无 | `libcudart`；`libcuda` 由宿主驱动提供 |

三者共享 CUDA context、batch/CSR graph 和 Python lazy plugin loader，但不共享
BLAS runtime。CPU `_native` 路径仍可使用 SciPy/OpenBLAS；这与独立 CUDA wheel
的依赖边界无关。`MDESCRIPTOR_BUNDLE_CUDA_RUNTIME=ON` 现在只复制
`libcudart.so.*`，并由 wheel verifier 拒绝 cuBLAS/cuBLASLt 文件或 ELF 依赖。

第一阶段支持矩阵：

- Linux x86_64；
- 单 GPU；
- 一个经过 CI 验证的 CUDA toolkit/runtime 主版本；
- 显式的 `CMAKE_CUDA_ARCHITECTURES`；
- 仅支持 `device="cuda"`；
- 暂不支持 Windows、macOS、ARM、多 GPU 和 GPU index。

具体 CUDA 版本和最低 compute capability 以实际部署 GPU/driver 与 CI 矩阵为准，并写入 plugin 发布说明。

## 11. 第一阶段范围

当前 CUDA plugin 已覆盖以下描述符：

```text
NeighborList
SphericalExpansion
SoapRadialSpectrum
SoapPowerSpectrum
NEP
DPA4
DPA4C
SOAP
SOAPTurbo
ACSF
ACE
CoulombMatrix
SineMatrix
EwaldSumMatrix
MBTR
LMBTR
ValleOganov
AtomicComposition
SortedDistances
SphericalExpansionByPair
LodeSphericalExpansion
EAD
SO3
SO4
SNAP
LBispectrum
MTP
C00PSMLFF
```

因此当前 registry 的 CUDA 支持矩阵为 28 个描述符；上面的 21 个扩展描述符
统一由 `extended_descriptors.cu` dispatch，并保持各自 CPU kernel 的 feature
layout、labels、周期 exact-self 语义和结果 level。

以下内容仍明确排除在 standalone GPU 第一阶段之外：

- 未列入支持矩阵的模型权重 GPU 化；已支持的 SOAPTurbo、ACE、MTP 和
  C00PSMLFF 只上传 descriptor 计算所需的扁平参数，不改变 CPU parser；
- GPU tensor 或异步公共接口；
- 多 GPU 和跨设备 graph cache。

本任务在上述共识之外增加了普通（非 spin/charge/dipole/polar）NEP descriptor
扩展，并完成了 21 个原 CPU-only 描述符的 CUDA 路径。它们沿用 CPU parser，
只把 descriptor 所需的扁平 coefficients/evaluator 数据上传到 device；不支持
的 ANN 或其他模型权重仍单独规划。NEP 的模型解析仍复用 CPU parser，
NEPAdapters 的 GPU parity 已在具备 NVIDIA runner 的环境中完成，性能门禁按
下文的多数据集协议复核。

模型 GPU 化未来沿用 `CudaExecutionContext`，但单独处理 `ModelSession`、device weights、checkpoint ABI 和模型数值验证。

## 12. 测试与性能门禁

建议新增独立 GPU 测试目录：

```text
tests/gpu/
  test_cuda_availability.py
  test_cuda_neighbor_list.py
  test_cuda_spherical.py
  test_cuda_soap_spectra.py
  test_cuda_nep.py
  test_cuda_cancellation.py
  test_cuda_errors.py
```

测试分为三层：

1. CPU contract/golden：所有环境运行，CPU golden 不更新；
2. GPU contract：CUDA runner 必须运行；
3. GPU benchmark：包含 H2D、neighbor graph、descriptor kernel、D2H 和总耗时。

GPU 对照必须严格检查：

- `level`；
- shape 和 feature count；
- labels；
- samples；
- structure ids；
- row offsets；
- pair 顺序。

数值使用独立 tolerance manifest。CPU workflow 不依赖 CUDA；涉及 CUDA 源码、构建配置或 GPU registry 能力的变更必须触发独立 GPU workflow。GPU runner 缺失或测试失败不能被静默 skip。

第一版 benchmark 只要求完整 `compute()` 在大 batch 上显示明确 GPU 收益，不对小 batch 设置硬性 speedup 阈值；正式性能门禁在固定硬件后再建立。

普通 NEP 的对照命令为：

```bash
.venv/bin/python scripts/benchmarking/benchmark_cuda_nep.py \
  --model src/mdescriptor/models/assets/nep89_20250409.txt \
  --structures 8 --atoms 64 --warmup 3 --repeat 7 --min-speedup 1.0
```

该命令同时检查 `rtol=1e-6, atol=1e-7` 和中位 steady-state `compute()`
速度；CUDA plugin、GPU 或 NEPAdapters CUDA backend 不可用时返回状态
`unavailable`，退出码为 2，不会把硬件缺失误报为通过。

2026-09-01 宿主机结果（RTX 2080 SUPER，driver 610.47，CUDA 13.3，WSL2，
`nep89_20250409.txt`）如下。矩阵每项为 warmup 1、repeat 3 的同步稳态
`compute()` 中位数；CPU 使用 1/16 native threads，GPU 与 NEPAdapters 使用
CUDA。精度列为 `max_abs_error / max_tolerance_ratio`，统一容差为
`atol=1e-7, rtol=1e-6`。

| 数据集 | 原子/帧 | CPU1 ms | CPU16 ms | GPU ms | GPU/CPU1 | GPU/CPU16 | GPU–CPU1 | NEPAdapters ms | GPU/NEPAdapters | GPU–NEPAdapters | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---|
| water3-periodic | 3/1 | 0.292 | 0.314 | 0.762 | 0.38x | 0.41x | 6.32e-8 / 0.242 | 0.589 | 0.77x | 9.84e-10 / 0.009 | 通过 |
| hea32-periodic | 32/1 | 0.652 | 0.410 | 2.059 | 0.32x | 0.20x | 1.19e-7 / 0.309 | 1.317 | 0.64x | 1.97e-8 / 0.082 | 精度通过；小批量较慢 |
| two-structure-mixed | 41/2 | 0.658 | 0.775 | 2.165 | 0.30x | 0.36x | 1.87e-7 / 0.694 | — | — | — | CPU/GPU 通过；GPU mixed path 通过 |
| carbon-pbc-frame34 | 64/1 | 1.685 | 0.779 | 2.685 | 0.63x | 0.29x | 4.02e-7 / 1.544 | 1.580 | 0.59x | 1.53e-7 / 0.724 | NEPAdapters 精度通过；CPU double path 不同 |
| carbon-pbc-first64 | 4096/64 | 57.202 | 5.535 | 29.108 | 1.97x | 0.19x | 4.43e-7 / 2.216 | 86.517 | 2.97x | 1.77e-7 / 0.935 | NEPAdapters 精度和速度通过 |
| carbon-pbc-all | 28337/450 | 427.530 | 34.792 | 250.575 | 1.71x | 0.14x | 4.90e-7 / 2.216 | 653.796 | 2.61x | 4.21e-7 / 2.030 | NEPAdapters 精度未通过；CPU16 快 7.20x |
| soap-diverse-300 | 1086/300 | 1.443 | 0.909 | 1.207 | 1.20x | 0.75x | 2.54e-7 / 0.623 | — | — | — | GPU/CPU 通过；全孤立 batch |
| random-periodic-8x64 | 512/8 | 1.644 | 0.696 | 1.076 | 1.53x | 0.65x | 1.74e-7 / 0.502 | 5.553 | 5.16x | 7.83e-8 / 0.226 | 通过 |

正式固定批次（warmup 3、repeat 7）重测结果为：shape `[512, 35]`，GPU 与
NEPAdapters `max_abs_error=7.825e-8`、最大容差比 `0.226`，中位耗时分别为
`0.9538 ms` 和 `5.3913 ms`，速度 `5.65x`，状态 `pass`。多数据集中的
`carbon-pbc-first64` 的 GPU 与 NEPAdapters 对照通过；但全量 carbon（450 帧、
28337 原子）出现少量严格超差，GPU 与 NEPAdapters 的最大绝对误差为
`4.21e-7`、最大容差比为 `2.030`，状态为 `fail`。全量 carbon 的 CPU1 与
NEPAdapters 对照也未通过（最大容差比 `2.733`），因此该数据集暴露出比前
64 帧更明显的算术路径/累加顺序差异。上一轮 CPU1 与 NEPAdapters CUDA 的
对照也曾严格失败；本轮单独调用 NEPAdapters CPU 后，CPU1/CPU16 与
NEPAdapters CPU 均通过，说明比较时必须区分 CPU 与 CUDA backend。GPU 与 CPU1 的 carbon 严格失败仍然是
因为 CPU canonical graph 保留 double 几何路径，而 NEP CUDA/NEPAdapters 使用
float32 中间路径，不能把这两个不同算术契约当作同一精度基准。

### 12.1 项目 CPU/GPU、NEPAdapters 和 GPUMD `nep` 交叉对照

针对 carbon 全量 450 帧（28337 个原子、35 维），使用同一个
`nep89_20250409.txt`，统一容差 `atol=1e-7, rtol=1e-6`。项目 CPU/GPU 和
NEPAdapters 使用同一批 ASE 结构；项目 CPU 使用 1/16 native threads，
NEPAdapters CPU 使用其默认线程策略。速度为稳态调用中位数；官方 GPUMD
`nep` 的速度取其 `Time used for predicting`，不是项目 Python 调用的 wall time。

| 实现 | 时间 | 相对速度说明 |
|---|---:|---|
| MDescriptor CPU 1 thread | 430.513 ms | 基准 |
| MDescriptor CPU 16 threads | 36.886 ms | 比 MDescriptor CPU1 快 11.67x |
| NEPAdapters CPU | 85.722 ms | 比 MDescriptor CPU1 快 5.02x；比 CPU16 慢 2.32x |
| MDescriptor GPU | 276.164 ms | 比 CPU1 快 1.56x |
| NEPAdapters GPU | 682.272 ms | MDescriptor GPU 快 2.47x |
| GPUMD `nep` | 1284.680 ms | MDescriptor GPU 快 4.65x |

| 对照 | 最大绝对误差 / 最大容差比 | 结果 |
|---|---:|---|
| MDescriptor CPU1 vs NEPAdapters CPU | `3.28e-15 / 1.43e-8` | 通过 |
| MDescriptor CPU16 vs NEPAdapters CPU | `3.28e-15 / 1.43e-8` | 通过 |
| MDescriptor GPU vs NEPAdapters GPU | `4.43e-7 / 2.088` | 未通过 |
| MDescriptor GPU vs GPUMD `nep` | `2.95e-6 / 8.425` | 未通过 |
| NEPAdapters GPU vs GPUMD `nep` | `2.95e-6 / 8.447` | 未通过 |
| NEPAdapters GPU vs NEPAdapters CPU | `5.38e-7 / 2.718` | 未通过 |

GPUMD 的 `gpumd` 主程序是分子动力学驱动器，没有 descriptor-only 的
`descriptor.out` 接口；因此 descriptor 对照使用同一 GPUMD 源码中的官方
`nep` 可执行文件。官方 `descriptor.out` 使用 `%g` 六位有效数字，本次另保留
一个只将输出改为 `%.17g` 的诊断版；17 位输出仍有约 `3e-6` 差异，说明
GPUMD 与项目 GPU/NEPAdapters GPU 的 float 几何和累加路径本身不同，而不只是
文本截断。

可复现入口为
[`benchmark_nep_implementations.py`](../scripts/benchmarking/benchmark_nep_implementations.py)：

```bash
MDESCRIPTOR_CUDA_PLUGIN_DIR="$PWD/build-cuda" \
  .venv/bin/python scripts/benchmarking/benchmark_nep_implementations.py \
  --gpumd-nep /path/to/GPUMD/build/nep \
  --gpumd-output-significant-digits 17 \
  --warmup 1 --repeat 3 \
  --output /tmp/mdescriptor-nep-implementations-carbon-all-highprec.json
```

本轮完整结果保存在
`/tmp/mdescriptor-nep-implementations-carbon-all-highprec.json`。

本轮按 [GPUMD 原始 NEP CUDA kernel](https://raw.githubusercontent.com/brucefan1983/GPUMD/master/src/main_nep/nep.cu)
和 [GPUMD NEP utilities](https://raw.githubusercontent.com/brucefan1983/GPUMD/master/src/utilities/nep_utilities.cuh)
核对了 float displacement、`sqrtf`、cutoff/Chebyshev 运算和邻居逐项累加规则；生产路径保留 NEPAdapters 的 grouped radial accumulation，
同时使用确定性的 32-thread lane-major cell order。周期展开和最终 replica reduction
现在都在 device 完成。直接切换到 GPUMD 的逐邻居径向
dot A/B 路径会降低对当前 NEPAdapters 目标的精度，因此未作为生产路径保留。

本次结果保存在 `/tmp/mdescriptor-nep-gpumd-formal-end2end.json` 和
`/tmp/mdescriptor-nep-gpumd-end2end.json`。

多数据集命令为：

```bash
MDESCRIPTOR_CUDA_PLUGIN_DIR="$PWD/build-cuda" \
  .venv/bin/python scripts/benchmarking/benchmark_cuda_nep_datasets.py \
  --warmup 1 --repeat 3 --output /tmp/mdescriptor-nep-gpumd-end2end.json
```

加入 carbon 全量 450 帧的命令为：

```bash
MDESCRIPTOR_CUDA_PLUGIN_DIR="$PWD/build-cuda" \
  .venv/bin/python scripts/benchmarking/benchmark_cuda_nep_datasets.py \
  --carbon-all --warmup 1 --repeat 3 --output /tmp/mdescriptor-carbon-full.json
```

该命令必须在宿主机 GPU 权限环境运行；容器/沙箱中看不到 NVIDIA device 时，结果
只能标记为 unavailable，不能代替宿主机门禁。

## 13. 实施顺序

| 阶段 | 内容 | 完成标准 |
|---|---|---|
| G0 | 固化 GPU fixtures、tolerance 和 feature layout | CPU golden 不变 |
| G1 | backend dispatcher、plugin loader、control bridge | CUDA plugin 可加载，错误结构化 |
| G2 | execution context、DeviceBatch、workspace | 不同大小 batch 可安全复用容量 |
| G3 | 完整 GPU NeighborGraph | triclinic、周期镜像和 pair 顺序一致 |
| G4 | NeighborList CUDA backend | pair samples 和 offsets 一致 |
| G5 | spherical coefficient pipeline | SphericalExpansion 通过 contract/tolerance |
| G6 | SOAP radial/power spectrum | 三个描述符通过 GPU 回归 |
| G7 | CUDA wheel、GPU CI、benchmark | plugin 可发布并可验证 |
| G8 | 普通 NEP descriptor GPU 扩展 | coefficients device model、CUDA kernel、parity benchmark |
| G9 | DPA4/DPA4C CUDA descriptor path | 自有 FP32 kernels、GPU 权重上传、parity gate 和 CUDA wheel 依赖审计 |
| G10 | 21 个 CPU-only 描述符 CUDA path | `extended_descriptors.cu`、payload flattening、CPU/CUDA contract 和 golden parity |

本文件记录设计共识和当前实现边界；standalone CUDA runner、GPU 数值 tolerance、
wheel 发布和跨平台 benchmark 仍应按上述门禁在目标硬件上完成验证。
