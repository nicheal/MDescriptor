# GPU backend design consensus / GPU 后端设计共识

状态：设计已确认；第一阶段代码已落地，GPU 硬件回归与发布矩阵仍需在具备
CUDA runner 的环境中执行。

当前实现包含 backend seam、惰性 CUDA plugin loader、可复用的 CUDA context /
batch / CSR+SoA graph、NeighborList CUDA kernel，以及
SphericalExpansion / SOAP radial / SOAP power 的共享 device coefficient
pipeline。CPU canonical graph 作为第一版的确定性语义来源并上传到 device；
这保留了现有 triclinic、周期镜像和 pair 顺序语义，后续可在不改变 backend
接口的情况下替换为 device cell-list 构建。

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

构建流程固定为：

```text
计算周期 image/cell 信息
→ 构建 cell list
→ 统计每个 center 的邻居数
→ exclusive scan
→ 按确定性顺序填充 CSR
→ 检查 cancellation
```

不能用 atomic append 的到达顺序作为最终 pair 顺序。GPU 结果中的 `offsets`、`atoms`、`shifts`、displacements 和 pair samples 必须与 CPU 语义一致。

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

第一阶段支持矩阵：

- Linux x86_64；
- 单 GPU；
- 一个经过 CI 验证的 CUDA toolkit/runtime 主版本；
- 显式的 `CMAKE_CUDA_ARCHITECTURES`；
- 仅支持 `device="cuda"`；
- 暂不支持 Windows、macOS、ARM、多 GPU 和 GPU index。

具体 CUDA 版本和最低 compute capability 以实际部署 GPU/driver 与 CI 矩阵为准，并写入 plugin 发布说明。

## 11. 第一阶段范围

只实现以下四个描述符：

```text
NeighborList
SphericalExpansion
SoapRadialSpectrum
SoapPowerSpectrum
```

明确排除：

- `NEP`；
- `DPA4`；
- `DPA4C`；
- 所有模型权重的 GPU 化；
- GPU tensor 或异步公共接口；
- 多 GPU 和跨设备 graph cache。

模型 GPU 化未来沿用 `CudaExecutionContext`，但单独处理 `ModelSession`、device weights、checkpoint ABI 和模型数值验证。

## 12. 测试与性能门禁

建议新增独立 GPU 测试目录：

```text
tests/gpu/
  test_cuda_availability.py
  test_cuda_neighbor_list.py
  test_cuda_spherical.py
  test_cuda_soap_spectra.py
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
| G8 | 单独规划 NEP/DPA4/DPA4C | 不与 standalone GPU 版本耦合 |

本文件记录设计共识和当前实现边界；CUDA runner、GPU 数值 tolerance、wheel
发布和 benchmark 仍应按上述门禁在目标硬件上完成验证。
