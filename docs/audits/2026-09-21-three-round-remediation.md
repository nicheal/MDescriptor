# 三轮审计整改台账

审计基准：`main` @ `91361804fa7e5e0367c6a2b142f162eec7a3eaa6`  
记录日期：2026-09-21

本台账合并前三轮审计中已经复核过的结论。第一轮提出的 CUDA CI 问题按要求不作为核心正确性整改项；本轮仅推进可信 runner 的触发和发布架构契约，未扩大 toolkit 支持矩阵。

## 已确认问题

| 编号 | 问题 | 影响 | 状态 |
| --- | --- | --- | --- |
| C1 | CUDA MLIP-4 MTP `eval_count > 65536` 时，host 仍然上传并启动 kernel，kernel 直接返回 | 输出保持为清零值，调用方得到静默错误结果 | 已修复并通过宿主机 GPU 回归 |
| C2 | CUDA MLIP-2 MTP `alpha_moments_count > 65536` 时同样静默返回 | 输出保持为清零值，调用方得到静默错误结果 | 已修复并通过宿主机 GPU 回归 |
| P1 | SphericalPair 对同一 `(edge, angular)` 在每个 `m` 上重复计算 radial basis | CUDA 计算量随 `2l+1` 无必要放大 | 已修复并通过 parity 回归 |
| P2 | local MBTR 每个 atom 线性扫描 structure offsets | 大 batch/大量 structure 时索引成本变成热点 | 已修复并通过 MBTR/LMBTR parity 回归 |
| P3 | SphericalPair 每次 compute 重复上传静态 GTO 常量 | 同一 backend 的重复 compute 有额外 H2D 开销（此前 host 侧 basis 构造也按调用生成） | 已修复：host basis 与 H2D payload 均按 backend 缓存，并通过重复 compute/重建回归 |
| P4 | C00PS self-correction 在 channel pair 内重复计算 radial value | 高阶/大邻域时重复工作明显 | 已修复：每个中心一次计算 radial values 和 self-correction scratch，并通过 CPU/GPU parity 回归 |
| P5 | C00PS CUDA 每次 compute 重新 flatten 分层 basis payload | 重复 host 分配/拷贝，静态 payload 命中时仍有 CPU 开销 | 已修复：payload 构造阶段 flatten，compute 直接复用连续数组，并通过 GPU 生命周期/parity 回归 |
| L1 | CUDA Python backend close 后仍保留 native implementation 和重 `_cuda_payload` | 生命周期结束后仍持有设备/模型相关对象，增加内存驻留和二次使用风险 | 已修复并通过生命周期单测 |
| L2 | model-backed adapter close 后仍保留 resolver 的模型 bytes | 大模型内容继续被 Python 对象引用 | 已修复并通过模型生命周期单测 |

## 本次推进

- C00PS 多阶段 coefficient/spectrum 拆分已完成：CUDA 在同一 stream 上先生成 per-center coefficient/self-correction workspace，再独立生成 radial/power-spectrum 输出；输出布局、labels、row offsets 和 public error surface 保持不变。RTX 2080 SUPER 上同批次稳态 benchmark（5 次 warm-up、20 次采样）为 moderate 3.748→3.836 ms、heavy 6.349→5.628 ms、angular 5.882→5.921 ms。

## 暂缓项与边界

- CPU C00PS 已增加单结构大 batch 的中心级自适应并行，并保留多结构的原有结构级调度。
- CUDA CI 已接入可信 `main` push/周调度，并默认按发布架构列表编译；toolkit 支持矩阵仍不额外扩张。

## 本轮验证要求

1. CUDA MTP 越界回归必须在宿主机 GPU 上确认：超过上限应在 kernel launch 前抛出明确错误，边界值仍可运行。
2. CPU 与 CUDA 现有 descriptor parity、生命周期及静态 payload 测试保持通过。
3. 任何性能改动不得改变输出布局、标签、结构/atom/pair row offsets 或 public error surface。
