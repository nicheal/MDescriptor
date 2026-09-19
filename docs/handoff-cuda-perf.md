# CUDA 性能与正确性 — 交接文档（2026-09-19 会话）

## 0. 现状

- `main` = `4e0991e`，已推送，工作树干净。
- 测试：`pytest tests -q` → **323 passed / 26 deselected**（deselected 为需外部参考包的 `reference` 标记）。
- 本会话 12 个提交：一次全库审计修复 + 一轮 CUDA 性能优化（含 Ewald 重写）。

## 1. 本会话完成了什么

### 1a. 审计修复（`f336a69`）

- **4×P1**：SOAP 绑定层参数不一致导致堆越界；Coulomb/Sine/Ewald 矩阵内核 `int` 乘法溢出；OpenMP 异常逃逸 `std::terminate`（统一为 `run_captured_structures`）；CUDA c00ps `l_max>20` 栈越界。
- **14×P2**：MLIP-4/MLIP-2/DPA4C checkpoint 加载期校验；CUDA generic MTP 静默零输出改为显式拒绝；DPA4 死存储删除；dpa4c `compute_mutex_` 一致性；NEP/MTP 模型缓存过期清理；SoapTurbo/MTP 懒初始化线程锁；`model_adapter` 异常面收口；`so3_radial_basis` 提升出结构循环。

### 1b. CUDA 性能系列（12 个提交）

| 提交 | 内容 | 实测收益（RTX 2080） |
|---|---|---|
| `0dec6bc` | 消除每线程 local 数组：attention 寄存器化、SOAP mu1nu1 两阶段、generic 按需 scratch | DPA4 1.39-1.62× |
| `9b923d1` | attention logit 预计算内核 + dpa4c 校验/缓冲复用 | +1.07-1.14× |
| `445f9d2` | SOAP 消除冗余超越函数（prefactor 提升，n_max 倍冗余） | 1.76-2.43× |
| `de98ad7` | SOAP 展开块级协作（每原子一 block，shared 协作） | 2.0-8.8× |
| `e2a9824` | SOAP 混合派发（密集→协作，稀疏→两阶段） | 稀疏批次 8.1× |
| `130a5c7` | 确定性分段归约（≤8 段，位级一致/ε 级） | 稀疏 9.07× |
| `802ec70` | SOAP schema 边界声明（GTO `r_cut>1`、基条件数） | — |
| `d132a9c` | 混合派发阈值修正（曾误删协作内核致密集回归） | 密集恢复 1109-1404ms |
| `f6cbdae` | Coulomb/Sine 对并行填充（每元素一线程 + 每结构后处理） | 5.75× → **0.5-0.65×（GPU 反超）** |
| `4e0991e` | **Ewald 两级并行重写**（宿主位级复刻 setup + 相位/实空间/倒数三内核） | **32584ms → 23.4ms（1393×）** |

顺带修复：EAD CUDA 省略 `parameters` 时崩溃（`rotational.py` 的 `_canonical_configuration`，随 `f6cbdae` 入库）。

### 1c. 关键教训（已固化到代码注释）

- Ewald 重写从失败到成功的转折点：**二分搜索前缀语义**（`prefix[middle] <= thread` 才对；`prefix[middle+1]` 错误版本单结构测试全对、多结构全错）。
- **单 scratch 基址 + 绝对偏移**取代多基址方案（后者连续 3 轮 offset 错位）。
- `workspace_buffer` 增长时会 free 旧指针 → 一次分配、切片使用。

## 2. 当前性能全景（8 帧 × 192 原子，RTX 2080）

| 描述符 | CPU (ms) | GPU (ms) | GPU/CPU | 状态 |
|---|---|---|---|---|
| EwaldSumMatrix | 29.7 | 23.4 | **0.79×** | ✅ 本会话修复 |
| CoulombMatrix / SineMatrix | 6.2 / 8.1 | ~13 | 0.5-0.65× | ✅ 本会话修复 |
| SphericalExpansion | 109.7 | 25.9 | **0.24×** | ✅ GPU 快 |
| SO4 / SNAP | 8.2 / 8.0 | 5.6 / 5.5 | ~0.68× | ✅ |
| SOAPTurbo | 13.4 | 22.8 | 1.70× | 可接受 |
| EAD | 42.1 | 77.3 | 1.84× | 可接受 |
| ACSF | 28.7 | 61.2 | 2.13× | 低优先 |
| **SphericalExpansionByPair** | 1035.6 | **2821.7** | **2.72×** | ⏳ 绝对时间最大 |
| **MBTR** | 66.6 | **2411.8** | **36.23×** | ⏳ 病态 |
| **LMBTR** | 65.1 | **1563.2** | **24.00×** | ⏳ 病态 |
| **ValleOganov** | 6.5 | **697.6** | **107.71×** | ⏳ 最严重比例 |
| **C00PSMLFF** | 38.6 | **286.8** | **7.43×** | ⏳ |
| **SO3** | 36.7 | **160.3** | **4.37×** | ⏳ |
| SortedDistances | 2.8 | 13.1 | 4.70× | 低优先（绝对小） |
| SOAP 密集单结构（1×256 原子, l_max=12） | 48 | ~1278 | **~26×** | ⏳ 架构上限 |

## 3. 下一步任务（按收益排序）

### 任务 A：MBTR / LMBTR / ValleOganov（首要，36×/24×/108×）

- **文件**：`cpp/cuda/src/extended_descriptors_mbtr.cu`
- **几何**：`mbtr_kernel` 每行一线程（MBTR/ValleOganov 为结构行、LMBTR 为原子行），行内**串行**遍历全部邻近对并逐点填充 `grid_n` 直方图——8 结构批次只有 8 个线程。
- **建议**：套用本会话已验证的模式之一——
  - 两阶段：phase1 每对（或每（行，对段））一线程计算贡献写 scratch，phase2 按行固定序归约（参考 SOAP `e2a9824`/`130a5c7`）；
  - 或每（行，grid 区间）一线程并行填充 + 原子累加（MBTR 的 Gaussian 只贡献局部 grid 区间，天然适合）。
- **验收**：`tests/test_mbtr_golden.py`、`tests/test_lmbtr_golden.py`、`tests/test_valleoganov_golden.py`（含 OpenMP 变体）+ 与 CPU 相对差 ≤1e-12。

### 任务 B：SphericalExpansionByPair（绝对时间最大，2.72×）

- **文件**：`cpp/cuda/src/extended_descriptors_basic.cu`（发射点）、`cpp/cuda/src/extended_descriptors_common.cuh:2448`（`spherical_pair_kernel`）。
- **几何**：已是每边一线程（`edge = blockIdx.x * blockDim.x + threadIdx.x`），2.7× 差距来自每边成本而非并行度。
- **建议**：先查是否有与 SOAP `445f9d2` 同款的冗余超越函数（GTO prefactor 对 `(angular, raw)` 重复计算）；再用 ncu（见 §5）看占用率/访存。相对差 1e-11 级，验证容差放宽到 1e-9。

### 任务 C：C00PSMLFF（7.43×）

- **文件**：`cpp/cuda/src/extended_descriptors_c00ps.cu`（`c00ps_mlff_kernel`，每中心一线程，每中心一条 coefficient row——与 SOAP 展开内核**高度同构**）。
- **建议**：直接套用 SOAP 的**块协作**（`de98ad7`）或**混合派发 + 分段归约**（`e2a9824`/`130a5c7`）模式；注意该内核之前修过 `l_max>20` 栈越界守卫，改造时保留。
- **验收**：`tests/test_c00ps_mlff.py`、`tests/test_c00psmlff_golden.py`。

### 任务 D：SO3（4.37×）

- **文件**：`cpp/cuda/src/extended_descriptors_rotational.cu`（`kind==0` 分支；bispectrum 计划在 `rotational_plan_cache`）。
- **建议**：先确认 SO3 实际走的内核（bispectrum 路径 vs 专用实现），多半是占用率问题。
- **验收**：`tests/test_so3_golden.py`、`tests/test_descriptor_symmetry.py`。

### 任务 E：SOAP 密集单结构（~26×，架构上限）

- 已做：块协作 + 分段归约 + 混合派发（见 §1b）。
- 剩余瓶颈：每 block 串行边循环 + 1 block/SM 占用率（37.9KB shared）。
- 下一步需要 **warp 级协作归约**，会改变求和顺序（~1e-11 级，与既有 GPU-vs-CPU 差同级）——**属设计决策，动手前先与用户确认**。

### 任务 F：ACSF / SortedDistances（低优先）

- 绝对时间 < 61ms；比值 2.1× / 4.7×。仅在 A-D 完成后考虑。

## 4. 非性能遗留（正确性安全，但需知道）

1. `dpa4c.cu` 构造期异步 H2D 依赖 pageable 暂存语义（`dpa4_common.cuh` 注释锚定）——若改 pinned 内存必须同步改拷贝方式。
2. 字符串匹配的错误路径映射：`core/descriptor.py` 的 `_input_error_path`、`core/backends.py` 的 `_looks_cancelled`。
3. `neighbor.cpp:50` 网格 `int` 线性化（纯理论，需数亿原子盒）。
4. OMP 体内剩余防御性 throw 死代码（已被 `run_captured_structures` 无害化）：`acsf.cpp:90`、`c00ps_mlff.cpp:467/481`、`ace.cpp:923`。
5. 超长文件：`dpa4.cpp` 2260 行、`mtp4.cpp` 1392 行、`soap.cpp` 1256 行。
6. 26 个 reference 测试从未在本机运行（需 `pip install dscribe==2.1.2` / `deepmd-kit[torch]==3.2.0` / `featomic==0.6.6` / `pyxtal_ff==0.2.3` / `nep-adapters==1.0.2`）。
7. `benchmarks/` 本地 2.6GB 中间产物（已 gitignore，可清理）。

## 5. 环境与工具

- **ncu/nsys 在 WSL2 被拦**：注册表 `RmProfilingAdminOnly=0` 已写入并重启，但 WSL 的 GPU-PV 直通层有独立限制。完整解锁需在 Windows NVIDIA 控制面板 → 桌面/开发人员 → **管理 GPU 性能计数器 → 允许所有用户访问**。之后可 `ncu --kernel-name regex:xxx --launch-count 1 --section SpeedOfLight --section Occupancy`。
- **compute-sanitizer 可用**（无需额外权限）：`compute-sanitizer --tool memcheck .venv/bin/python script.py` ——定位非法访存的首选。
- **段错误定位**：`python -X faulthandler script.py`（给出 Python 侧调用点；无 gdb 可用）。
- **`/tmp` 会被重启清空**：性能基线的 `.npy` 存到 `~/baselines/`（本会话被清过两次）。

## 6. 方法论（复用本会话流程）

### 诊断三步

1. **GPU/CPU 比率扫描**（`signal.alarm` 超时保护）先找病态路径，再谈优化——比函数级 profiling 更快定位。
2. 改动前保存基准输出 `.npy`；改动后 `np.array_equal` 位级对比（除已授权改变结合序的方案，用相对差 ≤1e-12 验收）。
3. 越界用 compute-sanitizer，段错误用 faulthandler；数值错误用 Python 复刻参考逐项对比中间量（实空间/相位/倒数分开验证）。

### 已验证的改造模式（按场景选择）

| 模式 | 参考提交 | 适用 |
|---|---|---|
| 块协作（每原子一 block，lane 分片累加器） | `de98ad7` | 每元素累加、需位级一致 |
| 两阶段 + 确定性分段归约（≤8 段） | `e2a9824`/`130a5c7` | 边数大、需确定性 |
| 混合派发（按每原子边数选路径） | `d132a9c` | 批次形状差异大 |
| 对并行填充（每 (structure,i,j) 一线程） | `f6cbdae` | 元素独立 |
| 宿主位级复刻 + 设备按串行累加序 | `4e0991e` | 有 CPU 参考需对齐 |
| 冗余超越函数提升（prefactor 按 angular 复用） | `445f9d2` | 内层有重复 pow/exp |

### 踩过的坑（务必避免）

1. **二分搜索**：找所属行用 `prefix[middle] <= idx`（右边界），不是 `prefix[middle+1]`——后者单结构测试发现不了。
2. **scratch 布局**：单基址 + 绝对偏移；并列数组不要交错（stride 3/1 不要合成 stride 4）。
3. **workspace 分配**：一次分配（含所有区域）再切片；二次调用 `workspace_buffer` 会 free 旧指针。
4. **零 grid**：空帧/空批次先判 `total > 0` 再 launch。
5. **C++ 大文件手术**：不要多段小补丁（本会话被搅坏 3 次）；整文件程序化组装或单锚点替换后**立即编译**；改完检查大括号平衡。
6. 性能改动三步验收：位级/容差 + 全量 `pytest` + 与前构建的基线和计时对照。

## 7. 快速上手

```bash
# 构建（CUDA 由 nvcc 自动检测，AUTO 模式）
.venv/bin/pip install -e . --no-build-isolation --no-deps

# 全量测试
.venv/bin/python -m pytest tests -q -p no:cacheprovider

# Ewald 自检（多结构是必须的——单结构测不出二分 bug）
# 8 帧 × 192 原子，permutation 三档，与 CPU 相对差应 ≤1e-12
```

**性能扫描脚本模板**：用 `signal.alarm` 包裹每个描述符的 computed，构造 8×(192 原子 12Å 周期盒)、species 1-8，同时跑 cpu/cuda 取 min-of-2，输出比率表——本会话用它在 20 秒内扫完全部 17 个描述符。
