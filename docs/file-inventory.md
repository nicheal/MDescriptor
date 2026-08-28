# MDescriptor 文件作用清单

本表按 `git ls-files` 统计，覆盖项目中当前纳入版本控制的文件。`.venv`、
`dist`、`__pycache__`、测试缓存和其他构建生成物未纳入；`_vendor` 下的文件是随
项目打包的 DPA4 推理实现，单独列出以保持文件级完整性。

## 根目录、工程配置与文档

| 文件 | 作用简述 |
|---|---|
| `.gitattributes` | 统一 Git 中文本文件的属性、换行或二进制识别规则。 |
| `.github/workflows/release.yml` | 构建并验证源码包和平台 wheel 的发布 CI 流程。 |
| `.github/workflows/test.yml` | 执行测试、静态检查、类型检查和文档清单检查的 CI 流程。 |
| `.gitignore` | 排除虚拟环境、缓存、编译产物和本地临时文件。 |
| `CMakeLists.txt` | 声明 C++17 原生扩展的源文件、编译选项、OpenMP 和安装位置。 |
| `LICENSE` | 项目 GPLv3 许可证全文。 |
| `README.md` | 项目简介、目录布局、安装方式、输入契约、描述符 API 和模型资源说明。 |
| `docs/descriptor-cpp-plan.md` | C++ 描述符边界、Python/C++ 分工、构建发布和后续重构计划。 |
| `docs/descriptor-inventory.md` | 从内置 registry 派生的描述符名称、后端、输出级别和能力清单。 |
| `docs/plan.md` | 项目剩余重构收口、数值冻结、模型资源、测试和验收计划。 |
| `pyproject.toml` | Python 包元数据、依赖、构建、wheel/sdist、pytest、Ruff 和 mypy 配置。 |

## C++ 头文件

| 文件 | 作用简述 |
|---|---|
| `cpp/include/mdescriptor/descriptor.hpp` | SOAP、SOAPTurbo、ACSF、C00PSMLFF、MTP 和相关 calculator 的选项、接口与生命周期声明。 |
| `cpp/include/mdescriptor/ace.hpp` | ACE1-compatible C++17 calculator 的参数 ABI、特征元数据和生命周期接口。 |
| `cpp/include/mdescriptor/detail/batch.hpp` | 扁平化结构批次视图及原子数、晶胞、周期性和有限值校验。 |
| `cpp/include/mdescriptor/detail/control.hpp` | 原生计算的取消、完成进度和 `CancelledError` 辅助接口。 |
| `cpp/include/mdescriptor/detail/math3.hpp` | 不依赖外部线性代数库的三维向量、矩阵、点积、行列式和逆矩阵工具。 |
| `cpp/include/mdescriptor/detail/species.hpp` | 元素种类列表校验、元素到类型编号映射和批次元素检查。 |
| `cpp/include/mdescriptor/dpa4c.hpp` | DPA4C C++ calculator 的参数 ABI、生命周期和批次计算接口。 |
| `cpp/include/mdescriptor/extra.hpp` | 矩阵、MBTR、EAD、MTP 和旋转不变描述符的选项、特征数与计算接口。 |
| `cpp/include/mdescriptor/local_descriptors.hpp` | 局部描述符类型、参数、pair 表以及原子组成、邻居表和球谐展开接口。 |
| `cpp/include/mdescriptor/mtp4.hpp` | MLIP-4 MTP JSON 模型的读取、元数据查询和描述符计算接口。 |
| `cpp/include/mdescriptor/mtp_cinf_coeffs.hpp` | MLIP-4 `RadialBasisCinf` 使用的 Chebyshev 展开系数常量。 |
| `cpp/include/mdescriptor/neighbor.hpp` | 周期邻居图、邻居视图、位移、晶胞偏移和邻居搜索接口。 |
| `cpp/include/mdescriptor/nep.hpp` | NEP 模型选项、模型元数据和 native calculator 接口。 |

## C++ 原生实现

| 文件 | 作用简述 |
|---|---|
| `cpp/src/bindings/module.cpp` | 使用 pybind11 将原生 calculator、数组计算函数和 `ComputeControl` 暴露给 Python。 |
| `cpp/src/common/control.cpp` | 实现原生计算控制对象的重置、取消、进度读取和完成计数。 |
| `cpp/src/common/descriptor_common.hpp` | 提供晶胞读取、按结构并行执行和统一取消检查辅助函数。 |
| `cpp/src/common/extra_common.hpp` | 提供矩阵范数、参考排序、晶胞和位置读取等额外描述符共享工具。 |
| `cpp/src/common/local_common.hpp` | 校验局部描述符的 cutoff、径向阶数和角向阶数等公共参数。 |
| `cpp/src/common/local_spherical_common.hpp` | 实现局部球谐描述符所需的 Gamma、合流超几何函数和径向积分数值工具。 |
| `cpp/src/common/matrix_common.hpp` | 实现对称矩阵特征值、矩阵排列、补零和矩阵描述符公共输出逻辑。 |
| `cpp/src/common/neighbor.cpp` | 构建周期扩展原子、空间网格和高效邻居图，并处理边界与自邻居。 |
| `cpp/src/model_backed/dpa4c.cpp` | 执行默认无 spin/无 compression DPA4C 的邻居、径向网络、moment 和不变量读出。 |
| `cpp/src/common/nep.cpp` | 解析 NEP 模型并执行径向/角向基函数、球谐和 NEP 特征累积。 |
| `cpp/src/standalone/acsf.cpp` | 实现 ACSF 的 G2、G3、G4、G5 原生计算。 |
| `cpp/src/standalone/ace.cpp` | 实现 ACE1.jl 标准 `Utils.rpi_basis` 路径的径向基、球谐和 RPI 不变量。 |
| `cpp/src/standalone/atomic_composition.cpp` | 计算按结构或按原子统计的元素组成描述符。 |
| `cpp/src/standalone/c00ps_mlff.cpp` | 实现 C00PS-MLFF 的径向和角向特征、归一化及 cutoff 变体。 |
| `cpp/src/standalone/coulomb_matrix.cpp` | 计算 Coulomb matrix 数值。 |
| `cpp/src/standalone/ead.cpp` | 实现 EAD 原子环境描述符。 |
| `cpp/src/standalone/ewald_sum_matrix.cpp` | 计算周期 Ewald sum matrix。 |
| `cpp/src/standalone/matrix_dispatch.cpp` | 按矩阵类型分派 Sine、Ewald 和 Coulomb matrix 计算。 |
| `cpp/src/standalone/mbtr.cpp` | 实现 MBTR、LMBTR 和 Valle-Oganov 的几何、权重、网格与归一化计算。 |
| `cpp/src/standalone/mtp.cpp` | 实现通用 MTP 特征以及 MLIP-2/MLMTPR potential 适配路径。 |
| `cpp/src/standalone/mtp4.cpp` | 读取 MLIP-4 MTP JSON 并计算其原生径向基与矩特征。 |
| `cpp/src/standalone/neighbor_list.cpp` | 将周期邻居图整理成 pair 记录和偏移表。 |
| `cpp/src/standalone/rotational_descriptors.cpp` | 实现 SO3、SO4、SNAP 和 LBispectrum 描述符。 |
| `cpp/src/standalone/sine_matrix.cpp` | 计算周期 Sine matrix 数值。 |
| `cpp/src/standalone/soap.cpp` | 实现 SOAP 密度、径向基、球谐展开、压缩和平均。 |
| `cpp/src/standalone/soap_turbo.cpp` | 实现 SOAP-Turbo 的上游兼容参数、径向增强和压缩模式。 |
| `cpp/src/standalone/sorted_distances.cpp` | 计算每个中心原子的排序邻居距离特征。 |
| `cpp/src/standalone/spherical_expansion.cpp` | 实现局部球谐/径向基球形展开及其派生谱描述符。 |
| `cpp/src/standalone/spherical_expansion_by_pair.cpp` | 计算按邻居 pair 组织的球形展开输出和 pair 记录。 |

## Python 边界与公共核心

| 文件 | 作用简述 |
|---|---|
| `scripts/build_reference_wheel.py` | 在隔离临时目录构建指定参考版本的 wheel，供数值基线生成使用。 |
| `scripts/check_descriptor_inventory.py` | 从 builtin registry 渲染或检查 `docs/descriptor-inventory.md`。 |
| `scripts/descriptor_reference.py` | 为双结构 golden 生成器提供 reference wheel、DPA evaluator 和结果规范化辅助。 |
| `scripts/generate_descriptor_goldens.py` | 用显式 reference wheel 生成双结构本地 benchmark snapshot。 |
| `scripts/promote_descriptor_golden.py` | 将 accepted snapshot 单向复制为独立的测试 golden fixture。 |
| `scripts/benchmarking/run_descriptor_benchmark.py` | 根据测试 fixture 执行 CPU 单线程 benchmark，记录 median/p95。 |
| `scripts/benchmarking/benchmark_openmp_small.py` | 在固定小批次上比较目标描述符的单线程/多线程精度与耗时。 |
| `scripts/run_benchmark.py` | 受兼容性保留的 benchmark runner 入口。 |
| `scripts/verify_wheel.py` | 在安装后的 wheel 环境中检查导入、基线、模型资源和输出契约。 |
| `src/mdescriptor/__init__.py` | 根包稳定公共导出：核心契约、异常、registry 查询和配置 factory。 |
| `src/mdescriptor/_native.py` | 未构建 native 扩展时的占位模块，给出明确安装/构建错误。 |
| `src/mdescriptor/_native.pyi` | 私有 native 扩展的静态类型存根，描述取消控制和异常接口。 |
| `src/mdescriptor/core/__init__.py` | 汇总并导出核心输入、结果、配置、生命周期、控制和 species 工具。 |
| `src/mdescriptor/core/adapter.py` | 连接公共 descriptor 与私有数值 kernel，统一显式参数、输出、执行选项和结果转换。 |
| `src/mdescriptor/core/control.py` | 暴露 native `ComputeControl`，并在未构建扩展时提供 Python fallback。 |
| `src/mdescriptor/core/descriptor.py` | 所有描述符共享的抽象基类，集中处理输入转换、生命周期、取消和结果类型检查。 |
| `src/mdescriptor/core/errors.py` | 定义配置、输入、模型、关闭和取消等公共异常层次。 |
| `src/mdescriptor/core/input.py` | 定义 `StructureBatch`，校验扁平数组、晶胞、周期性、offsets，并支持 ASE 转换。 |
| `src/mdescriptor/core/model_adapter.py` | 为需要模型的描述符提供模型资源解析、共享加载、session 管理和推理适配边界。 |
| `src/mdescriptor/core/options.py` | 定义输出/执行选项和可 JSON 化的不可变 `DescriptorConfiguration`。 |
| `src/mdescriptor/core/result.py` | 定义 `DescriptorResult`、atom/structure/pair sample 索引、metadata schema 和 dense/sparse 输出格式化。 |
| `src/mdescriptor/core/species.py` | 规范化元素种类、要求显式 species，并校验批次元素是否被描述符支持。 |

## Python 描述符入口与数值 kernel

| 文件 | 作用简述 |
|---|---|
| `src/mdescriptor/descriptors/__init__.py` | 描述符公共命名空间，按 registry 懒加载 standalone 和 model-backed 类。 |
| `src/mdescriptor/descriptors/_kernels/__init__.py` | 私有 kernel 包标记，不作为算法公共 API。 |
| `src/mdescriptor/descriptors/_kernels/c00ps_mlff.py` | C00PSMLFF kernel 的 Python 参数整理、native 调用、标签和 metadata。 |
| `src/mdescriptor/descriptors/_kernels/ace.py` | ACE 选项规范化、符号 species 映射、native 调用、标签和 metadata。 |
| `src/mdescriptor/descriptors/_kernels/core.py` | SOAP、ACSF kernel 以及径向基、权重、native 扩展发现和结果适配逻辑。 |
| `src/mdescriptor/descriptors/_kernels/dpa4.py` | DPA4 kernel，负责 checkpoint 注入、CPU 推理调用和输出 metadata。 |
| `src/mdescriptor/descriptors/_kernels/dpa4c.py` | DPA4C kernel，负责带 calibration/线程选项的 checkpoint 推理、C++/NumPy 路由和结果适配。 |
| `src/mdescriptor/descriptors/_kernels/extra.py` | 矩阵、MBTR/LMBTR、Valle-Oganov 等额外描述符的 Python kernel。 |
| `src/mdescriptor/descriptors/_kernels/local.py` | 原子组成、邻居表、排序距离和球形展开系列的 Python kernel。 |
| `src/mdescriptor/descriptors/_kernels/mtp.py` | MTP kernel，加载可选 MLIP-2/MLIP-4 模型并调用 native 特征计算。 |
| `src/mdescriptor/descriptors/_kernels/nep.py` | NEP kernel，解析模型元数据并调用 native NEP calculator。 |
| `src/mdescriptor/descriptors/_kernels/rotational.py` | EAD、SO3、SO4、SNAP 和 LBispectrum kernel。 |
| `src/mdescriptor/descriptors/_kernels/soap_turbo.py` | SOAPTurbo kernel 的每元素参数归一化、native 调用和输出描述。 |
| `src/mdescriptor/descriptors/standalone/__init__.py` | standalone 描述符的懒加载入口和名称到实现的映射。 |
| `src/mdescriptor/descriptors/standalone/acsf.py` | 将公共 ACSF 类绑定到 `AcsfKernel`。 |
| `src/mdescriptor/descriptors/standalone/ace.py` | 将公共 ACE 类绑定到 ACE1-compatible kernel。 |
| `src/mdescriptor/descriptors/standalone/c00ps_mlff.py` | 将公共 C00PSMLFF 类绑定到 `C00PSMlffKernel`。 |
| `src/mdescriptor/descriptors/standalone/local/__init__.py` | 导出 AtomicComposition、NeighborList、SortedDistances 和球形展开类。 |
| `src/mdescriptor/descriptors/standalone/many_body/__init__.py` | 导出 MBTR、LMBTR 和 ValleOganov 类。 |
| `src/mdescriptor/descriptors/standalone/matrices/__init__.py` | 导出 CoulombMatrix、SineMatrix 和 EwaldSumMatrix 类。 |
| `src/mdescriptor/descriptors/standalone/mtp.py` | 将公共 MTP 类绑定到带模型能力的 `MtpKernel`。 |
| `src/mdescriptor/descriptors/standalone/rotational/__init__.py` | 导出 EAD、SO3、SO4、SNAP 和 LBispectrum 类。 |
| `src/mdescriptor/descriptors/standalone/soap.py` | 将公共 SOAP 类绑定到 `SoapKernel`。 |
| `src/mdescriptor/descriptors/standalone/soap_turbo.py` | 将公共 SOAPTurbo 类绑定到 `SoapTurboKernel`。 |

## DPA4/DPA4C 公共模型适配层

| 文件 | 作用简述 |
|---|---|
| `src/mdescriptor/descriptors/model_backed/__init__.py` | model-backed 描述符包入口，隔离需要本地模型资源的实现。 |
| `src/mdescriptor/descriptors/model_backed/dpa.py` | DPA checkpoint 读取、schema/type map 校验、runtime 构建和批量推理主流程。 |
| `src/mdescriptor/descriptors/model_backed/dpa4/__init__.py` | DPA4 公共类的包入口。 |
| `src/mdescriptor/descriptors/model_backed/dpa4/descriptor.py` | 用统一 adapter 工厂绑定 DPA4 默认模型资源和 kernel。 |
| `src/mdescriptor/descriptors/model_backed/dpa4c/__init__.py` | DPA4C 公共类的包入口。 |
| `src/mdescriptor/descriptors/model_backed/dpa4c/descriptor.py` | 用统一 adapter 工厂绑定 DPA4C 默认模型资源和 kernel。 |
| `cpp/include/mdescriptor/dpa4.hpp` | DPA4 C++17/OpenMP calculator 的扁平权重 ABI、生命周期和批次计算接口。 |
| `cpp/include/mdescriptor/dpa4_wigner.hpp` | DPA4 低阶 Wigner-D 旋转与单项式/tensor 权重接口。 |
| `cpp/src/model_backed/dpa4.cpp` | 执行默认 DPA4 图的邻居、环境/径向网络、GIE、SO(2) block、SO(3) grid 和 readout。 |
| `cpp/src/model_backed/dpa4_wigner.cpp` | DPA4 l=1/2/3 Wigner-D 低阶多项式计算核心。 |
| `src/mdescriptor/descriptors/model_backed/graph.py` | 提供原子序数到元素符号的 DPA checkpoint type map 查找。 |
| `src/mdescriptor/descriptors/model_backed/nep/__init__.py` | NEP 公共类的包入口。 |
| `src/mdescriptor/descriptors/model_backed/nep/descriptor.py` | 用统一 model adapter 绑定 NEP 默认模型资源和 native kernel。 |

## 随包 vendored 的 DPA4 推理实现：`_vendor/dpa4desc`

以下文件是项目内裁剪的 DPA4/DPA4C checkpoint 解析与 NumPy fallback 依赖；除
许可证、数据文件和包入口外，均服务于 checkpoint 解析、邻居图、球谐/等变网络
和读出计算。DPA4C 的默认无 spin、无 compression 路径在解析后由 C++17/OpenMP
核心执行。

| 文件 | 作用简述 |
|---|---|
| `src/mdescriptor/descriptors/model_backed/_vendor/LICENSE-LGPL-3.0-or-later.txt` | vendored DPA4 代码采用的 LGPL-3.0-or-later 许可证。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/NOTICE.md` | vendored 代码来源、版权和第三方通知。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/__init__.py` | vendor 根包入口。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/__init__.py` | DPA4 descriptor vendor 包的懒加载入口。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/api.py` | checkpoint 到 descriptor evaluator 的公开适配 API、元素映射和权重装载。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/common.py` | checkpoint 类型标记和通用解析辅助。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/__init__.py` | 纯 NumPy DPA 模型运行时包入口。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/array_api.py` | 与数组后端无关的索引、scatter、take 和累加操作。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/common.py` | dtype/精度转换、native operation 基类和 NumPy 转换工具。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/descriptor/__init__.py` | 导出 DPA4、DPA4C 和 descriptor 基类。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/descriptor/base_descriptor.py` | 为 NumPy 后端实例化 descriptor 抽象基类。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/descriptor/dpa4.py` | DPA4 descriptor 的邻居图输入、等变网络执行和输出计算。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/descriptor/dpa4_nn/__init__.py` | DPA4 神经网络组件包入口。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/descriptor/dpa4_nn/activation.py` | gated activation、SwiGLU 等激活层。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/descriptor/dpa4_nn/attention.py` | 分段 attention、包络加权 softmax 和梯度隔离辅助。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/descriptor/dpa4_nn/attn_res.py` | 深度 attention residual 模块。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/descriptor/dpa4_nn/block.py` | SeZMP interaction block、ghost feature 交换和等变交互层。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/descriptor/dpa4_nn/cartesian.py` | Cartesian 基、边张量和 Cartesian tensor product。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/descriptor/dpa4_nn/edge_cache.py` | 缓存边特征、Wigner 数据、源节点 gate 和边类型特征。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/descriptor/dpa4_nn/embedding.py` | 元素类型嵌入和几何初始嵌入。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/descriptor/dpa4_nn/ffn.py` | 等变前馈网络及其权重序列化/装载。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/descriptor/dpa4_nn/grid_net.py` | grid product、grid MLP 和按角动量分块的投影计算。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/descriptor/dpa4_nn/indexing.py` | SO(3) 维度、degree/m 索引和 Wigner 变换索引表。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/descriptor/dpa4_nn/mlp.py` | SwiGLU MLP 隐层宽度推导和前向计算。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/descriptor/dpa4_nn/norm.py` | RMSNorm 与等变 RMSNorm。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/descriptor/dpa4_nn/projection.py` | embedding 与网格表示之间的投影器。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/descriptor/dpa4_nn/radial.py` | 径向 MLP 和 C3 cutoff 包络。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/descriptor/dpa4_nn/so2.py` | SO(2) 分块对角线性变换。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/descriptor/dpa4_nn/so3.py` | SO(3) 线性层、通道线性层和焦点线性层。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/descriptor/dpa4_nn/utils.py` | 神经网络初始化、范数和 dtype 辅助。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/descriptor/dpa4_nn/wignerd.py` | Wigner 小 d 矩阵多项式和低阶系数。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/descriptor/dpa4c.py` | DPA4C descriptor 的带 charge/spin 图推理。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/descriptor/dpa4c_nn/__init__.py` | DPA4C 专属神经网络组件包入口。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/descriptor/dpa4c_nn/bispectrum.py` | degree triple 枚举、耦合布局和 bispectrum 特征维度。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/descriptor/dpa4c_nn/charge_state.py` | charge/spin 状态嵌入、规范化和校验。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/descriptor/dpa4c_nn/geometry.py` | DPA4C 角向基、degree channel、矩索引和 STF 转换。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/descriptor/dpa4c_nn/pair_film.py` | 按元素对生成 FiLM 条件调制参数。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/descriptor/dpa4c_nn/readout.py` | 不变量 readout、Gram/bispectrum 构造和向量张量收缩。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/descriptor/dpa4c_nn/spin.py` | spin channel 推导、拆分、条件化和聚合。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/descriptor/make_base_descriptor.py` | 按数组类型生成 descriptor 基类。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/utils/__init__.py` | 汇总环境矩阵、邻居图、网络、区域和 embedding 工具。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/utils/env_mat.py` | 构造平滑环境矩阵和 cutoff 权重。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/utils/exclude_mask.py` | 原子/邻居 pair 排除类型的 mask。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/utils/lebedev.py` | 读取 Lebedev 球面积分规则。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/utils/lebedev_rules.npz` | Lebedev 角向积分节点和权重数据。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/utils/neighbor_graph/__init__.py` | 汇总 DPA 邻居图、CSR 和 segment 工具。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/utils/neighbor_graph/builder.py` | 从 dense 邻居四元组构建邻居图。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/utils/neighbor_graph/csr.py` | 邻居图 CSR 边结构构建、挂接和规范化。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/utils/neighbor_graph/from_ijs.py` | 从 i/j/s 邻居数组恢复图结构。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/utils/neighbor_graph/graph.py` | 邻居图布局、padding、frame 索引和节点所有权 mask。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/utils/neighbor_graph/pairs.py` | 生成中心原子与边的 pair 索引。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/utils/neighbor_graph/segment.py` | segment sum/mean/max、softmax 和槽位占用计算。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/utils/network.py` | sigmoid、softplus、Identity 和轻量网络抽象。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/utils/nlist.py` | ghost 坐标扩展、邻居表构建和 pair 排除。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/utils/region.py` | fractional/cartesian 坐标转换和晶胞面距离计算。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/utils/safe_gradient.py` | 避免开方和向量范数在零附近数值异常。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/utils/seed.py` | 为嵌套模块派生确定性随机种子。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/utils/spherical_harmonics.py` | 计算实球谐函数。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/dpmodel/utils/type_embed.py` | 元素类型 embedding 的 padding、索引和网络封装。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/env.py` | 设置 CPU 推理使用的全局浮点精度策略。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/utils/__init__.py` | vendor 通用工具包入口。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/utils/charge_state.py` | 校验 charge/spin 状态输入的形状和值域。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/utils/plugin.py` | 轻量 plugin/variant 注册机制。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/utils/version.py` | 检查 checkpoint 与运行时版本兼容性。 |
| `src/mdescriptor/descriptors/model_backed/_vendor/dpa4desc/weights.py` | 受限 checkpoint unpickler、权重占位对象和 tensor 重建。 |

## 模型资源与 registry

| 文件 | 作用简述 |
|---|---|
| `src/mdescriptor/models/__init__.py` | 导出模型资源、解析器、session 和内置模型路径/资源常量。 |
| `src/mdescriptor/models/assets/DPA4-Air-OMat24-v20260704.pt` | 随包提供的 DPA4 官方 checkpoint。 |
| `src/mdescriptor/models/assets/DPA4C-Air-OMat24-v20260819.pt` | 随包提供的 DPA4C 官方 checkpoint。 |
| `src/mdescriptor/models/assets/nep89_20250409.txt` | 随包提供的 NEP 模型参数文件。 |
| `src/mdescriptor/models/resolver.py` | 按显式路径/缓存/包内资源解析模型，并进行流式 SHA-256 校验。 |
| `src/mdescriptor/models/resource.py` | 定义命名或显式路径模型资源及其严格 JSON 序列化格式。 |
| `src/mdescriptor/models/session.py` | 实现共享已加载模型、不可变权重快照、独立 runtime session 和弱缓存。 |
| `src/mdescriptor/registry/__init__.py` | 提供内置 registry 查询、静态 GUI 元数据查询、描述符类加载和配置 factory。 |
| `src/mdescriptor/registry/builtins.py` | 唯一声明内置 28 个描述符的名称、导入路径、后端、级别、能力、资产策略和静态元数据。 |
| `src/mdescriptor/registry/info.py` | 定义不可变、JSON-safe 的 `DescriptorInfo` schema。 |
| `src/mdescriptor/registry/registry.py` | 实现可冻结、可继承、按名称查询的描述符 registry。 |
| `src/mdescriptor/registry/spec.py` | 定义 `AssetPolicy`、`DescriptorSpec` 和懒加载描述符类的规范。 |

## 测试辅助、测试用例与测试数据

| 文件 | 作用简述 |
|---|---|
| `tests/__init__.py` | 测试包入口。 |
| `tests/_public.py` | 测试统一导入的公共 API、描述符类、异常和配置类型。 |
| `tests/contracts/test_capabilities.py` | 检查 registry capability 声明与构造器实际行为一致，并验证 registry 扩展规则。 |
| `tests/contracts/test_descriptor_info.py` | 检查静态 GUI 元数据、版本、懒加载和 schema 不可变契约。 |
| `tests/contracts/test_model_resources.py` | 检查模型解析优先级、hash、缓存、不可变快照、session 隔离和失败重试。 |
| `tests/contracts/test_public_api.py` | 检查根包导出、懒加载、显式签名、统一 compute 边界、生命周期和 factory。 |
| `tests/contracts/test_result_schema.py` | 检查结果 labels、metadata、sample 索引、row offsets 和 JSON-safe 契约。 |
| `tests/data/mlip4_test_mtp.json` | 用于验证 MLIP-4 MTP 读取和特征前缀的测试模型。 |
| `tests/reference/test_dscribe_reference.py` | 与固定 DScribe 2.1.2 对比 SOAP、ACSF、矩阵、MBTR/LMBTR 和 Valle-Oganov。 |
| `tests/_golden.py` | 读取独立 descriptor golden、比较完整结果字段并验证非周期边界策略。 |
| `tests/test_golden_independence.py` | 防止测试 golden 重新依赖本地 benchmarks 路径或缺失自带输入/输出。 |
| `tests/test_all_descriptors.py` | 遍历所有描述符，检查 native/backend、输出形状、有限值、矩阵和取消行为。 |
| `tests/test_c00ps_mlff.py` | 检查 C00PSMLFF 形状、标签、平移/旋转不变性、模式和取消。 |
| `tests/test_descriptor_symmetry.py` | 用单个水分子对 registry 中全部描述符执行原子排列、平移和旋转对称性报告。 |
| `tests/test_dpa4.py` | 检查 DPA4 官方 checkpoint、几何/排列不变性、Torch-free 推理和 session 共享。 |
| `tests/test_dpa4c.py` | 检查 DPA4C golden、charge/spin、type map、calibration 和 checkpoint schema 错误。 |
| `tests/test_mtp.py` | 检查 MTP 不变性、MLIP-4 JSON、模型替换重载和多种 radial basis。 |
| `tests/test_neighbor_graph.py` | 对比 native 周期邻居图与 brute-force 结果，并检查 self pair 过滤。 |
| `tests/test_openmp_support.py` | 覆盖排除 DPA4/DPA4C 后 13 个目标描述符的 OpenMP 精度一致性。 |
| `tests/test_coulombmatrix_openmp.py` | 检查 CoulombMatrix 的 OpenMP 排列结果和小批次速度。 |
| `tests/test_ewaldsummatrix_openmp.py` | 检查 EwaldSumMatrix 的 OpenMP 数值稳定性和小批次速度。 |
| `tests/test_lmbtr_openmp.py` | 检查 LMBTR 的局部输出、OpenMP 精度和小批次速度。 |
| `tests/test_sinematrix_openmp.py` | 检查 SineMatrix 的 OpenMP 结果稳定性。 |
| `tests/test_valleoganov_openmp.py` | 检查 ValleOganov 的归一化、OpenMP 精度和小批次速度。 |
| `tests/test_nep.py` | 检查 NEP 本地模型加载、native 后端、缩放和同路径模型替换。 |
| `tests/test_*_golden.py` | 每个 canonical descriptor 的独立双结构准确性测试。 |
| `tests/test_reference.py` | 检查 native 核心对周期结构平移不变。 |
| `tests/test_soap_acsf_advanced.py` | 检查 SOAP/ACSF 高级参数、参考值、dtype 和稀疏输出等价性。 |

### Independent golden files

Each `tests/golden/<descriptor>/` directory contains an input NPZ, expected
output NPZ and manifest with the complete result schema, tolerance and
reference provenance. These fixtures never reference `benchmarks/`.
