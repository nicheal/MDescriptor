MDescriptor 剩余重构收口计划
1. 结论与边界
当前已完成 src/ 布局、28 个描述符注册、统一生命周期、_native 拆分、官方 DPA checkpoint 加载、三个默认资产打包及基础 wheel 验证；ACE 现作为独立 C++17 standalone 描述符接入。
尚未严格达成的部分集中在五处：registry 仍有重复声明、动态 factory 仍接收裸映射、结果索引契约不完整、模型共享层名实不符、数值参考与正式 wheel 验收仍可跳过。
本轮不改数值公式、默认算法值、邻居算法或 DPA 推理核心；不重命名私有 C++ *Calculator，也不强制删除 _kernels 等职责明确的内部 module。
2. 锁定的公共 interface
- 根包只导出核心类型、异常、DescriptorConfiguration、registry 类型及 builtin_registry、list_descriptors()、get_descriptor()、create_descriptor()；删除 BUILTIN_REGISTRY、BUILTIN_SPECS、species helper 和其他旧公开名，不提供兼容别名。
- 新增不可变 DescriptorConfiguration(schema_version, descriptor, parameters)。Descriptor.configuration 返回该对象；to_dict()/from_dict() 使用固定 JSON schema。factory 改为：
  create_descriptor(
      configuration: DescriptorConfiguration,
      *,
      registry: DescriptorRegistry = builtin_registry,
  ) -> Descriptor
  factory 不再接受 name+options 或裸映射。调用形状错误保留 Python TypeError；已声明参数的非法值抛 DescriptorConfigError。
- 所有算法构造器保持真实、显式、keyword-only 签名。公共运行选项只允许 output=OutputOptions(...) 和 execution=ExecutionOptions(...)，删除直接的 dtype/sparse/device/num_threads 及 Mapping coercion。
- MTP 在有模型和无模型模式下都强制显式 species=；其他固定 species 算法同样禁止首批输入推断。model= 严格限定为 None | PathLike | ModelResource。
- DPA4 的公共参数收敛为 model/output/execution；DPA4C 仅保留实际影响推理的 calibrate。checkpoint 中的 cutoff、channels、precision、type map、spin/charge 配置均由 loader 验证和读取，不再要求调用方重复声明。
- DescriptorResult 强制 len(labels) == values.shape[1]，包括空 labels 情况；samples 固定为二维连续 int64：
  - structure：[structure]
  - atom：[structure, local_atom]
  - pair：[structure, local_atom_1, local_atom_2, shift_a, shift_b, shift_c]
  pair 身份不再重复放入 metadata。metadata v1 固定包含 descriptor、backend、level、feature count、output、execution，以及可选 model/details；拒绝无法 JSON 化的对象，不再静默 str()。
3. 实施顺序
1. 先冻结数值证据
   - 从固定提交 60dccbb 在临时目录构建隔离 reference wheel。
   - 使用统一小型周期结构和显式构造参数生成 28 个描述符的 NPZ 基线；MTP 额外覆盖无模型和已提交 MLIP4 模型两种模式。
   - manifest 记录来源提交、输入、构造参数、算法默认值、模型 hash、reference wheel/hash、DPA reference evaluator/hash、labels、level、samples、row offsets 和容差。
   - 新增 DPA4 官方 checkpoint golden，并把现有 DPA4C golden 迁入相同格式。非 DPA 描述符使用固定 reference wheel；DPA wrapper 使用直接 `dpa4desc.DescriptorEvaluator`，并与官方 checkpoint golden 分层校验。生成脚本必须指定 reference wheel/checkpoint 并显式传入 --accept；CI 永不自动更新。
2. 收口 core 与 registry seam
   - 实现配置快照及严格的 Output/Execution 转换，统一 ComputeControl | None 类型。
   - 删除 DescriptorRegistry.instantiate()、child() 和 alias 支持；parent registry 改为真实只读回退，注册名与 parent 冲突立即报错。
   - _BUILTIN_SPECS 成为唯一声明数据；mdescriptor.descriptors.__all__、懒加载映射、文档表格和测试枚举全部从 builtin_registry 派生。
   - 固定 capability 词汇：所有内置项支持 sparse；MTP/NEP/DPA 支持 model；DPA 支持 spin/charge_spin 但仅使用 CPU；实际支持线程数的 C++ 描述符声明 num_threads；C++ 后端声明 cooperative_cancel。契约测试验证声明与构造行为一致。
3. 完成结果和 species 单一实现
   - 在 result.py 集中生成和校验三类 sample 索引。
   - 所有固定 species kernel 只调用 species.py 的规范化和 batch 校验，删除首批输入推断分支。
   - 关闭后保留不可变配置、metadata 和已解析 feature_count；未知维度始终为 None。
   - 稀疏能力和 SciPy 缺失在构造阶段失败，不允许先执行 dense kernel 再发现配置不受支持。
4. 落实模型资源深 module
   - ModelResource 区分“显式路径”和“命名资源”两种互斥形态。PathLike 只解析指定文件；命名资源按 MDESCRIPTOR_MODEL_CACHE、包内资产顺序解析。
   - 缓存项存在但 hash 错误立即抛 ModelLoadError；只有缓存项不存在时才回退包内资产。解析器使用流式 SHA-256，不实现网络。
   - resolver 产出包含实际 digest 和来源的内部 resolved identity；弱引用缓存按 (loader kind, loader schema, digest) 键控并加锁，失败不缓存。
   - LoadedModel 只保存验证后的不可变配置和 CPU 权重；每个实例创建独立 ModelSession，持有 device、runtime dtype 和运行缓存。关闭 session 后释放 runtime 与强引用，不影响其他 session。
   - MTP/NEP 在私有 C++ 层拆出可共享的只读模型对象，calculator/session 独立；DPA loader 拆成 CPU checkpoint 解析验证与 per-session runtime 构建。DPA 官方 `.pt` 由受限纯 Python/NumPy reader 解析，不导入 Torch，不实现联网下载。
5. 清理与派生材料
   - 删除静态 descriptor 名称表、测试 catalog、旧 inventory marker 和无真实资产的 MLIP2 跳过测试；保留已提交的 MLIP4 hard reference。
   - 文档生成器渲染 registry 的名称、目录组、资产策略、backend、level、capabilities 和 extra，并提供 --check 模式。
   - 增加 registry 驱动的受控 benchmark：CPU、单线程、固定输入、2 次预热、5 次测量，输出版本化 JSON 和 median/p95；release 上传报告，但不设置跨机器耗时门禁。
   - 私有 C++ 类名及现有明确职责的内部 module 保持不动；只增加模型共享所需 loader/binding，不触碰 numerical kernel。
4. 测试、CI 与完成标准
- 数值结构字段精确比较；standalone/NEP 使用 rtol=1e-9, atol=1e-11，DPA 使用 rtol=2e-5, atol=1e-5（float32 checkpoint 的跨平台绝对舍入容限）。所有适用的平移、旋转、排列不变性或等变性继续作为独立门禁。
- reference CI 在 Linux CPython 3.12 固定安装 dscribe==2.1.2 与 ase==3.26.0，硬性比较其可对应的 SOAP、ACSF、矩阵、MBTR/LMBTR 和 Valle–Oganov 实现；reference/model suite 不得通过 importorskip 静默成功。
- 模型测试覆盖解析优先级、损坏缓存、hash、受限 checkpoint unpickler/global allowlist、checkpoint schema/tensor shape/dtype/type map、两个活跃 session 共享 LoadedModel 但不共享 runtime、关闭隔离、弱缓存回收及失败后重试；DPA 测试不得导入 Torch。
- 所有 CPython 3.10–3.14 正式 wheel 在仓库外执行基础 import、契约、standalone baseline、NEP 计算和资产 hash 检查。Linux、Windows x86_64、macOS arm64 的 CPython 3.12 额外执行 DPA4 的 NumPy 路径和 DPA4C 默认 C++/NumPy fallback 计算；publish 必须依赖这些验证任务。
- 最终验收要求：28 个名称全部通过统一契约与基线；MTP 两种资产模式通过；三个默认模型均在 wheel 中且 hash 正确；registry 是名称、分类和 capability 唯一来源；根导入不加载 Torch 或模型；Ruff、mypy、文档派生检查和仓外 wheel 验收全部通过；硬门禁中不存在意外 skip。
5. 固定假设
- 项目未发布，不保留旧名、旧 factory、旧配置映射或过渡 shim。
- 60dccbb 是本次算法行为与默认值的冻结基准；只有本计划明确列出的 interface 删除不受旧签名约束。
- 不承诺同一实例线程安全，不 pickle 活跃 descriptor/native/session。
- 不实现联网下载、伪异步、自研 CUDA、Linux ARM64、macOS Intel 或 DPA 内核重写。
