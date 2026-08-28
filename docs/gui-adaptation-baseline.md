# MDescriptor GUI 适配正式设计基线

**状态：** 已确认的设计基线  
**日期：** 2026-08-28  
**适用范围：** MDescriptor 与材料结构描述符 GUI/backend 的集成

## 1. 目标与边界

MDescriptor 继续定位为纯描述符计算引擎。GUI/backend 可以查询描述符能力、构造描述符并执行计算，但数据集管理、任务调度和结果分析不进入 MDescriptor 核心。

~~~text
DeepMD / extxyz
        ↓
DatasetAdapter（GUI/backend）
        ↓
StructureBatch
        ↓
MDescriptor
        ↓
DescriptorResult
        ↓
结果存储与可视化（GUI/backend）
~~~

MDescriptor 的外部接口必须是小而稳定的深模块接口：调用方只需要了解描述符名称、配置、输入契约和结果契约，不需要了解 C++、NumPy、模型加载或具体 kernel。

## 2. 已确定的公共接口

保留现有接口：

~~~python
mdescriptor.list_descriptors()
mdescriptor.create_descriptor(configuration)
descriptor.compute(batch, control=control)
~~~

新增接口和常量：

~~~python
mdescriptor.describe_descriptor(name)
mdescriptor.get_runtime_info()

mdescriptor.API_VERSION
mdescriptor.CONFIGURATION_SCHEMA_VERSION
mdescriptor.DESCRIPTOR_INFO_SCHEMA_VERSION
mdescriptor.RESULT_SCHEMA_VERSION
~~~

describe_descriptor(name) 返回 JSON-safe dict。查询必须是静态操作，不得实例化描述符、加载模型、初始化 native kernel 或导入重型 runtime。

DescriptorInfo 可以作为 registry 内部的不可变实现类型；公共查询结果不强制暴露 dataclass。

## 3. DescriptorInfo Schema

描述符信息的顶层结构固定为：

~~~text
schema_version
name
display_name
description
category
level
backend
capabilities
parameters
execution
input
output
asset
~~~

不暴露 import_path、Python 类型、C++ 类型或前端组件名称。

示例：

~~~python
{
    "schema_version": 1,
    "name": "ACE",
    "display_name": "ACE",
    "description": "Atomic Cluster Expansion descriptor.",
    "category": "local",
    "level": "atom",
    "backend": "cpp",
    "capabilities": ["sparse", "num_threads", "cooperative_cancel"],
    "parameters": {
        "species": {
            "type": "species",
            "required": True,
            "description": "Chemical species included in the descriptor."
        },
        "N": {
            "type": "integer",
            "required": False,
            "default": 3,
            "minimum": 1
        },
        "rcut": {
            "type": "number",
            "required": False,
            "default": 5.0,
            "exclusiveMinimum": 0,
            "unit": "Å"
        }
    },
    "execution": {
        "devices": ["cpu"],
        "num_threads": True,
        "cooperative_cancel": True
    },
    "input": {
        "periodicity": ["isolated", "fully_periodic"],
        "mixed_periodicity": False,
        "spin": False,
        "charge_spin": False
    },
    "output": {
        "dtypes": ["float32", "float64"],
        "sparse": True
    },
    "asset": {
        "policy": "none"
    }
}
~~~

### 3.1 参数 Schema 规则

使用接近 JSON Schema 的受控子集：

~~~text
type
required
default
minimum / maximum
exclusiveMinimum / exclusiveMaximum
enum
unit
description
items
properties
~~~

允许的领域语义类型包括：

~~~text
integer
number
boolean
string
enum
species
model
array
object
~~~

Schema 只表达计算语义，不包含 React、Ant Design 或其他 GUI 组件名。GUI 根据 species、model 等语义类型选择元素选择器、文件选择器或通用控件。

所有 28 个 registry 描述符都必须提供合法 Schema。GUI 第一版重点优化 SOAP、ACSF、ACE、MTP、NEP、DPA4 和 DPA4C，其余描述符使用通用语义表单。

Schema 只暴露 canonical 参数名。历史别名可以继续保留给直接 Python 调用，但不进入默认 GUI 表单。

``type: "model"`` 参数的 JSON 形式固定为两种：文件选择器直接提交路径字符串
（表示显式本地路径），或提交 ``ModelResource.to_dict()`` 产生的带
``"__type__": "ModelResource"`` 标记的对象（表示命名/带摘要资源）。其它对象
形式拒绝解析。历史构造器对部分 per-species 数组接受的标量广播写法只用于兼容旧
配置；GUI 和新的持久化配置使用数组形式。

ACE 的 ``maxdeg`` 和 ``wL`` 同样以数组 schema 表达按 correlation order 的
值；直接 Python 调用和旧配置仍可提交单个数值作为广播简写。

确定性的默认值写入 default。依赖其他参数的值或运行时才能确定的值不伪造静态默认值；跨字段约束仍由描述符构造阶段验证。

## 4. Registry 与元数据来源

registry 是描述符名称、分类、能力和元数据的唯一来源。

DescriptorSpec 继续保存注册身份信息：

~~~text
name
import_path
asset_policy
backend
level
capabilities
optional_extra
~~~

复杂的描述信息由独立的 DescriptorInfo 保存并由 DescriptorSpec 引用。describe_descriptor() 将二者组合为完整的 JSON-safe 响应，避免重复维护 name、backend、level 等字段。

自定义 registry 可以继续支持没有 GUI Schema 的纯 Python 描述符；这类描述符不能作为 GUI 可配置描述符展示，必须在补齐 DescriptorInfo 后才进入 GUI registry。

## 5. 执行、输入和输出契约

### 5.1 Execution

execution.devices 是规范声明而非提示信息。描述符构造时必须拒绝不支持的 device，并由契约测试验证声明与实际行为一致。

当前第一版只承诺 CPU：

~~~python
"devices": ["cpu"]
~~~

backend="cpp" 表示实现方式，不表示 CUDA 支持。

现有的同步 compute() 和 ComputeControl 保留不变。GUI/backend 使用独立 worker process 执行长任务；优先使用 cooperative cancel，超时后由 Job Manager 终止 worker。

### 5.2 Input

描述信息按描述符暴露：

~~~text
periodicity
mixed_periodicity
spin
charge_spin
~~~

StructureBatch 继续只接受 isolated 或 fully periodic 结构，并拒绝 mixed periodicity。第一版不增加 check_input_compatibility()；DatasetAdapter 负责导入时扫描 frame 并生成详细诊断，MDescriptor 在 compute() 时继续做最终严格校验。

### 5.3 Output

DescriptorResult 继续作为唯一结果契约，保留：

~~~text
values
level
structure_ids
row_offsets
labels
samples
feature_count
metadata
~~~

这些字段已经足够支持 atom、structure、pair 三种粒度的可视化和 PCA 点回跳。不要增加 to_gui()、to_react() 或将大型数组转换成 JSON。

## 6. 模型资源与可复现性

asset 至少包含：

~~~text
policy: none / optional / required
parameter: model
allow_external: true / false
bundled_resources
file_extensions
~~~

GUI/backend 负责文件选择和权限处理，MDescriptor 负责模型解析和校验。

模型结果和缓存以实际 SHA-256 digest 作为 identity，并记录：

~~~text
resource reference
digest
source: explicit / cache / package
~~~

绝对路径只能作为诊断信息，不能作为缓存匹配的唯一依据。

NEP/MTP 的只读解析模型由原生层按 digest 共享；Python ``LoadedModel`` 只向声明
接收预加载权重的实现（当前为 DPA4/DPA4C）提供 CPU 权重，不把不会被 kernel 消费
的文本/JSON 副本放入 Python 权重缓存。

## 7. 版本与错误契约

使用独立版本号：

~~~text
api_version
configuration_schema_version
descriptor_info_schema_version
result_schema_version
~~~

配置 Schema 和结果 Schema 继续使用现有版本机制。新增可选字段不改变既有字段含义；未知的 descriptor-info Schema 版本必须拒绝解析。

配置和输入异常保留现有异常类型与字符串消息，同时增加可选结构化信息：

~~~text
code
path
details
to_dict()
~~~

示例：

~~~python
{
    "code": "invalid_value",
    "path": ["rcut"],
    "message": "rcut must be positive",
    "details": {}
}
~~~

path 使用 JSON 数组，以支持 trans.r0、D.wL 等嵌套参数。

## 8. GUI/backend 不属于 MDescriptor 的内容

以下功能由 GUI/backend 实现：

- DeepMD、extxyz 等文件加载器
- 数据集注册、索引、统计和质量检查
- 当前数据集/结构/任务状态
- Job Manager 和 worker 生命周期
- 数据集级 chunk 调度
- Zarr、NPY、SQLite 等存储
- PCA、UMAP、聚类和相似度分析
- GUI 专用序列化

当前 compute() 整体物化一个 StructureBatch 的结果。第一版由 backend 分 chunk 调用现有接口并负责合并 offsets、samples 和结构 ID，不新增 MDescriptor streaming API。

每次 DescriptorRun 的 backend 元数据至少记录：

~~~text
mdescriptor version
api_version
configuration_schema_version
descriptor_info_schema_version
result_schema_version
descriptor configuration
model digest
dataset identity
selection/chunk identity
~~~

StructureBatch.ids 必须由 DatasetAdapter 提供稳定、唯一、可复现的 dataset-scoped ID。MDescriptor 不理解 DeepMD frame 或 extxyz 行号。

``StructureBatch`` 和 ``DescriptorResult`` 都在构造时复制并冻结 NumPy 数组；
调用方修改输入或结果之前持有的数组不会改变正在执行或已经完成的任务。结果的
``values``、``samples`` 和 offsets 数组对外只读。

对支持 cooperative cancel 的描述符，``ComputeControl.total()`` 和
``completed()`` 以结构数为单位；取消以 ``CancelledError`` 结束，backend 应以
``execution.cooperative_cancel`` 决定是否展示软取消，超时仍由 worker 管理器兜底。

## 9. 实施顺序

1. 实现独立 DescriptorInfo、版本常量和 describe_descriptor()。
2. 为 28 个 registry 描述符补齐 Schema、execution/input/output/asset 元数据。
3. 增加设备约束、模型 identity 和结构化错误信息。
4. 增加 registry 驱动的契约测试与文档检查。
5. GUI/backend 接入；优先优化首批七个描述符。

## 10. 验收标准

- 所有 registry 描述符均可通过 describe_descriptor() 查询。
- 查询结果始终 JSON-safe，且不加载模型或 native runtime。
- Schema 参数与 public constructor 的 canonical 参数一致。
- 默认配置可以通过 DescriptorConfiguration 和 create_descriptor() 重建。
- 不支持的设备和输入在计算前被明确拒绝。
- 结构化错误可以定位到参数或嵌套字段。
- 模型缓存和结果复用以 digest 为准。
- progress/cancel 的声明与实际行为一致。
- 文档和测试均由 registry 驱动，避免产生第二套描述符清单。

## 11. 明确保持不变的核心

以下内容不因 GUI 适配而重构：

- 数值公式和 C++ kernel
- StructureBatch 的核心布局与周期性规则
- DescriptorResult 的行索引和样本语义
- 同步 Descriptor.compute() 接口
- DescriptorConfiguration 与现有 factory
- C++/NumPy backend 的内部实现
- DeepMD/extxyz 解析职责
- 打包、wheel 和 CI 发布体系
