# MDescriptor GUI 项目设计文档

> 规范说明：本文件的旧示例以[正式设计基线](gui-adaptation-baseline.md)为准。
> 基线定义唯一的 JSON 字段、版本、模型资源和执行契约；本文件只描述 GUI
> 产品层、数据集层和 IPC 层，不再另行定义 MDescriptor public API。

> 文档版本：v0.1  
> 目标平台：Windows x64  
> 计算引擎：MDescriptor（Python + C++17）  
> GUI 技术路线：React + TypeScript + Tauri 2 + Python Controller  
> 状态：架构设计阶段

---

## 1. 项目背景

MDescriptor 已经作为独立的材料结构描述符计算引擎开发和发布：

- GitHub：https://github.com/nicheal/MDescriptor
- PyPI：https://pypi.org/project/MDescriptor/

当前 MDescriptor 已具备较完整的计算层抽象，包括：

- `StructureBatch`
- `Descriptor`
- `DescriptorResult`
- `DescriptorRegistry`
- `DescriptorConfiguration`
- `ExecutionOptions`
- `OutputOptions`
- `ComputeControl`
- Python/C++17 计算后端
- pybind11 绑定
- scikit-build-core 构建
- cibuildwheel 多平台 wheel 发布
- PyPI 自动发布流程

因此 GUI 项目的定位不是重新实现描述符框架，而是在 MDescriptor 之上构建一个面向材料数据集管理、结构浏览、描述符计算和结果分析的 Windows 桌面应用。

---

# 2. 项目目标

GUI 第一阶段需要解决三个核心问题。

## 2.1 数据集管理

支持导入、注册和管理材料结构数据集。

首批支持：

- DeepMD-kit 数据格式
- extxyz 数据格式

后续可扩展：

- ASE trajectory
- XYZ
- POSCAR / XDATCAR
- CIF
- LAMMPS dump
- HDF5 / Zarr 等

数据集页面提供基本统计与质量信息，包括：

- 结构数量
- 元素组成
- 每帧原子数
- Energy / atom 分布
- Force magnitude 分布
- 每结构最大力
- Cell volume 分布
- PBC 状态
- Energy / Force / Virial 是否存在

原则：

> GUI 注册数据集，不默认复制原始数据。

---

## 2.2 全局数据集上下文

整个应用只能有一个当前活动数据集：

```text
activeDatasetId
```

用户在左侧数据集列表中切换数据集后，以下模块全部刷新：

```text
Dataset Statistics
Structure Browser
Descriptor Calculation
Descriptor Runs
Results
PCA / UMAP
```

同时维护：

```text
activeDatasetId
activeFrameIndex
activeDescriptorRunId
```

这三个状态构成 GUI 的主要 Workspace Context。

---

## 2.3 描述符计算和分析

用户可以：

1. 选择当前数据集；
2. 浏览其中结构；
3. 选择 MDescriptor 中的描述符；
4. 设置描述符参数；
5. 对当前结构或整个数据集计算；
6. 查看任务进度；
7. 取消支持 cooperative cancellation 的任务；
8. 查看结果；
9. 对 descriptor space 进行 PCA / UMAP 等分析；
10. 从 PCA / UMAP 中选择结构并反向跳转到 Structure Browser。

---

# 3. 产品定位

建议产品定位为：

> **MDescriptor Studio**  
> Materials Dataset & Descriptor Analysis Studio

它不是单纯的 Descriptor Calculator，而是围绕以下工作流组织：

```text
Dataset
   ↓
Structure
   ↓
Descriptor
   ↓
Descriptor Space
   ↓
Analysis / Visualization
```

SOAP、ACE、ACSF、MBTR、NEP、DPA4 等仅作为 Descriptor Plugin / Registry Item，不应该成为 GUI 的一级页面。

---

# 4. 总体架构

```text
┌────────────────────────────────────────────────────┐
│                  React Frontend                    │
│                                                    │
│ Dataset │ Explore │ Descriptor │ Results │ Jobs    │
│                                                    │
│ Ant Design / ECharts / Structure Viewer            │
└──────────────────────┬─────────────────────────────┘
                       │
                   Tauri IPC
                       │
┌──────────────────────▼─────────────────────────────┐
│                   Tauri 2                          │
│                                                    │
│ Windows Window                                     │
│ Native File Dialog                                 │
│ Process Management                                 │
│ App Lifecycle                                      │
│ Updater                                            │
└──────────────────────┬─────────────────────────────┘
                       │
                JSON IPC / Sidecar
                       │
┌──────────────────────▼─────────────────────────────┐
│              Python GUI Controller                 │
│                                                    │
│ DatasetService                                     │
│ DescriptorService                                  │
│ JobService                                         │
│ ResultService                                      │
│ AnalysisService                                    │
│ CacheService                                       │
│ SQLite                                             │
└──────────────────────┬─────────────────────────────┘
                       │
                 Python API
                       │
┌──────────────────────▼─────────────────────────────┐
│                   MDescriptor                      │
│                                                    │
│ StructureBatch                                     │
│ DescriptorRegistry                                 │
│ Descriptor                                         │
│ DescriptorResult                                   │
│ ComputeControl                                     │
│                                                    │
│ Python → pybind11 → C++17                          │
└────────────────────────────────────────────────────┘
```

---

# 5. 技术栈

## 5.1 前端

推荐：

```text
React
TypeScript
Vite
Ant Design
Zustand
ECharts
3Dmol.js
```

后期根据结构浏览需求，可将 3Dmol.js 替换为自研 Three.js Viewer。

### React

负责：

- 页面和组件
- 状态驱动 UI
- Dataset 切换联动
- Descriptor 参数表单
- Jobs
- Results

### TypeScript

所有 Python IPC 数据结构都应在 TypeScript 中有明确类型定义。

### Ant Design

负责：

- Layout
- Menu
- Table
- Tree
- Form
- Input
- Select
- Slider
- Modal
- Drawer
- Progress
- Tabs
- Notification
- Dropdown

### Zustand

管理 GUI Workspace 状态：

```ts
interface WorkspaceState {
    activeDatasetId: string | null
    activeFrameIndex: number
    activeDescriptorRunId: string | null
}
```

### ECharts

用于：

- Histogram
- Scatter
- PCA
- UMAP
- Heatmap
- Distribution
- Similarity Matrix
- Statistics

### 3Dmol.js

第一阶段用于：

- 原子结构显示
- Cell
- Ball-stick
- Atom click
- Atom highlight
- Label

后期如需要以下功能：

- supercell
- force vector
- neighbor environment
- cutoff sphere
- trajectory
- descriptor → atom color mapping

可考虑统一迁移至 Three.js。

---

# 6. Windows 桌面层

使用：

```text
Tauri 2
```

职责包括：

- Windows 窗口
- Native File Dialog
- App Menu
- Process Management
- Python Sidecar 启动
- 应用关闭时释放 Sidecar
- Windows installer
- 自动升级
- 日志目录入口
- About 页面

第一阶段不使用 Electron。

原因：

- 应用本身已经包含 Python/C++；
- Electron 会再附带 Chromium + Node；
- Tauri 对 Windows-only 项目更轻量；
- Windows 10/11 可使用 WebView2。

---

# 7. MDescriptor 与 GUI 的边界

## 7.1 MDescriptor 负责

```text
StructureBatch
Descriptor
DescriptorResult
DescriptorRegistry
DescriptorConfiguration
Descriptor capabilities
ExecutionOptions
OutputOptions
ComputeControl
Model management
C++ kernel
CPU threading
Numerical correctness
```

---

## 7.2 GUI 项目负责

```text
DeepMD loader
extxyz loader
Dataset registry
Dataset statistics
Dataset fingerprint
Structure Browser
Job orchestration
Descriptor UI
Result storage
PCA
UMAP
Cache
SQLite
Windows desktop
```

---

## 7.3 明确不放入 MDescriptor 的功能

不建议加入：

```python
mdescriptor.load_deepmd(...)
mdescriptor.load_extxyz(...)
```

MDescriptor 应保持：

> atomic descriptor engine

而数据格式和数据集管理属于 GUI / Application Layer。

---

# 8. MDescriptor GUI 集成前置要求

当前 MDescriptor 已具备稳定的计算契约；GUI 直接使用下面的 registry 和版本
接口，不再根据实现类反射生成表单。

## 8.1 `describe_descriptor(name)`

这是 GUI 使用的静态描述 API。

建议：

```python
mdescriptor.describe_descriptor("ACE")
```

返回：

```json
{
  "schema_version": 1,
  "name": "ACE",
  "level": "atom",
  "backend": "cpp",
  "asset": {
    "policy": "none",
    "parameter": null,
    "allow_external": false
  },

  "execution": {
    "devices": ["cpu"],
    "num_threads": true,
    "cooperative_cancel": true
  },

  "output": {
    "dtypes": ["float32", "float64"],
    "sparse": true
  },

  "parameters": {
    "species": {
      "type": "species",
      "required": true
    },
    "N": {
      "type": "integer",
      "default": 3,
      "minimum": 1
    },
    "maxdeg": {
      "type": "array",
      "items": {"type": "number"},
      "default": [8.0]
    },
    "rcut": {
      "type": "number",
      "default": 5.0,
      "exclusiveMinimum": 0,
      "unit": "Å"
    }
  }
}
```

GUI 根据 schema 动态生成参数表单。

禁止 GUI 使用：

```python
inspect.signature(...)
```

推断 descriptor UI。

---

## 8.2 Parameter Schema

以正式设计基线规定的受控字段为准：

```text
type
description
default
required
minimum
maximum
exclusiveMinimum
exclusiveMaximum
enum
unit
```

特殊参数类型：

```text
species
model
integer
number
boolean
string
enum
```

其中 `model` 的 JSON 字符串值表示显式本地路径；命名或带摘要资源使用
`ModelResource.to_dict()` 的 tagged object。GUI 新配置只写 canonical 字段，
历史 Python aliases 仅用于直接调用和旧配置兼容。

---

## 8.3 Execution Capability

Descriptor metadata 应明确：

```text
devices
num_threads
cooperative_cancel
```

不要通过：

```text
backend == cpp
```

推断 CPU/CUDA 支持。

---

## 8.4 Input Capability

输入能力直接读取 `describe_descriptor(name)["input"]`，不增加第二个能力 API：

```json
{
    "periodicity": ["isolated", "fully_periodic"],
    "mixed_periodicity": false,
    "spin": true,
    "charge_spin": true
}
```

GUI 在导入数据集时即可提前判断兼容性。

---

## 8.5 Runtime Information

GUI 启动时直接读取：

```python
mdescriptor.get_runtime_info()
```

例如：

```json
{
  "version": "0.2.1",
  "api_version": 1,
  "configuration_schema_version": 1,
  "descriptor_info_schema_version": 1,
  "result_schema_version": 1
}
```

用于 GUI 启动兼容检查。

---

# 9. Dataset Layer

## 9.1 Dataset 抽象

GUI 内部维护：

```python
Dataset
├── id
├── name
├── format
├── source_path
├── number_of_frames
├── elements
├── properties
├── fingerprint
└── metadata
```

---

## 9.2 DatasetAdapter

定义：

```python
class DatasetAdapter:

    def scan(self) -> DatasetMetadata:
        ...

    def __len__(self) -> int:
        ...

    def get_frame(self, index: int) -> DatasetFrame:
        ...

    def iter_frames(self):
        ...
```

首批：

```text
DatasetAdapter
├── DeepMDAdapter
└── ExtXYZAdapter
```

---

# 10. DatasetFrame

GUI 自己的数据结构可以比 `StructureBatch` 更丰富：

```python
@dataclass
class DatasetFrame:
    numbers: np.ndarray
    positions: np.ndarray
    cell: np.ndarray
    pbc: np.ndarray

    energy: float | None = None
    forces: np.ndarray | None = None
    virial: np.ndarray | None = None

    metadata: dict | None = None
```

送入 MDescriptor 时转换成：

```python
StructureBatch
```

Energy、Force、Virial 不进入 MDescriptor 的 `StructureBatch`。

---

# 11. Dataset → StructureBatch

单结构：

```text
DatasetFrame
    ↓
StructureBatch
    ↓
MDescriptor
```

批量结构：

```text
frame 0 ──┐
frame 1 ──┤
frame 2 ──┼─→ StructureBatch
...       │
frame N ──┘
```

利用 MDescriptor 已有：

```text
numbers
positions
cells
pbc
offsets
ids
```

作为标准 batch 输入。

---

# 12. Dataset 注册

原则：

> 注册，不复制。

例如：

```text
D:\datasets\GaAs\
```

GUI 保存：

```text
path
format
metadata
fingerprint
statistics cache
```

原数据只读。

---

# 13. Dataset Fingerprint

用于检测数据集变化。

建议综合：

```text
root path
file list
file size
mtime
frame count
关键文件 metadata
```

计算：

```text
SHA-256 fingerprint
```

如果原始数据发生变化：

```text
old fingerprint != new fingerprint
```

则：

- 标记统计缓存失效；
- 标记 descriptor result 可能失效；
- 提示重新扫描。

---

# 14. Dataset Statistics

第一版统计：

```text
Structures
Elements
Atoms / structure
Energy / atom
Force magnitude
Max force / structure
Cell volume
PBC
```

所有统计结果缓存。

禁止每次进入 Dataset 页面重新扫描整个数据集。

---

# 15. Workspace State

React 全局状态：

```ts
interface WorkspaceState {
    activeDatasetId: string | null
    activeFrameIndex: number
    activeDescriptorRunId: string | null
}
```

联动：

```text
DatasetSelector
      ↓
activeDatasetId
      ↓
┌───────────────┬───────────────┬───────────────┐
Statistics      Explore         Descriptor      Results
```

---

# 16. 主界面布局

建议：

```text
┌──────────────────────────────────────────────────────────────┐
│ MDescriptor Studio                              Jobs   ⚙     │
├────────────────┬─────────────────────────────────────────────┤
│ DATASETS       │ Current Dataset Context                     │
│                │ GaAs_train | DeepMD | 12480 | Ga As        │
│ GaAs_train ●   ├─────────────────────────────────────────────┤
│ Si             │                                             │
│ MoS2           │              Main Workspace                 │
│                │                                             │
│ + Add Dataset  │                                             │
└────────────────┴─────────────────────────────────────────────┘
```

一级页面：

```text
Datasets
Explore
Descriptors
Results
```

全局区域：

```text
Jobs
Settings
About
```

---

# 17. Dataset Page

包含：

## Summary

```text
Name
Format
Path
Structures
Elements
PBC
Energy
Force
Virial
```

## Statistics

```text
Energy / atom histogram
Force magnitude histogram
Atoms / structure
Cell volume
```

## Health

后续：

```text
NaN / Inf
Invalid cell
Short atomic distance
Extreme force
Duplicate structure
```

---

# 18. Structure Browser

布局：

```text
┌──────────────────────────────────────────────────────────┐
│ Frame 328 / 12480        ◀  ▶  Random                   │
├─────────────────────────────────┬────────────────────────┤
│                                 │ Structure Info         │
│                                 │                        │
│          3D Viewer              │ Formula                │
│                                 │ Atoms                  │
│                                 │ Energy / atom          │
│                                 │ Max Force              │
│                                 │ Volume                 │
├─────────────────────────────────┴────────────────────────┤
│ Atom Table                                               │
│ ID | Element | x | y | z | Fx | Fy | Fz | |F|           │
└──────────────────────────────────────────────────────────┘
```

MVP：

- previous
- next
- random
- frame index
- ball-stick
- cell
- structure properties

后续：

- click atom
- neighbor highlight
- force arrows
- cutoff sphere
- supercell
- atom descriptor
- trajectory

---

# 19. Descriptor 页面

## 19.1 Descriptor 列表

来自：

```python
mdescriptor.list_descriptors()
```

禁止前端写死：

```ts
["SOAP", "ACE", "ACSF", ...]
```

---

## 19.2 Descriptor Information

来自：

```python
mdescriptor.describe_descriptor(name)
```

用于显示：

```text
Name
Level
Backend
CPU/GPU
Model requirement
Sparse
Threads
Cancel support
```

---

## 19.3 动态参数表单

例如：

```text
ACE

Species
[ Ga, As               ]

Correlation order
[ 3                    ]

Maximum degree
[ 8                    ]

Cutoff
[ 5.0                  ] Å

Threads
[ 16                   ]

Output
float32 / float64

[ Calculate ]
```

---

# 20. Compute Scope

第一版：

```text
Current frame
Entire dataset
```

后续：

```text
Selected frames
Filtered structures
Selected atoms
```

---

# 21. DescriptorRun

每一次计算必须生成独立 Run。

数据库：

```text
descriptor_runs
────────────────────────────
id
dataset_id
descriptor_name
descriptor_version
engine_version
parameters_json
scope
status
created_at
started_at
finished_at
result_path
error_message
```

禁止新计算覆盖旧结果。

---

# 22. Job Manager

所有耗时操作统一为 Job。

包括：

```text
Descriptor Compute
PCA
UMAP
Dataset Scan
Dataset Statistics
```

状态：

```text
QUEUED
RUNNING
COMPLETED
FAILED
CANCELLED
```

Job：

```text
job_id
job_type
dataset_id
status
progress
message
started_at
finished_at
```

---

# 23. MDescriptor ComputeControl 集成

对于声明：

```text
cooperative_cancel
```

的描述符：

```python
control = mdescriptor.ComputeControl()

result = descriptor.compute(
    batch,
    control=control,
)
```

取消：

```python
control.cancel()
```

进度：

```python
control.completed()
control.total()
```

GUI：

```text
ACE · GaAs_train

████████████████░░░ 82 %

8200 / 10000

[ Cancel ]
```

没有 cooperative cancel capability 时：

```text
Cancel unavailable
```

---

# 24. Result Storage

MDescriptor 返回：

```text
DescriptorResult
```

其中：

```text
values
level
structure_ids
row_offsets
labels
metadata
samples
feature_count
```

GUI 不修改 MDescriptor 的 result contract。

---

# 25. 大数组原则

禁止大数组通过 JSON IPC 传给 React。

错误：

```text
100000 × 1024
      ↓
JSON
      ↓
React
```

正确：

```text
DescriptorResult
      ↓
.npy / Zarr
      ↓
Python Analysis
      ↓
仅把可视化所需数据送给 React
```

---

# 26. Result 文件组织

建议：

```text
%LOCALAPPDATA%\MDescriptorStudio\
│
├── database.sqlite
│
├── results\
│   └── run_<uuid>\
│       ├── metadata.json
│       ├── values.npy
│       ├── samples.npy
│       └── row_offsets.npy
│
├── analysis\
│
├── cache\
│
├── config\
│
└── logs\
```

数据规模较大后可切换为：

```text
Zarr
```

---

# 27. Result Metadata

至少记录：

```json
{
  "descriptor": "ACE",
  "engine_version": "0.2.1",
  "descriptor_info_schema": 1,
  "configuration": {},
  "dataset_id": "dataset_x",
  "dataset_fingerprint": "...",
  "shape": [12480, 512],
  "dtype": "float32",
  "created_at": "..."
}
```

用于：

- reproducibility
- cache validity
- compatibility
- bug trace

---

# 28. Cache Key

建议：

```text
SHA256(
    dataset_fingerprint
    descriptor_name
    descriptor_configuration
    MDescriptor_version
    scope
)
```

缓存命中：

```text
Existing compatible result found.

[ Use existing ]
[ Recalculate ]
```

---

# 29. Results 页面

MVP：

```text
Descriptor Summary
Descriptor Matrix
Distribution
PCA
Heatmap
```

后续：

```text
UMAP
t-SNE
Clustering
Similarity
Outlier
FPS
Dataset Coverage
```

---

# 30. Descriptor Level 与可视化

MDescriptor 已统一：

```text
structure
atom
pair
```

因此 GUI 根据：

```text
result.level
```

选择行为。

## structure

```text
one point = one structure
```

## atom

```text
one point = one atom
```

## pair

```text
one point = one atom pair
```

利用：

```text
result.samples
```

进行反向索引。

---

# 31. PCA

数据流：

```text
DescriptorResult
      ↓
Python PCA
      ↓
2D coordinates
      ↓
ECharts Scatter
```

Color by：

```text
Energy
Force
Volume
Frame
Element
Cluster
```

核心交互：

```text
PCA point click
      ↓
sample identity
      ↓
activeFrameIndex
      ↓
Structure Browser refresh
```

这是应用最重要的跨模块联动之一。

---

# 32. Heatmap

对于 atom descriptor：

```text
N_atoms × N_features
```

展示：

```text
Atom ID
   ↓
descriptor dimension →
```

MVP 只显示当前结构，避免一次渲染超大矩阵。

---

# 33. IPC 协议

推荐：

```text
React
  ↓
Tauri
  ↓
Python Sidecar
```

第一版不使用 FastAPI。

使用 versioned JSON protocol。

---

# 34. IPC Request

```json
{
  "protocol_version": 1,
  "id": 101,
  "method": "dataset.list",
  "params": {}
}
```

---

# 35. IPC Success Response

```json
{
  "protocol_version": 1,
  "id": 101,
  "result": []
}
```

---

# 36. IPC Error

```json
{
  "protocol_version": 1,
  "id": 101,
  "error": {
    "code": "DATASET_NOT_FOUND",
    "message": "Dataset does not exist."
  }
}
```

---

# 37. IPC Event

```json
{
  "protocol_version": 1,
  "event": "job.progress",
  "data": {
    "job_id": "job_001",
    "progress": 0.67,
    "completed": 6700,
    "total": 10000
  }
}
```

---

# 38. Backend Ready

Python sidecar 启动后：

```json
{
  "protocol_version": 1,
  "event": "backend.ready",
  "data": {
    "backend_version": "0.1.0",
    "mdescriptor_version": "0.2.1",
    "mdescriptor_api_version": 1
  }
}
```

Tauri 在收到该消息前，计算相关页面保持 disabled。

---

# 39. 第一批 IPC Methods

建议：

```text
system.info

dataset.list
dataset.register
dataset.remove
dataset.get
dataset.statistics
dataset.frame

descriptor.list
descriptor.describe
descriptor.submit

job.list
job.cancel

result.list
result.get

analysis.pca
```

后续：

```text
analysis.umap
analysis.cluster
analysis.fps
```

---

# 40. Python Controller

目录：

```text
backend/
└── mdescriptor_studio_backend/
    │
    ├── protocol/
    │
    ├── datasets/
    │   ├── base.py
    │   ├── deepmd.py
    │   └── extxyz.py
    │
    ├── services/
    │   ├── dataset_service.py
    │   ├── descriptor_service.py
    │   ├── job_service.py
    │   ├── result_service.py
    │   └── analysis_service.py
    │
    ├── storage/
    │   ├── database.py
    │   ├── cache.py
    │   └── migrations/
    │
    ├── mdescriptor_adapter.py
    │
    └── main.py
```

---

# 41. MDescriptor Adapter

GUI Controller 不应该在各处：

```python
import mdescriptor
```

建议集中：

```text
mdescriptor_adapter.py
```

职责：

```text
runtime info
list descriptor
describe descriptor
build descriptor
compute
cancel
convert errors
```

这样 GUI 项目对 MDescriptor 的依赖集中在一个边界。

---

# 42. SQLite

建议表：

```text
datasets
dataset_statistics
descriptor_runs
jobs
analysis_runs
settings
schema_version
```

---

# 43. Database Migration

从 v0.1 就必须有：

```text
schema_version
```

例如：

```text
1 → 2 → 3
```

启动时自动：

```python
migration_001_to_002()
```

升级程序不能破坏：

```text
Dataset Registry
Descriptor Results
Settings
Analysis History
```

---

# 44. 错误体系

GUI Controller 定义 Application Error Code：

```text
DATASET_NOT_FOUND
DATASET_CHANGED
INVALID_DATASET
UNSUPPORTED_FORMAT
UNSUPPORTED_PERIODICITY
MDESCRIPTOR_INCOMPATIBLE
DESCRIPTOR_CONFIGURATION_ERROR
MODEL_NOT_FOUND
OUT_OF_MEMORY
JOB_CANCELLED
RESULT_INCOMPATIBLE
INTERNAL_ERROR
```

前端显示用户友好信息。

Python traceback 进入日志，不直接展示为主要错误 UI。

---

# 45. Logging

建议：

```text
app.log
backend.log
engine.log
```

至少记录：

```text
GUI version
Backend version
MDescriptor version
Protocol version
Windows version
CPU
Memory
Dataset
Descriptor
Configuration
Job ID
Exception
```

GUI 提供：

```text
Help → Open Log Folder
```

---

# 46. Windows 数据目录

程序：

```text
C:\Program Files\MDescriptor Studio\
```

用户数据：

```text
%LOCALAPPDATA%\MDescriptorStudio\
```

原始 Dataset：

```text
用户指定位置
```

软件升级只替换程序，不删除用户数据。

---

# 47. 开发目录

建议独立仓库：

```text
MDescriptorStudio/
│
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Datasets/
│   │   │   ├── Explore/
│   │   │   ├── Descriptors/
│   │   │   └── Results/
│   │   │
│   │   ├── components/
│   │   ├── stores/
│   │   ├── services/
│   │   ├── types/
│   │   └── App.tsx
│   │
│   ├── package.json
│   └── vite.config.ts
│
├── backend/
│
├── src-tauri/
│
├── scripts/
│
├── tests/
│
├── docs/
│
└── README.md
```

MDescriptor 继续保持独立仓库：

```text
MDescriptor/
```

---

# 48. 开发模式

MDescriptor：

```bash
pip install -e ../MDescriptor
```

GUI Backend：

```bash
python backend/main.py
```

Frontend：

```bash
npm run dev
```

Tauri：

```bash
npm run tauri dev
```

开发阶段不运行 PyInstaller。

---

# 49. Release 模式

构建：

```text
React
  ↓
Vite Build
  ↓
frontend/dist

Python Controller
  +
MDescriptor fixed version
  ↓
PyInstaller
  ↓
backend.exe

Tauri
  ↓
bundle
  ↓
MDescriptorStudio-Setup.exe
```

---

# 50. MDescriptor 版本固定

开发：

```text
editable MDescriptor
```

Release：

```text
固定 MDescriptor 版本
```

例如：

```text
MDescriptorStudio 0.3.0
MDescriptor        0.2.1
Protocol           1
```

不要让最终用户自行：

```bash
pip install -U MDescriptor
```

修改桌面应用内部的计算引擎。

---

# 51. 自动升级

成熟后使用 Tauri Updater。

应用整体更新：

```text
React
Tauri
Python Controller
MDescriptor
```

一起升级。

避免：

```text
GUI 0.5
Backend 0.4
MDescriptor 0.3
```

这种混合版本环境。

---

# 52. About 页面

建议：

```text
MDescriptor Studio
Version 0.3.0

Backend        0.3.0
MDescriptor    0.2.1
Protocol       1

Windows x64
```

---

# 53. 测试体系

## MDescriptor

负责：

```text
numerical correctness
reference outputs
C++ correctness
```

## GUI Backend

负责：

```text
DeepMD parsing
extxyz parsing
StructureBatch conversion
Dataset statistics
Cache
Job
IPC
```

## Frontend

负责：

```text
state
forms
page transitions
dataset switching
PCA → structure linking
```

## Integration

必须测试：

```text
GUI
 ↓
Backend
 ↓
MDescriptor
 ↓
Result
 ↓
PCA
 ↓
Structure
```

---

# 54. 测试数据集

GUI repo 自带极小测试数据：

```text
tests/data/

deepmd_small/
    ~10 frames

extxyz_small.xyz
    ~10 frames
```

用于 CI 和开发调试。

---

# 55. 性能原则

必须遵守：

```text
Dataset
→ lazy loading

Structure
→ current frame only

Descriptor
→ batched compute

Result
→ file-backed storage

Visualization
→ only transfer required data
```

禁止：

```text
打开 Dataset
→ 全部结构进入 RAM
```

禁止：

```text
完整 descriptor matrix
→ JSON
→ React
```

---

# 56. MVP v0.1

v0.1 只完成：

## Dataset

- DeepMD
- extxyz
- add
- remove
- switch
- statistics
- fingerprint

## Explore

- frame navigation
- 3D structure
- cell
- atom table
- energy
- force

## Descriptor

- MDescriptor registry
- descriptor info
- dynamic form
- current frame
- entire dataset
- threading
- model path if needed

## Job

- submit
- progress
- cancel where supported
- error

## Result

- result history
- descriptor metadata
- descriptor matrix information
- PCA
- PCA → Structure
- heatmap

## Desktop

- Tauri Windows app
- setup.exe
- logs
- SQLite

---

# 57. v0.1 暂不实现

```text
UMAP
t-SNE
Clustering
FPS
Active Learning
Remote GPU
HPC
Multi-dataset PCA
Dataset editing
Dataset conversion
Trajectory animation
Advanced structure comparison
```

---

# 58. v0.2

重点：

> Descriptor Space Analysis

加入：

```text
UMAP
Similarity
Clustering
Outlier detection
FPS
Dataset coverage
Train/Test overlap
```

---

# 59. v0.3

重点：

> Dataset Quality

加入：

```text
Energy outlier
Force outlier
Short-distance detection
Cell anomaly
Duplicate detection
Structure similarity
Coverage analysis
```

---

# 60. 推荐开发顺序

## Phase 0：MDescriptor GUI Integration

完成：

```text
describe_descriptor()
parameter schema
execution devices
input capabilities
runtime info
cancel/progress consistency test
```

---

## Phase 1：Backend Skeleton

完成：

```text
Python Controller
IPC
MDescriptor Adapter
SQLite
logging
```

---

## Phase 2：Dataset

完成：

```text
DatasetAdapter
DeepMDAdapter
ExtXYZAdapter
Dataset registry
Statistics
Fingerprint
```

---

## Phase 3：React + Tauri Skeleton

完成：

```text
Window
Layout
Sidebar
Workspace State
Dataset switch
Backend connection
```

---

## Phase 4：Explore

完成：

```text
Structure Browser
Frame
3D Viewer
Structure Info
Atom Table
```

---

## Phase 5：Descriptor

完成：

```text
Descriptor registry
Dynamic Form
Submit
Configuration
```

---

## Phase 6：Job

完成：

```text
Progress
Cancel
Queue
Error
History
```

---

## Phase 7：Results

完成：

```text
Storage
PCA
Heatmap
Linked selection
```

---

## Phase 8：Packaging

完成：

```text
PyInstaller
Tauri build
setup.exe
clean Windows test
```

---

## Phase 9：Updater

应用稳定后加入：

```text
Tauri updater
signed release
automatic version check
```

---

# 61. 核心设计原则

整个项目必须长期坚持以下规则。

## Rule 1

MDescriptor 是计算引擎，不是 Dataset Framework。

## Rule 2

GUI 不复制 MDescriptor descriptor implementation。

## Rule 3

GUI 不硬编码 descriptor list。

## Rule 4

GUI 不硬编码 descriptor 参数。

## Rule 5

Descriptor UI 来自 MDescriptor machine-readable schema。

## Rule 6

所有大数组留在 Python / 文件系统侧。

## Rule 7

React 只负责展示和交互。

## Rule 8

Dataset 是整个 GUI 的全局 Workspace Context。

## Rule 9

所有耗时操作进入 JobManager。

## Rule 10

计算结果必须记录 engine/configuration/dataset fingerprint。

---

# 62. 最终核心数据流

```text
                  Dataset
                     │
                     ↓
               activeDataset
                     │
        ┌────────────┼────────────┐
        ↓            ↓            ↓
   Statistics     Explore     Descriptor
                     │            │
                     ↓            ↓
              activeFrame        Job
                                  │
                                  ↓
                           MDescriptor
                                  │
                                  ↓
                        DescriptorResult
                                  │
                                  ↓
                           Result Storage
                                  │
                                  ↓
                               PCA
                                  │
                                  ↓
                         Select PCA Point
                                  │
                                  ↓
                           activeFrame
                                  │
                                  ↓
                       Structure Browser
```

---

# 63. 结论

MDescriptor Studio 的核心不是“把 MDescriptor 包一个界面”，而是建立一个清晰的四层架构：

```text
React
   ↓
Tauri
   ↓
Python Application Layer
   ↓
MDescriptor
```

其中：

```text
React
= Presentation

Tauri
= Windows Desktop Shell

Python Controller
= Dataset / Job / Result / Analysis

MDescriptor
= Descriptor Compute Engine
```

目前 MDescriptor 已经具备较成熟的输入、registry、result、configuration、cancel 和 C++ 计算边界，因此 GUI 项目不需要重新设计计算抽象。

GUI 开发前，MDescriptor 最重要的补充是：

```text
describe_descriptor()
parameter schema
device capability
input capability
runtime/API version
```

完成这些以后，GUI 可以做到：

> 新增 Descriptor → Registry 注册 → 参数 Schema 注册 → GUI 自动出现

这将显著降低长期维护成本，并保证 MDescriptor 与 GUI 两个项目可以独立演化。

---

# 64. UI 视觉设计规范

本节定义 MDescriptor Studio 的视觉方向、设计语言和组件使用原则。

目标不是制作传统 Web Dashboard，而是构建一个具有 Windows 原生感、科研软件信息密度和现代数据分析工具交互方式的桌面 Scientific Workbench。

视觉定位：

> **Fluent Scientific Workbench**

关键词：

```text
Clean
Dense
Technical
Calm
Precise
Scientific
Windows-native
Data-first
```

整体风格参考：

```text
Windows 11 / Fluent 2
VS Code / Scientific IDE
Modern Data Studio
Materials Analysis Workbench
```

需要避免：

```text
传统灰色科研工具软件
典型后台管理系统
过度卡片化 Dashboard
大面积渐变
玻璃拟态
巨大圆角
重阴影
消费型 Web App 风格
```

---

# 65. 视觉设计核心原则

## 65.1 Data First

所有视觉设计必须优先服务：

```text
Structure
Parameters
Numbers
Plots
Jobs
Scientific Results
```

装饰性元素必须保持低优先级。

例如 PCA 页面：

```text
正确：
大面积 PCA Plot + 小面积 Inspector

错误：
巨大标题 + 多个统计 Card + 很小的 PCA 图
```

---

## 65.2 Workbench，而不是 Web Page

应用应表现为一个持续存在的科研工作空间：

```text
Dataset Context
     ↓
Workspace
     ↓
Inspector
```

而不是若干互相独立的网页。

主界面结构始终保持稳定：

```text
Sidebar
Context Bar
Main Workspace
Inspector
Status Bar
```

---

## 65.3 高信息密度但不过度拥挤

桌面科研软件允许比普通 Web App 更高的信息密度。

推荐：

```text
control height     32 px
font size          13–14 px
table row          30–34 px
section gap        24 px
page padding       20–24 px
```

避免所有控件都使用：

```text
40–48 px
```

的大尺寸触控风格。

---

# 66. 默认主题

第一版采用：

> **Light-first**

原因：

- 科研用户长期查看图表和表格；
- 结构信息、参数和数值在浅色背景下更清晰；
- ECharts 科学图默认浅色环境更自然；
- Windows 科研工作站普遍仍以浅色为主。

Dark Mode 可在 v0.2 或 v0.3 加入。

---

# 67. 基础颜色系统

## 67.1 Background

建议：

```text
App Background       #F5F6F8
Workspace             #FFFFFF
Secondary Surface     #FAFAFA
Elevated Surface      #FFFFFF
```

层级：

```text
Level 0
#F5F6F8
Application background

Level 1
#FFFFFF
Main workspace / panels

Level 2
#FFFFFF + subtle shadow
Drawer / Modal / Popover
```

禁止大量：

```text
Card inside Card
Gray Card inside White Card
```

---

## 67.2 Text

```text
Primary Text          #242424
Secondary Text        #616161
Tertiary Text         #8A8A8A
Disabled Text         #A6A6A6
```

数字、科学值和参数需要保持高对比度。

---

## 67.3 Border

```text
Primary Border        #E1E4E8
Secondary Border      #EAECF0
Divider               #EDEFF2
```

边框只用于：

```text
Table
Input
Panel separation
Inspector separation
```

不要给所有 Section 都加完整边框。

---

# 68. 主色系统

推荐使用 Fluent Blue：

```text
Primary               #0F6CBD
Primary Hover         #115EA3
Primary Active        #0C5A96
Primary Light         #EBF3FC
Primary Border        #B4D6F7
```

Primary 只用于：

```text
Active Dataset
Active Tab
Primary Button
Selected Atom
Selected PCA Point
Focus
Progress
Interactive Highlight
```

禁止将蓝色作为大面积背景。

---

# 69. 状态色

状态色与科研数据颜色必须区分。

推荐：

```text
Running        Blue
Completed      Green
Warning        Amber
Failed         Red
Cancelled      Gray
```

概念 token：

```text
Success        #107C10
Warning        #F0A000
Danger         #C42B1C
Info           #0F6CBD
Neutral        #8A8A8A
```

Job Status、Dataset Health 和错误提示统一使用这一套状态语义。

---

# 70. 科学数据配色原则

UI Accent 与 Scientific Color Map 必须完全分离。

例如：

```text
UI Primary
Blue
```

但 PCA 根据 Energy 着色时，不应简单使用 UI Primary Blue。

连续数据：

```text
Energy
Force
Volume
Descriptor magnitude
```

使用科学连续色图。

分类数据：

```text
Element
Cluster
Dataset
Phase
```

使用 categorical palette。

---

# 71. 元素颜色

Structure Viewer 使用统一 Chemical Element Color Mapping。

至少保证：

```text
H
C
N
O
Si
Ga
As
Mo
S
Cr
W
```

在所有结构视图中颜色保持一致。

禁止：

```text
一个页面 Ga 是蓝色
另一个页面 Ga 是绿色
```

---

# 72. 字体系统

Windows-only 项目建议：

```css
font-family:
  "Segoe UI",
  "Microsoft YaHei UI",
  sans-serif;
```

用于：

```text
UI
Menu
Label
Button
Table
Inspector
```

---

# 73. 数值与等宽字体

路径、日志、Descriptor labels 等可使用：

```text
Cascadia Mono
```

例如：

```text
D:\Datasets\GaAs\set.000
ACE[species=Ga,As]
```

科学数值使用：

```css
font-variant-numeric: tabular-nums;
```

确保：

```text
-4.28371
-4.09122
-3.98212
```

纵向严格对齐。

---

# 74. Typography Scale

建议：

```text
App Title             18–20 px / Semibold
Page Title            22–24 px / Semibold
Section Title         14–16 px / Semibold
Body                  14 px
Secondary             12–13 px
Caption               11–12 px
Table                  13 px
```

禁止：

```text
32–48 px 巨型 Dashboard 标题
```

应用是桌面工具，不是宣传页面。

---

# 75. 圆角

圆角保持克制：

```text
Button                4–6 px
Input                 4–6 px
Panel                 6 px
Card                  6–8 px
Dialog                8 px
```

推荐全局基础值：

```text
6 px
```

避免：

```text
16 px
20 px
24 px
```

的消费型 UI 圆角。

---

# 76. 间距系统

采用：

```text
4 px base spacing
```

常用值：

```text
4
8
12
16
24
32
```

推荐：

```text
Inline gap            8 px
Label → control       6–8 px
Control group         12 px
Section internal      16 px
Section separation    24–32 px
Page padding          20–24 px
Sidebar padding       12–16 px
```

所有页面必须遵循统一 spacing scale。

---

# 77. 阴影原则

阴影只用于真正的 elevated surface：

```text
Modal
Popover
Dropdown
Drawer
Floating menu
```

普通工作区域：

```text
不使用明显阴影
```

通过：

```text
background
divider
spacing
```

建立层级。

---

# 78. 主应用布局

建议：

```text
┌───────────────────────────────────────────────────────────────┐
│ Title / Global Actions                                  ─ □ × │
├──────────────┬──────────────────────────────────┬─────────────┤
│              │ Dataset Context Bar              │             │
│              ├──────────────────────────────────┤             │
│ Dataset      │                                  │ Inspector   │
│ Sidebar      │                                  │             │
│              │        Main Workspace            │             │
│              │                                  │             │
│              │                                  │             │
├──────────────┴──────────────────────────────────┴─────────────┤
│ Status Bar                                                    │
└───────────────────────────────────────────────────────────────┘
```

---

# 79. Sidebar

建议宽度：

```text
220–250 px
```

结构：

```text
DATASETS                      +

[ Search datasets... ]

▌ GaAs Training
  DeepMD · 12,480

  Si Training
  extxyz · 6,320

  MoS₂ AIMD
  extxyz · 2,000


[ + Add Dataset ]
```

---

# 80. Active Dataset 视觉

当前 Dataset 使用：

```text
浅蓝背景
2–3 px Primary accent bar
Primary text
```

例如：

```text
▌ GaAs Training
  DeepMD · 12,480
```

不要：

```text
整个 Sidebar Item 变成深蓝色按钮
```

---

# 81. Dataset Context Bar

顶部 Context Bar 高度建议：

```text
64–72 px
```

内容：

```text
GaAs Training Set

DeepMD · 12,480 structures · Ga As · PBC XYZ

Energy ✓   Force ✓   Virial ✓

D:\Datasets\GaAs
```

作用：

> 始终明确当前整个 Workspace 正在处理哪个 Dataset。

---

# 82. 一级导航

主导航：

```text
Datasets
Explore
Descriptors
Results
```

建议采用水平 Tab / Navigation。

视觉：

```text
Active:
Primary text
2 px blue underline

Inactive:
Neutral text
```

不要使用四个大号 filled buttons。

---

# 83. Right Inspector Pattern

整个应用统一使用：

> **Main Object + Right Inspector**

这是核心 UI Pattern。

### Structure

```text
3D Viewer
+
Structure Inspector
```

### Descriptor

```text
Descriptor Configuration
+
Descriptor Inspector
```

### Result

```text
PCA / Heatmap
+
Selected Sample Inspector
```

---

# 84. Inspector 宽度

建议：

```text
260–320 px
```

默认：

```text
280 px
```

允许用户折叠：

```text
[ > ]
```

在较小窗口中可自动进入 Drawer。

---

# 85. Inspector 视觉

Inspector 使用：

```text
white background
left border
section title
key/value layout
```

例如：

```text
STRUCTURE

Formula          Ga32As32
Atoms             64
Energy / atom    -4.283 eV
Max |F|           0.326 eV/Å
Volume           721.3 Å³
```

---

# 86. Section-Based Layout

MDescriptor Studio 应尽量少使用 Card。

推荐：

```text
Overview
──────────────────────────────────

Structures        12,480
Atoms             798,720
Elements          Ga, As


Distributions
──────────────────────────────────

[ Energy ]   [ Force ]
```

而不是：

```text
[Card Overview]
[Card Structures]
[Card Elements]
[Card Properties]
```

---

# 87. Card 使用场景

Card 仅用于：

```text
明确独立的 visualization panel
可拖动/可折叠模块
需要明确边界的 summary
```

例如：

```text
Energy histogram
Force histogram
PCA panel
```

即使使用 Card，也保持：

```text
6 px radius
1 px border
no heavy shadow
```

---

# 88. Dataset Page 视觉布局

推荐：

```text
┌─────────────────────────────────────────────────────────┐
│ Overview                                                │
│                                                         │
│ Structures   12,480        Elements     Ga · As         │
│ Atoms        798,720       Format       DeepMD          │
│ PBC          Yes           Properties   E · F · Virial  │
├─────────────────────────────────────────────────────────┤
│ Distributions                                           │
│                                                         │
│ Energy / atom              Force magnitude              │
│ [ histogram ]              [ histogram ]                │
│                                                         │
│ Volume                     Atoms / structure            │
│ [ histogram ]              [ histogram ]                │
├─────────────────────────────────────────────────────────┤
│ Structure List                                          │
│ [ table ]                                               │
└─────────────────────────────────────────────────────────┘
```

---

# 89. Explore Page 视觉布局

Explore 应以 3D Viewer 为绝对主视觉。

推荐：

```text
┌──────────────────────────────────────────────────────────┐
│ Frame 328 / 12480       ‹   ›   Random       Fit View   │
├───────────────────────────────────┬──────────────────────┤
│                                   │ STRUCTURE            │
│                                   │ Formula   Ga32As32   │
│                                   │ Atoms     64         │
│            3D Viewer              │ E / atom  -4.283 eV │
│                                   │ Max |F|   0.326      │
│                                   │ Volume    721.3 Å³   │
│                                   │                      │
├───────────────────────────────────┴──────────────────────┤
│ Atoms                                                   │
│ # | Element | x | y | z | Fx | Fy | Fz | |F|          │
└──────────────────────────────────────────────────────────┘
```

比例：

```text
Viewer : Inspector
≈ 70 : 30
```

---

# 90. 3D Viewer 视觉规范

默认：

```text
white / very light gray background
subtle cell lines
anti-aliased atoms
moderate ball-stick radius
```

Viewer controls：

```text
Reset
Fit
Cell
Bond
Labels
Supercell
```

应放在：

```text
top-right compact toolbar
```

不要使用大按钮遮挡结构。

---

# 91. Atom Selection

选中原子：

```text
Primary blue outline / halo
```

邻居：

```text
secondary highlight
```

选中状态必须同步 Inspector：

```text
SELECTED ATOM

Index       18
Element     As
Position    ...
Force       ...
```

---

# 92. Descriptor Page

推荐：

```text
┌────────────────────────────────────┬─────────────────────┐
│ ACE                                │ DESCRIPTOR INFO     │
│ Atomic Cluster Expansion           │                     │
│                                    │ Level      Atom     │
│ Configuration                      │ Backend    C++      │
│ ─────────────────────────────      │ Device     CPU      │
│ Species    [ Ga ] [ As ]           │ Features   512      │
│ N          [ 3 ]                   │                     │
│ maxdeg     [ 8 ]                   │ Capabilities        │
│ Cutoff     [ 5.000 ] Å             │ Sparse              │
│                                    │ Threads             │
│ Execution                          │ Cancelable          │
│ ─────────────────────────────      │                     │
│ Device     CPU                     │                     │
│ Threads    [ 16 ]                  │                     │
│ Output     float64                 │                     │
│ Scope      Entire Dataset          │                     │
│                                    │                     │
│                      [ Calculate ] │                     │
└────────────────────────────────────┴─────────────────────┘
```

---

# 93. Descriptor Form 规范

科研参数优先使用精确输入。

推荐：

```text
InputNumber
Select
MultiSelect
Checkbox
Radio
File Picker
```

避免大量使用 Slider。

例如：

```text
Cutoff
┌──────────────────┬────┐
│ 5.000            │ Å  │
└──────────────────┴────┘
```

---

# 94. Unit 视觉

单位统一作为 Input 的 suffix / addon：

```text
Å
eV
eV/Å
Å³
```

不要直接写入 value。

正确：

```text
[ 5.000 ] Å
```

错误：

```text
[ 5.000 Å ]
```

---

# 95. Model-backed Descriptor UI

如果 descriptor 需要模型：

```text
MODEL
──────────────────────────

● Built-in model

○ Custom model
  [ Browse... ]
```

模型路径作为 secondary text：

```text
D:\models\dpa4.pt
```

---

# 96. Results / PCA Page

必须 Plot-first。

推荐：

```text
PCA · ACE

Color by [ Energy / atom ▼ ]
X [ PC1 ▼ ]   Y [ PC2 ▼ ]

┌────────────────────────────────────────┬──────────────────┐
│                                        │ SELECTED SAMPLE  │
│                                        │                  │
│                                        │ Frame     1837   │
│              PCA Plot                  │ Formula   Ga32As32│
│                                        │ Energy    ...    │
│                                        │ Force     ...    │
│                                        │                  │
│                                        │ Open Explore     │
└────────────────────────────────────────┴──────────────────┘
```

PCA Plot 占工作区约：

```text
75–80 %
```

---

# 97. Plot Style

所有 ECharts 图统一：

```text
background       transparent / white
grid             very light
axis             neutral gray
axis text        12 px
tooltip          compact
legend           minimal
animation        subtle
```

禁止：

```text
heavy grid
3D bar chart
gradient background
large chart title
```

---

# 98. Plot Title

图表标题优先由页面 Section 提供：

```text
Energy / atom
```

而不是 ECharts 内部再加大标题。

这样页面排版更统一。

---

# 99. Tooltip

科学 Tooltip 统一布局：

```text
Frame          382
Energy / atom  -4.28317 eV
Max |F|         0.326 eV/Å
Volume          721.3 Å³
```

要求：

```text
fixed precision
tabular numbers
unit displayed
```

---

# 100. Selected PCA Point

选中点：

```text
larger marker
primary outline
high z-index
```

其他点保持原色。

Inspector 同步 Selected Sample。

---

# 101. Heatmap

Heatmap 使用：

```text
white surface
minimal axes
scientific colorbar
```

对于 atom descriptor：

```text
Y = Atom
X = Feature
```

Tooltip：

```text
Atom       18
Feature    128
Value      0.03481
```

---

# 102. Job Manager

推荐使用右侧 Drawer，而不是独立页面。

右上角：

```text
Jobs ②
```

打开：

```text
JOBS

Running

ACE · GaAs Training
██████████████░░░░  72%
7,200 / 10,000
02:31

[ Cancel ]


Dataset Statistics
██████████████████  100%
Completed
```

---

# 103. Job Progress Bar

高度保持：

```text
4–6 px
```

不要使用巨大的 Progress。

Color：

```text
Running      Primary
Completed    Success
Failed       Danger
Cancelled    Neutral
```

---

# 104. Global Status Bar

窗口底部可加入 24–28 px status bar：

```text
● Ready                       MDescriptor 0.3.0 | CPU 16 threads
```

运行中：

```text
● 2 jobs running              ACE 72% | CPU 16 threads
```

作用类似 IDE status bar。

---

# 105. Window Title Bar

Windows-only 环境可采用轻量 Fluent 风格：

```text
MDescriptor Studio
```

右侧：

```text
Jobs
Settings
Minimize
Maximize
Close
```

不建议放复杂菜单栏。

---

# 106. Mica / Acrylic 使用规则

可以在以下区域轻量使用：

```text
Title bar
Navigation background
Sidebar background
```

主 Workspace：

```text
必须保持实色
```

禁止给：

```text
PCA
Structure Viewer
Tables
Scientific Plots
```

使用透明 Acrylic 背景。

---

# 107. Table 视觉

Table 应保持高信息密度：

```text
row height          30–34 px
font                13 px
header              semibold
border              very light
hover               subtle gray
selected            light blue
```

不要给每行添加明显卡片。

---

# 108. Table 数字对齐

所有科学数值：

```text
right aligned
tabular numbers
```

文本：

```text
left aligned
```

例如：

```text
Frame     Atoms     Energy / atom    Max |F|
0            64           -4.2831       0.326
```

---

# 109. Empty State

空 Dataset：

```text
No datasets

Add a DeepMD or extxyz dataset to begin.

[ Add Dataset ]
```

保持简单。

不要使用大型插画。

---

# 110. Loading State

Dataset scan：

```text
Scanning dataset...

Reading metadata
4,320 / 12,480 structures
```

采用：

```text
small spinner
progress
status text
```

不要冻结整个应用。

---

# 111. Error State

错误信息分两级：

### 用户信息

```text
Dataset unavailable

The source directory could not be found.

D:\Datasets\GaAs

[ Locate Dataset ]
```

### 技术详情

```text
Show Details
```

展开：

```text
error code
trace id
technical message
```

完整 traceback 写入 logs。

---

# 112. Warning State

非阻塞问题：

```text
⚠ Dataset changed since descriptors were calculated.
```

使用 inline banner。

不要弹 Modal。

---

# 113. Modal 使用原则

Modal 只用于：

```text
Delete dataset
Discard result
Overwrite / destructive operation
Critical incompatible state
```

普通操作使用：

```text
Drawer
Popover
Inline panel
```

---

# 114. Notification

短暂事件：

```text
Dataset added
Descriptor job submitted
Result exported
```

使用右下 / 右上 Toast。

失败的关键任务应进入：

```text
Jobs
```

长期保留。

---

# 115. Icon 系统

建议统一使用：

> Fluent System Icons

图标：

```text
Dataset       Database / Folder
Explore       Cube
Descriptor    Grid / Function
Results       Chart Scatter
Jobs          Clock
Settings      Settings
Add           Add
Refresh       Arrow Sync
Delete        Delete
Model         Document / Cube
```

禁止混合：

```text
Fluent
Ant Design Icons
FontAwesome
Emoji
Random SVG
```

---

# 116. Icon 尺寸

推荐：

```text
Toolbar        16 px
Sidebar        18 px
Navigation     18 px
Empty state    24–32 px
```

禁止大型 64 px 功能图标。

---

# 117. Button Hierarchy

仅三类：

### Primary

```text
Calculate
Add Dataset
Confirm
```

### Secondary

```text
Browse
Recalculate
Open Explore
```

### Tertiary / Text

```text
Cancel
More
Reset
```

每个区域最多一个 Primary Button。

---

# 118. Destructive Button

Delete 使用：

```text
neutral button
```

确认 Modal 内才使用红色危险按钮。

避免整个界面出现大量红色。

---

# 119. Dark Mode

未来 Dark Mode 推荐：

```text
App BG           #1E1E1E
Workspace        #252526
Surface          #2D2D30
Border           #3F3F46
Primary Text     #F3F3F3
Secondary        #C8C8C8
Primary Blue     Fluent blue adjusted for dark
```

Scientific Plot 需要独立 dark palette。

不要简单做：

```text
CSS invert
```

---

# 120. Accessibility

最低要求：

```text
Keyboard navigation
Visible focus ring
Color contrast
Do not encode state only by color
Tooltip for icon-only buttons
```

例如：

```text
Failed
```

必须同时有：

```text
red icon + text "Failed"
```

而不能只有红点。

---

# 121. 最小窗口尺寸

建议：

```text
1280 × 760
```

推荐工作尺寸：

```text
1440 × 900
1920 × 1080
```

低于最小宽度时：

```text
Inspector collapses
Sidebar may become compact
```

---

# 122. High DPI

Windows 必须测试：

```text
100%
125%
150%
175%
200%
```

重点：

```text
3D Viewer
Canvas
ECharts
Table
Text
Tauri title bar
```

禁止使用大量固定 pixel position。

---

# 123. 响应式策略

这是桌面应用，不追求手机响应式。

只需要三档：

```text
Compact
1280–1440

Normal
1440–1920

Wide
>1920
```

主要响应方式：

```text
Inspector collapse
plot grows
table grows
sidebar fixed
```

---

# 124. Ant Design 定制

建议通过 `ConfigProvider` 统一设置 tokens。

初始建议：

```ts
const theme = {
  token: {
    colorPrimary: "#0F6CBD",

    colorBgBase: "#FFFFFF",
    colorBgLayout: "#F5F6F8",
    colorBgContainer: "#FFFFFF",

    colorText: "#242424",
    colorTextSecondary: "#616161",

    colorBorder: "#E1E4E8",
    colorBorderSecondary: "#EAECF0",

    borderRadius: 6,

    fontFamily:
      '"Segoe UI", "Microsoft YaHei UI", sans-serif',

    fontSize: 14,

    controlHeight: 32
  }
}
```

---

# 125. 不要让应用看起来像 Ant Design Admin

虽然底层使用 Ant Design，但必须进行以下视觉约束：

```text
少 Card
少巨大 Statistic
少复杂 Breadcrumb
少后台管理式 Menu
少深色 Sidebar
```

设计目标是：

```text
Scientific Workbench
```

而不是：

```text
Admin Dashboard
```

---

# 126. Branding

产品名建议：

```text
MDescriptor Studio
```

Logo 视觉可以表达：

```text
Atomic Structure
      ↓
Descriptor Vector
```

图形元素：

```text
atom nodes
bond lines
vector bars
radial rings
```

Logo 必须：

```text
simple
flat
recognizable at 16–24 px
```

---

# 127. 建议视觉基准页面

正式开发前，至少完成以下四个 High-Fidelity Mockup：

```text
1. Dataset Page
2. Explore Page
3. Descriptor Page
4. PCA Results Page
```

这四个页面确定后：

```text
color
spacing
typography
panel hierarchy
inspector pattern
plot styling
button hierarchy
```

基本可以固定。

---

# 128. Dataset 页面视觉基准

重点验证：

```text
Sidebar density
Context Bar
Statistics layout
Table density
Inspector
```

---

# 129. Explore 页面视觉基准

重点验证：

```text
3D Viewer dominance
Viewer toolbar
Structure Inspector
Atom Table
Atom Selection
```

---

# 130. Descriptor 页面视觉基准

重点验证：

```text
Dynamic Form
Section hierarchy
Unit input
Model picker
Execution parameters
Descriptor Inspector
```

---

# 131. PCA Results 页面视觉基准

重点验证：

```text
Plot-first layout
Color-by control
Tooltip
Selected Point
Inspector linkage
Open in Explore
```

---

# 132. UI 视觉验收标准

一个页面通过视觉验收，应满足：

```text
当前 Dataset 一眼可见
主要任务一眼可见
主要科学内容占最大面积
Primary Action 唯一明确
Inspector 不抢主内容
颜色用途明确
不存在无意义装饰
数字对齐
单位规范
信息密度适合桌面
```

---

# 133. 最终视觉方向总结

MDescriptor Studio 的最终视觉风格定义为：

> **Fluent Scientific Workbench**

其核心不是模仿某个现有软件，而是结合：

```text
Windows 11 Fluent
+
Scientific IDE
+
Materials Workbench
+
Data Visualization Studio
```

形成统一设计语言。

最终应给用户的感受是：

```text
这是一个真正的 Windows 科学计算桌面工具
```

而不是：

```text
这是一个网页套进了桌面窗口
```

视觉重点始终是：

```text
Dataset
Structure
Descriptor
Scientific Results
```

并通过统一的：

```text
Sidebar
Context Bar
Workspace
Inspector
Jobs Drawer
Status Bar
```

构建整个产品的视觉和交互骨架。
