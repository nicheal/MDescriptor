# Matbench Discovery 模型描述符核查笔记

- 核查日期：2026-08-26
- 范围：AlchemBERT、BAM-MP-core、eSEN、ESNet、GRACE、ORB、PET、TACE/TECE，以及 ALIGNN-FF 是否出现在 Matbench Discovery 的 models 页面。
- 资料原则：只采用模型官方仓库、官方模型 YAML/source、论文或 arXiv。模型页面本身无法直接抓取时，以 Matbench Discovery 官方仓库中的页面源码和模型 YAML 为准。

## 先说明“模型描述符”的含义

本文把描述符理解为模型直接消费的结构/化学输入，以及这些输入采用的几何基、局部不变量或等变特征。训练数据、层数、参数量和任务头只有在会改变输入或几何感受野时才归入“版本差异”；单纯的数据集变化不被当作新描述符。

## 页面收录与 ALIGNN-FF

Matbench Discovery 官方页面源码在 [`site/src/lib/models.svelte.ts`](https://github.com/janosh/matbench-discovery/blob/main/site/src/lib/models.svelte.ts) 中通过 `import.meta.glob('$root/models/[^_]**/[^_]*.yml')` 读取模型 YAML，并将 active 与非 active 条目都放入 `MODELS`；`ACTIVE_MODELS` 则只保留 active 条目。因此，“能否在 models 页面看到”与“是否进入当前 active leaderboard”是两个不同问题。

[`models/alignn_ff/alignn-ff.yml`](https://github.com/janosh/matbench-discovery/blob/main/models/alignn_ff/alignn-ff.yml) 明确记录了 `ALIGNN FF`，但其 `lifecycle` 是 `aborted`。所以结论是：

- ALIGNN-FF 出现在 models 页面所使用的模型注册集合中，作为历史/非 active 条目可被详情页读取；
- 它不属于当前 active leaderboard；
- 官方模型仓库是 [`usnistgov/alignn`](https://github.com/usnistgov/alignn)，对应论文为 [arXiv:2209.05554](https://arxiv.org/abs/2209.05554)。

## 总览

| 家族 | Matbench 条目/状态 | 输入表征 | 几何基与对称性判断 | 主要版本差异 |
|---|---|---|---|---|
| AlchemBERT | `alchembert` / active | 结构转自然语言，再由 `bert-base-cased` tokenizer 变成 token、attention mask | 文本输入；序列模型没有显式物理等变约束。文本中使用的晶格、对称性和距离多为刚体不变量，但序列化顺序不保证严格置换不变 | Matbench 只有一个条目；官方代码另有 `nl_angle` 辅助格式，但 Matbench 生成路径使用 `nl` |
| BAM-MP-core | `bam-mp-core` / active | 原子种类 one-hot、周期邻居位移 | Bessel 径向基 + 归一化边向量 spherical harmonics；E(3) 等变消息传递/张量积 | BAM 的 RACE/贝叶斯不确定性是消息传递与输出特性；Matbench 未列出另一套 BAM 描述符 |
| eSEN | `esen-30m-mp`、`esen-30m-oam` / active | 周期原子图：元素、相对位置和邻居边 | 连续径向基、平滑 envelope/cutoff、edge-aligned local frame 的 SO(2) 卷积；内部特征旋转等变，能量标量不变，保守版本用能量梯度得力 | MP 与 OAM 主要改变训练数据和训练阶段；官方 YAML 未给出每个 checkpoint 的完整径向/通道配置 |
| ESNet | `esnet` / active | 结构图 + 元素组成知识图谱嵌入（RotatE/KGE） | 距离 RBF、邻居键角 cosine RBF，部分实现含 spherical-harmonics equivariant update；完整 ESNet 是否严格物理等变不能由公开代码确认 | `eComformer`/`iComformer`、atom feature、KGE 和角度/晶格开关存在实现分支；Matbench YAML 没有提交 checkpoint 的完整 config |
| GRACE | `grace-1l-oam`、`grace-2l-mptrj`、`grace-2l-oam`、`grace-2l-oam-l`、`grace-3l-oam-l` 等 / 多数 active | 元素种类 + 局部原子环境的 ACE/graph-ACE 表征 | ACE 径向-角向张量基、图基函数和消息传递；能量为旋转/平移/置换不变量，内部张量通道为对称适配的等变表征 | 1L/2L/3L 改变图消息传递层数和感受野；OAM/MP 主要是训练数据与规模差异 |
| ORB | `orb-v2`、`orb-v2-mptrj`、`orb-v3` / active | 元素节点嵌入 + 相对边几何 | v2：归一化边方向 + Gaussian RBF；v3：Bessel radial × spherical-harmonic angular outer product。架构不显式强制旋转等变，主要从数据增强/训练中学习 | v2 与 v2-MPtrj 结构相同、数据不同；v3 改进边嵌入、减少层数并增宽，且 checkpoint 的邻居/保守力设置不同 |
| PET | `pet-oam-xl-1.0.0` / active | 每条有向键一个 edge/message token，结合元素和三维相对几何 | 邻居 token 聚合具有置换协变性；无显式旋转等变约束，通过数据增强学习；论文没有规定固定 ACE/球谐描述符 | Matbench 条目是 OAM、XL 规模；5 个 GNN 层、3 个 attention 层、cutoff 10 Å、adaptive neighbors 40 |
| TACE/TECE | `tace-oam-l`、`tace-v1-oam-m`、RRA preview、`tece-oam-rra-1.0` 等 / active 或 superseded | 元素和局部邻居几何，展开为 spherical tensor 或 irreducible Cartesian tensor；TECE 在 edge 上做 cluster expansion | ACE/ECE 多体展开；TACE 使用不可约张量通道，TECE/RRA 使用 edge expansion 与 SO(2)/复数旋转注意力；标量能量不变，内部通道等变 | `tace-v1` 被 `tace-oam-l` 取代；RRA preview 被 `tece-oam-rra-1.0` 取代；Lmax、层数、通道和注意力机制变化明显 |
| ALIGNN-FF | `alignn_ff` / aborted、非 active | 原子/邻居图 + line graph；line graph 显式表达相邻键之间的键角关系 | 距离和键角构造的图特征对刚体平移/旋转不变；标准 ALIGNN 不是显式 E(3) 等变网络 | 这是被中止的力场条目；Matbench YAML 给出半径 8 Å、最多 12 邻居 |

## 逐家族核查

### 1. AlchemBERT

**一手资料。** Matbench 条目见 [`models/alchembert/alchembert.yml`](https://github.com/janosh/matbench-discovery/blob/main/models/alchembert/alchembert.yml)。官方实现为 [`WangYuHang-WYH/AlchemBERT`](https://github.com/WangYuHang-WYH/AlchemBERT)，关键数据处理在 [`load_md_data.py`](https://raw.githubusercontent.com/WangYuHang-WYH/AlchemBERT/main/load_md_data.py) 和 [`str_matbench_data_generation.py`](https://raw.githubusercontent.com/WangYuHang-WYH/AlchemBERT/main/str_matbench_data_generation.py)；论文预印本见 [ChemRxiv](https://chemrxiv.org/engage/chemrxiv/article-details/67540a28085116a133a62b85)。

**输入表征。** 代码将晶体结构转换为自然语言描述，再用 `BertTokenizerFast.from_pretrained('bert-base-cased')` 产生 `input_ids` 和 `attention_mask`，最大长度为 512。Matbench 的 `nl` 路径包含：

- 化学式；
- space-group symbol；
- 晶格 `a,b,c,alpha,beta,gamma`；
- 每个 site 的元素、Wyckoff position、site symmetry 和等价 site 数；
- CrystalNN 邻居的元素与距离（Å）。

官方代码还定义了带邻居夹角的 `nl_angle` 格式，但 Matbench 生成函数选择的是 `nl`，因此不能把夹角描述当作该提交的实际输入。

**几何基/对称性。** 它不是固定长度的数值材料描述符，而是自然语言 token 序列。晶格长度/角度、距离和空间群等字段本身大多是刚体变换不变量；但是 BERT 的位置编码和文本序列化顺序意味着没有被显式实现为严格的原子置换不变量，也没有 E(3)/SE(3) 等变层。官方资料没有宣称 AlchemBERT 具有物理等变性。

**版本差异与不能确定。** Matbench Discovery 当前只列出一个 AlchemBERT 条目，没有可比较的 v1/v2 描述符版本。可以确定的是 benchmark path 使用 `nl`；不能从 Matbench YAML 单独确定 tokenizer 截断后是否造成某些长结构字段丢失，以及训练权重最终采用的全部数据清洗细节。

### 2. BAM-MP-core

**一手资料。** Matbench 条目见 [`models/bam/bam-mp-core.yml`](https://github.com/janosh/matbench-discovery/blob/main/models/bam/bam-mp-core.yml)。官方仓库为 [`myung-group/BAM-torch`](https://github.com/myung-group/BAM-torch)，论文见 [arXiv:2510.03046](https://arxiv.org/abs/2510.03046)。模型实现的核心路径在 [`bam_torch/model/models.py`](https://raw.githubusercontent.com/myung-group/BAM-torch/main/bam_torch/model/models.py)。

**输入和几何基。** 官方 README/config 和源码显示：节点以原子种类 one-hot 初始化；周期邻居边提供相对位移；边长经过 Bessel radial embedding 和 polynomial cutoff；归一化边方向经过 spherical harmonics；随后进入 equivariant interaction blocks、tensor/product basis。Matbench YAML 的图半径为 6 Å。

这不是将所有结构压成一组预先计算的角度标量，而是以 e3nn 风格的不可约表示和张量积在网络中构造多体角向信息。能量输出是 `0e` scalar，力和应力可以由能量梯度得到。官方 README 明确将 BAM 描述为 Bayesian E(3)-equivariant ML potential，并将 RACE（iterative restratification of many-body message passing）作为其消息传递核心。

**版本差异与不能确定。** `BAM-MP-core` 的 `MP` 是 MPtrj 训练变体；RACE 和 Bayesian uncertainty 属于模型/消息传递与不确定性机制，不是另一种输入描述符。Matbench YAML 给出了 cutoff 和参数量，但没有完整展开提交 checkpoint 的所有 hidden irreps、径向基数量、correlation 等值；README 中的默认/示例 config 不能无条件当作该 checkpoint 的全部内部配置。

### 3. eSEN

**一手资料。** Matbench 条目为 [`esen-30m-mp.yml`](https://github.com/janosh/matbench-discovery/blob/main/models/esen/esen-30m-mp.yml) 和 [`esen-30m-oam.yml`](https://github.com/janosh/matbench-discovery/blob/main/models/esen/esen-30m-oam.yml)。官方论文是 [arXiv:2502.12147](https://arxiv.org/abs/2502.12147)；官方实现位于 [`facebookresearch/fairchem`](https://github.com/facebookresearch/fairchem)，Fair Chemistry 文档的总入口为 [`fair-chem.github.io`](https://fair-chem.github.io/intro/)。

**输入和几何基。** eSEN 消费周期原子图：元素特征、相对位置和邻居边。论文/官方资料描述的关键几何设计包括连续径向基、平滑 envelope/cutoff，以及以 edge-aligned local frame 为基础的 SO(2) convolution。因而角向信息不是普通的固定键角表，而是通过局部参考系中的连续等变通道处理。内部特征对旋转是等变的，能量是旋转/平移/置换不变的；保守版本通过能量对坐标求梯度得到力，从而保持能量-力一致性。

**版本差异。** `esen-30m-mp` 的 YAML 记录了 MPtrj 预训练后再进行 conservative fine-tuning；`esen-30m-oam` 则使用 OMat24，并有后续 MPtrj+sAlex conservative fine-tuning。两者共享 eSEN 几何表征，主要差异是训练数据、训练阶段和权重，而不是另一套 descriptor。

**不能确定。** Matbench YAML 没有列出每个 30M checkpoint 的精确 radial basis 数、SO(2) 频率范围、通道 irreps 等内部超参数；本文只确认公开论文/实现所能支持的几何机制，不替代 checkpoint config 的逐字段复原。

### 4. ESNet

**一手资料。** Matbench 条目见 [`models/esnet/esnet.yml`](https://github.com/janosh/matbench-discovery/blob/main/models/esnet/esnet.yml)。官方仓库为 [`zzz-sl/ESNet`](https://github.com/zzz-sl/ESNet)，描述图构造的源码为 [`esnet/graphs.py`](https://raw.githubusercontent.com/zzz-sl/ESNet/main/esnet/graphs.py)，模型源码见 [`esnet/models/comformer.py`](https://raw.githubusercontent.com/zzz-sl/ESNet/main/esnet/models/comformer.py) 与 [`comformer_cga.py`](https://raw.githubusercontent.com/zzz-sl/ESNet/main/esnet/models/comformer_cga.py)，配置见 [`esnet/config.py`](https://raw.githubusercontent.com/zzz-sl/ESNet/main/esnet/config.py)。

**输入表征。** ESNet 是双模态模型：

1. 结构图分支：JARVIS/CGCNN 或 atomic-number 等原子特征，邻居边上的 Cartesian displacement，距离及可选晶格/角度信息；
2. 组成分支：元素比例/属性和 knowledge-graph embedding，README 和训练代码使用 RotatE/KGE 组成表示。

官方 `iComformer` 路径中可以看到：边距离经过 RBF，邻居长度经过 radial RBF，邻居键角的 cosine 经过 `[-1,1]` 区间的 RBF；另有元素 KGE 分支和 transformer/channel-spatial attention 融合。`eComformer` 路径还含 spherical-harmonics `0e+1o+2e` 的 equivariant update。

**几何基/对称性。** 距离 RBF 与键角 cosine 是刚体不变量；Cartesian displacement 用于建边和局部图。代码确实存在等变更新组件，但主模型还包括普通 scalar attention、组成 KGE 和 readout。因此仅凭官方仓库不能把整个 ESNet 认证为严格 E(3) 等变模型；更准确的说法是“含等变几何更新的双模态图 Transformer”，其完整物理对称性保证未被公开资料明确给出。

**版本差异与不能确定。** 官方 MP 训练脚本明确示例 `iComformer`、`use_lattice=True`、`use_angle=True`、`max_neighbors=25`，而通用默认配置又提供 `eComformer`、不同 cutoff 和 atom feature 选项；Matbench YAML 只记录半径 8 Å、最多 25 邻居，并没有提交 checkpoint 的完整 config。因此可以确定 ESNet 使用结构+组成双分支，但不能 100% 确认 Matbench 权重采用 `iComformer` 还是另一条实现分支，也不能把仓库默认配置的所有字段直接归给该权重。

### 5. GRACE

**一手资料。** Matbench 示例条目 [`grace-2l-mptrj.yml`](https://github.com/janosh/matbench-discovery/blob/main/models/grace/grace-2l-mptrj.yml)；官方仓库为 [`ICAMS/grace-tensorpotential`](https://github.com/ICAMS/grace-tensorpotential)，官方文档见 [`docs/index.md`](https://raw.githubusercontent.com/ICAMS/grace-tensorpotential/master/docs/index.md)，基础论文为 [Physical Review X, 14, 021036](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.14.021036)。

**输入和几何基。** GRACE（Graph Atomic Cluster Expansion）以元素种类和局部邻居环境为输入，把 ACE 的径向-角向展开与 graph basis functions 结合。基础论文强调 graph basis functions 依赖原子位置，并通过迭代张量分解产生消息传递；官方文档将 1L/2L/3L 描述为 local ACE 与 semi-local interaction 的不同层级。

因此其几何表示不是只保留距离的 descriptor，而是 ACE 风格的径向函数、角向/球谐张量通道和受控多体相关。最终能量是刚体变换和原子置换不变量；中间张量通道是按对称性组织的等变表征。具体 `lmax`、径向函数、channel 数和 correlation 截断随 checkpoint/config 变化，不能从通用 GRACE 名称推定。

**版本差异。**

- `1L` 是局部版本；`2L` 引入更大的 semi-local 感受野；`3L` 再增加一层消息传递并进一步扩大感受野。
- `grace-2l-mptrj` 与 `grace-2l-oam` 的核心几何族相同，训练数据不同。
- `-l`、OAM 和 MP 等后缀还可能对应模型规模、数据组合或训练配方，不能自动解释为新的基函数。

**不能确定。** Matbench YAML 记录了图半径和训练集合，但没有展开每个提交的完整 ACE/graph-ACE basis 配置；本文不把论文示例超参数当作所有 Matbench checkpoint 的确切值。

### 6. ORB

**一手资料。** Matbench 条目为 [`orb-v2.yml`](https://github.com/janosh/matbench-discovery/blob/main/models/orb/orb-v2.yml)、[`orb-v2-mptrj.yml`](https://github.com/janosh/matbench-discovery/blob/main/models/orb/orb-v2-mptrj.yml) 和 [`orb-v3.yml`](https://github.com/janosh/matbench-discovery/blob/main/models/orb/orb-v3.yml)；官方模型说明在 [`orb-models/MODELS.md`](https://raw.githubusercontent.com/orbital-materials/orb-models/main/MODELS.md)，论文为 [Orb v2](https://arxiv.org/html/2410.22570) 和 [Orb v3](https://arxiv.org/html/2504.06231)。

**Orb v2。** 节点输入是元素类型 embedding，不使用绝对坐标；边输入由归一化相对边向量和距离的 Gaussian RBF 组成，并使用平滑 cosine cutoff。没有绝对坐标使其具有平移不变性；周期性由邻居图/PBC 建边体现。官方论文明确说 Orb 通过数据学习 invariances，而不是在架构上施加严格旋转等变约束。因此 v2 不是 e3nn 式显式 E(3) 等变模型。

**Orb v3。** 官方模型说明和论文把 v2 的“归一化边向量 + 20 个 Gaussian RBF”替换/升级为 Bessel radial basis 与 spherical-harmonic angular embedding 的 outer product，使用 8 个 Bessel basis、`Lmax=3`。这提供了更明确的角向结构输入，但官方仍将 Orb 描述为 unconstrained/从数据学习对称性的一类模型，不能据此把 v3 归类为严格等变网络。

**版本差异。** `orb-v2` 与 `orb-v2-mptrj` 架构相同而训练数据不同；v2 YAML 为 15 layers、半径 10 Å、20 neighbors、50 RBF，v3 YAML 为 5 layers、半径 6 Å、最多 120 neighbors、8 RBF、512 features。v3 的 conservative/direct、邻居上限和 force/stress 头是 checkpoint/任务设置差异，不应混同为几何 descriptor 本身。

### 7. PET

**一手资料。** Matbench 条目为 [`pet-oam-xl-1.0.0.yml`](https://github.com/janosh/matbench-discovery/blob/main/models/pet/pet-oam-xl-1.0.0.yml)。官方代码仓库为 [`lab-cosmo/upet`](https://github.com/lab-cosmo/upet)，PET 的主论文见 [Nature Communications](https://www.nature.com/articles/s41467-025-65662-7)。

**输入表征。** PET（Point Edge Transformer）为每条有向键维护一个 edge/message token。token 由元素种类和三维相对几何形成，模型在邻居 edge token 上做 transformer 操作，再生成发往下一节点的消息；能量由边/层的输出聚合。邻居 token 的处理对邻居排列是 permutation-covariant 的。它不是预先固定的 ACE 或有限角度表；几何信息通过可学习 edge token 进入网络。

**几何基/对称性。** 论文明确指出 PET 不施加显式旋转对称约束，而是通过数据增强学习等变性；因此不能称为严格解析 E(3) 等变。相对坐标使平移处理自然，邻居集合聚合提供置换协变/不变结构，但具体旋转泛化依赖训练。论文还强调单层可表达很高的 body order/angular resolution，这与固定低阶角度描述符不同。

**版本差异与不能确定。** Matbench 当前条目是 `pet-oam-xl-1.0.0`：YAML 给出 `d_pet=640`、`num_gnn_layers=5`、`num_attention_layers=3`、cutoff 10 Å、adaptive neighbors 40 等配置；`XL` 是规模，`OAM` 是训练数据/配方。公开资料没有在 Matbench YAML 中逐项写出该 checkpoint 的全部 edge feature 参数，也没有证据表明它改用某个固定球谐/ACE 描述符，所以 exact radial/angular basis 应标为未公开或不适用。

### 8. TACE / TECE

**一手资料。** Matbench 条目包括 [`tace-oam-l.yml`](https://github.com/janosh/matbench-discovery/blob/main/models/tace/tace-oam-l.yml)、[`tace-v1-oam-m.yml`](https://github.com/janosh/matbench-discovery/blob/main/models/tace/tace-v1-oam-m.yml)、[`tace-oam-rra-preview.yml`](https://github.com/janosh/matbench-discovery/blob/main/models/tace/tace-oam-rra-preview.yml) 和 [`tece-oam-rra-1.0.yml`](https://github.com/janosh/matbench-discovery/blob/main/models/tace/tece-oam-rra-1.0.yml)。官方仓库为 [`xvzemin/tace`](https://github.com/xvzemin/tace)，TACE 论文见 [arXiv:2509.14961](https://arxiv.org/html/2509.14961)，TECE/RRA 论文见 [arXiv:2607.10664](https://arxiv.org/html/2607.10664)。

**TACE。** TACE（Tensor Atomic/Edge Cluster Expansion）把局部环境展开为 spherical tensors 或 irreducible Cartesian tensors，并用 ACE 式受控多体相关统一标量和张量建模。`tace-oam-l` YAML 明确给出 `n_layers=5`、`num_channel=64`、`lmax=5`、`correlation=2`，并支持 spherical 与 irreducible Cartesian tensors；`tace-v1-oam-m` 是较小且已 superseded 的版本，YAML 中的通道和角动量截断不同。能量输出是旋转/平移/置换不变量，内部不可约张量通道按 O(3)/SO(3) 对称性变换；奇偶约定和具体 irreps 需以实现配置为准。

**TECE/RRA。** 论文将 Edge Cluster Expansion（ECE）定义为直接在 edge features 上进行多体展开，并以 generalized asymmetric contraction 形成 edge-level 多体特征；Radial Rotary Attention（RRA）使用 SO(2) 理论与复数/旋转注意力，允许在边上增加角频率和 body order。Matbench 中的 `tace-oam-rra-preview` 已被 `tece-oam-rra-1.0` 取代，这说明它是从 TACE 的 atomic/tensor 路径转向 edge expansion/RRA 的演进，而不是简单更换训练数据。

**不能确定。** “TECE”在 Matbench YAML 中没有完整展开缩写，本文依据官方仓库的 “Tensor Atomic/Edge Cluster Expansion” 命名和 TECE/RRA 论文作对应；不能仅凭名称断言所有 TECE checkpoint 的确切 `lmax`、`mmax`、层数或 irreps。公开的 preview YAML 记录过 8 层、`num_interaction_channel=64`、product 256、`mmax=4`、`Lmax/lmax=4`，但这些值不应自动套用到 active `tece-oam-rra-1.0`，后者应以其 YAML/checkpoint config 为准。

### 9. ALIGNN-FF

**一手资料。** Matbench 条目见 [`models/alignn_ff/alignn-ff.yml`](https://github.com/janosh/matbench-discovery/blob/main/models/alignn_ff/alignn-ff.yml)，官方仓库为 [`usnistgov/alignn`](https://github.com/usnistgov/alignn)，论文为 [arXiv:2209.05554](https://arxiv.org/abs/2209.05554)。

**输入和几何基。** ALIGNN 使用原子/邻居图，并构造 line graph：原图中的相邻键在 line graph 中相连，因而可以显式传播键-键关系和键角信息。几何上通常使用邻居距离以及由相邻边得到的角度关系；这些量在刚体平移/旋转下保持不变。它属于普通 ALIGNN 图网络，不是显式 E(3) 等变架构。

**页面状态。** 官方 YAML 的 `architecture_types` 是 `gnn`，图半径为 8 Å、最多 12 邻居，但 `lifecycle: aborted`。所以 ALIGNN-FF 确实在 models 页面注册/历史条目中出现，不能说“完全不存在”；若问题限定为 active models/active leaderboard，则答案是否定的。

## 横向结论

1. **显式对称性最强的几类。** BAM、eSEN、GRACE、TACE/TECE 都把角向信息放入 spherical/不可约张量或局部等变通道中，能量是标量不变量；它们与 ORB/PET 的“从数据学习旋转性质”路线不同。
2. **非传统/混合输入。** AlchemBERT 把结构序列化为自然语言；ESNet 把结构图与元素知识图谱嵌入融合。这两者都不应被简化成“只用原子坐标的 GNN descriptor”。
3. **ORB 与 PET。** 两者都不应标为严格解析旋转等变：ORB v2/v3 通过边几何和训练学习对称性，PET 明确依靠数据增强学习等变性。ORB v3 虽然使用 spherical harmonics edge embedding，但这不等价于整个网络具有严格 E(3) 等变保证。
4. **版本后缀的含义。** `MP`、`OAM`、`MPtrj`、`sAlex` 多数表示训练数据/配方；`1L/2L/3L`、`XL`、`v2/v3` 才更可能涉及层数、规模或边嵌入的实质变化。不能把每个后缀都当作全新描述符。
5. **ALIGNN-FF 的页面结论。** models 页面源码包含 active 与非 active YAML；ALIGNN-FF 是 aborted 历史条目，因此“页面收录”与“当前排行榜可用”必须分开报告。

## 复核边界

- Matbench Discovery 页面在本次环境中不能直接稳定抓取，因此页面收录结论来自官方站点源码和官方模型 YAML，而不是第三方列表。
- Matbench YAML 通常公开 cutoff、邻居数、训练集和模型规模，但不总是公开提交 checkpoint 的完整内部 config。因此 BAM、eSEN、ESNet、GRACE、PET、TECE 的精确 basis 数、hidden irreps 或实现分支，凡未由对应 checkpoint/source 明确给出者，均已在上文标注为不能确定。
- ESNet 最明显的歧义是 `iComformer`/`eComformer` 及角度、晶格和 atom-feature 开关；公开 MP 训练脚本支持 `iComformer` 路径，但 Matbench YAML 没有提供足以排除其他分支的完整 config。
- TACE/TECE 是仍在快速演进的官方代码/论文家族；`tace-v1` 和 RRA preview 的参数不能直接转移到 active `tece-oam-rra-1.0`。

## 附录：完整模型键清单与描述符归类（2026-08-26）

官方生成注册表 [`matbench_discovery/enums.py`](https://raw.githubusercontent.com/janosh/matbench-discovery/main/matbench_discovery/enums.py) 当前给出 64 个 non-aborted model key。模型目录还保留历史 [`alignn_ff/alignn-ff.yml`](https://github.com/janosh/matbench-discovery/blob/main/models/alignn_ff/alignn-ff.yml)，其状态为 `aborted`，故完整 metadata roster 按 65 个条目报告；它不计入 64 个当前 registry 条目。

下表按“描述符家族”合并只改变数据集、规模或训练配方的条目；反引号中的 key 是注册表中的精确名称。`Z` 表示元素/原子序数，RBF/Bessel 表示径向基，`Y_lm` 表示球谐基，irrep 表示不可约表示。

| 描述符家族 | 完整 model key | 输入/几何描述符与对称性 |
|---|---|---|
| Voronoi RF | `voronoi_rf` | Magpie 组成特征 + 松弛不变 Voronoi tessellation 特征（配位数、键角等）+ random forest；固定手工描述符。|
| Wrenformer | `wrenformer` | 组成、空间群、Wyckoff 位置等符号/晶体学字段；Transformer，不消费局部笛卡尔坐标。|
| AlchemBERT | `alchembert` | 化学式、空间群、晶格、Wyckoff/site symmetry、邻居距离的自然语言 token；无显式 E(3) 等变。|
| CGCNN | `cgcnn`, `cgcnn_p` | 元素 embedding + 邻居距离的 Gaussian/RBF 边特征；`cgcnn_p` 主要增加结构扰动训练，不是新几何基。|
| MEGNet/BOWSR | `megnet`, `bowsr` | MEGNet 的原子节点、距离边和 global state；`bowsr` 是围绕 MEGNet 的贝叶斯结构优化流程，不是独立描述符。|
| ALIGNN | `alignn`, `alignn_ff`† | 原子图 + line graph，在相邻键之间传播键角关系；主要是距离/角度标量图特征，不是显式 E(3) 等变。`alignn_ff` 为历史 aborted 条目。|
| EMA-GNN | `ema_gnn` | one-hot 元素节点 + Gaussian 径向边特征 + learned global feature；标量 GNN。|
| ESNet | `esnet` | 晶体图（`Z`、相对位移、距离及实现可选的角度/晶格信息）+ 组成级 RotatE 知识图谱 embedding；Matbench checkpoint 的具体分支/开关未完全公开。|
| M3GNet/MatterSim | `m3gnet`, `mattersim_v1_5m` | 原子、二体边和三体角度/邻域图；MatterSim 官方 model card 将发布架构说明为 M3GNet。|
| CHGNet | `chgnet_0_3_0` | 原子/键/角图，径向与角向 basis expansion；磁矩/电荷信息参与 charge-informed 多任务表征，但不是用户必须提供的额外结构坐标；非完整球谐 E(3) 架构。|
| MatRIS | `matris_10m_mp`, `matris_10m_oam`, `matris_v050_mptrj` | invariant GNN 中对三体距离/角度做 attention；不是完整不可约张量等变描述符。|
| DPA3 | `dpa3_v1_mptrj`, `dpa3_v1_openlam`, `dpa3_v2_mptrj`, `dpa3_v2_openlam`, `dpa_3_1_mptrj`, `dpa_3_1_3m_ft` | DPA3 的 node/edge/angle RepFlow、类型 embedding、径向/角向局部环境；中间表征可等变，最终局部 descriptor 为旋转不变量。|
| DPA4/SeZM | `dpa_4_0_pro_mptrj`, `dpa_4_0_1_pro_mptrj` | SO(3)-equivariant edge-conditioned message passing；位移、平滑 cutoff、径向基、edge-aligned frame 和高阶角向信息，最终输出标量 descriptor。|
| AlphaNet | `alphanet_v1_mptrj`, `alphanet_v1_oam` | 局部完整 frame 的几何标量化 + Gaussian/Bessel 径向基 + 可学习 frame transition；局部坐标构造保证等变路线。|
| NequIP/NequiX | `nequip_mp_l_0_1`, `nequip_oam_l_0_1`, `nequip_oam_xl_0_1`, `nequix_mp_1`, `nequix_mp_1_pft` | 径向基 × `Y_lm` 球谐滤波器 + Clebsch–Gordan tensor products/irreps，E(3) 等变；`PFT` 是 fine-tuning，`L/XL` 是规模或数据变体。|
| Allegro | `allegro_mp_l_0_1`, `allegro_oam_l_0_1` | NequIP 风格 radial + spherical harmonics/tensor products 的严格局部 E(3) 等变描述符；不做迭代长程 message passing。|
| SevenNet/EquFlash | `sevennet_0`, `sevennet_l3i5`, `sevennet_mf_ompa`, `sevennet_omni_i12`, `equflash_29m_oam`, `equflashv2_45m_oam` | SevenNet/EquFlash 的 NequIP-like 径向 Bessel + 球谐/irreps；FlashTP/cuEquivariance 是张量积实现优化，MF/Omni/OAM 多为数据或任务变体。|
| EqNorm | `eqnorm_mptrj` | Bessel 径向基 + `l=0…3` spherical-harmonic irreps；混合 invariant/equivariant 层和归一化。|
| HIENet | `hienet` | hybrid invariant/equivariant 图表示，Bessel radial + `l=0…3` irreps；梯度输出力。|
| MACE | `mace_mp_0`, `mace_mpa_0` | ACE-like 高阶 E(3) 等变 radial + spherical harmonics + tensor products；MP/MPA 是训练数据变体。|
| GNoME | `gnome` | NequIP-GNoME：e3nn-JAX 径向/球谐/irrep 特征（公开 YAML 含 `l=0,1,2`），E(3) 等变。|
| GRACE | `grace_1l_oam`, `grace_2l_mptrj`, `grace_2l_oam`, `grace_2l_oam_l`, `grace_3l_oam_l` | graph-ACE 的 Chebyshev radial + spherical harmonics、star/tree 多体基；能量满足平移/旋转/反演/置换不变性，内部使用对称适配张量。|
| TACE/TECE | `tace_oam_l`, `tace_oam_rra_preview`, `tace_v1_oam_m`, `tece_oam_rra_1_0` | spherical/irreducible Cartesian tensor 的 TACE/ECE 多体展开；RRA 使用 edge-level SO(2)/复数旋转注意力。`v1` 与 preview 已被后续条目 supersede。|
| eSEN | `esen_30m_mp`, `esen_30m_oam` | spherical representations、edge-aligned SO(2) convolution、连续径向基和平滑 cutoff；内部等变、能量标量不变。|
| EquiformerV3 | `equiformer_v3_mp`, `equiformer_v3_oam` | SE(3)-equivariant graph attention，径向边基 × 球谐/irreps，并在球面特征上做 equivariant attention；DeNS 是训练目标，不是新 descriptor。|
| EquiformerV2/EqV2 | `eqv2_s_dens_mp`, `eqv2_m_omat_salex_mp` | eSCN/EquiformerV2 的 equivariant irreps、径向 + 球谐边表示、attention renormalization 和 separable `S²` 操作；DeNS 仍是训练配方。|
| ORB | `orb_v2`, `orb_v2_mptrj`, `orb_v3` | v2 为归一化相对边向量 + Gaussian RBF；v3 为 Bessel radial × spherical-harmonic angular outer product。含角向基但整体不施加严格解析 E(3) 约束，旋转性质主要从数据学习。|
| PET | `pet_oam_xl_1_0_0` | 有向边 token 融合元素和相对三维几何，再用 transformer 聚合；无显式解析旋转等变约束，依靠数据增强/训练学习旋转泛化。|
| BAM | `bam_mp_core` | Bayesian E(3)-equivariant RACE；one-hot 元素、Bessel radial、球谐/不可约张量通道；Bayesian uncertainty 是输出/UQ 特性，不是 descriptor。|

† `alignn_ff` 不在 64 个 non-aborted 自动 registry key 中，但在 models 目录及页面 loader 的历史 metadata 范围内；参见上文的页面状态说明。上述 29 行合计 65 个 model key。

## 定量统计：明确描述符覆盖率

### 统计口径

- 分母为 65 个 metadata 条目，即 64 个 non-aborted registry key 加上历史 aborted 的 `alignn_ff`。
- “明确”表示官方 YAML、官方实现或原始论文明确给出该输入/几何构造；不能仅凭模型名称推测。
- 下列特征统计允许重叠：一个模型可以同时具有 radial basis、spherical harmonics 和 irreps；所以特征覆盖率不应相加。
- `esnet` 的结构图和 radial geometry 是明确的，但角度/球谐分支取决于实现配置；统计中将其从“已确认角向/球谐”排除，并单独提示配置不确定。

### 互斥的主表示路线

| 主表示路线 | 条目数 | 占 65 | 条目/家族 |
|---|---:|---:|---|
| 显式旋转/张量表示（irrep、等变 tensor、local frame 或 SO(2) angular representation） | 44 | 67.7% | DPA3/4 (8)、AlphaNet (2)、NequIP/NequiX (5)、Allegro (2)、SevenNet/EquFlash (6)、EqNorm (1)、HIENet (1)、MACE (2)、GNoME (1)、GRACE (5)、TACE/TECE (4)、eSEN (2)、EquiformerV3 (2)、EqV2 (2)、BAM (1) |
| 普通标量图表示（元素 + 距离/角度标量，未显式使用 irreps） | 13 | 20.0% | CGCNN (2)、MEGNet/BOWSR (2)、ALIGNN/ALIGNN-FF (2)、EMA-GNN (1)、M3GNet/MatterSim (2)、CHGNet (1)、MatRIS (3) |
| 学习旋转性质但不作严格解析等变约束 | 4 | 6.2% | ORB v2/v2-MPtrj/v3 (3)、PET (1) |
| 符号序列或手工晶体学/结构指纹 | 3 | 4.6% | AlchemBERT、Wrenformer、Voronoi RF |
| 结构图 + 组成知识图谱 embedding | 1 | 1.5% | ESNet；其具体 iComformer/eComformer 分支未由 Matbench YAML 完全确定 |

这五类合计 65。这里把 ESNet 单列，是因为它不能安全地归入普通标量图或已确认的严格等变图；若取得 checkpoint config 并确认 `eComformer` 分支，应再把它移入等变路线。

### 可重叠的明确特征统计

| 特征 | 明确条目数 | 占 65 | 计数解释 |
|---|---:|---:|---|
| 元素身份/化学组成 | 65 | 100.0% | 所有条目都以元素、组成或等价化学类型作为输入；Voronoi RF 用 Magpie，AlchemBERT/Wrenformer 用组成字段，其余多用 atom embedding/one-hot。 |
| 数值 radial/pair-distance basis | 61 | 93.8% | 明确写出 Gaussian/RBF、Bessel、Chebyshev 或 radial embedding 的图模型；不把 AlchemBERT 的文本距离、PET 的未公开 edge basis、Wrenformer 的晶体学字段和 Voronoi 手工特征混入。 |
| 角向/三体信息（含键角、球谐、局部 frame） | 54 | 83.1% | 从 ALIGNN 的 line graph、M3GNet/CHGNet/MatRIS 的三体几何，到 DPA、tensor/irrep、ACE 和 spherical angular channels；ESNet 的可选角度分支不计入确认值。 |
| spherical harmonics 或等价球面/不可约角向基 | 35 | 53.8% | 其中 31 个可直接对应 spherical harmonics；TACE/TECE 的 spherical/Cartesian tensor 与 RRA angular representation 另计入广义 35。 |
| 显式 symmetry-aware tensor/irrep/local-frame 表示 | 44 | 67.7% | 与上面的主路线一致；包括内部等变但最终输出 invariant scalar 的 DPA3、GRACE 等。不能把这 44 个都简写成同一种“严格 E(3) descriptor”，因为 TACE/TECE 部分路径是 SO(2)，DPA3 最终 descriptor 是 invariant。 |
| 明确三体/高阶 interaction 或 ACE/ECE 展开 | 28 | 43.1% | Voronoi RF、ALIGNN、M3GNet/MatterSim、CHGNet、MatRIS、DPA3/4、GRACE、TACE/TECE、MACE。 |
| 明确 ACE/ECE 或 ACE-equivalent 高阶展开 | 11 | 16.9% | GRACE (5)、MACE 的 ACE-equivalent 高阶消息 (2)、TACE/TECE (4)；若只按名称严格限定为 graph-ACE/ECE，则不把 MACE 算入。 |
| 明确 local frame/edge-aligned frame | 6 | 9.2% | AlphaNet (2)、DPA4 (2)、eSEN (2)。 |
| 纯距离型标量图（没有明确角向/张量通道） | 5 | 7.7% | CGCNN/CGCNN-P (2)、MEGNet/BOWSR (2)、EMA-GNN (1)。 |
| 标量键角/三体图，但不是显式张量等变 | 8 | 12.3% | ALIGNN/ALIGNN-FF (2)、M3GNet/MatterSim (2)、CHGNet (1)、MatRIS (3)。 |
| 符号/晶体学字段或手工指纹 | 3 | 4.6% | AlchemBERT、Wrenformer、Voronoi RF。 |
| 组成知识图谱 embedding | 1 | 1.5% | ESNet 的 RotatE/KGE 分支。 |
| 旋转行为主要由数据学习、非严格解析等变 | 4 | 6.2% | ORB 三个条目和 PET；ORB-v3 虽使用 angular spherical-harmonic edge embedding，也不因此自动成为严格 E(3) 网络。 |

### 读表时最重要的结论

1. “有距离”不等于“有丰富描述符”：61 个条目明确使用 radial/pair-distance basis，但只有 54 个明确引入角向信息，只有 35 个进入球面/不可约角向表示。
2. ACE/ECE 明确家族只有 11 个条目；其余等变模型更多是 tensor-product message passing 或 equivariant attention，而不是显式 ACE basis。
3. `MP`、`OAM`、`MPtrj`、`OMat`、`sAlex`、`PFT`、`DeNS`、`MF/Omni` 多数改变数据、微调或训练目标，不应重复计为新的 descriptor。
