# MDescriptor 与 DScribe、PyXtal、Featomic 的对应关系

## 结论

这四个项目不处在同一层：MDescriptor、DScribe 和 Featomic 计算原子/晶体
表示；PyXtal 负责生成、读取和变换晶体结构，是上游结构提供者。

DScribe 官方当前列出的 8 个描述符为 Coulomb matrix、Sine matrix、Ewald
sum matrix、ACSF、SOAP、MBTR、LMBTR 和 Valle-Oganov；它们都在 MDescriptor
中有同名实现。仓库的 reference suite 还对这些实现与固定的 DScribe 2.1.2
进行了数值比较（覆盖的配置见测试文件）。

Featomic 当前 calculator reference 列出的 8 个 calculator 为 Spherical
expansion、Spherical expansion by pair、LODE spherical expansion、SOAP
radial spectrum、SOAP power spectrum、Atomic Composition、Neighbor List 和
Sorted distance vector；它们对应 MDescriptor 的局部/邻居描述符族。Featomic
的输出模型是 metatensor `TensorMap`，MDescriptor 则统一返回自己的
`DescriptorResult`，所以同名不代表 API 或特征布局可直接互换。

PyXtal 生成或读取 `pyxtal` 结构，并可导出为 ASE `Atoms` 或 pymatgen
`Structure`。典型数据流是：

```text
PyXtal  --to_ase()-->  ASE Atoms  -->  MDescriptor.StructureBatch.from_ase()
                                      --> MDescriptor / DScribe / Featomic
```

PyXtal 支持 0D--3D 和部分周期结构；当前 MDescriptor 的输入契约只接受全
周期或全非周期结构，混合周期性会被拒绝。

## 描述符级对应

| MDescriptor | DScribe | Featomic | 关系 |
|---|---|---|---|
| `SOAP` | `dscribe.descriptors.SOAP` | 最接近 `SoapPowerSpectrum` | DScribe 方向有仓库数值对照；Featomic 方向是同一 SOAP 家族，但参数和输出模型不同 |
| `ACSF` | `dscribe.descriptors.ACSF` | 无当前同名 calculator | DScribe 方向有仓库数值对照 |
| `CoulombMatrix` | `CoulombMatrix` | 无当前同名 calculator | 直接同名 |
| `SineMatrix` | `SineMatrix` | 无当前同名 calculator | 直接同名 |
| `EwaldSumMatrix` | `EwaldSumMatrix` | 无当前同名 calculator | 直接同名 |
| `MBTR` | `MBTR` | 无当前同名 calculator | 直接同名 |
| `LMBTR` | `LMBTR` | 无当前同名 calculator | 直接同名 |
| `ValleOganov` | `ValleOganov` | 无当前同名 calculator | 直接同名 |
| `AtomicComposition` | 无公开同名描述符 | `AtomicComposition` | Featomic 方向直接同名 |
| `NeighborList` | 无公开描述符类 | `Neighbor List` | Featomic 方向直接同名；这是邻居图/辅助表示 |
| `SortedDistances` | 无公开同名描述符 | `Sorted distance vector` | Featomic 方向直接对应 |
| `SphericalExpansion` | 无公开同名 calculator | `Spherical expansion` | Featomic 方向直接同名 |
| `SphericalExpansionByPair` | 无公开同名 calculator | `Spherical expansion by pair` | Featomic 方向直接同名 |
| `SoapRadialSpectrum` | 无公开同名 calculator | `SOAP radial spectrum` | Featomic 方向直接同名 |
| `SoapPowerSpectrum` | SOAP 的底层/输出家族 | `SOAP power spectrum` | Featomic 方向直接同名 |
| `LodeSphericalExpansion` | 无公开同名 calculator | `LODE spherical expansion` | Featomic 方向直接同名 |
| `EAD` | 无直接对应 | 无直接对应 | 独立的旋转不变实现 |
| `SO3` | 无直接对应 | 可用球谐密度相关构造相近表示 | 不是同名、同参数或可直接替换的实现 |
| `SO4` | 无直接对应 | 可用 bispectrum/密度相关构造相近表示 | 不是 drop-in replacement |
| `SNAP` | 无直接对应 | 可用 bispectrum/密度相关构造相近表示 | 不是 drop-in replacement |
| `LBispectrum` | 无直接对应 | 可用 bispectrum/密度相关构造相近表示 | 不是 drop-in replacement |
| `MTP` | 无直接对应 | 无直接对应 | 可选模型/势函数描述符 |
| `C00PSMLFF` | 无直接对应 | 无直接对应 | 独立实现 |
| `NEP` | 无直接对应 | 无直接对应 | 模型绑定描述符 |
| `DPA4` | 无直接对应 | 无直接对应 | 模型绑定描述符 |
| `DPA4C` | 无直接对应 | 无直接对应 | 模型绑定描述符 |

“可用球谐密度相关构造相近表示”是概念层面的对应，不是数值兼容承诺。
Featomic 官方说明其 `DensityCorrelations` 可以从 spherical expansion
构造更高 body-order 的相关和 bispectrum；这与 MDescriptor 的 SO3/SO4/
SNAP/LBispectrum 共享数学家族，但具体基函数、截断、归一化、特征顺序和
输出容器仍需逐项对齐。

## 官方来源

- [DScribe 官方描述符列表](https://singroup.github.io/dscribe/)
- [Featomic calculator reference](https://docs.metatensor.org/featomic/latest/references/calculators/index.html)
- [Featomic 核心概念与 TensorMap 输出](https://docs.metatensor.org/featomic/latest/explanations/concepts.html)
- [Featomic density correlations 与 bispectrum](https://docs.metatensor.org/featomic/latest/how-to/density-correlations.html)
- [PyXtal 使用文档](https://pyxtal.readthedocs.io/en/stable/Usage.html)

## 仓库依据

- [MDescriptor 描述符清单](descriptor-inventory.md)
- [DScribe 参考对照测试](../tests/reference/test_dscribe_reference.py)
- [MDescriptor 局部描述符适配层](../src/mdescriptor/descriptors/_kernels/local.py)
- [MDescriptor 输入契约](../src/mdescriptor/core/input.py)
