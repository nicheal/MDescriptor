# Numerical baselines / 数值基线

The repository has a committed numerical golden for every descriptor.
An `external_static` entry means the values came from an independent
upstream/source oracle.  The 21 provider-backed entries keep the older
project snapshot as a secondary contract fixture and add an
`external_manifest.json` sidecar as the primary numerical oracle.
The pinned provider tests remain a separate runtime smoke check.

当前 28 个描述符全部有外部静态数值 golden：7 个沿用已提交的
source-derived NPZ，另外 21 个由锁定版本的独立 provider 生成
sidecar。21 个旧 project snapshot 仍保留，用于覆盖原有布局、
metadata 与非周期契约；它们不再被冒充为 upstream 数值来源。

| Descriptor | Committed golden | Independent oracle | Package/source | CI marker |
|---|---|---|---|---|
| `SOAP` | external static | DScribe | `dscribe` (2.1.2) | `dscribe` |
| `SOAPTurbo` | external static | soap_turbo-master | `soap_turbo` (pinned source archive) | — |
| `ACSF` | external static | DScribe | `dscribe` (2.1.2) | `dscribe` |
| `ACE` | external static | ACE1.jl | `ACE1.jl` (0.12.5) | — |
| `CoulombMatrix` | external static | DScribe | `dscribe` (2.1.2) | `dscribe` |
| `SineMatrix` | external static | DScribe | `dscribe` (2.1.2) | `dscribe` |
| `EwaldSumMatrix` | external static | DScribe | `dscribe` (2.1.2) | `dscribe` |
| `MBTR` | external static | DScribe | `dscribe` (2.1.2) | `dscribe` |
| `LMBTR` | external static | DScribe | `dscribe` (2.1.2) | `dscribe` |
| `ValleOganov` | external static | DScribe | `dscribe` (2.1.2) | `dscribe` |
| `AtomicComposition` | external static | Featomic | `featomic` (0.6.6) | `featomic` |
| `NeighborList` | external static | Featomic | `featomic` (0.6.6) | `featomic` |
| `SortedDistances` | external static | Featomic | `featomic` (0.6.6) | `featomic` |
| `SphericalExpansion` | external static | Featomic | `featomic` (0.6.6) | `featomic` |
| `SphericalExpansionByPair` | external static | Featomic | `featomic` (0.6.6) | `featomic` |
| `SoapRadialSpectrum` | external static | Featomic | `featomic` (0.6.6) | `featomic` |
| `SoapPowerSpectrum` | external static | Featomic | `featomic` (0.6.6) | `featomic` |
| `LodeSphericalExpansion` | external static | Featomic | `featomic` (0.6.6) | `featomic` |
| `EAD` | external static | PyXtal_FF | `pyxtal_ff` (0.2.3) | `pyxtalff` |
| `SO3` | external static | PyXtal_FF | `pyxtal_ff` (0.2.3) | `pyxtalff` |
| `SO4` | external static | PyXtal_FF | `pyxtal_ff` (0.2.3) | `pyxtalff` |
| `SNAP` | external static | PyXtal_FF | `pyxtal_ff` (0.2.3) | `pyxtalff` |
| `LBispectrum` | external static | LAMMPS/PyXtal_FF | `lammps + pyxtal_ff` (pinned source archive + 0.2.3) | — |
| `MTP` | external static | MLIP-4 | `MLIP-4` (pinned source archive) | — |
| `C00PSMLFF` | external static | licensed external MLFF | `local-only input` (user-supplied) | — |
| `NEP` | external static | nep-adapters | `nep-adapters` (1.0.2) | `nepadapters` |
| `DPA4` | external static | deepmd-kit | `deepmd-kit` (3.2.0) | — |
| `DPA4C` | external static | deepmd-kit | `deepmd-kit` (3.2.0) | — |

The NPZ comparisons check values and all result identity fields
(`samples`, labels, level, structure ids and row offsets).  Contract,
periodicity, symmetry and lifecycle tests remain separate because a
numeric golden cannot prove those behaviours.
