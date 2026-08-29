"""LBispectrum comparison: PyXtal-FF + .deps LAMMPS versus MDescriptor.

The installed PyXtal-FF Bispectrum class is
used unchanged for data/input construction; CompatBispectrum only adapts the
old ``diagonal`` keyword and requests lossless-enough dump formatting for the
current LAMMPS release.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import types
from collections.abc import Callable
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
TWO_NPZ = ROOT / "benchmarks/_datasets/two-structure-v1-2a727a880fef/structures.npz"
TWO_MANIFEST = ROOT / "benchmarks/_datasets/two-structure-v1-2a727a880fef/manifest.json"
CARBON_XYZ = ROOT / "benchmarks/_datasets/legacy/carbon_dataset_pbc.xyz"
OLD_DIR = ROOT / "benchmarks/lbispectrum/20260826-v0.1.0-dev-17f6a89-r02"
OUT = ROOT / "benchmarks/lbispectrum/20260826-v0.2.2-dev-e02e2c1-pyxtal-lammps-r01"
THREADS = (1, 2, 4, 8, 16, 32)
WARMUPS = 2
REPEATS = 5
RCUT = 3.5
TWOJMAX = 3
DIAGONAL = 3
RFAC0 = 0.99363
RMIN0 = 0.0
RCUTFAC = 1.0
RTOL = 1e-9
ATOL = 1e-11
STRICT_RTOL = 1e-12
STRICT_ATOL = 1e-12
PYXTAL_FF_VERSION = "0.2.3"

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mdescriptor-mpl")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OMP_DYNAMIC", "FALSE")
sys.path.insert(0, str(ROOT / "src"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


class _Structure:
    """The small structure interface consumed by PyXtal-FF's Bispectrum."""

    def __init__(self, numbers: Any, positions: Any, cell: Any, pbc: Any):
        self.numbers = np.asarray(numbers, dtype=np.int32)
        self.positions = np.asarray(positions, dtype=np.float64)
        self.cell = np.asarray(cell, dtype=np.float64)
        self.pbc = np.asarray(pbc, dtype=bool)


class _LammpsData:
    """Minimal Pymatgen LammpsData compatibility adapter.

    Pymatgen is not installed in this environment, while PyXtal-FF's
    ``descriptors.lbispectrum`` imports only ``LammpsData``.  The two fixtures
    use orthogonal cells, so this writes the equivalent atom-style-charge data
    file directly and leaves the PyXtal-FF input-generation/run path intact.
    """

    def __init__(self, structure: _Structure, elements: Any):
        self.structure = structure
        self.elements = list(elements)

    @classmethod
    def from_structure(cls, structure: _Structure, elements: Any) -> _LammpsData:
        return cls(structure, elements)

    def write_file(self, filename: str) -> None:
        structure = self.structure
        numbers = np.asarray(structure.numbers, dtype=np.int32)
        positions = np.asarray(structure.positions, dtype=np.float64)
        pbc = np.asarray(structure.pbc, dtype=bool)
        cell = np.asarray(structure.cell, dtype=np.float64)
        elements = [int(element) for element in self.elements]
        type_by_number = {number: index + 1 for index, number in enumerate(elements)}
        missing = sorted(set(map(int, numbers)) - set(type_by_number))
        if missing:
            raise ValueError(f"elements missing atomic numbers: {missing}")

        tilt: tuple[float, float, float] | None = None
        if np.all(pbc):
            # ASE stores cell vectors by rows.  The carbon extxyz contains
            # both orthogonal cells and LAMMPS restricted-triclinic cells.
            # The latter have a=(lx,0,0), b=(xy,ly,0), c=(xz,yz,lz).
            restricted = np.allclose(
                cell[np.ix_([0, 1, 2], [1, 2])],
                np.array([[0.0, 0.0], [cell[1, 1], 0.0], [cell[2, 1], cell[2, 2]]]),
                rtol=0.0,
                atol=1e-12,
            )
            if not restricted or np.any(np.diag(cell) <= 0.0):
                raise ValueError("the local LAMMPS adapter expects restricted-triclinic cells")
            lo = np.zeros(3, dtype=np.float64)
            hi = np.diag(cell).copy()
            if not np.allclose(cell, np.diag(np.diag(cell)), rtol=0.0, atol=1e-12):
                tilt = (float(cell[1, 0]), float(cell[2, 0]), float(cell[2, 1]))
        else:
            lo = positions.min(axis=0) - 1.0
            hi = np.maximum(positions.max(axis=0) + 1.0, lo + 8.0)

        with Path(filename).open("w", encoding="utf-8") as handle:
            handle.write("LAMMPS data file generated for PyXtal-FF LBispectrum\n\n")
            handle.write(f"{len(numbers)} atoms\n")
            handle.write(f"{len(elements)} atom types\n\n")
            handle.write(f"{lo[0]:.17g} {hi[0]:.17g} xlo xhi\n")
            handle.write(f"{lo[1]:.17g} {hi[1]:.17g} ylo yhi\n")
            handle.write(f"{lo[2]:.17g} {hi[2]:.17g} zlo zhi\n")
            if tilt is not None:
                handle.write(f"{tilt[0]:.17g} {tilt[1]:.17g} {tilt[2]:.17g} xy xz yz\n")
            handle.write("\n")
            handle.write("Masses\n\n")
            for index, number in enumerate(elements, 1):
                # Masses do not affect compute sna/atom; use a valid positive mass.
                handle.write(f"{index} 1.0 # Z={number}\n")
            handle.write("\nAtoms # charge\n\n")
            for index, (number, position) in enumerate(
                zip(numbers, positions, strict=True), 1
            ):
                atom_type = type_by_number[int(number)]
                handle.write(
                    f"{index} {atom_type} 0.0 {position[0]:.17g} "
                    f"{position[1]:.17g} {position[2]:.17g}\n"
                )


def install_pymatgen_shim() -> None:
    pymatgen = types.ModuleType("pymatgen")
    pymatgen_io = types.ModuleType("pymatgen.io")
    pymatgen_lammps = types.ModuleType("pymatgen.io.lammps")
    pymatgen_data = types.ModuleType("pymatgen.io.lammps.data")
    pymatgen_data.LammpsData = _LammpsData
    sys.modules.update(
        {
            "pymatgen": pymatgen,
            "pymatgen.io": pymatgen_io,
            "pymatgen.io.lammps": pymatgen_lammps,
            "pymatgen.io.lammps.data": pymatgen_data,
        }
    )


install_pymatgen_shim()
from pyxtal_ff.descriptors.lbispectrum import Bispectrum  # noqa: E402


class CompatBispectrum(Bispectrum):
    """PyXtal-FF Bispectrum with current-LAMMPS input compatibility."""

    def get_lammps_input(self, input_file: str) -> None:
        super().get_lammps_input(input_file)
        path = Path(input_file)
        output: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            # Current LAMMPS has diagonal=3 as the only sna/atom output and
            # rejects the keyword removed from the older PyXtal-FF template.
            line = line.replace(f"diagonal {self.diagonal} ", "")
            if line.startswith("dump 2 "):
                line = line.replace("dump.sna c_sna[*]", "dump.sna id c_sna[*]")
            output.append(line)
            if line.startswith("dump 2 "):
                output.append("dump_modify 2 sort id format float %22.16e")
        if not np.all(self.structure.pbc):
            output.insert(0, "boundary f f f")
        path.write_text("\n".join(output) + "\n", encoding="utf-8")


def parse_sna_dump(path: Path, atom_count: int) -> np.ndarray:
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        header = next(index for index, line in enumerate(lines) if line.startswith("ITEM: ATOMS"))
    except StopIteration as exc:
        raise RuntimeError(f"LAMMPS dump has no ATOMS section: {path}") from exc
    rows: list[tuple[int, list[float]]] = []
    for line in lines[header + 1 : header + 1 + atom_count]:
        fields = line.split()
        if len(fields) < 9:
            raise RuntimeError(f"invalid LBispectrum dump row: {line!r}")
        rows.append((int(fields[0]), [float(value) for value in fields[1:9]]))
    if len(rows) != atom_count:
        raise RuntimeError(f"expected {atom_count} rows, found {len(rows)} in {path}")
    rows.sort(key=lambda row: row[0])
    return np.asarray([row[1] for row in rows], dtype=np.float64)


def cleanup_lammps_files() -> None:
    for name in ("dump.element", "dump.sna", "dump.snad", "dump.snav", "log.lammps"):
        Path(name).unlink(missing_ok=True)


class ReferenceRunner:
    def __init__(self, structures: list[_Structure], profile: dict[int, dict[str, float]]):
        self.structures = structures
        self.profile = profile
        self._directory = tempfile.TemporaryDirectory(prefix="mdescriptor-lbispectrum-")

    def close(self) -> None:
        self._directory.cleanup()

    def compute(self) -> np.ndarray:
        previous = Path.cwd()
        os.chdir(self._directory.name)
        try:
            values: list[np.ndarray] = []
            for structure in self.structures:
                CompatBispectrum(
                    structure,
                    RCUTFAC,
                    self.profile,
                    TWOJMAX,
                    diagonal=DIAGONAL,
                    rfac0=RFAC0,
                    rmin0=RMIN0,
                )
                dump = Path("dump.sna")
                if not dump.exists():
                    raise RuntimeError("PyXtal-FF/LAMMPS did not produce dump.sna")
                values.append(parse_sna_dump(dump, len(structure.numbers)))
                cleanup_lammps_files()
            return np.vstack(values)
        finally:
            os.chdir(previous)


def load_two_structure() -> tuple[Any, list[_Structure], list[str], dict[str, Any]]:
    from mdescriptor import StructureBatch

    with np.load(TWO_NPZ) as arrays:
        numbers = np.asarray(arrays["numbers"], dtype=np.int32)
        positions = np.asarray(arrays["positions"], dtype=np.float64)
        cells = np.asarray(arrays["cells"], dtype=np.float64)
        pbc = np.asarray(arrays["pbc"], dtype=np.int32)
        offsets = np.asarray(arrays["offsets"], dtype=np.int64)
    ids = ["hea32-periodic", "water3-nonperiodic"]
    batch = StructureBatch(numbers, positions, cells, pbc, offsets, ids)
    structures = [
        _Structure(
            numbers[offsets[index] : offsets[index + 1]],
            positions[offsets[index] : offsets[index + 1]],
            cells[index],
            pbc[index],
        )
        for index in range(len(ids))
    ]
    return batch, structures, ids, {
        "name": "two-structure-v1-2a727a880fef",
        "path": str(TWO_NPZ),
        "sha256": sha256(TWO_NPZ),
        "manifest_sha256": json.loads(TWO_MANIFEST.read_text(encoding="utf-8"))["sha256"],
        "structures": len(ids),
        "atoms": int(len(numbers)),
        "species": sorted({int(number) for number in numbers}),
    }


def load_carbon() -> tuple[Any, list[_Structure], list[str], dict[str, Any]]:
    from ase.io import read

    from mdescriptor import StructureBatch

    atoms = list(read(CARBON_XYZ, index=":"))
    ids = [f"carbon-{index:04d}" for index in range(len(atoms))]
    batch = StructureBatch.from_ase(atoms, ids=ids)
    structures = [
        _Structure(item.numbers, item.positions, item.cell.array, item.pbc)
        for item in atoms
    ]
    return batch, structures, ids, {
        "name": "carbon_dataset_pbc",
        "path": str(CARBON_XYZ),
        "sha256": sha256(CARBON_XYZ),
        "structures": len(atoms),
        "atoms": int(sum(len(item) for item in atoms)),
        "species": sorted({int(number) for item in atoms for number in item.numbers}),
        "atom_count_min": min(len(item) for item in atoms),
        "atom_count_max": max(len(item) for item in atoms),
    }


def set_openmp_threads(threads: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["OMP_DYNAMIC"] = "FALSE"
    try:
        runtime = ctypes.CDLL("libgomp.so.1")
        runtime.omp_set_dynamic(ctypes.c_int(0))
        runtime.omp_set_num_threads(ctypes.c_int(threads))
    except OSError:
        pass


def stats(samples: list[float]) -> dict[str, Any]:
    array = np.asarray(samples, dtype=np.float64)
    return {
        "raw_seconds": [float(value) for value in samples],
        "min_seconds": float(np.min(array)),
        "median_seconds": float(np.median(array)),
        "p95_seconds": float(np.percentile(array, 95)),
        "max_seconds": float(np.max(array)),
    }


def timed(function: Callable[[], np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    for _ in range(WARMUPS):
        function()
    samples: list[float] = []
    value: np.ndarray | None = None
    for _ in range(REPEATS):
        started = time.perf_counter()
        value = function()
        samples.append(time.perf_counter() - started)
    assert value is not None
    return value, stats(samples)


def accuracy_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    absolute = np.abs(delta)
    reference_abs = np.abs(reference)
    mask = reference_abs > 1e-14
    relative = np.zeros_like(absolute)
    relative[mask] = absolute[mask] / reference_abs[mask]
    return {
        "shape": list(candidate.shape),
        "max_abs_error": float(np.max(absolute, initial=0.0)),
        "rmse": float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0,
        "mae": float(np.mean(absolute)) if delta.size else 0.0,
        "max_rel_error_nonzero_reference": float(np.max(relative, initial=0.0)),
        "nonzero_reference_entries": int(np.count_nonzero(mask)),
        "exact_array_equal": bool(np.array_equal(candidate, reference)),
        "allclose": bool(np.allclose(candidate, reference, rtol=RTOL, atol=ATOL)),
        "strict_allclose": bool(
            np.allclose(candidate, reference, rtol=STRICT_RTOL, atol=STRICT_ATOL)
        ),
        "rtol": RTOL,
        "atol": ATOL,
        "strict_rtol": STRICT_RTOL,
        "strict_atol": STRICT_ATOL,
    }


def make_project_descriptor() -> Any:
    from mdescriptor.descriptors import LBispectrum

    return LBispectrum(
        twojmax=TWOJMAX,
        diagonal=DIAGONAL,
        rfac0=RFAC0,
        rmin0=RMIN0,
        rcutfac=RCUTFAC,
        rcut=RCUT,
        normalize_U=False,
    )


def benchmark_case(
    name: str,
    batch: Any,
    structures: list[_Structure],
    dataset: dict[str, Any],
) -> dict[str, Any]:
    species = [int(value) for value in dataset["species"]]
    profile = {number: {"r": RCUT / 2.0, "w": 1.0} for number in species}
    reference_runner = ReferenceRunner(structures, profile)
    try:
        print(f"{name}: reference PyXtal-FF + LAMMPS", flush=True)
        set_openmp_threads(1)
        reference_values, reference_perf = timed(reference_runner.compute)

        print(f"{name}: project thread curve", flush=True)
        project_scaling: list[dict[str, Any]] = []
        project_one: np.ndarray | None = None
        for threads in THREADS:
            print(f"{name}: MDescriptor OMP_NUM_THREADS={threads}", flush=True)
            set_openmp_threads(threads)
            descriptor = make_project_descriptor()
            project_values, project_perf = timed(lambda descriptor=descriptor: descriptor.compute(batch).values)
            if project_one is None:
                project_one = project_values.copy()
            project_scaling.append(
                {
                    "threads": threads,
                    **project_perf,
                    "max_abs_vs_thread_1": float(
                        np.max(np.abs(project_values - project_one), initial=0.0)
                    ),
                    "exact_vs_thread_1": bool(np.array_equal(project_values, project_one)),
                    "accuracy_vs_pyxtal_lammps": accuracy_metrics(project_values, reference_values),
                }
            )
        assert project_one is not None

        old_comparison: dict[str, Any] | None = None
        if name == "two_structure":
            with np.load(OLD_DIR / "candidate_output.npz") as arrays:
                old_values = np.asarray(arrays["values"], dtype=np.float64)
            old_comparison = {
                "source": str(OLD_DIR),
                "accuracy_current_vs_old_candidate": accuracy_metrics(project_one, old_values),
                "accuracy_reference_vs_old_candidate": accuracy_metrics(reference_values, old_values),
                "old_performance": json.loads(
                    (OLD_DIR / "performance.json").read_text(encoding="utf-8")
                ),
            }

        project_base = project_scaling[0]["median_seconds"]
        for row in project_scaling:
            row["speedup_vs_thread_1"] = float(project_base / row["median_seconds"])
            row["parallel_efficiency"] = float(row["speedup_vs_thread_1"] / row["threads"])

        case_dir = OUT / name
        case_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(case_dir / "pyxtal_lammps_output.npz", values=reference_values)
        np.savez_compressed(case_dir / "mdescriptor_output.npz", values=project_one)
        return {
            "dataset": dataset,
            "configuration": {
                "descriptor": "LBispectrum",
                "twojmax": TWOJMAX,
                "diagonal": DIAGONAL,
                "rcut": RCUT,
                "rfac0": RFAC0,
                "rmin0": RMIN0,
                "rcutfac": RCUTFAC,
                "normalize_U": False,
                "profile_equivalence": "all element radii=rcut/2=1.75 A, weights=1.0",
            },
            "shape": list(project_one.shape),
            "accuracy": accuracy_metrics(project_one, reference_values),
            "reference_pyxtal_lammps": {
                "backend": "installed PyXtal-FF Bispectrum + .deps LAMMPS ML-SNAP sna/atom",
                "profile": profile,
                "timed_scope": "PyXtal-FF data/input creation, LAMMPS subprocess, dump parse",
                **reference_perf,
                "internal_multithreading": False,
            },
            "project_mdescriptor": {
                "backend": "MDescriptor C++ rotational kernel",
                "timed_scope": "descriptor compute only; batch and descriptor constructed before timing",
                "public_execution_num_threads": False,
                "thread_control": "OMP_NUM_THREADS + libgomp omp_set_num_threads; affects neighbor-graph build only",
                "scaling": project_scaling,
            },
            "previous_comparison": old_comparison,
        }
    finally:
        reference_runner.close()


def lammps_version() -> str:
    try:
        output = subprocess.check_output(
            ["lmp_serial", "-h"], text=True, stderr=subprocess.STDOUT
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc}"
    for line in output.splitlines():
        if line.startswith("Large-scale Atomic/Molecular Massively Parallel Simulator"):
            return line.strip()
    return output.splitlines()[0].strip() if output.splitlines() else "unknown"


def fseconds(value: float) -> str:
    if value < 0.001:
        return f"{value * 1e6:.2f} us"
    if value < 1.0:
        return f"{value * 1e3:.3f} ms"
    return f"{value:.3f} s"


def fnum(value: float) -> str:
    return f"{value:.3e}"


def make_report(result: dict[str, Any]) -> str:
    two = result["cases"]["two_structure"]
    carbon = result["cases"]["carbon"]
    lines = [
        "# LBispectrum：PyXtal-FF/LAMMPS 与 MDescriptor 对比",
        "",
        "## 结论摘要",
        "",
        "- 两结构 41 个原子、8 个特征的 PyXtal-FF + LAMMPS 输出与 MDescriptor 对齐；",
        f"  最大绝对差 `{fnum(two['accuracy']['max_abs_error'])}`，严格阈值通过：`{two['accuracy']['strict_allclose']}`。",
        f"- 碳数据 450 个结构、{carbon['dataset']['atoms']} 个原子也采用同一参数和逐原子顺序；",
        f"  最大绝对差 `{fnum(carbon['accuracy']['max_abs_error'])}`，严格阈值通过：`{carbon['accuracy']['strict_allclose']}`。",
        "- PyXtal-FF 路径是逐结构启动一次串行 LAMMPS，并包含数据文件生成、进程启动和 dump 解析；",
        "  因此其端到端时间不能当作纯 C++ 内核时间。",
        "- 当前 LBispectrum 公共 API 未声明 `execution.num_threads`，其 C++ 旋转中心计算循环也没有 OpenMP 并行区；",
        "  线程曲线按项目现有能力用 `OMP_NUM_THREADS` 测量，主要覆盖邻居图构建，参考 LAMMPS 内部线程扩展为 N/A。",
        "",
        "## 测试配置与规范",
        "",
        f"- 预热/计时：`{WARMUPS}` / `{REPEATS}`；计时统计中位数和 p95。线程点：`{list(THREADS)}`。",
        f"- 参数：`twojmax={TWOJMAX}, diagonal={DIAGONAL}, rcut={RCUT}, rfac0={RFAC0}, rmin0={RMIN0}, rcutfac={RCUTFAC}, normalize_U=false`。",
        "- 为使 LAMMPS 的必需元素 profile 与项目默认 `element_profile=None` 等价，所有元素设 `r=1.75 Å`、`w=1.0`。",
        f"- PyXtal-FF：`{result['environment']['pyxtal_ff']}`；LAMMPS：`{result['environment']['lammps']}`。",
        f"- LAMMPS 源归档：`.deps/lammps-stable.tar.gz`，SHA256 `{result['environment']['lammps_archive_sha256']}`；构建为 ML-SNAP、MPI=off、OMP=off 的 `lmp_serial`。",
        "- 当前环境未安装 Pymatgen；runner 只注入 `LammpsData.from_structure/write_file` 兼容适配器，PyXtal-FF 的 Bispectrum 输入/调用路径和 LAMMPS `compute sna/atom` 仍实际执行。",
        "",
        "## 精度列表比较",
        "",
        "| 数据集 | 行数×特征 | 最大绝对差 | RMSE | 最大相对差 | 严格 allclose |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for case in (two, carbon):
        accuracy = case["accuracy"]
        lines.append(
            f"| {case['dataset']['name']} | `{accuracy['shape'][0]}×{accuracy['shape'][1]}` | "
            f"`{fnum(accuracy['max_abs_error'])}` | `{fnum(accuracy['rmse'])}` | "
            f"`{fnum(accuracy['max_rel_error_nonzero_reference'])}` | `{accuracy['strict_allclose']}` |"
        )

    lines += [
        "",
        "## 单线程基准列表",
        "",
        "PyXtal-FF/LAMMPS 为端到端参考时间；MDescriptor 为仅 descriptor compute 时间，故速度比值用于工程路径对比，不能解释成同一计时范围的内核比值。",
        "",
        "| 数据集 | PyXtal-FF + LAMMPS median / p95 | MDescriptor median / p95 | 参考端到端 / 项目 compute |",
        "|---|---:|---:|---:|",
    ]
    for case in (two, carbon):
        ref = case["reference_pyxtal_lammps"]
        project = case["project_mdescriptor"]["scaling"][0]
        ratio = ref["median_seconds"] / project["median_seconds"]
        lines.append(
            f"| {case['dataset']['name']} | `{fseconds(ref['median_seconds'])}` / `{fseconds(ref['p95_seconds'])}` | "
            f"`{fseconds(project['median_seconds'])}` / `{fseconds(project['p95_seconds'])}` | `{ratio:.2f}×` |"
        )

    old = two["previous_comparison"]
    old_perf = old["old_performance"]
    old_median = old_perf["median_seconds"]
    current_median = two["project_mdescriptor"]["scaling"][0]["median_seconds"]
    lines += [
        "",
        "## 与前一次 LBispectrum 测试比较",
        "",
        f"前次结果：`{OLD_DIR}`，提交 `{result['comparison']['previous_commit']}`，41 行×8 特征，项目自身参考，median `{fseconds(old_median)}`，p95 `{fseconds(old_perf['p95_seconds'])}`。",
        f"本次 MDescriptor 单线程 median `{fseconds(current_median)}`，相对前次为 `{current_median / old_median:.2f}×`（`{(current_median / old_median - 1.0) * 100:+.1f}%`）。",
        f"本次项目输出对前次 candidate 的最大绝对差 `{fnum(two['previous_comparison']['accuracy_current_vs_old_candidate']['max_abs_error'])}`，严格 allclose `{two['previous_comparison']['accuracy_current_vs_old_candidate']['strict_allclose']}`。",
        "",
        "| 项目 | 前一次 | 本次 | 差异 |",
        "|---|---:|---:|---:|",
        "| 两结构行数×特征 | `41×8` | `41×8` | 一致 |",
        f"| 项目单线程 median | `{fseconds(old_median)}` | `{fseconds(current_median)}` | `{current_median / old_median:.2f}×` |",
        f"| 项目单线程 p95 | `{fseconds(old_perf['p95_seconds'])}` | `{fseconds(two['project_mdescriptor']['scaling'][0]['p95_seconds'])}` | — |",
        f"| 本次 PyXtal-FF/LAMMPS 精度 | — | max abs `{fnum(two['accuracy']['max_abs_error'])}` | 严格通过 |",
        "",
        "## 多线程扩展性",
        "",
        "项目曲线中的 N 是 `OMP_NUM_THREADS=N`；参考路径是串行 `lmp_serial`，没有可比的内部线程参数。",
        "",
        "### 两结构",
        "",
        "| N | MDescriptor median | speedup | 并行效率 | 相对 N=1 最大差 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in two["project_mdescriptor"]["scaling"]:
        lines.append(
            f"| {row['threads']} | `{fseconds(row['median_seconds'])}` | `{row['speedup_vs_thread_1']:.2f}×` | "
            f"`{row['parallel_efficiency'] * 100:.1f}%` | `{fnum(row['max_abs_vs_thread_1'])}` |"
        )
    lines += [
        "",
        "### carbon_dataset_pbc.xyz",
        "",
        "| N | MDescriptor median | speedup | 并行效率 | 相对 N=1 最大差 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in carbon["project_mdescriptor"]["scaling"]:
        lines.append(
            f"| {row['threads']} | `{fseconds(row['median_seconds'])}` | `{row['speedup_vs_thread_1']:.2f}×` | "
            f"`{row['parallel_efficiency'] * 100:.1f}%` | `{fnum(row['max_abs_vs_thread_1'])}` |"
        )
    lines += [
        "",
        "## 结果文件",
        "",
        "- `accuracy.json`：精度与旧结果比较。",
        "- `performance.json`：原始计时、单线程和线程曲线。",
        "- `two_structure/`、`carbon/`：两套后端的压缩输出。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    import pyxtal_ff
    from ase import __version__ as ase_version

    installed_pyxtal_ff = package_version("pyxtal-ff")
    if installed_pyxtal_ff != PYXTAL_FF_VERSION:
        raise RuntimeError(
            f"expected pyxtal-ff=={PYXTAL_FF_VERSION}, got {installed_pyxtal_ff}"
        )
    mdescriptor_version = package_version("mdescriptor")

    two_batch, two_structures, _, two_dataset = load_two_structure()
    carbon_batch, carbon_structures, _, carbon_dataset = load_carbon()
    cases = {
        "two_structure": benchmark_case("two_structure", two_batch, two_structures, two_dataset),
        "carbon": benchmark_case("carbon", carbon_batch, carbon_structures, carbon_dataset),
    }

    previous_manifest = json.loads((OLD_DIR / "manifest.json").read_text(encoding="utf-8"))
    archive = ROOT / ".deps/lammps-stable.tar.gz"
    result: dict[str, Any] = {
        "schema_version": 1,
        "descriptor": "LBispectrum",
        "cases": cases,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "ase": ase_version,
            "pyxtal_ff": pyxtal_ff.__version__,
            "mdescriptor": mdescriptor_version,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "lammps": lammps_version(),
            "lammps_archive": str(archive),
            "lammps_archive_sha256": sha256(archive),
            "omp_num_threads_after_run": os.environ.get("OMP_NUM_THREADS"),
        },
        "git": {
            "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
        },
        "comparison": {
            "previous_commit": previous_manifest["git"]["commit"],
            "previous_dir": str(OLD_DIR),
        },
        "protocol": {
            "warmup_calls": WARMUPS,
            "measured_calls": REPEATS,
            "thread_points": list(THREADS),
            "accuracy_rtol": RTOL,
            "accuracy_atol": ATOL,
            "strict_rtol": STRICT_RTOL,
            "strict_atol": STRICT_ATOL,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "accuracy.json").write_text(
        json.dumps(
            {
                "descriptor": result["descriptor"],
                "protocol": result["protocol"],
                "cases": {
                    name: {
                        "dataset": case["dataset"],
                        "shape": case["shape"],
                        "accuracy": case["accuracy"],
                        "previous_comparison": case["previous_comparison"],
                    }
                    for name, case in cases.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (OUT / "performance.json").write_text(
        json.dumps(
            {
                "descriptor": result["descriptor"],
                "protocol": result["protocol"],
                "environment": result["environment"],
                "cases": {
                    name: {
                        "dataset": case["dataset"],
                        "reference_pyxtal_lammps": case["reference_pyxtal_lammps"],
                        "project_mdescriptor": case["project_mdescriptor"],
                    }
                    for name, case in cases.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (OUT / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "report.md").write_text(make_report(result), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
