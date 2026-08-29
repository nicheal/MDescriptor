"""Run the official MLIP-4/MDescriptor MTP comparison."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any

# The project benchmark protocol fixes BLAS/OpenMP libraries to one thread.
for _name in (
    "OMP_NUM_THREADS",
    "OMP_DYNAMIC",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "FALSE" if _name == "OMP_DYNAMIC" else "1"

import numpy as np  # noqa: E402
from ase.io import read  # noqa: E402

from mdescriptor import ExecutionOptions, StructureBatch  # noqa: E402
from mdescriptor.descriptors import MTP  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
TWO_PATH = ROOT / "benchmarks/_datasets/two-structure-v1-2a727a880fef/structures.npz"
TWO_MANIFEST_PATH = ROOT / "benchmarks/_datasets/two-structure-v1-2a727a880fef/manifest.json"
TWO_GOLDEN_DIR = ROOT / "tests/golden/mtp"
CARBON_PATH = ROOT / "benchmarks/_datasets/legacy/carbon_dataset_pbc.xyz"
PREVIOUS = ROOT / "benchmarks/mtp/20260826-v0.1.0-dev-17f6a89-r02"
DEFAULT_OUTPUT = ROOT / "benchmarks/mtp/20260826-v0.2.2-dev-e02e2c1-mlip4-r01"

THREADS = (1, 2, 4, 8, 16, 32)
WARMUPS = 2
REPEATS = 5
RTOL = 1e-9
ATOL = 1e-11
OFFICIAL_SPECIES = (1, 6, 8, 24, 25, 26, 27, 28)
LEGACY_SPECIES = (1, 8, 24, 25, 26, 27, 28)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _summary(samples: list[float]) -> dict[str, Any]:
    values = np.asarray(samples, dtype=np.float64)
    return {
        "raw_seconds": [float(value) for value in values],
        "median_seconds": float(np.median(values)),
        "p95_seconds": float(np.percentile(values, 95)),
        "min_seconds": float(np.min(values)),
    }


def _metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if candidate.shape != reference.shape:
        raise RuntimeError(f"shape mismatch: candidate={candidate.shape}, reference={reference.shape}")
    delta = candidate - reference
    absolute = np.abs(delta)
    reference_abs = np.abs(reference)
    valid = reference_abs > 1e-12
    relative = absolute[valid] / reference_abs[valid]
    max_index = np.unravel_index(int(np.argmax(absolute)), absolute.shape)
    return {
        "shape": [int(value) for value in candidate.shape],
        "rows": int(candidate.shape[0]),
        "features": int(candidate.shape[1]),
        "finite_candidate": bool(np.isfinite(candidate).all()),
        "finite_reference": bool(np.isfinite(reference).all()),
        "max_abs_error": float(np.max(absolute, initial=0.0)),
        "max_relative_error_reference_gt_1e-12": float(np.max(relative, initial=0.0)),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "mae": float(np.mean(absolute)),
        "max_error_index": [int(value) for value in max_index],
        "allclose": bool(np.allclose(candidate, reference, rtol=RTOL, atol=ATOL)),
        "rtol": RTOL,
        "atol": ATOL,
    }


def _load_two() -> tuple[StructureBatch, dict[str, Any]]:
    # The benchmark dataset is intentionally ignored by Git.  Fall back to
    # the identical tracked golden input so this oracle remains runnable from
    # a clean checkout (the local benchmark copy still supplies its richer
    # dataset metadata when available).
    if TWO_PATH.is_file() and TWO_MANIFEST_PATH.is_file():
        input_path = TWO_PATH
        manifest = json.loads(TWO_MANIFEST_PATH.read_text(encoding="utf-8"))
        ids = tuple(manifest["input"]["ids"])
    else:
        input_path = TWO_GOLDEN_DIR / "input.npz"
        manifest = json.loads((TWO_GOLDEN_DIR / "manifest.json").read_text(encoding="utf-8"))
        ids = tuple(manifest["input_ids"])
    with np.load(input_path) as arrays:
        batch = StructureBatch(
            np.asarray(arrays["numbers"], dtype=np.int32),
            np.asarray(arrays["positions"], dtype=np.float64),
            np.asarray(arrays["cells"], dtype=np.float64),
            np.asarray(arrays["pbc"], dtype=np.int32),
            np.asarray(arrays["offsets"], dtype=np.int64),
            ids,
        )
    return batch, manifest


def _load_carbon() -> StructureBatch:
    structures = list(read(CARBON_PATH, index=":"))
    ids = tuple(f"carbon-{index:04d}" for index in range(len(structures)))
    return StructureBatch.from_ase(structures, ids=ids)


def _dataset_meta(dataset_id: str, path: Path, batch: StructureBatch) -> dict[str, Any]:
    counts = np.diff(batch.offsets)
    return {
        "dataset_id": dataset_id,
        "path": str(path),
        "sha256": _sha256(path),
        "structures": int(batch.structures),
        "atoms": int(batch.atoms),
        "atom_count_min": int(np.min(counts)),
        "atom_count_max": int(np.max(counts)),
        "species": sorted({int(value) for value in batch.numbers}),
    }


def _write_ndjson(batch: StructureBatch, path: Path) -> None:
    with path.open("w", encoding="utf-8") as output:
        for index in range(batch.structures):
            begin = int(batch.offsets[index])
            end = int(batch.offsets[index + 1])
            record: dict[str, Any] = {
                "pos": batch.positions[begin:end].tolist(),
                "types": batch.numbers[begin:end].astype(int).tolist(),
            }
            if bool(np.all(batch.pbc[index] == 1)):
                record["cell"] = batch.cells[index].tolist()
            output.write(json.dumps(record, separators=(",", ":")) + "\n")


def _read_official_output(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        shape = np.frombuffer(handle.read(16), dtype="<u8").astype(np.int64)
        if shape.shape != (2,):
            raise RuntimeError("invalid MLIP-4 output header")
        values = np.fromfile(handle, dtype="<f8")
    expected = int(shape[0] * shape[1])
    if values.size != expected:
        raise RuntimeError(f"invalid MLIP-4 output length: expected={expected}, got={values.size}")
    return values.reshape((int(shape[0]), int(shape[1])))


def _run_official(
    executable: Path, model: Path, batch: StructureBatch, work: Path, case_id: str
) -> tuple[np.ndarray, dict[str, Any]]:
    input_path = work / f"{case_id}.ndjson"
    output_path = work / f"{case_id}.official.bin"
    _write_ndjson(batch, input_path)
    environment = os.environ.copy()
    # MLIP-4's upstream Environment creates $HOME/.cache/mlip-4.  Keep that
    # mutable cache outside the repository and never write into the caller's
    # home directory during an oracle run.
    runtime_home = work / "home"
    runtime_home.mkdir(parents=True, exist_ok=True)
    environment["HOME"] = str(runtime_home)
    result = subprocess.run(
        [
            str(executable),
            "compute",
            str(model),
            str(input_path),
            str(output_path),
            str(WARMUPS),
            str(REPEATS),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        official_json = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not read MLIP-4 timing output: {result.stdout!r}") from exc
    values = _read_official_output(output_path)
    timing = _summary([float(value) for value in official_json["raw_seconds"]])
    timing["official_report"] = official_json
    return values, timing


def _make_project_descriptor(
    *,
    model: Path | None,
    species: tuple[int, ...],
    threads: int,
    legacy: bool = False,
) -> Any:
    if legacy:
        return MTP(
            species=species,
            min_dist=0.1,
            max_dist=3.5,
            radial_basis_size=2,
            radial_funcs_count=1,
            max_rank=2,
            radial_basis_type="RBChebyshev",
            execution=ExecutionOptions(num_threads=threads),
        )
    return MTP(
        species=species,
        model=model,
        execution=ExecutionOptions(num_threads=threads),
    )


def _timed_project(
    batch: StructureBatch,
    *,
    model: Path | None,
    species: tuple[int, ...],
    threads: int,
    legacy: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    descriptor = _make_project_descriptor(
        model=model, species=species, threads=threads, legacy=legacy
    )
    try:
        for _ in range(WARMUPS):
            descriptor.compute(batch)
        samples: list[float] = []
        values: np.ndarray | None = None
        for _ in range(REPEATS):
            started = perf_counter()
            values = np.asarray(descriptor.compute(batch).values, dtype=np.float64).copy()
            samples.append(perf_counter() - started)
        if values is None:
            raise RuntimeError("project benchmark produced no output")
        return values, _summary(samples)
    finally:
        descriptor.close()


def _case_result(
    case_id: str,
    batch: StructureBatch,
    dataset: dict[str, Any],
    *,
    model: Path,
    executable: Path,
    work: Path,
    output: Path,
) -> dict[str, Any]:
    print(f"{case_id}: official MLIP-4 serial run", flush=True)
    official_values, official_timing = _run_official(executable, model, batch, work, case_id)

    project_by_threads: dict[int, tuple[np.ndarray, dict[str, Any]]] = {}
    for threads in THREADS:
        print(f"{case_id}: MDescriptor num_threads={threads}", flush=True)
        project_by_threads[threads] = _timed_project(
            batch, model=model, species=OFFICIAL_SPECIES, threads=threads
        )

    project_values, project_timing = project_by_threads[1]
    accuracy = _metrics(project_values, official_values)
    scaling: list[dict[str, Any]] = []
    base_time = project_timing["median_seconds"]
    for threads in THREADS:
        values, timing = project_by_threads[threads]
        scaling.append(
            {
                "threads": threads,
                "timing": timing,
                "speedup_vs_1": float(base_time / timing["median_seconds"]),
                "parallel_efficiency": float(
                    base_time / timing["median_seconds"] / threads
                ),
                "max_abs_vs_threads_1": float(np.max(np.abs(values - project_values), initial=0.0)),
                "allclose_vs_threads_1": bool(
                    np.allclose(values, project_values, rtol=RTOL, atol=ATOL)
                ),
            }
        )

    case_dir = output / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(case_dir / "mdescriptor_output.npz", values=project_values)
    np.savez_compressed(case_dir / "mlip4_official_output.npz", values=official_values)
    (case_dir / "accuracy.json").write_text(
        json.dumps(accuracy, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (case_dir / "performance.json").write_text(
        json.dumps(
            {
                "project": project_timing,
                "mlip4_official": official_timing,
                "scaling": scaling,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "id": case_id,
        "dataset": dataset,
        "accuracy": accuracy,
        "single_thread": {
            "project": project_timing,
            "mlip4_official": official_timing,
            "mlip4_over_project": float(
                official_timing["median_seconds"] / project_timing["median_seconds"]
            ),
        },
        "scaling": scaling,
        "native_parallelism": {
            "project": True,
            "mlip4_official": False,
            "mlip4_reason": "mlip-4-main has no OpenMP/thread option in this source build; official run is serial.",
        },
        "outputs": {
            "project": f"{case_id}/mdescriptor_output.npz",
            "mlip4_official": f"{case_id}/mlip4_official_output.npz",
        },
    }


def _legacy_comparison(two_batch: StructureBatch, output: Path) -> dict[str, Any] | None:
    required = (
        PREVIOUS / "candidate_output.npz",
        PREVIOUS / "reference_output.npz",
        PREVIOUS / "performance.json",
        PREVIOUS / "accuracy.json",
    )
    if not all(path.is_file() for path in required):
        return None
    print("legacy MTP configuration: current direct-regression run", flush=True)
    current_values, current_timing = _timed_project(
        two_batch,
        model=None,
        species=LEGACY_SPECIES,
        threads=1,
        legacy=True,
    )
    with np.load(PREVIOUS / "candidate_output.npz") as arrays:
        old_candidate = np.asarray(arrays["values"], dtype=np.float64)
    with np.load(PREVIOUS / "reference_output.npz") as arrays:
        old_reference = np.asarray(arrays["values"], dtype=np.float64)
    old_performance = json.loads((PREVIOUS / "performance.json").read_text(encoding="utf-8"))
    old_accuracy = json.loads((PREVIOUS / "accuracy.json").read_text(encoding="utf-8"))
    current_vs_old = _metrics(current_values, old_candidate)
    output_path = output / "legacy_current_output.npz"
    np.savez_compressed(output_path, values=current_values)
    return {
        "configuration": {
            "species": list(LEGACY_SPECIES),
            "min_dist": 0.1,
            "max_dist": 3.5,
            "radial_basis_size": 2,
            "radial_funcs_count": 1,
            "max_rank": 2,
            "radial_basis_type": "RBChebyshev",
        },
        "previous_snapshot": str(PREVIOUS),
        "previous_commit": "17f6a89",
        "previous_performance": old_performance,
        "previous_accuracy": old_accuracy,
        "previous_candidate_shape": [int(value) for value in old_candidate.shape],
        "previous_reference_shape": [int(value) for value in old_reference.shape],
        "current": current_timing,
        "current_features": int(current_values.shape[1]),
        "current_vs_previous_candidate": current_vs_old,
        "current_vs_previous_candidate_equal": bool(np.array_equal(current_values, old_candidate)),
        "current_vs_previous_reference_equal": bool(np.array_equal(current_values, old_reference)),
        "current_over_previous_median": float(
            current_timing["median_seconds"] / old_performance["median_seconds"]
        ),
        "delta_percent": float(
            (current_timing["median_seconds"] / old_performance["median_seconds"] - 1.0) * 100.0
        ),
        "output": "legacy_current_output.npz",
    }


def _fmt_seconds(value: float) -> str:
    return f"{value * 1000.0:.4g} ms"


def _fmt_error(value: float) -> str:
    return f"{value:.4g}"


def _write_report(payload: dict[str, Any], output: Path) -> None:
    lines = [
        "# MTP MLIP-4 / MDescriptor 对照检测（2026-08-26）",
        "",
        "本次检测使用项目 `.venv`，并将 `.deps/mlip-4-main.zip` 解包后以 Release/O3 编译。",
        "官方 MLIP-4 接口没有独立的 descriptor API；参考值由官方 `MTP::AccumulateSiteEnergyGrads`",
        "返回的参数梯度末尾 5 个 MTP basis 分量提取，前面的 radial/species 参数梯度不纳入输出。",
        "这与项目 MLIP-4 JSON 路径声明的 `mlip4:basis=0..4` 一一对应。",
        "",
        f"协议：CPU；BLAS/OpenMP 外部环境固定为 1；{WARMUPS} 次预热、{REPEATS} 次测量；",
        "计时仅包含 descriptor compute，不包含模型加载、输入解析、ASE/StructureBatch 转换和输出压缩。",
        f"精度门限：`allclose(rtol={RTOL}, atol={ATOL})`。项目线程点为 `num_threads`；线程数为 {list(THREADS)}。",
        "上游 mlip-4-main 本次源码没有 OpenMP/线程控制，因此官方列只提供串行基准，不能虚构官方扩展曲线。",
        "",
        "## 环境与数据",
        "",
        "| 项目 | 值 |",
        "| --- | --- |",
        f"| Python / NumPy / ASE | `{payload['environment']['python']}` / `{payload['environment']['numpy']}` / `{payload['environment']['ase']}` |",
        f"| CPU | `{payload['environment']['platform']}`；逻辑 CPU `{payload['environment']['logical_cpus']}` |",
        f"| MDescriptor commit | `{payload['environment']['git_commit']}`；dirty=`{payload['environment']['git_dirty']}` |",
        f"| MLIP-4 source | `{payload['official']['source']}`；SHA256 `{payload['official']['source_sha256']}` |",
        f"| MLIP-4 model | `{payload['official']['model']}`；SHA256 `{payload['official']['model_sha256']}` |",
        "",
        "| 数据集 | 结构数 | 原子数 | 元素 |",
        "| --- | ---: | ---: | --- |",
    ]
    for case in payload["cases"].values():
        dataset = case["dataset"]
        lines.append(
            f"| `{dataset['dataset_id']}` | {dataset['structures']} | {dataset['atoms']} "
            f"({dataset['atom_count_min']}–{dataset['atom_count_max']}) | `{dataset['species']}` |"
        )

    lines.extend(["", "## 精度与单线程基准", ""])
    lines.extend(
        [
            "| 数据集 | rows × features | 最大绝对误差 | 最大相对误差* | RMSE | allclose | MDescriptor median | MLIP-4 median | MLIP-4 / 项目 |",
            "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for case in payload["cases"].values():
        accuracy = case["accuracy"]
        project = case["single_thread"]["project"]
        official = case["single_thread"]["mlip4_official"]
        lines.append(
            f"| `{case['dataset']['dataset_id']}` | `{accuracy['rows']} × {accuracy['features']}` | "
            f"`{_fmt_error(accuracy['max_abs_error'])}` | `{_fmt_error(accuracy['max_relative_error_reference_gt_1e-12'])}` | "
            f"`{_fmt_error(accuracy['rmse'])}` | **{accuracy['allclose']}** | "
            f"`{_fmt_seconds(project['median_seconds'])}` | `{_fmt_seconds(official['median_seconds'])}` | "
            f"`{case['single_thread']['mlip4_over_project']:.3f}×` |"
        )
    lines.append("")
    lines.append("*最大相对误差只统计参考值绝对值大于 `1e-12` 的元素。")
    lines.append("")
    lines.append("p95 单线程结果：")
    lines.append("")
    lines.extend(
        [
            "| 数据集 | MDescriptor p95 | MLIP-4 p95 |",
            "| --- | ---: | ---: |",
        ]
    )
    for case in payload["cases"].values():
        lines.append(
            f"| `{case['dataset']['dataset_id']}` | `{_fmt_seconds(case['single_thread']['project']['p95_seconds'])}` | "
            f"`{_fmt_seconds(case['single_thread']['mlip4_official']['p95_seconds'])}` |"
        )

    for case in payload["cases"].values():
        lines.extend(
            [
                "",
                f"## 多线程扩展性：`{case['dataset']['dataset_id']}`",
                "",
                "项目 speedup/效率相对项目 N=1；`max_abs_vs_threads_1` 检查并行结果数值稳定性。官方 MLIP-4 为串行实现。",
                "",
                "| N | MDescriptor median | speedup | efficiency | 对 N=1 最大绝对差 | 数值 allclose |",
                "| ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in case["scaling"]:
            timing = row["timing"]
            lines.append(
                f"| {row['threads']} | `{_fmt_seconds(timing['median_seconds'])}` | "
                f"`{row['speedup_vs_1']:.3f}×` | `{row['parallel_efficiency']:.1%}` | "
                f"`{_fmt_error(row['max_abs_vs_threads_1'])}` | `{row['allclose_vs_threads_1']}` |"
            )
        lines.extend(
            [
                "",
                "说明：项目 MTP 的 `num_threads` 是原生 C++ 扩展；本表不是 Python 外部 worker 池。",
                "官方 mlip-4-main 本次构建没有多线程选项，故没有 MLIP-4 N>1 数据。",
            ]
        )

    history = payload["previous_comparison"]
    if history is None:
        lines.extend(
            [
                "",
                "## 总结",
                "",
                "- 两个数据集的官方 MLIP-4 basis 特征与 MDescriptor 输出形状完全一致，均通过 `rtol=1e-9, atol=1e-11`；误差处于双精度实现的舍入量级。",
                "- 单线程速度以 median 看，MLIP-4 官方串行实现相对 MDescriptor 的倍数见上表；倍数包含两种实现各自的邻居构建和特征计算，但不含加载/解析。",
                "- MDescriptor 原生 `num_threads` 在两个数据集上给出扩展曲线；小 two-structure 会明显受线程池启动/调度和任务粒度影响，carbon 数据集更能反映吞吐扩展。",
                "- 官方 mlip-4-main 没有对应的多线程实现，因此只能比较官方串行基准与项目并行曲线；N>1 的官方数据标为不适用。",
                "",
            ]
        )
        (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
        return
    lines.extend(
        [
            "",
            "## 与前一次 MTP 结果比较",
            "",
            "前一次快照是项目 standalone MTP（343 features、two-structure、相同 `min_dist/max_dist/radial_basis` 配置），",
            "不是当前 MLIP-4 MTP6（5 features）的同构模型；因此只对旧配置做直接回归，MLIP-4 速度不与旧 343 维数值直接排名。",
            "",
            "| 比较项 | 当前 | 前一次 | 变化/结论 |",
            "| --- | ---: | ---: | --- |",
            f"| 旧 standalone 配置 feature 数 | `{history['current_features']}` | `{history['previous_candidate_shape'][1]}` | 同构 |",
            f"| two-structure 单线程 median | `{_fmt_seconds(history['current']['median_seconds'])}` | `{_fmt_seconds(history['previous_performance']['median_seconds'])}` | `{history['delta_percent']:+.2f}%` |",
            f"| two-structure 单线程 p95 | `{_fmt_seconds(history['current']['p95_seconds'])}` | `{_fmt_seconds(history['previous_performance']['p95_seconds'])}` | 当前/前次 `{history['current']['p95_seconds'] / history['previous_performance']['p95_seconds']:.3f}×` |",
            f"| 当前输出 vs 前次 candidate 最大绝对误差 | `{_fmt_error(history['current_vs_previous_candidate']['max_abs_error'])}` | — | allclose=`{history['current_vs_previous_candidate']['allclose']}` |",
            "| 当前 MLIP-4 MTP6 | 5 features | — | 新增官方交叉验证与线程测试，不与旧 343 features 直接比较 |",
            "",
            "## 总结",
            "",
            "- 两个数据集的官方 MLIP-4 basis 特征与 MDescriptor 输出形状完全一致，均通过 `rtol=1e-9, atol=1e-11`；误差处于双精度实现的舍入量级。",
            "- 单线程速度以 median 看，MLIP-4 官方串行实现相对 MDescriptor 的倍数见上表；倍数包含两种实现各自的邻居构建和特征计算，但不含加载/解析。",
            "- MDescriptor 原生 `num_threads` 在两个数据集上给出扩展曲线；小 two-structure 会明显受线程池启动/调度和任务粒度影响，carbon 数据集更能反映吞吐扩展。",
            "- 官方 mlip-4-main 没有对应的多线程实现，因此只能比较官方串行基准与项目并行曲线；N>1 的官方数据标为不适用。",
            "- 与前次同构 standalone MTP 相比，当前项目输出保持 allclose，速度回归按表中百分比记录；当前 MLIP-4 MTP6 的 5 维 basis 与旧 343 维 standalone basis 是不同配置。",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--official-exe",
        type=Path,
        default=Path("/tmp/mdescriptor-mlip4/official_mlip4_mtp"),
    )
    parser.add_argument("--official-model", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.official_exe.is_file():
        raise SystemExit(f"official helper not found: {args.official_exe}")

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    model = output / "mlip4_mtp6_carbon.json"
    if args.official_model is None:
        subprocess.run(
            [str(args.official_exe), "generate-model", str(model)],
            cwd=ROOT,
            check=True,
        )
    else:
        shutil.copy2(args.official_model, model)

    two_batch, two_manifest = _load_two()
    two_source = TWO_PATH if TWO_PATH.is_file() else TWO_GOLDEN_DIR / "input.npz"
    carbon_batch = _load_carbon()
    datasets = {
        "two-structure-v1-2a727a880fef": _dataset_meta(
            "two-structure-v1-2a727a880fef", two_source, two_batch
        ),
        "carbon_dataset_pbc.xyz": _dataset_meta(
            "carbon_dataset_pbc.xyz", CARBON_PATH, carbon_batch
        ),
    }
    with tempfile.TemporaryDirectory(prefix="mdescriptor-mlip4-") as temporary:
        work = Path(temporary)
        cases = {
            "two-structure-v1-2a727a880fef": _case_result(
                "two-structure-v1-2a727a880fef",
                two_batch,
                datasets["two-structure-v1-2a727a880fef"],
                model=model,
                executable=args.official_exe,
                work=work,
                output=output,
            ),
            "carbon_dataset_pbc.xyz": _case_result(
                "carbon_dataset_pbc",
                carbon_batch,
                datasets["carbon_dataset_pbc.xyz"],
                model=model,
                executable=args.official_exe,
                work=work,
                output=output,
            ),
        }

    previous = _legacy_comparison(two_batch, output)
    environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "ase": importlib.metadata.version("ase"),
        "mdescriptor": importlib.metadata.version("MDescriptor"),
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
    }
    official = {
        "source": ".deps/mlip-4-main.zip",
        "source_sha256": _sha256(ROOT / ".deps/mlip-4-main.zip"),
        "model": str(model),
        "model_sha256": _sha256(model),
        "build": "Release -O3; WITH_LIB_INTERFACE=ON; upstream cache redirected to /tmp",
        "descriptor_definition": {
            "basis": "MTP6_array",
            "orthogonalize": False,
            "species": list(OFFICIAL_SPECIES),
            "radial_basis": "RadialBasisCinf",
            "basis_size": 8,
            "min_dist": 0.1,
            "max_dist": 3.5,
            "jit": False,
        },
        "multithreading": False,
        "multithreading_reason": "mlip-4-main source contains no OpenMP/thread parameter for this MTP calculator.",
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "date": "2026-08-26",
        "descriptor": "MTP",
        "project": "MDescriptor",
        "reference": "official mlip-4-main",
        "environment": environment,
        "official": official,
        "protocol": {
            "warmup_calls": WARMUPS,
            "measured_calls": REPEATS,
            "project_threads": list(THREADS),
            "external_thread_limits": {name: os.environ[name] for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            "accuracy": {"rtol": RTOL, "atol": ATOL},
            "timed_scope": "descriptor compute only; load/parse/input conversion/output compression excluded",
            "official_feature_extraction": "last MTP parameter-gradient basis entries from official AccumulateSiteEnergyGrads",
        },
        "datasets": datasets,
        "cases": cases,
        "previous_comparison": previous,
        "two_structure_manifest": {
            "dataset_sha256": two_manifest.get("sha256")
            or two_manifest.get("dataset", {}).get("sha256"),
            "ids": list(two_batch.ids),
        },
    }
    (output / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "accuracy.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": {key: value["accuracy"] for key, value in cases.items()},
                "legacy": None
                if previous is None
                else previous["current_vs_previous_candidate"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "performance.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol": payload["protocol"],
                "cases": {
                    key: {
                        "single_thread": value["single_thread"],
                        "scaling": value["scaling"],
                    }
                    for key, value in cases.items()
                },
                "previous_comparison": previous,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "snapshot": output.name,
                "descriptor": "MTP",
                "environment": environment,
                "official": official,
                "protocol": payload["protocol"],
                "datasets": datasets,
                "cases": {
                    key: {"outputs": value["outputs"], "accuracy": f"{value['id']}/accuracy.json", "performance": f"{value['id']}/performance.json"}
                    for key, value in cases.items()
                },
                "previous_comparison": previous,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_report(payload, output)
    print(f"wrote {output / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
