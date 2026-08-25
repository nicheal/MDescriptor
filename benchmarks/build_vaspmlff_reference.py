"""Build the Linux C00PS reference library from ``.deps/vaspmlff.zip``.

The archive ships a Windows DLL and the Fortran source used to produce it.
This helper extracts the source into a temporary directory, compiles the
standalone C API with gfortran, and writes the resulting Linux shared library
under ``.deps/vaspmlff-build/``.  The input archive is never modified.

Run from the project root with::

    .venv/bin/python benchmarks/build_vaspmlff_reference.py
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / ".deps" / "vaspmlff.zip"
VASP_SOURCE_ARCHIVE = ROOT / ".deps" / "vasp.6.6.0.tgz"
OUTPUT_DIR = ROOT / ".deps" / "vaspmlff-build"
OUTPUT_LIBRARY = OUTPUT_DIR / "libvaspmlff.so"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _extract(archive: Path, destination: Path) -> Path:
    with ZipFile(archive) as package:
        members = package.namelist()
        for member in members:
            target = (destination / member).resolve()
            if destination.resolve() not in target.parents and target != destination.resolve():
                raise ValueError(f"unsafe archive member: {member}")
        package.extractall(destination)
    source_root = destination / "vaspmlff"
    if not (source_root / "src" / "c_api.F90").exists():
        raise FileNotFoundError("archive does not contain vaspmlff/src/c_api.F90")
    return source_root


def _compile(source_root: Path, compiler: str) -> Path:
    source_dir = source_root / "src"
    flags = [
        "-O2",
        "-fPIC",
        "-ffree-form",
        "-ffree-line-length-none",
        "-cpp",
        "-Wno-align-commons",
        "-fopenmp",
        "-fallow-argument-mismatch",
        "-std=legacy",
        f"-J{source_dir}",
        f"-I{source_dir}",
    ]
    sources = [
        "ml_ff_prec.F",
        "ml_ff_constant.F",
        "vaspmlff_stubs.F",
        "ml_ff_struct.F",
        "ml_asa2.F",
        "ml_ff_math_core.F",
        "blas_lite.F",
        "compute_c00ps.F90",
        "c_api.F90",
    ]
    objects: list[Path] = []
    for name in sources:
        source = source_dir / name
        if not source.exists():
            raise FileNotFoundError(source)
        object_path = source_dir / f"{source.stem}.o"
        subprocess.run(
            [compiler, *flags, "-c", str(source), "-o", str(object_path)],
            check=True,
            cwd=source_root,
        )
        objects.append(object_path)

    library = source_root / "libvaspmlff.so"
    subprocess.run(
        [compiler, "-shared", "-fopenmp", "-o", str(library), *(str(item) for item in objects)],
        check=True,
        cwd=source_root,
    )
    return library


def _apply_compatibility_patches(source_root: Path) -> list[str]:
    """Apply source-alignment and memory-sizing fixes to the extracted source.

    The ZIP contains a small C API driver around the VASP MLFF kernels.  Its
    original driver used convenient standalone defaults rather than the
    VASP 6.6.0 defaults.  The descriptor kernels themselves are retained;
    these patches only restore the values and bookkeeping used by the
    original ``ML_FF`` path.

    ``SET_SIZE_EST_POINT`` and ``SET_EST_DATA_POINT`` use slightly different
    cutoff-boundary arithmetic.  The former can undercount by one, while the
    latter writes the corresponding neighbour row.  Reserving one structure's
    worth of extra rows prevents an out-of-bounds write without changing any
    neighbour, radial, or angular value.
    """

    source = source_root / "src" / "compute_c00ps.F90"
    text = source.read_text(encoding="utf-8")
    patches: list[str] = []

    def replace_once(old: str, new: str, description: str) -> None:
        nonlocal text
        if text.count(old) != 1:
            raise ValueError(f"unexpected source while applying {description}")
        text = text.replace(old, new)
        patches.append(description)

    replace_once(
        "         LOGICAL  :: LAFILT2  = .FALSE.\n"
        "      END TYPE C00PS_PARAMS",
        "         LOGICAL  :: LAFILT2  = .FALSE.\n"
        "         ! VASP 6.6.0 default: ML_LSIC = .TRUE. when ML_LFAST=.FALSE.\n"
        "         LOGICAL  :: LSIC     = .TRUE.\n"
        "      END TYPE C00PS_PARAMS",
        "restore VASP default PS self-interaction correction flag",
    )
    replace_once(
        "         INTEGER  :: IBROAD1 = 0\n",
        "         INTEGER  :: IBROAD1 = 2\n",
        "restore VASP default radial Gaussian broadening mode",
    )
    replace_once(
        "         INTEGER  :: IBROAD2 = 0\n",
        "         INTEGER  :: IBROAD2 = 2\n",
        "restore VASP default angular Gaussian broadening mode",
    )
    replace_once(
        "         INTEGER  :: MROW_C00, MROW_PS\n",
        "         INTEGER  :: MROW_C00, MROW_PS\n"
        "         INTEGER  :: MROW_PS_SIC\n",
        "add VASP PS self-interaction workspace size",
    )
    if text.count("ALLOCATE(WION(MTYP))  ;  WION = 1.0_q") != 2:
        raise ValueError("unexpected WION initialisation count")
    text = text.replace(
        "ALLOCATE(WION(MTYP))  ;  WION = 1.0_q",
        "ALLOCATE(WION(MTYP))  ;  WION = 0.5_q/(0.5_q*0.5_q)",
    )
    patches.append("restore VASP ML_SION=0.5 Gaussian width (WION=2)")

    replace_once(
        "      !--- SIC dummy arrays (unused, but needed for D0CLM_SOAP signature) ---\n"
        "      ALLOCATE(CTX%LVAR_SIC(1:P%MRB2, 1:P%MRB2, 0:P%LMAX2, 1:MTYP, 1:MTYP))\n"
        "      ALLOCATE(CTX%LFLAG_VAR_SIC(1:P%MRB2, 1:P%MRB2, 0:P%LMAX2, 1:MTYP, 1:MTYP))\n"
        "      CTX%LVAR_SIC = 0\n"
        "      CTX%LFLAG_VAR_SIC = .FALSE.\n",
        "      !--- PS self-interaction mapping, as in ML_FF%LVAR_SIC ---\n"
        "      ALLOCATE(CTX%LVAR_SIC(1:P%MRB2, 1:P%MRB2, 0:P%LMAX2, 1:MTYP, 1:MTYP))\n"
        "      ALLOCATE(CTX%LFLAG_VAR_SIC(1:P%MRB2, 1:P%MRB2, 0:P%LMAX2, 1:MTYP, 1:MTYP))\n"
        "      CTX%LVAR_SIC = 0\n"
        "      CTX%LFLAG_VAR_SIC = .FALSE.\n"
        "      CTX%MROW_PS_SIC = 0\n"
        "      IF (P%LSIC) THEN\n"
        "         DO INTYP0 = 1, MTYP\n"
        "            DO L = 0, P%LMAX2\n"
        "               DO IRB = 1, CTX%NRB2(L)\n"
        "                  DO JRB = IRB, CTX%NRB2(L)\n"
        "                     CTX%MROW_PS_SIC = CTX%MROW_PS_SIC + 1\n"
        "                     CTX%LVAR_SIC(JRB,IRB,L,INTYP0,INTYP0) = CTX%MROW_PS_SIC\n"
        "                     CTX%LFLAG_VAR_SIC(JRB,IRB,L,INTYP0,INTYP0) = .TRUE.\n"
        "                  END DO\n"
        "               END DO\n"
        "            END DO\n"
        "         END DO\n"
        "      END IF\n"
        "      IF (CTX%MROW_PS_SIC == 0) CTX%MROW_PS_SIC = 1\n",
        "restore VASP LVAR_SIC triangular same-species mapping",
    )

    replace_once(
        "      ! dummy (size-1) arrays for coupling / SIC\n"
        "      REAL(q) :: CLM_COUPLE(1), C00_COUPLE(1), PS_COUPLE(1)\n"
        "      REAL(q) :: PS_SIC(1), PS_SIC_COUPLE(1)\n",
        "      ! coupling stubs remain size-one; SIC follows VASP's PS mapping\n"
        "      REAL(q) :: CLM_COUPLE(1), C00_COUPLE(1), PS_COUPLE(1)\n"
        "      REAL(q), ALLOCATABLE :: PS_SIC(:), PS_SIC_COUPLE(:)\n",
        "allocate VASP-sized PS self-interaction workspaces",
    )
    replace_once(
        "      EPS_TOL = 1.0E-10_q\n\n      !--- 1) neighbour lists ---",
        "      EPS_TOL = 1.0E-10_q\n"
        "      ALLOCATE(PS_SIC(CTX%MROW_PS_SIC))\n"
        "      ALLOCATE(PS_SIC_COUPLE(1))\n\n"
        "      !--- 1) neighbour lists ---",
        "allocate per-atom VASP PS self-interaction workspaces",
    )
    replace_once(
        "              .FALSE., CTX%LVAR, CTX%LVAR_SIC, P%LWINDOW2, &\n",
        "              P%LSIC, CTX%LVAR, CTX%LVAR_SIC, P%LWINDOW2, &\n",
        "enable VASP PS self-interaction accumulation in D0CLM_SOAP",
    )
    replace_once(
        "              P%LAFILT2, LADD, .FALSE., CTX%LFLAG_VAR, CTX%LFLAG_VAR_SIC, &\n"
        "              CTX%L_LNRB2, P%LMAX2, .FALSE., CTX%LVAR, CTX%LVAR_SIC, &\n"
        "              CTX%MLNRB2, P%MRB2, NTYP, &\n"
        "              CTX%NLNRB2, CTX%NRB2, CTX%NRB_LNRB2, NTYP, &\n",
        "              P%LAFILT2, LADD, .FALSE., CTX%LFLAG_VAR, CTX%LFLAG_VAR_SIC, &\n"
        "              CTX%L_LNRB2, P%LMAX2, P%LSIC, CTX%LVAR, CTX%LVAR_SIC, &\n"
        "              CTX%MLNRB2, P%MRB2, NTYP, &\n"
        "              CTX%NLNRB2, CTX%NRB2, CTX%NRB_LNRB2, NTYP, &\n",
        "enable VASP PS self-interaction subtraction in D0PS_SOAP",
    )
    replace_once(
        "              CTX%MROW_CLM2, 1, 1, 1, &\n"
        "              NNEIB_EST2, CTX%NRB2, CTX%NSPL2, NTYP, &\n",
        "              CTX%MROW_CLM2, 1, 1, CTX%MROW_PS_SIC, &\n"
        "              NNEIB_EST2, CTX%NRB2, CTX%NSPL2, NTYP, &\n",
        "pass VASP PS self-interaction workspace size to D0CLM_SOAP",
    )
    replace_once(
        "      DEALLOCATE(CLM1, CLM2, C00, PS)\n",
        "      DEALLOCATE(CLM1, CLM2, C00, PS, PS_SIC, PS_SIC_COUPLE)\n",
        "deallocate per-atom VASP PS self-interaction workspaces",
    )

    old = "      MROW_EST1 = MNEIB_EST1 + 1\n      MROW_EST2 = MNEIB_EST2 + 1"
    new = (
        "      ! The legacy size estimator can undercount by one at a cutoff\n"
        "      ! boundary. Reserve a structure-sized guard for the neighbour rows.\n"
        "      MROW_EST1 = MNEIB_EST1 + N_ATOMS + 1\n"
        "      MROW_EST2 = MNEIB_EST2 + N_ATOMS + 1"
    )
    if text.count(old) != 1:
        raise ValueError("unexpected compute_c00ps.F90 neighbour-row assignment")
    text = text.replace(old, new)
    patches.append("extra neighbour-row capacity in compute_c00ps.F90; arithmetic unchanged")
    source.write_text(text, encoding="utf-8")
    return patches


def main() -> int:
    if not ARCHIVE.exists():
        raise SystemExit(f"reference archive does not exist: {ARCHIVE}")
    compiler = os.environ.get("FC", "gfortran")
    with tempfile.TemporaryDirectory(prefix="vaspmlff-build-") as temporary:
        source_root = _extract(ARCHIVE, Path(temporary))
        patches = _apply_compatibility_patches(source_root)
        library = _compile(source_root, compiler)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(library, OUTPUT_LIBRARY)
    metadata = {
        "archive": str(ARCHIVE.resolve()),
        "archive_sha256": _sha256(ARCHIVE),
        "vasp_source_archive": str(VASP_SOURCE_ARCHIVE.resolve())
        if VASP_SOURCE_ARCHIVE.exists()
        else None,
        "vasp_source_archive_sha256": _sha256(VASP_SOURCE_ARCHIVE)
        if VASP_SOURCE_ARCHIVE.exists()
        else None,
        "library": str(OUTPUT_LIBRARY.resolve()),
        "library_sha256": _sha256(OUTPUT_LIBRARY),
        "compiler": compiler,
        "source_patches": patches,
        "flags": [
            "-O2",
            "-fPIC",
            "-ffree-form",
            "-ffree-line-length-none",
            "-cpp",
            "-Wno-align-commons",
            "-fopenmp",
            "-fallow-argument-mismatch",
            "-std=legacy",
        ],
    }
    (OUTPUT_DIR / "build.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
