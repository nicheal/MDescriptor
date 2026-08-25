# Real-data descriptor benchmark

The reproducible entry point is:

```bash
.venv/bin/python benchmarks/run_real_data_baseline.py \
  --dataset benchmarks/carbon_dataset_pbc.xyz
```

The default dataset is `benchmarks/carbon_dataset_pbc.xyz`. It is parsed as
extxyz, including `Lattice`, `pbc`, `force`, `energy`, and `virial` fields. The
current file has 450 periodic structures and 28,337 carbon atoms. Its SHA256
is recorded in every snapshot manifest.

Each run creates a new immutable directory under
`benchmarks/baselines/carbon_dataset_pbc/<snapshot>/`:

- `input/source.extxyz` and `input/canonical.npz` preserve the exact input;
- `cases/<case>/raw.npz` preserves the package-native numeric output;
- `cases/<case>/normalized.npz` stores the dense float64 comparison output;
- `cases/<case>/output.json` stores level, labels, shape, and backend metadata;
- `cases/<case>/logs/` stores cold-subprocess stdout/stderr;
- `manifest.json` stores hashes, versions, parameters, timings, status, and comparisons.

The top-level `benchmarks/baselines/carbon_dataset_pbc/manifest.json` is an
append-only snapshot index. Existing snapshots are never overwritten.
When a very slow case is materialized in a separate snapshot, the root-level
`benchmarks/baselines/carbon_dataset_pbc/comparisons.json` records the
cross-snapshot comparison and its input/model hashes.

The first-phase cases are MDescriptor, DScribe SOAP, Featomic SOAP power
spectrum, PyXtal_FF ACSF, VASPMLFF C00PS, NEP-Adapters, and DPA4/DPA4C. The
C00PS pair is a strict reference comparison: MDescriptor is configured with
the VASP default carbon setting (`r_cut=5`, `MRB=8`, radial `LMAX=2`, angular
`LMAX=4`, BP cutoff, and both block normalizations enabled), while the
reference is loaded from `.deps/vaspmlff.zip`. Its archive SHA256 and the
Linux shared-library SHA256 are written to the case manifest.

The archive includes a Windows DLL, so Linux runs first need:

```bash
.venv/bin/python benchmarks/build_vaspmlff_reference.py
```

The build uses the ZIP's Fortran source and records compiler flags. The build
also records the SHA256 of `.deps/vasp.6.6.0.tgz` and restores the VASP 6.6.0
defaults that the standalone ZIP driver had replaced: `ML_IBROAD1/2=2`,
`ML_SION1/2=0.5` (therefore `WION=2`), and `ML_LSIC=.TRUE.` with the triangular
same-species `LVAR_SIC` map. A separate memory-sizing patch reserves extra
neighbour rows because the legacy size estimator can undercount at a cutoff
boundary; it does not change descriptor arithmetic. The reference C API is
run in one forked child per structure because the legacy Linux build is not
safely reusable across multiple calls. The resulting timing therefore
includes that isolation cost, and the manifest records the route. Cleanup is
intentionally skipped before the isolated worker exits because the legacy
Fortran finalizer is unstable at Python interpreter shutdown.

Exact pass/fail is reported only for the same NEP checkpoint and the same DPA
checkpoint, plus the strict C00PS reference pair. SOAP, SOAP power spectrum,
and ACSF comparisons are retained as numerical metrics but are marked
nominal-only because package normalization and feature layout are not
guaranteed to be identical. For DPA4, the cross-backend equivalence tolerance is `rtol=2e-5` and
`atol=2e-5`; DPA4C uses the same relative tolerance with `atol=1e-5`. These
values accommodate the observed official CPU/PyTorch floating-point roundoff
while retaining a strict same-checkpoint comparison.

`deepmd-kit==3.2.0` is used through its official PyTorch backend. The benchmark
environment must provide matching `torch==2.11.0`, `mpich`, `e3nn`, and
`vesin[torch]` packages. DPA4C is routed through deepmd-kit's official graph
descriptor ABI because the installed public `eval_descriptor` helper uses the
checkpoint sentinel `sel=999999` and otherwise attempts an invalid dense
allocation; this route and its package versions are recorded in
`cases/*/output.json`. The runner never substitutes MDescriptor's NumPy
evaluator for an official deepmd-kit result.

For a smoke run while developing an adapter:

```bash
.venv/bin/python benchmarks/run_real_data_baseline.py \
  --cases mdescriptor_soap,dscribe_soap \
  --cold-repeats 1 --warmup 1 --repeat 1 \
  --snapshot-id smoke
```

For a deliberately materialization-only run of a very slow implementation,
use `--skip-warm --skip-per-structure`; the manifest then records both omitted
timing lanes explicitly instead of presenting them as zero.

The snapshots produced on 2026-08-25 are intentionally materialization/short-
cold-lane baselines: DPA4 MDescriptor takes about 1,765 s for the full file
and official deepmd-kit DPA4 about 839 s on this CPU environment, so their
warm and per-structure lanes were not repeated. They remain valid saved
outputs and cold-kernel measurements, but are not claims of completing the
default five-cold/two-warmup/ten-warm-measurement protocol. Remove the skip
flags (and use a new snapshot ID) when a full timing campaign is required.

The original `benchmarks/soap_diverse_dataset_300.xyz` remains a separate
dataset. It currently contains ordinary XYZ frames without `Lattice` or
`pbc`; it is not silently mixed into this periodic baseline.

The source-aligned C00PS materialization on 2026-08-25 is stored in
`benchmarks/baselines/carbon_dataset_pbc/20260825-c00ps-v66-aligned-clm/`. Both
outputs have shape `[28337, 142]`. Under the strict `rtol=1e-8, atol=1e-8`
check, the comparison passes with `max_abs=8.9204869585e-09`,
`RMSE=5.3069689151e-10`, `MAE=1.9691665870e-10`, and cosine similarity
`0.9999999999999999`. In the one cold materialization lane, the optimized
project and reference kernel times were `1.2035 s` and `9.7082 s`,
respectively; total times were `2.8450 s` and `18.9100 s`. The project path
now follows VASP's `CLM` accumulation followed by PS channel dot products,
while retaining the same descriptor values. The earlier
`20260825-c00ps-reference` snapshot records the un-aligned standalone wrapper,
and `20260825-c00ps-v66-aligned` records the source-aligned arithmetic before
this CLM speed optimization; neither supersedes the latest `-clm` baseline.
