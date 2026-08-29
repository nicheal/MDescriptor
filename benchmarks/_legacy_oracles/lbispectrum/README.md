# LBispectrum upstream oracle

This oracle uses the installed PyXtal-FF `Bispectrum` class to generate and
execute LAMMPS `compute sna/atom`.  LAMMPS is built exclusively from
`.deps/lammps-stable.tar.gz` with ML-SNAP enabled.  The local Python adapter
supplies PyXtal-FF's optional `LammpsData` dependency and compatibility changes
needed by the newer LAMMPS input syntax; it does not calculate the descriptor.

Build and run from the repository root:

```bash
lmp_bin="$(benchmarks/_legacy_oracles/lbispectrum/build_lammps.sh)"
PATH="$(dirname "$lmp_bin"):$PATH" \
  .venv/bin/python benchmarks/_legacy_oracles/lbispectrum/generate_golden.py
```

For the larger local benchmark datasets, use the comparison runner with the
same `PATH`:

```bash
PATH="$(dirname "$lmp_bin"):$PATH" \
  .venv/bin/python benchmarks/_legacy_oracles/lbispectrum/run_comparison.py
```

The required Python distribution is locked as `pyxtal-ff==0.2.3` in
`../sources.lock.json`.  The runner checks and records both PyXtal-FF and
LAMMPS versions in its output manifest.
