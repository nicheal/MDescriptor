# SOAPTurbo upstream oracle

The numerical oracle is the unmodified Fortran implementation in
`.deps/soap_turbo-master.zip`.  `soap_turbo_reference.f90` is a C ABI bridge;
the LAPACK shims only map the upstream BLAS/LAPACK calls to the SciPy OpenBLAS
already present in the project environment.

Build and run from the repository root:

```bash
library="$(benchmarks/_legacy_oracles/soapturbo/build.sh)"
.venv/bin/python benchmarks/_legacy_oracles/soapturbo/generate_golden.py \
  --official-library "$library"
```

For the larger local benchmark datasets, use the comparison runner:

```bash
.venv/bin/python benchmarks/_legacy_oracles/soapturbo/run_comparison.py \
  --official-library "$library" \
  --official-source /tmp/mdescriptor-soapturbo-upstream/soap_turbo-master \
  --output-dir benchmarks/_legacy_oracles/soapturbo-upstream/reproduction
```

The upstream routine accepts a prepared neighbor list.  The runner records
that preparation outside the upstream kernel timer and compares the official
`get_soap` output against MDescriptor using identical neighbor vectors.
