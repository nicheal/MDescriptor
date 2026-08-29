# Independent upstream numerical oracles

This directory keeps the source adapters and reproducible build/run entry
points used to produce legacy benchmark references.  These programs are not
part of MDescriptor and must not import or reuse its descriptor arithmetic for
the upstream side of a comparison.

| Descriptor | Independent upstream | Directory |
| --- | --- | --- |
| SOAPTurbo | `soap_turbo-master.zip` Fortran `get_soap` | `soapturbo/` |
| LBispectrum | PyXtal-FF `Bispectrum` driving LAMMPS `compute sna/atom` | `lbispectrum/` |
| MTP | `mlip-4-main.zip` `MTP::AccumulateSiteEnergyGrads` | `mtp/` |

`sources.lock.json` records the exact local archives and hashes.  Upstream
sources stay in `.deps`; this directory contains only the bridge code needed
to compile and call them.  Build products go to `/tmp` and are never treated
as source artifacts.

