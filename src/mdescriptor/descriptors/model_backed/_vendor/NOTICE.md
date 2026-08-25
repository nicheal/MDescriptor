# Vendored DPA4/DPA4C CPU inference

The `dpa4desc` package in this directory is derived from the DPA4/DPA4C
descriptor extraction made from deepmd-kit 3.2.0 and supplied in
`.deps/dpa4-descriptor.zip`.

It is used as a private, CPU-only NumPy inference core. The source carries
its upstream `LGPL-3.0-or-later` SPDX notices. The corresponding license text
is included in `LICENSE-LGPL-3.0-or-later.txt`.

The package is adapted only to relocate its private imports under MDescriptor;
the public MDescriptor API and model/resource lifecycle remain outside this
vendored code.
