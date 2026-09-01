# MDescriptor-CUDA

`MDescriptor-CUDA` is the optional Linux CUDA plugin for MDescriptor. It
contains the CUDA extension and the CUDA user-space runtime libraries needed
by that extension. The NVIDIA driver remains a host requirement and is not
redistributed.

Build from the repository root, selecting the architectures required by the
deployment GPUs:

```bash
python -m build packaging/cuda --wheel --outdir dist \
  -Ccmake.define.CMAKE_CUDA_ARCHITECTURES=75
```

The resulting wheel depends on the matching `MDescriptor` release. Install
both with `pip`; a full CUDA Toolkit is not required on the target machine.

The current CUDA wheel build targets Linux. The wheel must still run with a
compatible NVIDIA driver and a host permission environment in which
`nvidia-smi` can see the GPU.

