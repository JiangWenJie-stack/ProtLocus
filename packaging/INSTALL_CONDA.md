# Install ProtLocus from a local Conda channel

This channel contains CPU, OpenCL, and CUDA packages for Linux x86_64.

Requires glibc 2.28 or newer. The standalone native binary archives require glibc 2.35 or newer.

Extract the channel archive and enter the directory containing `linux-64/` and `noarch/`, then run:

```bash
CHANNEL="file://$(pwd)"
conda create -n protlocus --override-channels --strict-channel-priority \
  -c "$CHANNEL" -c conda-forge \
  protlocus=0.1.0 protlocus-gpu=0.1.0 protlocus-cuda=0.1.0
conda activate protlocus
protlocus --version
protlocus-gpu --version
protlocus-cuda --version
```

For CPU only, request `protlocus=0.1.0`. For OpenCL or CUDA, request `protlocus-gpu=0.1.0` or `protlocus-cuda=0.1.0`; each GPU package also installs the same-version CPU package. Dependencies are downloaded from conda-forge, so installation normally requires internet access.

```bash
protlocus genome.fa proteins.fa > cpu.paf
protlocus-gpu --gpu genome.fa proteins.fa > opencl.paf
protlocus-cuda --gpu genome.fa proteins.fa > cuda.paf
```

The host must provide a compatible GPU and driver. An unavailable device can cause CPU fallback; a successful exit alone does not prove GPU execution. Look for `[M::gpu] using device` and ensure no `GPU unavailable` warning appears.

The OpenCL loader must find the vendor ICD files. If they are installed in `/etc/OpenCL/vendors` but are not discovered automatically, use:

```bash
export OCL_ICD_VENDORS=/etc/OpenCL/vendors
```

This setting does not install a driver. CUDA requires an NVIDIA driver compatible with the packaged CUDA runtime. 

## Offline examples

The complete mouse mitochondrial chromosome and E. coli K-12 MG1655 genome, together with their protein queries, are installed with the CPU package. No download is required to run them.

```bash
protlocus-example --list
protlocus-example mouse-mt
protlocus-example ecoli
```

Each run writes GFF3, PAF, logs, and a validation report into a new result directory. Use `--outdir DIRECTORY` to choose another location.
