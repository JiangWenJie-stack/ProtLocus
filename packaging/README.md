# ProtLocus package variants

Choose the CPU package for standard alignment and HMM-guided searches. Add a GPU package to use an OpenCL or CUDA device:

- `protlocus`: CPU aligner, HMM rechecker, companion tools, and complete offline examples.
- `protlocus-gpu`: OpenCL aligner, invoked as `protlocus-gpu`.
- `protlocus-cuda`: CUDA aligner, invoked as `protlocus-cuda`.

Each GPU package installs the same-version CPU package as a dependency. All three commands can coexist in one Conda environment. Example data are shared between the backends.

Conda packages require Linux x86_64 with glibc 2.28 or newer. Native binary archives require glibc 2.35 or newer. GPU drivers and vendor ICD files are supplied by the host system.

See [Conda installation instructions](INSTALL_CONDA.md), [native bundle instructions](BINARY_README.md), or the [main README](../README.md) for setup, examples, and usage.
