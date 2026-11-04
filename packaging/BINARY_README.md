# ProtLocus Linux binary bundle

ProtLocus searches genomic DNA with protein sequences or protein-family HMM profiles and produces alignments and candidate gene models.

## Requirements

Requires Linux x86_64 with glibc 2.35 or newer, zlib, and Python 3.9+. OpenCL also needs an OpenCL loader and a vendor GPU driver/ICD. CUDA also needs `libcudart.so.12` and a compatible NVIDIA driver. Install the required drivers and runtime libraries on the host before using a GPU backend.

For Linux systems with glibc 2.28–2.34, use the Conda packages instead of these native executables. If this complete bundle contains `conda-channel/`, enter that directory and follow its `README.md`.

## Set up and run

From the extracted bundle directory:

```bash
export PATH="$PWD/bin:$PATH"
```

The CPU, OpenCL, and CUDA executables are named `protlocus`, `protlocus-gpu`, and `protlocus-cuda`. The complete bundle includes all three; single-backend bundles include their selected executable.

```bash
# CPU alignment and gene features:
protlocus -t 8 genome.fa proteins.fa > alignments.paf
protlocus -t 8 --gff genome.fa proteins.fa > prediction.gff3

# Optional GPU backends:
protlocus-gpu --gpu -t 8 genome.fa proteins.fa > opencl.paf
protlocus-cuda --gpu -t 8 genome.fa proteins.fa > cuda.paf
```

Use `--help` with the selected program for its options. GPU commands may fall back to CPU when no device is available; check for `[M::gpu] using device` in the diagnostic log.

## Complete offline examples

The complete mouse mitochondrial chromosome and E. coli K-12 MG1655 genome, plus their protein queries, are included in `share/protlocus/examples/`. No download is needed.

With the CPU or full bundle:

```bash
protlocus-example mouse-mt
protlocus-example ecoli
```

For a single-backend GPU bundle, select that backend explicitly:

```bash
protlocus-example ecoli --backend opencl --outdir ecoli-opencl
# Or, with the CUDA bundle:
protlocus-example ecoli --backend cuda --outdir ecoli-cuda
```

Each run writes GFF3, PAF, a log, and a validation report into a new result directory. Use `--outdir DIRECTORY` to choose a fresh location. GPU examples require a working device and fail if initialization cannot be confirmed. See [example instructions](share/protlocus/examples/README.md) for reference accessions and expected gene loci.

## Check the installation

```bash
python3 verify.py
# Also check the selected GPU and its native test suite:
python3 verify.py --backend cuda --require-gpu
# Or: python3 verify.py --backend opencl --require-gpu
```

## Documentation

The complete bundle includes the full source under `src/`; source-build instructions apply to that directory. See `docs/USAGE_zh.md`, `docs/MANUAL_zh.md`, and `docs/GPU_zh.md` for usage and GPU details. These guides are in Chinese. The license is in `LICENSE.txt`.
