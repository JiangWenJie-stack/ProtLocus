# ProtLocus

ProtLocus finds protein-coding genes in genomic DNA using protein sequences or protein-family HMM profiles. It supports splice-aware protein-to-genome alignment, HMM-guided candidate searches, and alignment-derived gene models. 

## Features

- Align proteins to a genome with support for introns and frameshifts.
- Search with HMMER3 protein-family profiles and screen profile libraries.
- Recheck HMM candidates with the bundled `protlocus-recheck` program.
- Produce PAF alignments, GFF3 gene features, and predicted protein sequences.
- Run on CPU or use optional OpenCL and CUDA backends.
- Try complete, bundled genome examples without internet access.

## Installation

The precompiled packages support Linux x86_64. Choose one installation method below.

### Conda channel archive

Download and extract `protlocus-0.1.0-conda-channel.tar.gz`, then install from its local channel:

```bash
tar -xzf protlocus-0.1.0-conda-channel.tar.gz
cd conda-channel
conda create -n protlocus --override-channels --strict-channel-priority \
  -c "file://$(pwd)" -c conda-forge \
  protlocus=0.1.0 protlocus-gpu=0.1.0 protlocus-cuda=0.1.0
conda activate protlocus
protlocus --version
```

Requires glibc 2.28 or newer. Installation downloads dependencies from conda-forge. For CPU only, request just `protlocus=0.1.0`; the OpenCL and CUDA packages are optional and can coexist with the CPU package. GPU drivers must be installed on the host. See [Conda installation details](packaging/INSTALL_CONDA.md).

### Native binary bundle

Download the `linux-x86_64-full` archive for all three backends, or a single-backend archive. Extract it, enter the extracted directory, and add its programs to PATH:

```bash
export PATH="$PWD/bin:$PATH"
```

Native binaries require glibc 2.35 or newer, zlib, and Python 3.9+. OpenCL additionally requires an OpenCL loader and vendor driver/ICD; CUDA requires `libcudart.so.12` and a compatible NVIDIA driver. The complete bundle also includes the source under `src/`.

### Build from source

Install a C99 compiler, GNU Make, zlib development headers, and Python 3.9+. On Ubuntu/Debian, run from the source directory:

```bash
sudo apt-get install build-essential zlib1g-dev python3
make -j2
make check
export PATH="$PWD:$PWD/tools:$PATH"
protlocus --version
```

## Offline examples

The package includes both complete reference DNA sequences and their protein queries:

- **Mouse mitochondrial chromosome**: 16,299 bp, with a COX1 query.
- **E. coli K-12 MG1655 genome**: 4,641,652 bp, with a lacZ query.

The FASTA data total about 4.72 MB. No download is needed to run either example:

```bash
protlocus-example --list
protlocus-example mouse-mt
protlocus-example ecoli
```

Each command creates a result directory containing `result.gff3`, `alignments.paf`, `run.log`, and `report.json`. The runner selects the correct genetic code and checks that the expected gene locus was recovered. To repeat an example, choose a fresh output directory:

```bash
protlocus-example ecoli --outdir ecoli-results --threads 4
```

The mouse example uses mitochondrial DNA and demonstrates intronless gene alignment. See [example instructions](examples/README.md) for reference accessions, expected loci, and GPU options.

## Align proteins to a genome

Supply a genome in FASTA format and protein queries in FASTA format:

```bash
protlocus -t 8 genome.fa proteins.fa > alignments.paf
protlocus -t 8 --gff genome.fa proteins.fa > prediction.gff3
```

For repeated searches, build a reusable genome index:

```bash
protlocus -t 8 -d genome.mpi genome.fa
protlocus -I -u -t 8 --accuracy --gff genome.mpi proteins.fa > prediction.gff3
```

Use `--accuracy` for higher-recall alignment or `--compact` for compact genomes. Run `protlocus --help` for all options.

## Search with a protein-family HMM library

Supply your genome and an HMMER3 profile library:

```bash
protlocus_hmm_pipeline.sh \
  -b protlocus -r protlocus-recheck \
  -g genome.mpi -m library.hmm -o result -t 8 -G genome.fa
```

The pipeline produces `result.final.paf`, `result.genes.gff3`, `result.genes.faa`, and diagnostic logs. Add `-M compact` for compact genomes. For prokaryotic gene models, add `-X "--no-introns"`.

The default rechecking engine uses the bundled `protlocus-recheck`; external HMMER is not required. The optional `-R hmmer` engine requires `hmmfetch` and `hmmsearch`. GNU `time` provides resource statistics when available. Run `protlocus_hmm_pipeline.sh -h` for pipeline options.

## GPU execution

With the matching package, runtime libraries, and driver installed:

```bash
protlocus-gpu --gpu -t 8 genome.fa proteins.fa > opencl.paf
protlocus-cuda --gpu -t 8 genome.fa proteins.fa > cuda.paf

# Run a bundled example on a GPU:
protlocus-example ecoli --backend cuda --outdir ecoli-cuda
protlocus-example ecoli --backend opencl --outdir ecoli-opencl
```

The main GPU commands may fall back to CPU if no device is available. Check the log for `[M::gpu] using device` and ensure there is no `GPU unavailable` warning. The example runner treats an unconfirmed GPU initialization as a failure. Small examples are intended to check functionality, not measure speed.

For OpenCL, the loader must find the vendor ICD files. If they are installed in `/etc/OpenCL/vendors` but are not found automatically:

```bash
export OCL_ICD_VENDORS=/etc/OpenCL/vendors
```

To build a GPU backend from source, install its development libraries/toolkit and rebuild:

```bash
make clean
make -j2 gpu=1       # OpenCL
# Or: make clean && make -j2 cuda=1
./protlocus --gpu genome.fa proteins.fa > gpu.paf
```

Source builds use the name `protlocus` for all backends; rebuild after `make clean` when switching backends. See the [GPU guide](docs/GPU_zh.md) for details.

## Documentation and interpretation

- [Usage guide](docs/USAGE_zh.md) and [full manual](docs/MANUAL_zh.md) (Chinese).
- [GPU guide](docs/GPU_zh.md) and [technical notes](docs/TECHNICAL_NOTES.md).
- Command help: `protlocus --help`, `protlocus-recheck -h`, and `protlocus_hmm_pipeline.sh -h`.
- Source manual page: `man ./protlocus.1`.

HMM-guided searches detect genes with matching protein-family evidence. On intron-rich genomes, use the resulting models as domain-anchored candidates or annotation hints; inspect and validate them before treating them as complete gene annotations.

## License and citation

ProtLocus is MIT licensed; see [LICENSE.txt](LICENSE.txt). It derives from miniprot and retains the upstream copyright notice. Example reference data and their source records are described in [the examples README](examples/README.md).


