> Historical technical notes. See [the project README](../README.md) for current setup and capabilities. Benchmark results below are inherited reports, not new validation. Commands run from the repository root.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](../LICENSE.txt)
[![Based on miniprot 0.18-r281](https://img.shields.io/badge/based%20on-miniprot%200.18--r281-lightgrey)](https://github.com/lh3/miniprot)

# ProtLocus

**ProtLocus** is an unofficial, standalone derivative of
[lh3/miniprot](https://github.com/lh3/miniprot) — a splice-aware
protein-to-genome aligner. It keeps the upstream command line and default
behavior unchanged, and adds:

- an **opt-in `--accuracy` mode** that trades some speed for higher recall on
  repetitive/paralog-rich genomes;
- **HMMER3 profile queries** (`query.hmm`, single profile or a whole library),
  mapped through the same pipeline with position-specific Viterbi scoring;
- a **built-in prescreen for whole-library HMM scans (`--prescreen --tier=...`,
  3.0.0)** that generates candidate loci in-process from a six-frame ORF
  spaced-k-mer index (MMseqs2-style prefilter + banded alignment + E-value),
  replacing the external mmseqs + `--cand-table` step;
- an **opt-in OpenCL GPU backend (`--gpu`)** that runs the terminal-extension
  and gap DP on the device and, since 2.0.0, is faster than multi-threaded CPU
  SIMD on large library scans;
- an **opt-in CUDA GPU coarse-chain prefilter (`--gpu-prefilter`, 4.0.0,
  requires `make cuda=1`)** that scores candidate chains with a batched
  gapless X-drop kernel and keeps only the top-K (or an adaptive K) by score.

> ProtLocus is not affiliated with the original miniprot authors. If you don't pass
> `--accuracy`, `--gpu` or a `.hmm` query, output is intended to match upstream miniprot.

The complete Chinese user manual is [MANUAL_zh.md](MANUAL_zh.md).

## Table of Contents

- [Introduction](#intro)
- [What's different from upstream miniprot](#whats-different)
- [Installation](#install)
  - [Source distribution](#install-release)
  - [From source](#install-source)
  - [GPU backend](#install-gpu)
- [Quick start](#quickstart)
- [HMMER3 profile queries](#hmm)
- [Accuracy mode](#accuracy)
- [Benchmarks](#benchmarks)
- [Further reading](#further-reading)
- [Citing](#cite)
- [License and provenance](#license)

## <a name="intro"></a>Introduction

ProtLocus aligns a protein sequence against a genome with affine gap penalty,
splicing, and frameshift support. It is primarily intended for annotating
protein-coding genes in a new species using known genes from other species,
and is similar in purpose to GeneWise/Exonerate but maps proteins to whole
genomes and is much faster at the residue-alignment step. The upstream mapping policy is not
tuned for distant-homolog search by default; `--accuracy` in this fork exists
to push recall further on repetitive/paralog-rich genomes at some speed cost
(see below).

## <a name="whats-different"></a>What's different from upstream miniprot

| Area | Upstream `miniprot` | ProtLocus |
| --- | --- | --- |
| Default command line | seed → chain → refine → DP → filter | unchanged (intent-compatible) |
| `--accuracy` mode | n/a | opt-in: repetitive-seed rescue, per-contig/strand chaining, relaxed candidate retention, no-trusted-anchor DP rescue, suspicious-junction re-refinement, composite ranking, `--profile` stage timing (see [ACCURACY.md](ACCURACY.md)) |
| Query type | protein FASTA | protein FASTA **or HMMER3 profiles** (`.hmm`, one or many); profile consensus + emission-variant seeds, position-specific Viterbi scoring, batched per query |
| CPU extension DP | 128-bit SSE2/NEON | automatic 256-bit AVX2 dispatch on supported x86_64 CPUs, with the 128-bit path retained as fallback |
| GPU backend | n/a | opt-in OpenCL (`make gpu=1`, `--gpu`): device-side terminal-extension pipeline (persistent multi-queue executor, warp-synchronous kernel on NVIDIA) + batched gap DP + optional HMM profile DP (`MP_HMM_GPU=1`). CUDA (`make cuda=1`) covers the gap DP only. |
| 32-bit DP | has a documented correctness edge case | fixed for long alignments |
| CIGAR backtrack | trailing deletion length counted in nt | fixed (counted in codons; upstream latent bug) |

`--accuracy` is designed to be safe to turn on for annotation/recall-sensitive
work and safe to leave off for anything that needs upstream-identical output.
See [ACCURACY.md](ACCURACY.md) for the full mechanism list, every `--acc-*`
tuning knob, and how to reproduce the ablation studies.

## <a name="install"></a>Installation

### <a name="install-release"></a>Source distribution

This repository contains source only. Build the programs as below; no prebuilt release or Conda package is supplied here. Programs report `0.1.0`; see the project README for the release workflow.

### <a name="install-source"></a>From source

Requires a C99 compiler, `zlib`, and an x86_64 (SSE2) or ARM (NEON) CPU.
On x86_64, AVX2 is detected at run time and used automatically for 16-bit
terminal-extension DP; no extra build flag is needed, and the same binary
continues to run on SSE2-only CPUs.

```sh
make -j              # dynamic CPU build
make -j static=1     # fully static CPU build (needs static libz/libc)
make check           # builds + runs the C and Python test suites

./protlocus test/DPP3-hs.gen.fa.gz test/DPP3-mm.pep.fa.gz > aln.paf
```

For some toolchains, the default `gcc-4.8.5` on CentOS 7 may compile a binary
that is very slow on certain sequences; gcc-10+ has more stable performance.

### <a name="install-gpu"></a>GPU backend

```sh
# OpenCL backend (recommended GPU build). No OpenCL headers are needed: only
# libOpenCL.so at link time and a vendor ICD at run time.
make clean && make -j gpu=1
./protlocus --gpu --gpu-device=0 -t56 ref.mpi library.hmm > out.paf
make gpu=1 check      # runs the GPU-vs-CPU test suite if a device is present

# CUDA backend (gap DP only; no extension pipeline / HMM GPU scoring)
make clean && make -j cuda=1 NVCC=/path/to/nvcc CUDA_HOME=/path/to/cuda
```

`--gpu` is opt-in at run time. Everything the device cannot take (oversized
requests, non-matching scoring options, OpenCL failures) falls back to the CPU
SIMD kernels automatically. Extension and HMM scores are bit-for-bit those of
the CPU kernels; the CIGAR-producing gap DP returns the same score but may pick
a different equal-score path (see [Benchmarks](#benchmarks)). What runs on the
device:

1. **Terminal-extension DP** (`ns_global_gs16b` with `NS_F_EXT_LEFT/RIGHT`):
   every extension request of a query is queued, input preparation runs on the
   device, and a persistent executor with `NS_EXT_GPU_QUEUES` (default 8)
   command queues pools requests from all mapping threads into shared arenas,
   advancing them a few rows per launch so the device always holds many
   queries at once. On NVIDIA devices a warp-synchronous kernel (inline PTX)
   is built; if that fails, or with `NS_EXT_GPU_NO_WARP=1`, a portable
   work-group kernel is used.
2. **Gap DP** between anchors (`ns_gpu_global`) when the problem has at least
   `--gpu-min-cells` cells (default 1M).
3. **HMM profile Viterbi DP** when `MP_HMM_GPU=1` is set and the profile has
   at most 1023 match states (off by default: it competes with the extension
   pipeline for the device and was a net loss in our tests).

If you link against a Conda `libOpenCL.so` that has no ICD configured, the
program prints `GPU unavailable (clGetPlatformIDs failed ... -1001)` and runs
on the CPU; build with the system compiler/linker or point `OCL_ICD_VENDORS`
at your vendor ICD.

## <a name="quickstart"></a>Quick start

```sh
# build the genome index once (recommended for anything but a one-off run)
./protlocus -t16 -d genome.mpi genome.fna

# ordinary mapping (same intent as upstream miniprot)
./protlocus -Iut16 --gff genome.mpi protein.faa > aln.gff

# higher-recall mode for repetitive/paralog-rich genomes
./protlocus -Iut16 --accuracy --profile --gff genome.mpi protein.faa > aln.gff
```

`-I` lets ProtLocus pick a max intron size from the genome size; drop it and
use `-G` if you want to set it explicitly. See `man ./protlocus.1` or
`./protlocus --help` for the full option list.

ProtLocus can optionally take splice scores computed with
[minisplice](https://github.com/lh3/minisplice):
```sh
minisplice predict -t16 -c vi2-7k.kan.cali vi2-7k.kan genome.fa.gz > score.tsv
./protlocus -Iut16 --gff -j2 --spsc=score.tsv genome.fa.gz proteins.faa > align.gff
```

HMMER3 profiles can also be supplied as a query file ending in `.hmm`; see the
next section.

## <a name="hmm"></a>HMMER3 profile queries

A query file whose name ends in `.hmm` (lower case) is parsed as HMMER3 text
profiles. The file may hold a single profile or a whole library (e.g.
`cat *.hmm > library.hmm`); every profile is mapped as an independent query
using the same multi-threaded mini-batch pipeline as FASTA input, and `.hmm`
and FASTA query files can be mixed on one command line:

```sh
./protlocus -t16 genome.mpi family.hmm > family.paf
cat hmm_library/*.hmm > library.hmm
./protlocus-gpu -t56 --gpu genome.mpi library.hmm > library.paf
```

How it works:

- **Parsing**: `NAME`, `LENG`, `ALPH amino`, the `HMM` header line, match
  emissions and transitions per position, terminated by `//`. `COMPO` is
  skipped. The PAF query name is `NAME` and the query length is `LENG`.
- **Seeding**: the consensus (highest-emission residue per position) is
  sketched like a protein, plus up to 64 variant sequences that substitute
  near-tie residues (second-best emission within 6 score units), so seeds are
  not lost at ambiguous columns.
- **Scoring**: candidate regions are scored with a position-specific Viterbi
  DP over match/insert/delete states on the translated region; the Viterbi
  score is reported in `AS:i`. All candidate regions of one query are scored
  in a single batched pass (CPU by default; `MP_HMM_GPU=1` uses one OpenCL
  work-group per region for profiles with <= 1023 match states, longer
  profiles fall back to the CPU).
- **Traceback/output**: the residue alignment and CIGAR use the consensus
  sequence, so `--gff`, `--aln`, `--trans` and `-u` work unchanged.

Validate recall on a family-specific truth set before production use: the
profile score is used for ranking, but seeding still depends on the consensus
being recognisable in the genome.

### Whole-library scans (3.0)

For a full Pfam-A library against a genome, the genome-wide seed/chain stage is
the bottleneck. `--prescreen` replaces it with a built-in MMseqs2-style
profile-vs-ORF search: the genome is six-frame translated into stop-free ORFs
(>=30 aa), indexed by spaced 6-mers, and each profile is prefiltered
(similar-k-mer generation from the PSSM, double hit on one ORF diagonal),
aligned with a banded gapped Smith-Waterman and given a Karlin-Altschul
E-value. Only loci below the tier cutoff are refined and aligned.

```sh
./protlocus -t56 -k5 -M0 -d genome.sensitive.mpi genome.fna      # sensitive index, once per genome
./protlocus -t56 -u --compact --prescreen --tier=balanced genome.sensitive.mpi Pfam-A.hmm > hits.paf
```

`--tier` picks the prescreen E-value cutoff: `fast` E<=0.1, `balanced`
E<=100 (default), `sensitive` E<=10000 (`--cand-max-evalue=FLOAT` for any
other cutoff; `--ps-dump=FILE` writes the generated candidate table). The
prescreen itself costs ~10 s and 0.6 GB on a 12 Mb genome (`--ps-sens 7.5`;
lowering it to 5.7 cuts that to 2 s but loses 5 recall points). Hits are then
subjected to the `--compact` DP-score floor (AS>=15) plus the profile-score
floor `--acc-min-ps=60` (default for HMM queries; `--acc-min-ps=0` disables).
Yeast R64 vs Pfam-A 38 (30134 families, 56 threads, reference = hmmsearch
domain calls on the annotated proteome):

| tier | wall | peak RSS | candidate loci | precision | recall |
|---|---|---|---|---|---|
| `fast` (E<=0.1) | 36 s | 3.4 GB | 37 k | 0.988 | 0.873 |
| `balanced` (E<=100) | 1.4 min | 5.2 GB | 603 k | 0.956 | 0.923 |
| `sensitive` (E<=10000) | 10.5 min | 8.9 GB | 5.9 M | 0.936 | 0.965 |

#### External candidate table

`--cand-table=FILE` injects loci from any external search instead (TSV:
query, chrom, start0, end0, strand[, evalue]); `--prescreen` and
`--cand-table` are mutually exclusive. With an mmseqs2 table built once at
E<=10000, `--tier` maps to `fast` E<=100, `balanced` E<=1000, `sensitive`
E<=10000 (43 s P0.984 R0.871 / 2.5 min P0.966 R0.919 / 13.5 min P0.947
R0.959 on the same benchmark, excluding the mmseqs run itself):

```sh
esl-translate -l 30 genome.fna > orfs.faa
mmseqs convertmsa Pfam-A.seed.gz pfam_msa --identifier-field 0
mmseqs msa2profile pfam_msa pfam_prof --match-mode 1
mmseqs createdb orfs.faa orfsdb
mmseqs search pfam_prof orfsdb res tmp -s 7.5 -e 10000 --max-seqs 4000
mmseqs convertalis pfam_prof orfsdb res res.tsv --format-output query,target,evalue,bits
awk -F'\t' 'BEGIN{OFS="\t"}{print $2,$1,$3,$4}' res.tsv > res.swapped.tsv
python3 tools/mmseqs2cand.py --mmseqs res.swapped.tsv --orfs orfs.faa --strip-suffix '' -o cand.tsv
./protlocus -t56 -u --compact --tier=balanced --cand-table=cand.tsv genome.sensitive.mpi Pfam-A.hmm > hits.paf
```

## <a name="accuracy"></a>Accuracy mode

```sh
./protlocus -Iut16 --accuracy --profile --gff genome.fna proteins.faa > predictions.gff
```

On our 4-genome real-data benchmark (mouse/rat/frog/zebrafish, 1114 proteins,
56 threads), the shipped default configuration recovers every query mapped
by an uncapped, fully-exhaustive accuracy run (`oracle_only=0`, zero recall
regressions) while keeping a bounded terminal-extension retry budget for
speed. See [ACCURACY.md](ACCURACY.md) for:

- the six accuracy mechanisms and their `--acc-no-*` ablation switches,
- every `--acc-*` tuning knob and its default,
- the composite ranking formula and `--profile` stage-timing fields,
- RNA-junction (`--spsc`) conversion and truth-set evaluation tooling in
  [tools/](../tools/).

## <a name="benchmarks"></a>Benchmarks

Two write-ups document the accuracy/speed tradeoffs in more depth (in Chinese):

- [RESEARCH_PAPER.md](RESEARCH_PAPER.md) — recall gains from each
  `--accuracy` mechanism vs. upstream `miniprot`, and where the speed cost
  comes from.
- [SPEED_OPTIMIZATION_REPORT.md](SPEED_OPTIMIZATION_REPORT.md) — profiling
  and the specific optimizations shipped by default (e.g. skipping a
  redundant terminal-extension retry for no-trusted-anchor candidates cuts
  wall time ~20-36% with zero measured recall loss).

Headline numbers (`-t56`, 1114 proteins, 4 real genomes, median of 3 repeats):

| Mode | Aggregate wall time | Recall vs. uncapped oracle |
| --- | --- | --- |
| `--accuracy` (this build's default) | 65.2 s | 0 queries lost |
| `--accuracy`, uncapped/exhaustive | 158.3 s | reference |

In a separate paired three-run benchmark on the same four genomes, automatic
AVX2 dispatch reduced terminal-extension DP time by 12.4-16.6% per genome
(14.3% for the sum of medians) and aggregate end-to-end time by 14.6%, with
byte-identical PAF output. This optimization is automatic and does not change
the accuracy/speed policy selected by the command line.

### GPU vs CPU on a Swiss-Prot profile library (2.0.0)

Mouse genome GRCm39 (`.mpi` index), `--accuracy -I -u -t56`, Swiss-Prot
cluster-representative HMMER3 profiles sorted by cluster size, one NVIDIA L20
shared with other users, node load 70-125:

| Profiles | CPU wall | GPU wall (`--gpu`) | GPU device busy | Speed-up |
| --- | --- | --- | --- | --- |
| 1,000 | 25.6 s | 25.4 s | 17.8 s | 1.0x |
| 5,000 | 153.8 s | 120.6 s | 103.7 s | 1.28x |
| 83,230 (full library, 15 GB) | 2296 s / 105,946 CPU-s | 2206 s / 41,049 CPU-s | 1881 s | 1.04x wall, 2.58x CPU time |

GPU reruns are byte-identical. GPU vs CPU PAF differ in ~0.7% of lines, almost
all at the same locus with a different equal-score alignment path (tie-break
in the CIGAR-producing gap DP; the extension DP itself is bit-identical);
~0.04% of lines have shifted boundaries. Small inputs are CPU/GPU neutral
because the device is not saturated; at 5000+ profiles the (shared) device
becomes the bottleneck (`device_s` ≈ wall), which is why the full-library wall
time only improves 4% while CPU time drops 61% - a dedicated or faster GPU
scales further. See `MANUAL_zh.md` section 11 for details.

## <a name="further-reading"></a>Further reading

- [ACCURACY.md](ACCURACY.md) — accuracy mode reference
- `docs/MANUAL_zh.md` — complete user manual (中文)
- [RESEARCH_PAPER.md](RESEARCH_PAPER.md) — algorithm background and recall study (中文)
- [SPEED_OPTIMIZATION_REPORT.md](SPEED_OPTIMIZATION_REPORT.md) — profiling and speed optimizations (中文)
- [NEWS.md](NEWS.md) — release notes
- [protlocus.1](../protlocus.1) man page / `man ./protlocus.1`

Background on the underlying algorithm (indexing, chaining, splicing DP) is
unchanged from upstream miniprot; see the
[original paper](https://doi.org/10.1093/bioinformatics/btad014) or the
[upstream manpage](https://lh3.github.io/miniprot/miniprot.html).

## <a name="cite"></a>Citing

ProtLocus builds on miniprot. If you use it, please cite the original paper:

> Li, H. (2023) Protein-to-genome alignment with miniprot. *Bioinformatics*, **39**, btad014. [PMID: 36648328](https://pubmed.ncbi.nlm.nih.gov/36648328/)

Please note the published paper evaluated miniprot-0.7; this fork is based on
miniprot-0.18-r281 and adds the accuracy/GPU features described above, so
absolute numbers will differ.

## <a name="license"></a>License and provenance

MIT license, see [LICENSE.txt](../LICENSE.txt). This repository is a source
fork of [lh3/miniprot](https://github.com/lh3/miniprot) (Copyright (c)
2022- Dana-Farber Cancer Institute); the original copyright notice is
retained as required by the license. All `--accuracy`/GPU-backend additions
in this repository are separate, clearly opt-in code paths layered on top of
the unmodified upstream default behavior.

## Limitations

* The initial conditions of dynamic programming are not technically correct,
  which may result in suboptimal residue alignment in rare cases (inherited
  from upstream miniprot).
* `--accuracy` and `--gpu` are both opt-in and independently tunable/ablatable
  (see [ACCURACY.md](ACCURACY.md)); defaults are conservative but are
  data-set hypotheses, not universal biological constants — validate on your
  own truth data before production use.
* The GPU backend requires OpenCL (NVIDIA tested). It pays off on large
  inputs (thousands of queries); for small inputs it is neutral. With `--gpu`
  about 0.7% of PAF lines differ from the CPU build (equal-score alignment
  paths, see Benchmarks).
* The CUDA build covers the gap DP only and is kept for reference.
* HMM profile queries seed from the consensus; highly divergent families with
  no recognisable consensus k-mers can be missed even if the profile would
  score them.
