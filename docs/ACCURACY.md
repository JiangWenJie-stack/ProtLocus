# Accuracy experiment guide

ProtLocus is an experimental, standalone derivative of miniprot 0.18-r281. The ordinary command line keeps the upstream mapping policy. `--accuracy` enables sensitivity and model-selection changes that should be evaluated on truth data before production use.

## Build and test

```sh
make -j
make check
```

`make check` covers 16/32-bit DP equivalence, overflow fallback, ranking, microexon detection, the bundled DPP3 output checksum, profiling, repetitive-seed rescue, trusted-anchor rescue, resource limits, splice conversion, and GFF3 evaluation.

## Accuracy mode

```sh
./protlocus -Iut16 --accuracy --profile --gff genome.fa proteins.fa > predictions.gff
```

Accuracy mode enables:

1. **Repetitive-seed rescue.** Posting lists rejected by `-c` are sampled uniformly and deterministically. Baseline anchors are never displaced. `--acc-repeat-cap` limits each list and `--acc-anchor-budget` limits all added anchors for one query.
2. **Per-sequence chaining.** Coarse anchors are partitioned by contig and strand before pre-chaining and chaining. A chain cannot cross a virtual-sequence boundary.
3. **Relaxed candidate retention.** `--acc-pri-ratio` and `--acc-best-n` retain more distant or paralogous candidates until DP can distinguish them.
4. **Trusted-anchor fallback.** A refined chain with no tight three-anchor block receives bounded full-region DP instead of being discarded. `--acc-rescue-cells` caps nucleotide-by-amino-acid DP cells per candidate.
5. **Suspicious-model refinement.** Models with a noncanonical junction, frameshift, internal stop, or short query insertion are realigned with bounded full-region DP. Replacements are ordered by composite rank, intron-neutral alignment score, then spliced-DP score. `--acc-microexon` defines the short-insertion trigger.
6. **Composite ranking.** Raw `AS` and `ms` scores remain intact. Candidate ordering uses coverage, complete ends, splice classes, frameshifts, and internal stops. The bonus portion (coverage/end/splice bonuses) is capped at `--acc-rerank-margin` above raw `dp_max` so composite ranking can only break near-ties, never override a much better raw-scoring candidate; penalties (frameshift/stop/noncanonical) are not capped. The resulting score is emitted as `ra:i` in PAF records.

The composite score is

```text
dp_max
+ floor(cov_bonus * aligned_query / query_length)
+ end_bonus for an aligned start with an adjacent start codon
+ end_bonus for an aligned end with an adjacent stop codon
+ splice_bonus for each GT-AG junction
+ ceil(splice_bonus / 2) for each GC-AG or AT-AC junction
- noncan_penalty for every other junction
- fs_penalty * frameshift_count
- stop_penalty * internal_stop_count
```

Defaults are intentionally conservative and are data-set hypotheses, not universal biological constants. Tune them on held-out genes.

## Ablation options

| Option | Effect |
| --- | --- |
| `--acc-no-repeat` | Disable repetitive posting rescue |
| `--acc-no-vid-chain` | Disable contig/strand partitioned chaining |
| `--acc-no-rescue` | Disable no-trusted-anchor full-DP fallback |
| `--acc-no-junc-refine` | Disable suspicious-model second-pass DP |
| `--acc-no-rerank` | Disable composite ranking and `ra:i` |
| `--acc-repeat-cap=NUM` | Maximum anchors sampled from one filtered posting list |
| `--acc-anchor-budget=NUM` | Maximum added repetitive anchors per query |
| `--acc-rescue-cells=NUM` | Maximum cells for one full-DP rescue or refinement |
| `--acc-pri-ratio=FLOAT` | Candidate score ratio retained before DP |
| `--acc-best-n=NUM` | Maximum retained secondary candidates |
| `--acc-microexon=NUM` | Maximum short query insertion that triggers refinement |
| `--acc-cov-bonus=NUM` | Full-query coverage ranking bonus |
| `--acc-splice-bonus=NUM` | GT-AG ranking bonus per junction |
| `--acc-noncan-penalty=NUM` | Noncanonical junction ranking penalty |
| `--acc-fs-penalty=NUM` | Ranking penalty per frameshift |
| `--acc-stop-penalty=NUM` | Ranking penalty per internal stop |
| `--acc-end-bonus=NUM` | Ranking bonus for each complete protein end |
| `--acc-rerank-margin=NUM` | Cap on total ranking bonus above raw `dp_max` |

## Profiling

`--profile` writes one `[M::profile]` record per query file to stderr. Timing fields are accumulated thread CPU-stage time and may exceed wall time. Workload fields count query seeds, baseline plus rescued anchors, coarse chains, refinement candidates, exact anchors, DP candidates, fallback DP, suspicious-model realignment, and output records.

```sh
./protlocus --accuracy --profile genome.mpi proteins.fa > out.paf 2> profile.log
grep '^\[M::profile\]' profile.log
```

`km_max_blocks`/`km_max_capacity` report the worst per-thread kalloc free-list length/arena size observed at a mini-batch boundary (`-K`, default 2M query bases per batch). Each worker's arena is destroyed only at batch boundaries, so many small candidate regions accumulated by `--accuracy` (repeat rescue, anchor-rescue fallback) within one batch can fragment the free list; both `kmalloc`/`kfree` in `kalloc.c` are O(free-list length), so a growing `km_max_blocks` across a run is consistent with `align_ext_s` scaling worse than linearly in candidate count. `--no-kalloc` bypasses the arena (falls back to libc `malloc`/`free`) and is a quick way to A/B test whether free-list fragmentation, rather than the DP itself, drives the slowdown; a smaller `-K` forces more frequent arena resets and has the same diagnostic effect.

## Splice evidence

Use RNA junction evidence whenever it is available. Convert STAR `SJ.out.tab`:

```sh
python3 tools/junctions_to_spsc.py --format star SJ.out.tab > splice.spsc
./protlocus -I --accuracy -j2 --spsc=splice.spsc genome.mpi proteins.fa > out.paf
```

BED6 input is interpreted as a 0-based half-open intron interval:

```sh
python3 tools/junctions_to_spsc.py --format bed6 introns.bed > splice.spsc
```

The converter emits `contig offset strand D|A score`. STAR support is transformed with `round(scale * log2(1 + unique + multi_weight * multi))` plus an annotation bonus, then capped.

## Evaluation

Truth is a tab-separated file with one protein query per row. CDS intervals are 1-based closed coordinates and should use the same stop-codon convention as ProtLocus GFF3.

```text
query	contig	strand	cds
protein1	chr1	+	101-150,201-260
protein2	chr7	-	900-980,1101-1200
```

```sh
python3 tools/evaluate_accuracy.py \
  --truth truth.tsv \
  --prediction predictions.gff
```

Reported metrics include top-1 and oracle locus recall, exact CDS endpoints, exact exon models, and junction precision/recall. `--boundary-tolerance` permits near-boundary matching.

Run a reproducible baseline/accuracy comparison and all feature ablations with:

```sh
python3 tools/run_accuracy_benchmark.py \
  --truth truth.tsv --spsc splice.spsc --ablations \
  --threads 16 genome.mpi proteins.fa
```

Without truth CDS annotations, `tools/paf_metrics.py` gives ground-truth-free proxy
metrics (recall, coverage, completeness, frameshifts/stops, `ra:i`) for comparing
arbitrary PAF files, e.g. a baseline vs several tightened/ablated `--accuracy` runs:

```sh
python3 tools/paf_metrics.py summary genome_baseline.paf genome_accuracy.paf
python3 tools/paf_metrics.py compare --baseline genome_baseline.paf \
  genome_accuracy.paf genome_accuracy_tight.paf
```

`tools/filter_paf.py` post-filters an existing PAF by coverage/completeness/`ra:i`/
frameshift-stop count, to drop low-quality fragmentary hits without rerunning miniprot:

```sh
python3 tools/filter_paf.py --min-cov 0.5 --min-ra 0 genome_accuracy.paf > genome_accuracy.filtered.paf
```

The runner writes each GFF3 result, stderr log, command, profile counters, metrics, and `summary.json` under `accuracy-benchmark/`.

## Benchmark design

Evaluate held-out sets separately for near and distant homologs, repetitive families, close paralogs, single- and multi-exon genes, microexons, long introns, assembly gaps, processed pseudogenes, and known frameshifts. Report locus recall together with exact junction and CDS-model precision; aggregate alignment identity alone hides the errors this mode targets.

## Correctness fixes

The standalone tree also fixes the splice-score length assertion, SSE2 32-bit comparison and lane insertion, and automatically selects 32-bit global DP when the theoretical substitution score approaches the signed 16-bit limit. Long high-scoring proteins may therefore intentionally differ from upstream output.

## Limitations

This mode does not infer orthology or synteny from a single protein/genome pair. Resolve close paralogs with reciprocal searches and gene-neighborhood evidence after mapping. Suspicious-model refinement is bounded local full DP around an existing chain, not an unconstrained whole-genome microexon search. RNA-derived `--spsc` evidence remains the strongest available way to improve ambiguous splice boundaries.