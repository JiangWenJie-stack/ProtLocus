# Development notes

## 0.1.0 — initial standalone source release preparation

The source project reports `0.1.0`. Public release assets are prepared by the GitHub release workflow. The features below were carried over from the supplied source snapshot. Older entries are retained as historical development notes.


Gene-model builder on top of HMM-library hits (`tools/protlocus_genemodel.py`):

 * Turns the stage-2 `final.paf` into GFF3 gene models (`gene/mRNA/exon/CDS`)
   plus protein FASTA, using only the genome and the library hits (stdlib
   Python >= 3.6, one process per contig via `-t`). Steps: (1) hits ->
   exon blocks with codon phase; hits overlapping on one strand are merged
   when frame/structure agree, otherwise the lower-significance one is
   dropped; hits with frameshifts/in-frame stops are skipped (`--allow-fs`).
   (2) adjacent evidence units are linked: first by direct fusion (same
   frame, no stop codon in between), otherwise by a DP over GT/GC..AG
   introns keeping the reading frame open (splice-site PWM, codon/dicodon
   log-odds, intron-length prior). A spliced link must beat the
   "two independent genes" baseline by `--link-min`. (3) both ends are
   extended by ORF: 3' to the first in-frame stop, 5' to the ATG with the
   best cumulative coding score (`--start-slack`); optional one-intron
   spliced end extension (auto: only when the genome has enough genuine
   introns).
 * Everything is self-trained on the genome's own hits: splice-site PWMs
   (seeded from a generic prior, refined by EM), intron-length distribution,
   codon and dicodon log-odds. Alignment-introns in compact genomes are
   mostly artifacts (yeast: 25 of 1280 distinct hit introns match the
   annotation), so too few high-confidence seeds means "no trainable
   introns": generic prior, conservative max intron (2 kb), hits with
   unsupported introns dropped, intron prior cost scaled by the observed
   intron density, spliced end extension off. `--no-introns` for prokaryotes
   (`--start-codons ATG,GTG,TTG` optional).
 * One-command use: `protlocus_hmm_pipeline.sh ... -G genome.fa` adds stage 3
   (`OUTPREFIX.genes.gff3/.faa`). `tools/evaluate_genemodel.py` compares a
   GFF3 with RefSeq/TAIR CDS (nt/exon/gene level, plus recall restricted to
   genes covered by hits).
 * Measured (Pfam-A hits, E<=1e-5, versus RefSeq CDS; defaults):
   S. cerevisiae: nt sens 0.84 / spec 0.99; 87.7% of predicted genes have a
   CDS chain identical to the reference, 69% of all reference genes
   recovered exactly (80.5% of those covered by a hit). E. coli
   (`--no-introns`): nt 0.89/0.98, 65% of predictions identical. A. thaliana
   (intron-rich): nt sens 0.55 / spec 0.81, but only ~22% of predicted
   genes are exact: most CDS lies outside Pfam domains, and missing 5'/3'
   and inter-domain exons are the dominant error (spliced end extension
   with the codon/PWM model gains ~+5pp nt sensitivity at ~-0.5pp
   specificity; more permissive settings trade precision away). Genes
   without any Pfam hit are not found. For intron-rich genomes treat the
   output as protein-domain-anchored gene skeletons / hints (e.g. for
   Augustus or as evidence next to an ab initio model), not final models.

Small-genome recall preset (prokaryotes, fungi and other compact genomes):

 * New `--small-genome=auto|on|off` (default auto) and `--small-genome-size=NUM`
   (default 50M). With `--prescreen`, when the genome total length is at or
   below the threshold (or `on` forces it), the mapper switches to
   `--ps-kmer=5 --acc-min-ps=0 --acc-min-score=0 --acc-min-pos-ratio=0`;
   any of these four flags given explicitly on the command line is kept.
   Weak hits are passed through for `protlocus-recheck` (E<=1e-5) to
   arbitrate. Attribution experiments on E. coli and yeast against a
   six-frame hmmsearch reference showed the stage-1 output floor
   (`--acc-min-ps=60`) was the single largest recall cost on small genomes
   (~5.7-5.8pp); this preset recovers +4.4-4.7pp domain recall at unchanged
   purity and minute-scale runtime (E. coli 0.7555 -> 0.8023, yeast
   0.7367 -> 0.7809; reference hmm6f@1e-5 0.9317/0.8854, hmm6f@1e-10
   0.7733/0.7174). Do not combine open thresholds with a loosened
   sensitive-tier prescreen: candidate explosion inflates the recheck
   Z (unique peptides) and makes the E-value cutoff stricter, reducing
   final recall.

 * Validated against the frozen benchmark arms (E. coli, Pfam-A): auto
   trigger output is byte-identical to the `k5_open` arm, `--small-genome=off`
   is byte-identical to the v5.0 baseline arm, and an explicit `--ps-kmer=6`
   overrides the preset k while keeping the open thresholds.

Release 5.0.0 (20 October 2026)
-------------------------------

Candidate + HMM recheck pipeline for whole-library scans: ProtLocus stays the
ultra-fast high-sensitivity candidate generator, and a new rescoring stage
gives every surviving hit a statistically calibrated significance. No change
to the aligner core; FASTA and single-profile behaviour is unchanged. New:

 * `protlocus-recheck` (recheck.c, `make protlocus-recheck`): native C
   recheck engine with no external dependency. It takes the PAF produced by
   an HMM-library run with `--trans` (each hit followed by a `##STA` line
   carrying the spliced, frameshift-skipped translation of the aligned
   genome region) and rescores every candidate peptide against its own
   family profile only, using an independent implementation of the plan-7
   statistical model (Eddy, PLoS Comput Biol 2011): multihit local profile
   configuration with occupancy-weighted entry, target-length N/C/J model,
   full Forward in log space, null1 geometric length model, a
   posterior-weighted null2 biased-composition correction (ByExpectation,
   from a full Backward pass; -n disables), and the exponential-tail
   P-value from the profile's own `STATS LOCAL FORWARD tau lambda`
   calibration line. Filters on E = Z * P (default 1e-5);
   -Z defaults to the number of unique candidate peptides (pass the species'
   six-frame ORF count for a genome-wide yardstick). Output PAF carries
   `fB:f:` (Forward bits) and `fE:f:` (E-value); `-a` keeps rejected lines
   with `hp:i:0`; `-T` writes a per-candidate TSV (peptide length, in-frame
   stop count, Forward stats). Differences from hmmsearch's numbers:
   sequence-level Forward score and null2 (the candidate peptide is treated
   as one envelope, not per-domain envelopes); decoy-FDR runs are the
   ground-truth calibration check.
 * Compared with a genome-wide six-frame hmmsearch, the recheck skips the
   all-profiles-times-all-ORFs cross product (each peptide is scored against
   exactly one profile, ~1/20000 of the compute on Pfam-A) and, because
   `##STA` peptides are spliced, intron-containing domains that a
   single-frame ORF scan truncates are rescored intact.
 * `tools/protlocus_hmm_pipeline.sh`: one-command driver running stage 1
   (`protlocus -I -u --accuracy|--compact --prescreen --tier=... --outs=0
   -N 1000 --outn=1000 --acc-pri-ratio=0.1 --trans`, the frozen benchmark
   operating point) and stage 2 (`protlocus-recheck`), with /usr/bin/time -v
   resource logs per stage. `-R hmmer` switches stage 2 to
   `tools/hmmrecheck.py`, which drives HMMER's hmmfetch/hmmsearch per family
   (best domain i-Evalue, same 1e-5 yardstick) -- kept for cross-validating
   the native engine against hmmsearch.
 * Motivation (decoy-FDR study on 7 genomes, see release notes): raw
   balanced-tier hit-level FDR grows with genome repeat content (16%-77%),
   while AS-score calibration alone collapses controlled recall on large
   genomes (human 0.646 -> 0.119 at FDR<=5%). Rechecking candidates with the
   family profile restores hmmsearch-grade precision while keeping the
   candidate generator's speed and spliced-domain recall (yeast, Pfam-A,
   hmmsearch-based recheck: recall 0.746 -> 0.736, hit same-family purity
   0.25 -> 0.89, whole pipeline ~3 min on 56 threads).

Release 4.0.0 (25 September 2026)
---------------------------------

GPU-accelerated coarse-chain prefilter, ported from the pre-3.0.0 prototype
and reworked as a per-query call so it composes with the 3.0.0 pipeline
(prescreen/cand-table/PSSM rerank paths are unaffected; the prefilter runs
only on the plain FASTA seed/chain path). New:

 * `--gpu-prefilter`: after coarse chaining, score each candidate's anchors
   with a batched GPU gapless X-drop kernel (CUDA only; requires
   `make cuda=1`) and keep only the top-K chains by gapless score, with a
   per-contig/strand diversity floor so alternate loci are not all dropped.
 * `--gpu-prefilter-topk=NUM` (default 32): K, the number of chains kept
   per query.
 * `--gpu-prefilter-diversity=NUM` (default 1): minimum surviving chains per
   distinct target/strand group.
 * `--gpu-prefilter-xdrop=NUM` (default 40): X-drop threshold for the GPU
   gapless extension.
 * `--gpu-prefilter-adaptive`: instead of a fixed K, pick K in {1,2,4,8,K}
   per query from the score margin between the best and runner-up chain.
 * This is a simplified per-query re-implementation of the original
   cross-query-batched prefilter design; it is not wired into the
   `worker_coarse_for`/`worker_finish_for` split of the pre-3.0.0 prototype,
   so throughput characteristics differ from the earlier benchmark numbers.
 * Builds without `cuda=1` (default CPU-only, or `gpu=1` OpenCL) accept the
   flags but `ns_gpu_gapless_supported()` returns false, so
   `--gpu-prefilter` is disabled with a warning and mapping proceeds
   unfiltered.

Release 3.0.0 (23 September 2026)
---------------------------------

Candidate-table (injection) mode for whole-library HMM scans. FASTA and
single-profile behaviour is unchanged. New and changed:

 * `--prescreen`: built-in replacement for the mmseqs step. The genome is
   six-frame translated into stop-free ORFs (>=30 aa, `--ps-min-orf`) and
   indexed by spaced 6-mers (`--ps-kmer`); each profile then goes through an
   MMseqs2-style prefilter (similar k-mers from the PSSM with a sensitivity
   threshold `--ps-sens` 7.5, two hits on one ORF diagonal, ungapped score
   >=15, top `--ps-max-seqs` 4000 ORFs), a banded gapped alignment
   (gap 11/1, `--ps-band` 32) and a Karlin-Altschul E-value with the same
   BLOSUM62 parameters MMseqs2 uses. Loci with E <= tier cutoff are injected
   exactly like a `--cand-table`; `--ps-dump=FILE` writes that table.
   Prescreen E-values are on their own scale, so `--tier` maps to
   E<=0.1/100/10000 (fast/balanced/sensitive, default balanced). Yeast vs
   Pfam-A 38 (56 threads, `--compact`), whole run including the ~10 s
   prescreen: fast 36 s P0.988 R0.873; balanced 1.4 min P0.956 R0.923;
   sensitive 10.5 min P0.936 R0.965 -- each tier matches or beats the
   mmseqs-table tier below without the external mmseqs run. Mutually
   exclusive with `--cand-table`.
 * `--cand-table=FILE`: external candidate loci (query, chrom, start0, end0,
   strand[, evalue]) replace the genome-wide seed/chain stage; only listed
   loci are refined and aligned, unlisted queries get no hit. Tables come
   from a fast profile-vs-ORF mmseqs2 search via `tools/mmseqs2cand.py`.
 * `--tier=fast|balanced|sensitive` (E<=100/1000/10000) and
   `--cand-max-evalue=FLOAT` filter the table by its 6th (E-value) column, so
   one E<=10000 table serves every operating point. Yeast vs Pfam-A 38
   (56 threads, `--compact`): fast 43 s P0.984 R0.871; balanced 2.5 min
   P0.966 R0.919; sensitive 13.5 min P0.947 R0.959.
 * `--acc-min-ps=NUM` (default 60): profile-score floor for HMM hits applied
   on top of the DP-score floor; a no-op for protein queries; 0 disables.
 * `--thr-table`/`--thr-fallback`/`--thr-use-ps` (per-family decoy tables)
   remain available but are no longer part of the recommended protocol:
   the global AS>=15 + ps>=60 floors dominate every decoy-table operating
   point measured.
 * Makefile: `hmm.o` now depends on the headers (a stale object after an
   option-struct change used to crash right after the candidate table load).

Release 2.0.0 (14 September 2026)
---------------------------------

Second major release of `ProtLocus`. Ordinary FASTA command-line behavior is
unchanged from 1.0.0 (and from upstream miniprot 0.18-r281); the DPP3 smoke
test output is byte-identical to 1.0.0. New and changed:

 * HMMER3 profile queries. A query file ending in `.hmm` is parsed as one or
   many HMMER3 profiles (e.g. `cat *.hmm > library.hmm`) and every profile is
   mapped as an independent query through the same multi-threaded mini-batch
   pipeline as FASTA input. Seeds come from the profile consensus plus
   near-tie emission variants; candidate regions are scored with a
   position-specific Viterbi DP whose score is reported in `AS:i`. The PAF
   query name/length are the profile `NAME`/`LENG`. All output formats
   (`--gff`, `--aln`, `--trans`, `-u`) work with profile queries.
 * Batched HMM profile scoring: all candidate regions of a query are scored in
   one pass (CPU by default; `MP_HMM_GPU=1` moves the profile DP to an OpenCL
   work-group per region when the profile has <= 1023 match states).
 * GPU terminal-extension pipeline (`--gpu`, OpenCL). The 16-bit x-drop
   extension DP that dominates `--accuracy` run time (~74% of CPU time on a
   200-profile mouse benchmark) is now executed on the device: input
   preparation runs on the GPU, requests from all mapping threads are pooled
   into shared arenas and advanced by a persistent multi-queue executor
   (`NS_EXT_GPU_QUEUES`, default 8), and on NVIDIA devices a warp-synchronous
   kernel (inline PTX, automatic fallback) is used. Results equal the CPU
   kernel bit-for-bit; PAF output is byte-identical across GPU reruns.
 * With this pipeline the GPU build is now faster than CPU SIMD on large
   library scans: on the mouse genome (GRCm39) with 56 threads, 5000
   Swiss-Prot cluster profiles took 120.6 s (GPU) vs 153.8 s (CPU), and the
   full 83,230-profile library (15 GB) took 2206 s / 41,049 CPU-s (GPU) vs
   2296 s / 105,946 CPU-s (CPU) on a heavily shared node where the shared L20
   was the bottleneck (device busy 85% of wall). Small inputs (<= ~1000
   queries) are CPU/GPU neutral. See MANUAL_zh.md for the full table.
 * Bug fix (affects all backends including upstream miniprot): the CIGAR
   backtrack pushed the trailing deletion with its length in nucleotides
   instead of codons, so a sub-alignment that starts with a deletion
   over-counted its reference span by 2x and tripped the `mp_extra_cal`
   assertion. Only the GPU tie-break path produced such alignments in
   practice, which is why it surfaced during the full-library scan.
 * Bug fix: the OpenCL HMM kernel was launched with 5 of 6 `__local`
   arguments and silently fell back to the CPU in 1.0.0.
 * Build: `make static=1` produces a fully static CPU executable; the
   non-OpenCL build of `nasw-ext-gpu.c` compiles again; the Makefile OBJS
   line no longer starts with a tab.
 * New diagnostics: `NS_EXT_GPU_DEBUG=1|2|3` (pipeline statistics / per-launch
   log / device-side input verification), `NS_EXT_GPU_STRICT=1` (abort instead
   of silent CPU fallback), `NS_EXT_GPU_NO_WARP=1`, `MP_HMM_GPU_DEBUG=1`,
   `NS_GPU_DEBUG_CHECK=1` (CIGAR span check with a full dump on mismatch).
 * Known difference: with `--gpu`, ~0.7% of PAF lines differ from the CPU
   build, almost all at the same locus with a different equal-score
   alignment path inside (`cg`/`cs`/`ms`/`np`, columns 10-11); ~0.04% of
   lines have shifted boundaries and 1 of 83,230 profiles was hit by the CPU
   build only. The terminal-extension results themselves are bit-identical;
   the differences come from tie-breaking in the CIGAR-producing gap DP.
   GPU reruns are byte-identical.
 * The CUDA backend (`make cuda=1`) only covers the batched gap DP; the
   extension pipeline and HMM GPU scoring require OpenCL.

(2.0.0: based on upstream 0.18-r281)



Release 1.0.0 (ProtLocus, first public release)
--------------------------------------------------------

First public release of `ProtLocus`, a standalone derivative of
miniprot 0.18-r281 (below). Ordinary command-line behavior is unchanged from
upstream. New, opt-in additions:

 * `--accuracy` mode: repetitive-seed rescue, per-contig/strand-isolated
   chaining, relaxed candidate retention, bounded no-trusted-anchor DP
   rescue, suspicious-junction re-refinement, composite ranking, and
   `--profile` stage timing. See ACCURACY.md.
 * Bounded terminal-extension retry budget for `--accuracy` (default
   `--acc-end-retry=89`, `--acc-end-ext=10000`): keeps every query recovered
   by an uncapped/exhaustive run while cutting aggregate wall time roughly in
   half on a 4-genome/1114-protein benchmark.
 * Default-on skip of a redundant terminal-extension retry for
   no-trusted-anchor ("rescued") candidates: ~20-36% additional wall-time
   reduction with zero measured recall regression on the same benchmark.
 * Automatic AVX2 acceleration for 16-bit terminal-extension DP on supported
   x86_64 CPUs, with the existing SSE2/NEON implementation retained as the
   runtime fallback. Paired three-run tests reduced extension-DP time by
   12.4-16.6% across four real genomes with byte-identical PAF output.
 * Optional, experimental GPU backend for the batched residue-alignment DP:
   OpenCL (`make gpu=1`) or CUDA (`make cuda=1`), `--gpu`. Correct
   (byte-identical to CPU) but currently slower than CPU SIMD on realistic
   workloads; shipped as opt-in, not built by the conda package.
 * Fix for a 32-bit DP correctness edge case in long alignments.

See RESEARCH_PAPER.md and SPEED_OPTIMIZATION_REPORT.md for the underlying
studies, and README.md for installation/usage.

(1.0.0: based on upstream 0.18-r281)



Release 0.18-r281 (11 July 2025)
--------------------------------

Older miniprot failed to count in-frame stop codons in deletions. This release
fixes the issue (#75).

(0.18: 11 July 2025, r281)



Release 0.17-r279 (15 June 2025)
--------------------------------

Notable changes:

 * Bugfix: older version may output overlapping exons with `--spsc`.

(0.17: 15 June 2025, r279)



Release 0.16-r273 (22 May 2025)
-------------------------------

Notable changes:

 * Improvement: `--spsc` now works with and optimized for minisplice output.
   Note that depending on scoring in `--spsc`, miniprot may generate 1bp
   introns in extremely rare cases. This happens to 3 out of 4.6 million
   aligned introns with zebrafish minisplice scores.

This version produces alignment identical to v0.13 unless `--spsc` is used.

(0.16: 22 May 2025, r273)



Release 0.15-r270 (18 April 2025)
---------------------------------

Notable change:

 * Improvement: better alignment in tandem gene duplications (#35). This
   slightly improves overall junction accuracy on all test datasets in the
   miniprot paper. Thank the NCBI EGAPx team for providing test cases and
   explaining the relevance in gene annotation.

(0.15: 18 April 2025, r270)



Release 0.14-r265 (7 March 2025)
--------------------------------

Notable changes:

 * Bug fix: support chromosomes longer than 2Gbp (#59)

 * EXPERIMENTAL feature: read splice scores from a file specified by `--spsc`
   and consider the scores during residue alignment. The feature makes it
   possible to apply advanced splice models and to improve miniprot alignment.

 * Improvement: documented the C APIs and added an example program on using the
   APIs (#69).

This version produces alignment identical to v0.13, except for long chromosomes
or when the new feature is used.

(0.14: 7 March 2025, r265)



Release 0.13-r248 (6 March 2024)
--------------------------------

Notable changes:

 * New feature: added option -T to specify a non-standard NCBI translation
   table (#56 and #57). As this is an indexing option, the binary index format
   has to be changed accordingly. Miniprot will reject indices built with
   previous versions.

 * Improvement: properly handle reference deletions involving in-frame stop
   codons (#58). Older versions would not penalize these stop codons. This
   change also improves junction accuracy especially for distant homologs.

 * Bug fix: in the GFF3 output, CDS now includes stop codons (#55). Note the in
   GTF, CDS excludes stop codons.

 * Bug fix: suppress an extra amino acid in the --trans or --aln output (#47).
   In rare cases, this may lead to memory violation.

(0.13: 6 March 2024, r248)



Release 0.12-r237 (24 June 2023)
--------------------------------

Notable changes:

 * New feature: added option --no-cs to disable the cs tag. This tag is not as
   useful as the cs tag for nucleotide alignment because it does not encode the
   matching amino acids.

 * New feature: output the number of frameshifts and in-frame stop codons in
   the PAF output. It is non-trivial to parse in-frame stop codons.

 * Bugfix: fixed malformatted protein sequences when --gtf and --trans are both
   specified (#45).

(0.12: 24 June 2023, r237)



Release 0.11-r234 (18 April 2023)
---------------------------------

Notable changes:

 * New feature: added option --trans to output translated protein sequences. It
   is possible to extract these sequences from the --aln output but the --trans
   output is smaller and more convenient.

 * Bugfix: infinite error messages if a wrong option is in use.

 * Improvement: better error messages given nonexisting query files (#40).

(0.11: 18 April 2023, r234)



Release 0.10-r225 (3 April 2023)
--------------------------------

Notable change:

 * Bugfix: rare segmentation fault (#38 and #39). This bug affected all
   previous versions of miniprot.

(0.10: 3 April 2023, r225)



Release 0.9-r223 (9 March 2023)
-------------------------------

Notable change:

 * Bugfix: some query proteins were not outputted with option `-u`.

(0.9: 9 March 2023, r223)



Release 0.8-r220 (6 March 2023)
-------------------------------

Notable changes:

 * Improvement: slightly improved the sensitivity to distant homolog at a minor
   cost of specificity. On the human-zebrafish dataset, we gained 1.2% junction
   sensivity at the cost of 0.2% of specificity.

 * New feature: added option --aln to output residue alignment.

 * New feature: added option -I to automatically set the maximum intron size to
   sqrt(GenomeSize) * 3.6, where GenomeSize is the total length of the
   nucleotide sequences. For smaller genomes, a smaller threshold leads to
   higher accuracy. This option is not the default because the reference is not
   always a genome.

(0.8: 6 March 2023, r220)



Release 0.7-r207 (25 December 2022)
-----------------------------------

Notable changes:

 * Improvement: replaced open syncmers with modimers. This simplifies the code
   and slightly reduces the memory at comparable k-mer sampling rate. This
   changes the index format.

 * Improvement: fine tune parameters for higher sensitivity at a minor cost of
   junction accuracy: a) only index ORFs >= 30bp; b) reduced max k-mer
   occurrences from 50k to 20k; c) sample k-mers at a rate of 50%; d) reduced
   min number of k-mers from 5 to 3; e) add a bonus chaining score for anchors
   on the same reference block.

 * Improvement: adjust the max k-mer occurrence dynamically per protein.

 * Improvement: implemented 2-level chaining like minimap2 and minigraph. This
   reduces chaining time.

 * Bugfix: fixed a rare off-by-1 memory violation

 * Bugfix: fixed a memory leak

Overall, miniprot becomes faster at slightly higher peak memory usage. It is
more sensitive to distant homologs, though the junction accuracy of additional
alignment is usually lower. Also importantly, the index format of miniprot has
been changed. Miniprot will throw an error if you use miniprot with pre-built
indices generated with older versions.

(0.7: 25 December 2022, r207)



Release 0.6-r185 (12 December 2022)
-----------------------------------

Notable changes:

 * Improvement: for each protein, only output alignments close to the best
   alignment. Also added option --outs to tune the threshold.

 * New feature: output GTF with option --gtf.

(0.6: 22 December 2022, r185)



Release 0.5-r179 (17 October 2022)
----------------------------------

Notable changes:

 * Improvement: more detailed splice model considering G|GTR..YYYNYAG|. This is
   not enabled by default. Added option `-j` to change the splice model.

 * Added the miniprot preprint. Available at http://arxiv.org/abs/2210.08052

(0.5: 17 October 2022, r179)



Release 0.4-r165 (5 October 2022)
---------------------------------

This version implements a better splice model and pays a little more effort in
aligning terminal exons. It improves both sensitivity and specificity by a few
percent.

Other notable changes:

 * Breaking change: changed -C to scale the splice model

 * Bugfix: implemented option -w (#12)

 * Bugfix: reduced the indexing time for highly fragmented genomes (#12)

 * New feature: output a Rank attribute in GFF

(0.4: 5 October 2022, r165)



Release 0.3-r137 (22 September 2022)
------------------------------------

Notable changes:

 * Improvement: fine tune parameters and heuristics: higher non-GT-AG penalty,
   higher frameshift penalty, higher penalty on in-frame stop codons and a
   small penalty on long terminal introns. Miniprot is a little more sensitive
   and a little more accurate, at a minor cost of performance.

 * New feature: richer GFF output. Miniprot now reports per-exon alignment
   score, number of frameshifts, number of in-frame stop codons and
   non-canonical donor/acceptor sequences.

 * New feature: added option `--outn` to control the number of alignments per
   protein to output.

 * New feature: added option `-P` to change the ID prefix in GFF output (#6).

 * Bug fix: fixed a segmentation fault when there are no k-mers on a reference
   sequence (#4).

(0.3: 22 September 2022, r137)



Release 0.2-r116 (12 September 2022)
------------------------------------

Notable changes:

 * New feature: output GFF3 with option `--gff`. PAF alignments are embedded in
   `##PAF` lines in the output GFF3.

 * Improvement: give a bonus score -B if extension reaches the end of a protein.
   BWA-MEM and minimap2 both have this heuristic.

 * Improvement: pay more effort to the first and the last exons. This increases
   the sensitivity at the cost of performance.

 * Improvement: increased non-canonical splicing penalty -C from 6 to 11. This
   increases overall specificity.

 * Improvement: rank an alignment with the DP score disregarding introns. This
   is to reduce the effect pseudogenes. Minimap2 uses the same strategy.

 * Bug fixes: fixed incorrect CIGAR in corner cases and patched a minor memory
   leak.

This version works on several real datasets to decent accuracy without crashing
or memory leaks. It is ready for more users.

(0.2: 12 September 2022, r116)



Release 0.1-r97 (9 September 2022)
----------------------------------

This is the first public release of miniprot, a mapper for aligning proteins to
a large genome. This release has a few issues and is generally not recommended
for production uses.

(0.1: 9 September 2022, r97)
