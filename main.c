#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mppriv.h"
#include "ketopt.h"

static ko_longopt_t long_options[] = {
	{ "gff",             ko_no_argument,       301 },
	{ "xdrop",           ko_required_argument, 302 },
	{ "outn",            ko_required_argument, 303 },
	{ "gff-only",        ko_no_argument,       304 },
	{ "gff-delim",       ko_required_argument, 305 },
	{ "J2",              ko_required_argument, 306 },
	{ "gtf",             ko_no_argument,       307 },
	{ "outs",            ko_required_argument, 308 },
	{ "max-skip",        ko_required_argument, 309 },
	{ "no-pre-chain",    ko_no_argument,       310 },
	{ "aln",             ko_no_argument,       311 },
	{ "max-intron-out",  ko_required_argument, 312 },
	{ "outc",            ko_required_argument, 313 },
	{ "ie-coef",         ko_required_argument, 314 },
	{ "trans",           ko_no_argument,       315 },
	{ "no-cs",           ko_no_argument,       316 },
	{ "spsc",            ko_required_argument, 317 },
	{ "spsc0",           ko_required_argument, 318 },
	{ "spsc-max",        ko_required_argument, 319 },
	{ "accuracy",        ko_no_argument,       320 },
	{ "profile",         ko_no_argument,       321 },
	{ "acc-repeat-cap",  ko_required_argument, 322 },
	{ "acc-anchor-budget", ko_required_argument, 323 },
	{ "acc-rescue-cells", ko_required_argument, 324 },
	{ "acc-no-repeat",   ko_no_argument,       325 },
	{ "acc-no-vid-chain", ko_no_argument,      326 },
	{ "acc-no-rescue",   ko_no_argument,       327 },
	{ "acc-no-rerank",   ko_no_argument,       328 },
	{ "acc-cov-bonus",   ko_required_argument, 329 },
	{ "acc-splice-bonus", ko_required_argument, 330 },
	{ "acc-noncan-penalty", ko_required_argument, 331 },
	{ "acc-fs-penalty",  ko_required_argument, 332 },
	{ "acc-stop-penalty", ko_required_argument, 333 },
	{ "acc-end-bonus",   ko_required_argument, 334 },
	{ "acc-no-junc-refine", ko_no_argument,     335 },
	{ "acc-rerank-margin", ko_required_argument, 342 },
	{ "acc-end-ext",     ko_required_argument, 343 },
	{ "acc-no-end-ext-cap", ko_no_argument,     344 },
	{ "acc-end-retry",   ko_required_argument, 345 },
	{ "acc-end-ext-retry", ko_no_argument,      346 },
	{ "acc-end-retry-win", ko_required_argument, 347 },
	{ "acc-end-aa-cap",  ko_no_argument,         348 },
	{ "acc-end-aa-ext",  ko_required_argument,   349 },
	{ "acc-end-dyn",     ko_no_argument,         351 },
	{ "acc-end-ext-min", ko_required_argument,   352 },
	{ "acc-end-nt-per-aa", ko_required_argument, 353 },
	{ "acc-end-aa-min",  ko_required_argument,   354 },
	{ "acc-end-aa-ratio", ko_required_argument,  355 },
	{ "acc-strict-multi-exon", ko_no_argument,   362 },
	{ "acc-intron-evidence-min", ko_required_argument, 363 },
	{ "acc-intron-evidence-ratio", ko_required_argument, 364 },
	{ "acc-intron-abs-max", ko_required_argument, 365 },
	{ "acc-strict-span-ratio", ko_required_argument, 366 },
	{ "acc-min-score",   ko_required_argument,     368 },
	{ "acc-min-pos-ratio", ko_required_argument,   369 },
	{ "acc-rescue-np",   ko_required_argument,     370 },
	{ "acc-min-np",      ko_required_argument,     371 },
	{ "hmm-seed-pos-loss", ko_required_argument,   372 },
	{ "hmm-seed-loss",   ko_required_argument,     373 },
	{ "hmm-seed-max",    ko_required_argument,     374 },
	{ "hmm-seed-max2",   ko_required_argument,     375 },	{ "acc-no-trim-intron", ko_no_argument,      376 },	{ "compact",         ko_no_argument,         367 },
	{ "acc-no-refine-local", ko_no_argument,     377 },
	{ "acc-no-refine-pssm", ko_no_argument,      378 },	{ "acc-refine-min-seg", ko_required_argument, 379 },	{ "acc-refine-xdrop", ko_required_argument, 380 },
	{ "acc-trim-side-ratio", ko_required_argument, 381 },
	{ "acc-rescue-score",    ko_required_argument, 382 },
	{ "thr-table",       ko_required_argument, 383 },
	{ "thr-fallback",    ko_required_argument, 384 },
	{ "acc-coarse-n",    ko_required_argument, 385 },
	{ "thr-use-ps",      ko_no_argument,       386 },
	{ "acc-no-end-ext-retry", ko_no_argument,  387 },
	{ "cand-table",      ko_required_argument, 388 },
	{ "acc-min-ps",      ko_required_argument, 389 },
	{ "tier",            ko_required_argument, 390 },
	{ "cand-max-evalue", ko_required_argument, 391 },
	{ "prescreen",       ko_no_argument,       392 },
	{ "ps-sens",         ko_required_argument, 393 },
	{ "ps-kmer",         ko_required_argument, 394 },
	{ "ps-min-orf",      ko_required_argument, 395 },
	{ "ps-max-seqs",     ko_required_argument, 396 },
	{ "ps-min-ungapped", ko_required_argument, 397 },
	{ "ps-band",         ko_required_argument, 398 },
	{ "ps-dump",         ko_required_argument, 399 },
	{ "ps-kmer-thr",     ko_required_argument, 403 },
	{ "ps-kmer-max",     ko_required_argument, 404 },
	{ "small-genome",    ko_required_argument, 412 },
	{ "small-genome-size", ko_required_argument, 413 },
	{ "acc-ps-rescue",   ko_required_argument, 405 },
	{ "no-cand-diag",    ko_no_argument,       406 },
	{ "gpu-prefilter",            ko_no_argument,       407 },
	{ "gpu-prefilter-topk",       ko_required_argument, 408 },
	{ "gpu-prefilter-diversity",  ko_required_argument, 409 },
	{ "gpu-prefilter-xdrop",      ko_required_argument, 410 },
	{ "gpu-prefilter-adaptive",   ko_no_argument,       411 },
	{ "fast",            ko_no_argument,         350 },
	{ "acc-pri-ratio",   ko_required_argument, 336 },
	{ "acc-best-n",      ko_required_argument, 337 },
	{ "acc-microexon",   ko_required_argument, 338 },
	{ "gpu",             ko_no_argument,       339 },
	{ "gpu-device",      ko_required_argument, 340 },
	{ "gpu-min-cells",   ko_required_argument, 341 },
	{ "help",            ko_no_argument,       400 },
	{ "version",         ko_no_argument,       401 },
	{ "no-kalloc",       ko_no_argument,       501 },
	{ "dbg-qname",       ko_no_argument,       502 },
	{ "dbg-no-refine",   ko_no_argument,       503 },
	{ "dbg-aflt",        ko_no_argument,       504 },
	{ "dbg-anchor",      ko_no_argument,       505 },
	{ "dbg-chain",       ko_no_argument,       506 },
	{ 0, 0, 0 }
};

static inline int64_t mp_parse_num2(const char *str, char **q)
{
	double x;
	char *p;
	x = strtod(str, &p);
	if (*p == 'G' || *p == 'g') x *= 1e9, ++p;
	else if (*p == 'M' || *p == 'm') x *= 1e6, ++p;
	else if (*p == 'K' || *p == 'k') x *= 1e3, ++p;
	if (q) *q = p;
	return (int64_t)(x + .499);
}

static inline int64_t mp_parse_num(const char *str)
{
	return mp_parse_num2(str, 0);
}

static void print_usage(FILE *fp, const mp_idxopt_t *io, const mp_mapopt_t *mo, int n_threads)
{
	fprintf(fp, "Usage: protlocus [options] <ref.fa> <query.faa> [...]\n");
	fprintf(fp, "Options:\n");
	fprintf(fp, "  Indexing:\n");
	fprintf(fp, "    -k INT       k-mer size [%d]\n", io->kmer);
	fprintf(fp, "    -M INT       modimisers bit (sample rate = 1/2**M) [%d]\n", io->mod_bit);
	fprintf(fp, "    -L INT       min ORF length to index [%d]\n", io->min_aa_len);
	fprintf(fp, "    -T INT       NCBI translation table (1 through 33) [%d]\n", io->trans_code);
	fprintf(fp, "    -b INT       bits per block [%d]\n", io->bbit);
	fprintf(fp, "    -d FILE      save index to FILE []\n");
	fprintf(fp, "  Mapping:\n");
	fprintf(fp, "    -S           no splicing (applying -G1k -J1k -e1k)\n");
	fprintf(fp, "    -c NUM       max k-mer occurrence [%d]\n", mo->max_occ);
	fprintf(fp, "    -G NUM       max intron size; override -I [200k]\n");
	fprintf(fp, "    -I           set max intron size to 3.6*sqrt(refLen)\n");
	fprintf(fp, "    -w FLOAT     weight of log gap penalty [%g]\n", mo->chn_coef_log);
	fprintf(fp, "    -n NUM       minimum number of syncmers in a chain [%d]\n", mo->min_chn_cnt);
	fprintf(fp, "    -m NUM       min chaining score [%d]\n", mo->min_chn_sc);
	fprintf(fp, "    -l INT       k-mer size for the second round of chaining [%d]\n", mo->kmer2);
	fprintf(fp, "    -e NUM       max extension for 2nd round of chaining and alignment [%d]\n", mo->max_ext);
	fprintf(fp, "    -p FLOAT     min secondary-to-primary score ratio [%g]\n", mo->pri_ratio);
	fprintf(fp, "    -N NUM       consider at most INT secondary alignments [%d]\n", mo->best_n);
	fprintf(fp, "  Accuracy mode:\n");
	fprintf(fp, "    --accuracy   enable repeat rescue, per-sequence chaining, anchor fallback and reranking\n");
	fprintf(fp, "    --profile    report stage timing and workload counters\n");
	fprintf(fp, "    --acc-repeat-cap=NUM    sample at most NUM anchors from each filtered repeat [%d]\n", mo->acc_repeat_cap);
	fprintf(fp, "    --acc-anchor-budget=NUM cap extra rescued anchors per query [%ld]\n", (long)mo->acc_anchor_budget);
	fprintf(fp, "    --acc-rescue-cells=NUM  cap full-DP fallback cells per candidate [%ld]\n", (long)mo->acc_rescue_cells);
	fprintf(fp, "    --acc-pri-ratio=FLOAT   relaxed candidate score ratio [%g]\n", mo->acc_pri_ratio);
	fprintf(fp, "    --acc-best-n=NUM        retain at most NUM secondary candidates [%d]\n", mo->acc_best_n);
	fprintf(fp, "    --acc-coarse-n=NUM      with PSSM refine, carry up to NUM coarse chains into refine and re-rank them by PSSM evidence before truncating to -N; <=-N disables [%d]\n", mo->acc_coarse_n);
	fprintf(fp, "    --acc-microexon=NUM     realign short query insertions up to NUM aa [%d]\n", mo->acc_microexon_max);
	fprintf(fp, "    --acc-rerank-margin=NUM cap composite-score bonus over raw dp_max [%d]\n", mo->acc_rerank_margin);
	fprintf(fp, "    --acc-end-ext=NUM      initial speculative end-search window [%d]\n", mo->acc_end_ext);
	fprintf(fp, "    --acc-end-retry=NUM    retry budget when capped query is unmapped [%d]\n", mo->acc_end_retry);
	fprintf(fp, "    --acc-end-retry-win=NUM nt window cap for the io_end 2nd-try short-exon DP [%d]\n", mo->acc_end_retry_win);
	fprintf(fp, "    --acc-end-ext-retry     also retry rescued candidates' end-extension with io_end (slower)\n");
	fprintf(fp, "    --acc-end-aa-cap        cap the aa-side extent of the first end-extension DP call for rescued candidates (experimental)\n");
	fprintf(fp, "    --acc-end-aa-ext=NUM    aa-side cap used by --acc-end-aa-cap [%d]\n", mo->acc_end_aa_ext);
	fprintf(fp, "    --acc-end-dyn           scale the end-extension nt/aa caps with query protein length instead of a fixed cap (experimental)\n");
	fprintf(fp, "    --acc-end-ext-min=NUM   floor for the --acc-end-dyn nt-side cap [%d]\n", mo->acc_end_ext_min);
	fprintf(fp, "    --acc-end-nt-per-aa=FLOAT nt-per-aa coefficient for the --acc-end-dyn nt-side cap [%g]\n", mo->acc_end_nt_per_aa);
	fprintf(fp, "    --acc-end-aa-min=NUM    floor for the --acc-end-dyn aa-side cap (needs --acc-end-aa-cap) [%d]\n", mo->acc_end_aa_min);
	fprintf(fp, "    --acc-end-aa-ratio=FLOAT query-length fraction for the --acc-end-dyn aa-side cap [%g]\n", mo->acc_end_aa_ratio);
	fprintf(fp, "    --acc-strict-multi-exon only prefer a multi-exon hit over a single-exon hit when every junction is canonical and the score margin covers every intron; also drops a multi-exon candidate whose widest intron exceeds its own aligned exonic evidence (improves purity, off by default)\n");
	fprintf(fp, "    --acc-intron-evidence-min=NUM   with --acc-strict-multi-exon, only check introns at least this long against the evidence ratio below [1000]\n");
	fprintf(fp, "    --acc-intron-evidence-ratio=NUM with --acc-strict-multi-exon, drop a candidate if any single intron exceeds its flanking or whole-candidate exonic nt times NUM [1.0]\n");
	fprintf(fp, "    --acc-intron-abs-max=NUM        with --acc-strict-multi-exon, drop a candidate if any single intron exceeds NUM nt regardless of exonic evidence, <=0 disables [1500]\n");
	fprintf(fp, "    --acc-no-trim-intron            with --compact, drop (instead of trim at) a spurious junction: the whole candidate is discarded as before\n");
	fprintf(fp, "    --acc-no-refine-local           with --compact, let the refine stage keep the window-wide best chain even if it leaves the coarse locus (old behaviour)\n");
	fprintf(fp, "    --acc-no-refine-pssm            with --compact (HMM mode), pick refine chains by chain score only, ignoring ungapped PSSM diagonal evidence\n");
	fprintf(fp, "    --acc-refine-min-seg INT        PSSM diagonal segment counts as evidence if >= INT half-bits [%d]\n", mo->acc_refine_min_seg);
	fprintf(fp, "    --acc-refine-xdrop INT          X-drop for the PSSM diagonal extension, half-bits [%d]\n", mo->acc_refine_xdrop);
	fprintf(fp, "    --acc-trim-side-ratio FLOAT     with --compact, also trim at a junction whose intron exceeds the exonic nt on its smaller side times FLOAT; <=0 disables [%g]\n", mo->acc_trim_side_ratio);
	fprintf(fp, "    --acc-strict-span-ratio=FLOAT   drop a hit whose genomic span exceeds FLOAT times its aligned query length in nt (excess must also reach --acc-intron-evidence-min); <=0 disables [%g]\n", mo->acc_span_ratio);
	fprintf(fp, "    --acc-min-score=NUM     drop a hit whose DP score (PAF AS) is below NUM, a significance floor for queries with no true homolog; <=0 disables [%d]\n", mo->acc_min_out_score);
	fprintf(fp, "    --acc-min-pos-ratio=FLOAT drop a hit whose positive-substitution fraction (3*np/blen) is below FLOAT; <=0 disables [%g]\n", mo->acc_min_pos_ratio);
	fprintf(fp, "    --acc-rescue-np=NUM     exempt a hit from the two floors above when its positive count (PAF np) >= NUM; <=0 disables [%d]\n", mo->acc_rescue_np);
	fprintf(fp, "    --acc-rescue-score=NUM  exempt a hit from the pos-ratio floor when its DP score (PAF AS) >= NUM; <=0 disables [%d]\n", mo->acc_rescue_score);
	fprintf(fp, "    --acc-min-np=NUM        unconditionally drop a hit whose positive count (PAF np) < NUM; <=0 disables [%d]\n", mo->acc_min_np);
	fprintf(fp, "    --thr-table=FILE        per-query DP-score floor (TSV: query_name<TAB>min_AS), e.g. decoy-calibrated per-family thresholds; replaces --acc-min-score for listed queries []\n");
	fprintf(fp, "    --thr-fallback=NUM      floor (PAF AS) for queries absent from --thr-table; <=0 keeps --acc-min-score [%d]\n", mo->thr_fallback);
	fprintf(fp, "    --thr-use-ps            apply the score floor to the HMM profile Viterbi score (PAF ps) INSTEAD of DP score (PAF AS); ps is a fully local, uncalibrated score (~40 half-bits by chance on long windows), so prefer --acc-min-ps\n");
	fprintf(fp, "    --acc-min-ps=NUM        drop a profile-query hit whose HMM Viterbi score (PAF ps) is below NUM, on top of the AS floor/--thr-table; <=0 disables; no effect on protein queries [%d]\n", mo->acc_min_ps);
	fprintf(fp, "    --acc-ps-rescue=AS[,POSR] exempt a hit from --acc-min-ps when AS>=AS and positive ratio>=POSR (ps is unnormalised; short models fail ps>=60 with strong alignments); AS<=0 disables [%d,%.2f]\n", mo->acc_ps_rescue_score, mo->acc_ps_rescue_posr);
	fprintf(fp, "    --cand-table=FILE       external candidate loci (TSV: query_name<TAB>chrom<TAB>start0<TAB>end0<TAB>strand[<TAB>evalue], e.g. from mmseqs2cand.py); replaces the\n");
	fprintf(fp, "                            genome-wide seed/chain stage: only listed loci (refined within +-max_ext, see -e) are aligned; unlisted queries get no hit []\n");
	fprintf(fp, "    --tier=STR              sensitivity tier = E-value cutoff on the candidate loci (fast/balanced/sensitive). With --cand-table (mmseqs E-values):\n");
	fprintf(fp, "                            E<=100/1000/10000, yeast/Pfam-A (56 thr, --compact) 43s P0.984 R0.871 / 2.5min P0.966 R0.919 / 13.5min P0.947 R0.959.\n");
	fprintf(fp, "                            With --prescreen (built-in E-values): E<=0.1/100/10000, 36s P0.988 R0.873 / 1.4min P0.956 R0.923 / 10.5min P0.936 R0.965 []\n");
	fprintf(fp, "    --cand-max-evalue=FLOAT same as --tier with an arbitrary E-value cutoff (ignored when --tier is given); <=0 keeps every listed locus [%g]\n", mo->cand_max_evalue);
	fprintf(fp, "  Built-in prescreen (HMM queries; replaces mmseqs + --cand-table):\n");
	fprintf(fp, "    --prescreen             build a six-frame ORF spaced-k-mer index of the genome and generate each profile's candidate loci in-process\n");
	fprintf(fp, "                            (MMseqs2-style profile k-mer prefilter -> ungapped diagonal -> banded gapped alignment -> E-value); --tier selects the cutoff,\n");
	fprintf(fp, "                            default balanced (E<=100) when neither --tier nor --cand-max-evalue is given\n");
	fprintf(fp, "    --small-genome=STR      auto/on/off: with --prescreen on a genome <= --small-genome-size, switch to the small-genome recall preset\n");
	fprintf(fp, "                            --ps-kmer=5 --acc-min-ps=0 --acc-min-score=0 --acc-min-pos-ratio=0 (flags you set explicitly are kept).\n");
	fprintf(fp, "                            Weak hits are passed through for protlocus-recheck (-e 1e-5) to arbitrate; validated on prokaryote/fungal\n");
	fprintf(fp, "                            genomes: +4.4-4.7pp domain recall vs balanced at the same purity and minute-scale runtime [auto]\n");
	fprintf(fp, "    --small-genome-size=NUM genome-size threshold that triggers --small-genome=auto [50M]\n");
	fprintf(fp, "    --ps-sens=FLOAT         prefilter sensitivity, mmseqs -s scale; sets the profile k-mer score threshold [%g]\n", mo->ps.sens);
	fprintf(fp, "    --ps-kmer=INT           spaced k-mer size 5..7 (index has 20^k buckets: k=6 0.5GB, k=7 10GB) [%d]\n", mo->ps.kmer);
	fprintf(fp, "    --ps-kmer-thr=NUM       explicit k-mer score threshold in half-bits, overrides --ps-sens; <=0 derives from --ps-sens (k=6: 22; --tier=sensitive lowers it by 1) [%d]\n", mo->ps.kmer_thr);
	fprintf(fp, "    --ps-kmer-max=NUM       cap on similar k-mers generated per profile window [%d]\n", mo->ps.kmer_max);
	fprintf(fp, "    --ps-min-orf=INT        ignore stop-free ORFs shorter than INT aa (esl-translate -l) [%d]\n", mo->ps.min_orf);
	fprintf(fp, "    --ps-min-ungapped=NUM   ungapped diagonal score floor, half-bits (mmseqs --min-ungapped-score) [%d]\n", mo->ps.min_ungapped);
	fprintf(fp, "    --ps-max-seqs=NUM       ORFs per profile passed to the gapped stage (mmseqs --max-seqs) [%d]\n", mo->ps.max_seqs);
	fprintf(fp, "    --ps-band=INT           gapped alignment band (aa) around the best ungapped diagonal; <=0 = full matrix [%d]\n", mo->ps.band);
	fprintf(fp, "    --ps-dump=FILE          also write the generated candidate table (same TSV as --cand-table, 6th column = E-value) []\n");
	fprintf(fp, "    --no-cand-diag          do not seed a prescreen locus from its best ungapped PSSM diagonal when the 5-mer refine chain is empty (divergent members)\n");
	fprintf(fp, "  HMM profile queries:\n");
	fprintf(fp, "    --hmm-seed-max=NUM      profile-derived k-mer seeds per k-window in the coarse stage; 0 = consensus only [%d]\n", mo->hmm_seed_max);
	fprintf(fp, "    --hmm-seed-max2=NUM     same budget for the refine stage [%d]\n", mo->hmm_seed_max2);
	fprintf(fp, "    --hmm-seed-loss=NUM     total emission loss budget per seed, units of 2*ln(p) [%d]\n", mo->hmm_seed_win_loss);
	fprintf(fp, "    --hmm-seed-pos-loss=NUM per-position emission loss allowed for an alternative residue group [%d]\n", mo->hmm_seed_pos_loss);
	fprintf(fp, "    --acc-no-repeat|--acc-no-vid-chain|--acc-no-rescue|--acc-no-rerank|--acc-no-junc-refine|--acc-no-end-ext-cap\n");
	fprintf(fp, "    --acc-no-end-ext-retry          skip the io_end 2nd-try end-extension DP for rescued candidates even under --compact (which enables it)\n");
	fprintf(fp, "    --fast       accuracy mode tuned for speed (implies --accuracy; equivalent to --acc-end-dyn --acc-end-aa-cap, scaling end-extension caps with query length; trades minor recall for ~25%% less align time)\n");
	fprintf(fp, "    --compact    identification preset for compact, gene-dense, intron-poor genomes such as yeast (implies --accuracy;\n");
	fprintf(fp, "                 raises anchor-rescue budgets for recall, enables --acc-strict-multi-exon and --acc-strict-span-ratio=4\n");
	fprintf(fp, "                 to reject hits spilling across neighboring genes, sets --acc-min-score=15 --acc-min-pos-ratio=0.50\n");
	fprintf(fp, "                 to suppress junk best-hits of families absent from the genome, and scales max intron with genome size as -I)\n");
	fprintf(fp, "  GPU (requires a 'make gpu=1' OpenCL build):\n");
	fprintf(fp, "    --gpu        run terminal-extension DP and large gap DP on the OpenCL device (MP_HMM_GPU=1 adds HMM scoring)\n");
	fprintf(fp, "    --gpu-device=INT     zero-based OpenCL GPU index [%d]\n", mo->gpu_device);
	fprintf(fp, "    --gpu-min-cells=NUM  minimum DP cells sent to the GPU [%ld]\n", (long)mo->gpu_min_cells);
	fprintf(fp, "  GPU candidate prefilter (requires a 'make cuda=1' CUDA build; independent of --gpu/OpenCL):\n");
	fprintf(fp, "    --gpu-prefilter      rank coarse-chain candidates per query by a GPU gapless X-drop score and keep the\n");
	fprintf(fp, "                         top --gpu-prefilter-topk before the refine/align stages; exact DP stays on CPU (or GPU with --gpu)\n");
	fprintf(fp, "    --gpu-prefilter-topk=NUM  keep at most NUM candidates per query (this is the historical 'K' parameter) [%d]\n", mo->gpu_prefilter_top_k);
	fprintf(fp, "    --gpu-prefilter-diversity=NUM  reserve NUM survivors per contig/strand group [%d]\n", mo->gpu_prefilter_min_per_group);
	fprintf(fp, "    --gpu-prefilter-xdrop=NUM  gapless extension X-drop threshold, half-bits [%d]\n", mo->gpu_prefilter_xdrop);
	fprintf(fp, "    --gpu-prefilter-adaptive  pick K per query (1..min(topk,4)) as the number of candidates in contention, calibrated per batch\n");
	fprintf(fp, "                            to the genome's background gapless score and paralog richness, instead of a fixed topk\n");
	fprintf(fp, "  Alignment:\n");
//	fprintf(fp, "    -A           no alignment\n");
	fprintf(fp, "    -O INT       gap open penalty [%d]\n", mo->go);
	fprintf(fp, "    -E INT       gap extension (a k-long gap costs O+k*E) [%d]\n", mo->ge);
	fprintf(fp, "    -J INT       intron open penalty [%d]\n", mo->io);
	fprintf(fp, "    -F INT       penalty for frameshifts or in-frame stop codons [%d]\n", mo->fs);
	fprintf(fp, "    -C FLOAT     weight of splice penalty; 0 to ignore splice signals [%g]\n", mo->sp_scale);
	fprintf(fp, "    -B INT       bonus score for alignment reaching query ends [%d]\n", mo->end_bonus);
	fprintf(fp, "    -j INT       splice model: 2=vertebrate/insect, 1=general, 0=none (see manual) [%d]\n", mo->sp_model);
	fprintf(fp, "    --spsc=FILE  splice score file in format \"ctg offset +|- D|A score\" []\n");
	fprintf(fp, "    --spsc0=INT  splice score for sites not in --spsc [%d]\n", mo->sp_null_bonus);
	fprintf(fp, "  Input/output:\n");
	fprintf(fp, "    -t INT       number of threads [%d]\n", n_threads);
	fprintf(fp, "    query.hmm    HMMER3 profile query; a file with many profiles (e.g. cat *.hmm) is mapped in parallel batches\n");
	fprintf(fp, "    --gff        output in the GFF3 format\n");
	fprintf(fp, "    --gtf        basic GTF output without detailed alignment\n");
	fprintf(fp, "    --aln        output residue alignment\n");
	fprintf(fp, "    --trans      output translated protein sequences (skipping frameshift)\n");
	fprintf(fp, "    -P STR       prefix for IDs in GFF3 [%s]\n", mo->gff_prefix);
	fprintf(fp, "    -u           print unmapped query proteins in PAF\n");
	fprintf(fp, "    --outn=NUM   output up to min{NUM,-N} alignments per query [%d]\n", mo->out_n);
	fprintf(fp, "    --outs=FLOAT output if score at least FLOAT*bestScore [%g]\n", mo->out_sim);
	fprintf(fp, "    --outc=FLOAT output if at least FLOAT fraction of query is aligned [%g]\n", mo->out_cov);
	fprintf(fp, "    -K NUM       query batch size [2M]\n");
}

int main(int argc, char *argv[])
{
	int32_t c, i, set_I = 0, set_G = 0, keep_io = 0, n_threads = 4, no_trim_intron = 0, no_refine_local = 0, no_refine_pssm = 0, no_end_ext_retry = 0, use_prescreen = 0, tier = 0, set_ps_thr = 0;
	int32_t small_genome = 0, set_sg_kmer = 0, set_sg_ps = 0, set_sg_score = 0, set_sg_posr = 0; // --small-genome: 0=auto, 1=on, -1=off; set_sg_* remember explicit flags the preset must not override
	int64_t small_genome_size = 50000000; // --small-genome-size
	ketopt_t o = KETOPT_INIT;
	mp_mapopt_t mo;
	mp_idxopt_t io;
	mp_idx_t *mi;
	char *fn_idx = 0, *fn_spsc = 0, *fn_thr = 0, *fn_cand = 0;
	mp_thr_tab_t *thr_tab = 0;
	mp_cand_tab_t *cand_tab = 0;
	mp_prescreen_t *prescreen = 0;

	mp_start();
	mp_mapopt_init(&mo);
	mp_idxopt_init(&io);
	while ((c = ketopt(&o, argc, argv, 1, "k:M:L:s:l:b:T:t:d:c:n:m:K:p:N:SAO:E:J:C:F:G:e:uB:P:w:j:g:I", long_options)) >= 0) {
		if (c == 'k') io.kmer = atoi(o.arg);
		else if (c == 'M') io.mod_bit = atoi(o.arg);
		else if (c == 'L') io.min_aa_len = atoi(o.arg);
		else if (c == 'b') io.bbit = atoi(o.arg);
		else if (c == 'T') io.trans_code = atoi(o.arg);
		else if (c == 'd') fn_idx = o.arg;
		else if (c == 't') n_threads = atoi(o.arg);
		else if (c == 'l') mo.kmer2 = atoi(o.arg);
		else if (c == 'c') mo.max_occ = mp_parse_num(o.arg);
		else if (c == 'G') mo.bw = mo.max_intron = mp_parse_num(o.arg), set_G = 1;
		else if (c == 'I') set_I = 1;
		else if (c == 'n') mo.min_chn_cnt = mp_parse_num(o.arg);
		else if (c == 'm') mo.min_chn_sc = mp_parse_num(o.arg);
		else if (c == 'K') mo.mini_batch_size = mp_parse_num(o.arg);
		else if (c == 'p') mo.pri_ratio = atof(o.arg);
		else if (c == 'N') mo.best_n = mp_parse_num(o.arg);
		else if (c == 'S') mo.flag |= MP_F_NO_SPLICE, mo.bw = mo.max_intron = mo.max_ext = 1000, mo.io = mo.io_end = 10000, set_G = 1;
		else if (c == 'A') mo.flag |= MP_F_NO_ALIGN;
		else if (c == 'O') mo.go = atoi(o.arg);
		else if (c == 'E') mo.ge = atoi(o.arg);
		else if (c == 'J') mo.io = atoi(o.arg), keep_io = 1;
		else if (c == 'C') mo.sp_scale = atof(o.arg);
		else if (c == 'F') mp_mapopt_set_fs(&mo, atoi(o.arg));
		else if (c == 'B') mo.end_bonus = atoi(o.arg);
		else if (c == 'e') mo.max_ext = mp_parse_num(o.arg);
		else if (c == 'P') mo.gff_prefix = o.arg;
		else if (c == 'u') mo.flag |= MP_F_SHOW_UNMAP;
		else if (c == 'w') mo.chn_coef_log = atof(o.arg);
		else if (c == 'j') mo.sp_model = atoi(o.arg);
		else if (c == 'g') mo.max_gap = mp_parse_num(o.arg);
		else if (c == 301) mo.flag |= MP_F_GFF; // --gff
		else if (c == 302) mo.xdrop = atoi(o.arg); // --xdrop
		else if (c == 303) mo.out_n = mp_parse_num(o.arg); // --outn
		else if (c == 308) mo.out_sim = atof(o.arg); // --outs
		else if (c == 304) mo.flag |= MP_F_GFF | MP_F_NO_PAF; // --gff-only
		else if (c == 305) mo.gff_delim = o.arg[0]; // --gff-delim
		else if (c == 306) mo.io_end = atoi(o.arg), keep_io = 1; // --J2
		else if (c == 307) mo.flag |= MP_F_GTF; // --gtf
		else if (c == 309) mo.max_chn_max_skip = mp_parse_num(o.arg); // --max-skip
		else if (c == 310) mo.flag |= MP_F_NO_PRE_CHAIN; // --no-pre-chain
		else if (c == 311) mo.flag |= MP_F_SHOW_RESIDUE; // --aln
		else if (c == 312) mo.max_intron_flank = (mp_parse_num(o.arg) + 1) / 2; // --max-intron-out
		else if (c == 313) mo.out_cov = atof(o.arg); // --outc
		else if (c == 314) mo.ie_coef = atof(o.arg); // --ie-coef
		else if (c == 315) mo.flag |= MP_F_SHOW_TRANS; // --trans
		else if (c == 316) mo.flag |= MP_F_NO_CS; // --no-cs
		else if (c == 317) fn_spsc = o.arg; // --spsc
		else if (c == 319) mo.sp_max_bonus = atoi(o.arg); // --spsc-max
		else if (c == 320) mo.flag |= MP_F_ACCURACY; // --accuracy
		else if (c == 321) mo.flag |= MP_F_PROFILE; // --profile
		else if (c == 322) mo.acc_repeat_cap = mp_parse_num(o.arg); // --acc-repeat-cap
		else if (c == 323) mo.acc_anchor_budget = mp_parse_num(o.arg); // --acc-anchor-budget
		else if (c == 324) mo.acc_rescue_cells = mp_parse_num(o.arg); // --acc-rescue-cells
		else if (c == 325) mo.acc_flags &= ~MP_ACC_REPEAT_RESCUE; // --acc-no-repeat
		else if (c == 326) mo.acc_flags &= ~MP_ACC_VID_CHAIN; // --acc-no-vid-chain
		else if (c == 327) mo.acc_flags &= ~MP_ACC_ANCHOR_RESCUE; // --acc-no-rescue
		else if (c == 328) mo.acc_flags &= ~MP_ACC_RERANK; // --acc-no-rerank
		else if (c == 329) mo.acc_cov_bonus = atoi(o.arg); // --acc-cov-bonus
		else if (c == 330) mo.acc_splice_bonus = atoi(o.arg); // --acc-splice-bonus
		else if (c == 331) mo.acc_noncan_penalty = atoi(o.arg); // --acc-noncan-penalty
		else if (c == 332) mo.acc_fs_penalty = atoi(o.arg); // --acc-fs-penalty
		else if (c == 333) mo.acc_stop_penalty = atoi(o.arg); // --acc-stop-penalty
		else if (c == 334) mo.acc_end_bonus = atoi(o.arg); // --acc-end-bonus
		else if (c == 335) mo.acc_flags &= ~MP_ACC_JUNC_REFINE; // --acc-no-junc-refine
		else if (c == 336) mo.acc_pri_ratio = atof(o.arg); // --acc-pri-ratio
		else if (c == 337) mo.acc_best_n = mp_parse_num(o.arg); // --acc-best-n
		else if (c == 338) mo.acc_microexon_max = mp_parse_num(o.arg); // --acc-microexon
		else if (c == 342) mo.acc_rerank_margin = atoi(o.arg); // --acc-rerank-margin
		else if (c == 343) mo.acc_end_ext = mp_parse_num(o.arg); // --acc-end-ext
		else if (c == 344) mo.acc_flags &= ~MP_ACC_END_EXT_CAP; // --acc-no-end-ext-cap
		else if (c == 345) mo.acc_end_retry = mp_parse_num(o.arg); // --acc-end-retry
		else if (c == 346) mo.acc_flags |= MP_ACC_END_EXT_RETRY; // --acc-end-ext-retry
		else if (c == 347) mo.acc_end_retry_win = mp_parse_num(o.arg); // --acc-end-retry-win
		else if (c == 348) mo.acc_flags |= MP_ACC_END_AA_CAP; // --acc-end-aa-cap
		else if (c == 349) mo.acc_end_aa_ext = mp_parse_num(o.arg); // --acc-end-aa-ext
		else if (c == 351) mo.acc_flags |= MP_ACC_END_DYN_WIN; // --acc-end-dyn
		else if (c == 352) mo.acc_end_ext_min = mp_parse_num(o.arg); // --acc-end-ext-min
		else if (c == 353) mo.acc_end_nt_per_aa = atof(o.arg); // --acc-end-nt-per-aa
		else if (c == 354) mo.acc_end_aa_min = mp_parse_num(o.arg); // --acc-end-aa-min
		else if (c == 355) mo.acc_end_aa_ratio = atof(o.arg); // --acc-end-aa-ratio
		else if (c == 362) mo.acc_flags |= MP_ACC_STRICT_MULTI_EXON; // --acc-strict-multi-exon
		else if (c == 363) mo.acc_intron_evidence_min = mp_parse_num(o.arg); // --acc-intron-evidence-min
		else if (c == 364) mo.acc_intron_evidence_ratio = atof(o.arg); // --acc-intron-evidence-ratio
		else if (c == 365) mo.acc_intron_abs_max = mp_parse_num(o.arg); // --acc-intron-abs-max
		else if (c == 366) mo.acc_span_ratio = atof(o.arg); // --acc-strict-span-ratio
		else if (c == 368) mo.acc_min_out_score = mp_parse_num(o.arg), set_sg_score = 1; // --acc-min-score
		else if (c == 369) mo.acc_min_pos_ratio = atof(o.arg), set_sg_posr = 1; // --acc-min-pos-ratio
		else if (c == 370) mo.acc_rescue_np = mp_parse_num(o.arg); // --acc-rescue-np
		else if (c == 371) mo.acc_min_np = mp_parse_num(o.arg); // --acc-min-np
		else if (c == 372) mo.hmm_seed_pos_loss = mp_parse_num(o.arg); // --hmm-seed-pos-loss
		else if (c == 373) mo.hmm_seed_win_loss = mp_parse_num(o.arg); // --hmm-seed-loss
		else if (c == 374) mo.hmm_seed_max = mp_parse_num(o.arg); // --hmm-seed-max
		else if (c == 375) mo.hmm_seed_max2 = mp_parse_num(o.arg); // --hmm-seed-max2
		else if (c == 376) no_trim_intron = 1; // --acc-no-trim-intron
		else if (c == 377) no_refine_local = 1; // --acc-no-refine-local
		else if (c == 378) no_refine_pssm = 1; // --acc-no-refine-pssm
		else if (c == 379) mo.acc_refine_min_seg = mp_parse_num(o.arg); // --acc-refine-min-seg
		else if (c == 380) mo.acc_refine_xdrop = mp_parse_num(o.arg); // --acc-refine-xdrop
		else if (c == 381) mo.acc_trim_side_ratio = atof(o.arg); // --acc-trim-side-ratio
		else if (c == 382) mo.acc_rescue_score = mp_parse_num(o.arg); // --acc-rescue-score
		else if (c == 383) fn_thr = o.arg; // --thr-table
		else if (c == 384) mo.thr_fallback = mp_parse_num(o.arg); // --thr-fallback
		else if (c == 385) mo.acc_coarse_n = mp_parse_num(o.arg); // --acc-coarse-n
		else if (c == 386) mo.thr_use_ps = 1; // --thr-use-ps
		else if (c == 387) no_end_ext_retry = 1; // --acc-no-end-ext-retry
		else if (c == 388) fn_cand = o.arg; // --cand-table
		else if (c == 389) mo.acc_min_ps = mp_parse_num(o.arg), set_sg_ps = 1; // --acc-min-ps
		else if (c == 390) { // --tier: resolved after the loop because the E-value table differs between --cand-table and --prescreen
			if (strcmp(o.arg, "fast") == 0) tier = 1;
			else if (strcmp(o.arg, "balanced") == 0) tier = 2;
			else if (strcmp(o.arg, "sensitive") == 0) tier = 3;
			else {
				fprintf(stderr, "[ERROR]\033[1;31m --tier must be fast, balanced or sensitive (got '%s')\033[0m\n", o.arg);
				return 1;
			}
		}
		else if (c == 391) mo.cand_max_evalue = atof(o.arg); // --cand-max-evalue
		else if (c == 392) use_prescreen = 1; // --prescreen
		else if (c == 393) mo.ps.sens = atof(o.arg), set_ps_thr = 1; // --ps-sens
		else if (c == 394) mo.ps.kmer = atoi(o.arg), set_sg_kmer = 1; // --ps-kmer
		else if (c == 395) mo.ps.min_orf = atoi(o.arg); // --ps-min-orf
		else if (c == 396) mo.ps.max_seqs = mp_parse_num(o.arg); // --ps-max-seqs
		else if (c == 397) mo.ps.min_ungapped = atoi(o.arg); // --ps-min-ungapped
		else if (c == 398) mo.ps.band = atoi(o.arg); // --ps-band
		else if (c == 399) mo.ps.dump_fn = o.arg; // --ps-dump
		else if (c == 403) mo.ps.kmer_thr = atoi(o.arg), set_ps_thr = 1; // --ps-kmer-thr
		else if (c == 404) mo.ps.kmer_max = mp_parse_num(o.arg); // --ps-kmer-max
		else if (c == 412) { // --small-genome
			if (strcmp(o.arg, "auto") == 0) small_genome = 0;
			else if (strcmp(o.arg, "on") == 0) small_genome = 1;
			else if (strcmp(o.arg, "off") == 0) small_genome = -1;
			else {
				fprintf(stderr, "[ERROR]\033[1;31m --small-genome must be auto, on or off (got '%s')\033[0m\n", o.arg);
				return 1;
			}
		}
		else if (c == 413) small_genome_size = mp_parse_num(o.arg); // --small-genome-size
		else if (c == 405) { // --acc-ps-rescue=AS[,POSR]
			char *p;
			mo.acc_ps_rescue_score = strtol(o.arg, &p, 10);
			if (*p == ',') mo.acc_ps_rescue_posr = atof(p + 1);
		}
		else if (c == 406) mo.flag |= MP_F_NO_CAND_DIAG; // --no-cand-diag
		else if (c == 367) { // --compact: identification preset for compact, gene-dense, intron-poor genomes (e.g. yeast)
			mo.flag |= MP_F_ACCURACY;
			mo.acc_repeat_cap = 1000000; // high-recall rescue: keep repeat-filtered anchors
			mo.acc_anchor_budget = 10000000;
			mo.acc_flags |= MP_ACC_END_EXT_RETRY | MP_ACC_STRICT_MULTI_EXON | MP_ACC_TRIM_INTRON | MP_ACC_REFINE_LOCAL | MP_ACC_REFINE_PSSM;
			mo.acc_end_retry = 1000;
			if (mo.acc_span_ratio <= 0.0f) mo.acc_span_ratio = 4.0f;
			if (mo.acc_min_out_score <= 0) mo.acc_min_out_score = 15; // yeast Pfam sweep: 15/0.50 is the Pareto knee (11.2% no-hit, 93.5% purity>=0.9)
			if (mo.acc_min_pos_ratio <= 0.0f) mo.acc_min_pos_ratio = 0.50f;
			set_I = 1; // scale max intron with genome size instead of the flat 200k default
		}
		else if (c == 350) { // --fast: bundled speed-for-accuracy preset (query-length-scaled end-extension caps; measured ~25% mean wall-time cut, less recall cost than a fixed 4500nt/800aa cap)
			mo.flag |= MP_F_ACCURACY;
			mo.acc_flags |= MP_ACC_END_DYN_WIN | MP_ACC_END_AA_CAP;
		}
		else if (c == 339) mo.flag |= MP_F_GPU; // --gpu
		else if (c == 340) mo.gpu_device = atoi(o.arg); // --gpu-device
		else if (c == 341) mo.gpu_min_cells = mp_parse_num(o.arg); // --gpu-min-cells
		else if (c == 407) mo.flag |= MP_F_GPU_PREFILTER; // --gpu-prefilter
		else if (c == 408) mo.gpu_prefilter_top_k = mp_parse_num(o.arg); // --gpu-prefilter-topk
		else if (c == 409) mo.gpu_prefilter_min_per_group = mp_parse_num(o.arg); // --gpu-prefilter-diversity
		else if (c == 410) mo.gpu_prefilter_xdrop = mp_parse_num(o.arg); // --gpu-prefilter-xdrop
		else if (c == 411) mo.flag |= MP_F_GPU_PREFILTER | MP_F_GPU_PREFILTER_ADAPTIVE; // --gpu-prefilter-adaptive
		else if (c == 501) mp_dbg_flag |= MP_DBG_NO_KALLOC; // --no-kalloc
		else if (c == 502) mp_dbg_flag |= MP_DBG_QNAME; // --dbg-qname
		else if (c == 503) mp_dbg_flag |= MP_DBG_NO_REFINE; // --dbg-no-refine
		else if (c == 504) mp_dbg_flag |= MP_DBG_MORE_DP; // --dbg-aflt
		else if (c == 505) mp_dbg_flag |= MP_DBG_ANCHOR; // --dbg-anchor
		else if (c == 506) mp_dbg_flag |= MP_DBG_CHAIN; // --dbg-chain
		else if (c == 318) { // --spsc0
			int32_t s;
			s = atoi(o.arg);
			mo.sp_null_bonus = s < 0? s : -s;
		} else if (c == 's') {
			fprintf(stderr, "Option '-s' is deprecated.\n");
		} else if (c == 400) {
			print_usage(stdout, &io, &mo, n_threads);
			return 0;
		} else if (c == 401) {
			printf("%s\n", PROTLOCUS_VERSION);
			return 0;
		} else {
			fprintf(stderr, "[WARNING]\033[1;31m unrecognized option: %s\033[0m\n", argv[o.i-1]);
		}
	}
	if (no_trim_intron) mo.acc_flags &= ~MP_ACC_TRIM_INTRON; // after the loop so it overrides --compact regardless of option order
	if (no_refine_local) mo.acc_flags &= ~MP_ACC_REFINE_LOCAL;
	if (no_refine_pssm) mo.acc_flags &= ~MP_ACC_REFINE_PSSM;
	if (no_end_ext_retry) mo.acc_flags &= ~MP_ACC_END_EXT_RETRY;
	if (use_prescreen && fn_cand) {
		fprintf(stderr, "[ERROR]\033[1;31m --prescreen and --cand-table are mutually exclusive\033[0m\n");
		return 1;
	}
	if (use_prescreen && (mo.ps.kmer < 5 || mo.ps.kmer > 7)) {
		fprintf(stderr, "[ERROR]\033[1;31m --ps-kmer must be 5, 6 or 7\033[0m\n");
		return 1;
	}
	if (use_prescreen && tier == 0 && mo.cand_max_evalue <= 0.0) tier = 2; // balanced unless the user chose a tier or an explicit cutoff
	if (tier > 0) { // prescreen E-values are calibrated differently from the mmseqs tables (yeast/Pfam-A: 0.1/100/10000 match or beat fast/balanced/sensitive)
		static const double tier_ev_cand[4] = { 0.0, 100.0, 1000.0, 10000.0 }, tier_ev_ps[4] = { 0.0, 0.1, 100.0, 10000.0 };
		mo.cand_max_evalue = use_prescreen? tier_ev_ps[tier] : tier_ev_cand[tier];
		// sensitive: k-mer threshold 22->21 half-bits (yeast/Pfam-A: fam-e2e +50 genes at the same runtime; 20 costs 1.5x time and 2.4x intergenic hits)
		if (use_prescreen && tier == 3 && !set_ps_thr) mo.ps.kmer_thr = mp_ps_kmer_thr(&mo.ps) - 1;
	}
	if (mp_mapopt_check(&mo) < 0) return 1;
	if ((mo.flag & (MP_F_GPU | MP_F_GPU_PREFILTER)) && !ns_gpu_init(mo.gpu_device)) {
		fprintf(stderr, "[WARNING]\033[1;31m GPU unavailable (%s); using CPU DP\033[0m\n", ns_gpu_error());
		mo.flag &= ~(MP_F_GPU | MP_F_GPU_PREFILTER | MP_F_GPU_PREFILTER_ADAPTIVE);
	} else if (mo.flag & (MP_F_GPU | MP_F_GPU_PREFILTER)) {
		if ((mo.flag & MP_F_GPU_PREFILTER) && !ns_gpu_gapless_supported()) {
			fprintf(stderr, "[WARNING]\033[1;31m --gpu-prefilter requires a 'make cuda=1' build; ignoring --gpu-prefilter*\033[0m\n");
			mo.flag &= ~(MP_F_GPU_PREFILTER | MP_F_GPU_PREFILTER_ADAPTIVE);
		}
		if ((mo.flag & MP_F_GPU) && mp_verbose >= 3)
			fprintf(stderr, "[M::gpu] using device %d: %s\n", mo.gpu_device, ns_gpu_name());
	}
	if (argc - o.ind == 0 || (argc - o.ind == 1 && fn_idx == 0)) {
		print_usage(stderr, &io, &mo, n_threads);
		return 1;
	}

	if (ns_make_tables(io.trans_code) < 0) {
		if (mp_verbose >= 1)
			fprintf(stderr, "[ERROR]\033[1;31m failed to find translation table %d\033[0m\n", io.trans_code);
		return 1;
	}
	mi = mp_idx_load(argv[o.ind], &io, n_threads);
	if (mi == 0) {
		if (mp_verbose >= 1)
			fprintf(stderr, "[ERROR]\033[1;31m failed to open/build the index\033[0m\n");
		return 1;
	}
	if (set_I && !set_G) mp_mapopt_set_max_intron(&mo, mi->nt->l_seq);
	if (use_prescreen && small_genome >= 0) { // --small-genome: open-threshold recall preset for prokaryote-sized genomes (see ACCURACY.md)
		int32_t sg_on = small_genome == 1 || (small_genome == 0 && mi->nt->l_seq <= small_genome_size);
		if (sg_on) {
			if (!set_sg_kmer) {
				mo.ps.kmer = 5;
				if (tier == 3 && !set_ps_thr) mo.ps.kmer_thr = mp_ps_kmer_thr(&mo.ps) - 1; // re-derive the sensitive-tier threshold for the new k
			}
			if (!set_sg_ps) mo.acc_min_ps = 0;
			if (!set_sg_score) mo.acc_min_out_score = 0;
			if (!set_sg_posr) mo.acc_min_pos_ratio = 0.0f;
			if (mp_verbose >= 2)
				fprintf(stderr, "[M::%s] small-genome mode%s (genome %.1f Mb, threshold %.0f Mb): --ps-kmer=%d --acc-min-ps=%d --acc-min-score=%d --acc-min-pos-ratio=%g; weak hits are kept, filter with protlocus-recheck (E<=1e-5)\n",
					__func__, small_genome == 1? " (forced on)" : "", mi->nt->l_seq / 1e6, small_genome_size / 1e6,
					mo.ps.kmer, mo.acc_min_ps, mo.acc_min_out_score, mo.acc_min_pos_ratio);
		}
	} else if (small_genome == 1 && mp_verbose >= 1)
		fprintf(stderr, "[WARNING]\033[1;31m --small-genome=on has no effect without --prescreen\033[0m\n");
	if (mp_verbose >= 3) mp_idx_print_stat(mi, mo.max_occ);
	if (fn_idx != 0) mp_idx_dump(fn_idx, mi);
	if (fn_spsc != 0) mp_set_spsc(fn_spsc, mi, &mo, keep_io);
	if (fn_thr != 0) {
		thr_tab = mp_thr_tab_read(fn_thr);
		if (thr_tab == 0) {
			if (mp_verbose >= 1)
				fprintf(stderr, "[ERROR]\033[1;31m failed to open threshold table %s\033[0m\n", fn_thr);
			mp_idx_destroy(mi);
			return 1;
		}
		mo.thr_tab = thr_tab;
	}
	if (fn_cand != 0) {
		cand_tab = mp_cand_tab_read(fn_cand, mi->nt, mo.cand_max_evalue);
		if (cand_tab == 0) {
			if (mp_verbose >= 1)
				fprintf(stderr, "[ERROR]\033[1;31m failed to open candidate table %s\033[0m\n", fn_cand);
			mp_thr_tab_destroy(thr_tab);
			mp_idx_destroy(mi);
			return 1;
		}
		mo.cand_tab = cand_tab;
	} else if (use_prescreen) {
		prescreen = mp_prescreen_build(mi->nt, &mo.ps, n_threads);
		mo.prescreen = prescreen;
	} else if (mo.cand_max_evalue > 0.0 && mp_verbose >= 1) {
		fprintf(stderr, "[WARNING]\033[1;31m --tier/--cand-max-evalue has no effect without --cand-table or --prescreen\033[0m\n");
	}
	for (i = o.ind + 1; i < argc; ++i) {
		int32_t res = mp_map_file(mi, argv[i], &mo, n_threads); // .hmm files (one or many profiles) are detected by suffix
		if (res != 0) {
			fprintf(stderr, "[M::%s] ERROR during mapping %s (check files exists and are amino acid fastas or HMMER3 profiles)\n", __func__, argv[i]);
			mp_prescreen_destroy(prescreen);
			mp_cand_tab_destroy(cand_tab);
			mp_thr_tab_destroy(thr_tab);
			mp_idx_destroy(mi);
			return 1;
		}
	}
	mp_prescreen_destroy(prescreen);
	mp_cand_tab_destroy(cand_tab);
	mp_thr_tab_destroy(thr_tab);
	mp_idx_destroy(mi);

	if (mp_verbose >= 3) {
		fprintf(stderr, "[M::%s] Version: %s\n", __func__, PROTLOCUS_VERSION);
		fprintf(stderr, "[M::%s] CMD:", __func__);
		for (i = 0; i < argc; ++i)
			fprintf(stderr, " %s", argv[i]);
		fprintf(stderr, "\n[M::%s] Real time: %.3f sec; CPU: %.3f sec; Peak RSS: %.3f GB\n", __func__, mp_realtime(), mp_cputime(), mp_peakrss() / 1024.0 / 1024.0 / 1024.0);
	}
	return 0;
}
