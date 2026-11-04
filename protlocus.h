#ifndef PROTLOCUS_H
#define PROTLOCUS_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define PROTLOCUS_VERSION "0.1.0"
#define MP_VERSION PROTLOCUS_VERSION // compatibility alias for the inherited C API

#define MP_F_NO_SPLICE    0x1
#define MP_F_NO_ALIGN     0x2
#define MP_F_SHOW_UNMAP   0x4
#define MP_F_GFF          0x8
#define MP_F_NO_PAF       0x10
#define MP_F_GTF          0x20
#define MP_F_NO_PRE_CHAIN 0x40
#define MP_F_SHOW_RESIDUE 0x80
#define MP_F_SHOW_TRANS   0x100
#define MP_F_NO_CS        0x200
#define MP_F_ACCURACY     0x400
#define MP_F_PROFILE      0x800
#define MP_F_GPU          0x1000
#define MP_F_NO_CAND_DIAG 0x2000 // --no-cand-diag: do not seed anchors from the prescreen diagonal when the refine chain is empty
#define MP_F_GPU_PREFILTER 0x4000 // --gpu-prefilter: GPU gapless X-drop candidate prefilter (CUDA backend only); reduces coarse-chain candidates per query before the refine stage
#define MP_F_GPU_PREFILTER_ADAPTIVE 0x8000 // --gpu-prefilter-adaptive: pick K from score margin (mp_gapless_adaptive_k) instead of a fixed --gpu-prefilter-topk

#define MP_ACC_REPEAT_RESCUE 0x1
#define MP_ACC_VID_CHAIN     0x2
#define MP_ACC_ANCHOR_RESCUE 0x4
#define MP_ACC_RERANK        0x8
#define MP_ACC_JUNC_REFINE   0x10
#define MP_ACC_END_EXT_CAP   0x20
#define MP_ACC_END_EXT_RETRY 0x40 // opt-in: also retry rescued (no-trusted-anchor) candidates with io_end (off by default)
#define MP_ACC_END_AA_CAP    0x80 // opt-in: cap the (uncapped by default) aa-side extent of the first end-extension DP call for rescued candidates (off by default)
#define MP_ACC_END_DYN_WIN   0x100 // opt-in: scale the nt-side/aa-side end-extension caps with query protein length instead of using a fixed cap (off by default)
#define MP_ACC_STRICT_MULTI_EXON 0x200 // opt-in: only prefer a multi-exon hit over a single-exon hit when every junction is canonical and the score margin covers every intron; also drops any multi-exon candidate that has any single intron exceeding the nt evidence of its own two immediately flanking exons (see acc_intron_evidence_min/ratio), off by default
#define MP_ACC_TRIM_INTRON   0x400 // opt-in (needs STRICT_MULTI_EXON): cut a candidate at every spurious junction right after alignment and keep the best exon block, instead of letting mp_reject_spurious_intron() drop the whole candidate
#define MP_ACC_REFINE_LOCAL  0x800 // opt-in: in mp_refine_reg() keep the best kmer2 chain that overlaps the coarse locus (fallback: window max) so a degenerate-seed chain elsewhere in the +-max_ext window cannot hijack the candidate
#define MP_ACC_REFINE_PSSM   0x1000 // opt-in (HMM mode): rank refine chains by X-drop ungapped PSSM evidence on their anchor diagonals before chain score
#define MP_ACC_ALL           0x3f

#define MP_FEAT_CDS       0
#define MP_FEAT_STOP      1

#define MP_BITS_PER_AA    4
#define MP_BLOCK_BONUS    2

#define MP_CODON_STD 1
#define MP_IDX_MAGIC "MPI\3"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct { uint64_t x, y; } mp128_t;
typedef struct { int32_t n, m; mp128_t *a; } mp128_v;
typedef struct { int32_t n, m; uint64_t *a; } mp64_v;

typedef struct {
	int32_t bbit; // block bit
	int32_t min_aa_len; // ignore ORFs shorter than this
	int32_t kmer, mod_bit;
	uint32_t trans_code;
} mp_idxopt_t;

typedef struct mp_hmm_s mp_hmm_t;
typedef struct mp_hmm_file_s mp_hmm_file_t;
typedef struct mp_thr_tab_s mp_thr_tab_t;
typedef struct mp_cand_tab_s mp_cand_tab_t;
typedef struct mp_prescreen_s mp_prescreen_t;

typedef struct {
	uint32_t vid; // cid<<1 | rev
	int64_t vs, ve; // oriented half-open nt coordinates, same convention as mp_reg1_t
	double ev; // best (smallest) E-value supporting this locus, or -1 when unknown
	int32_t aqs, aqe; // --prescreen only: query [aqs,aqe) of the best ungapped diagonal segment (aqs<0: none); fallback anchors when the 5-mer refine chain is empty
	int32_t asc; // half-bit PSSM score of that segment
	int64_t avs; // oriented nt start of the codon aligned to query position aqs
} mp_cand1_t;

// Built-in mmseqs-like profile prescreen (--prescreen): six-frame ORF k-mer index + profile k-mer prefilter +
// banded gapped alignment + Karlin-Altschul E-value; replaces the external mmseqs --cand-table for HMM queries.
typedef struct {
	int32_t kmer; // spaced k-mer size, 5..7 (mmseqs spaced patterns)
	int32_t min_orf; // ignore stop-free ORFs shorter than this many aa (esl-translate -l)
	float sens; // mmseqs -s; sets kmer_thr when kmer_thr <= 0
	int32_t kmer_thr; // profile k-mer score threshold in half-bits (<=0: derive from sens)
	int32_t kmer_max; // cap on similar k-mers generated per query window
	int32_t min_ungapped; // --min-ungapped-score, half-bits
	int32_t max_seqs; // --max-seqs: ORFs per query passed to the gapped stage
	int32_t band; // gapped alignment band (aa) around the best ungapped diagonal (<=0: full matrix)
	int32_t go, ge; // gap open/extend, half-bits (mmseqs 11/1)
	const char *dump_fn; // write the generated candidate table here (same TSV as --cand-table)
} mp_psopt_t;

typedef struct {
	uint32_t flag;
	int64_t mini_batch_size;
	int32_t max_occ;
	int32_t max_gap; // max gap on the query protein, in aa
	int32_t max_intron;
	int32_t min_max_intron, max_max_intron;
	int32_t bw; // bandwidth, in aa
	int32_t max_ext;
	int32_t max_ava;
	int32_t min_chn_cnt;
	int32_t max_chn_max_skip;
	int32_t max_chn_iter;
	int32_t min_chn_sc;
	float chn_coef_log;
	float mask_level;
	int32_t mask_len;
	float pri_ratio;
	float out_sim, out_cov;
	int32_t best_n, out_n;
	int32_t kmer2;
	uint32_t acc_flags;
	int32_t acc_repeat_cap;
	int64_t acc_anchor_budget, acc_rescue_cells;
	float acc_pri_ratio;
	int32_t acc_best_n, acc_microexon_max;
	int32_t acc_coarse_n; // with REFINE_PSSM: keep up to this many coarse chains for refine, then truncate to best_n by PSSM evidence (<=best_n disables)
	int32_t acc_cov_bonus, acc_splice_bonus, acc_noncan_penalty;
	int32_t acc_fs_penalty, acc_stop_penalty, acc_end_bonus;
	int32_t acc_rerank_margin;
	int32_t acc_end_ext; // initial cap on speculative terminal-exon search; also the ceiling for the dynamic nt-side cap below
	int32_t acc_end_retry; // retry budget when capped candidates would otherwise leave the query unmapped
	int32_t acc_end_retry_win; // nt window cap for the io_end 2nd-try short-terminal-exon DP
	int32_t acc_end_aa_ext; // opt-in cap on the aa-side extent of the first end-extension DP call for rescued candidates; also the ceiling for the dynamic aa-side cap below
	int32_t acc_end_ext_min; // floor for the dynamic (query-length-scaled) nt-side cap; see MP_ACC_END_DYN_WIN
	float acc_end_nt_per_aa; // nt-side dynamic cap = clamp(qlen * acc_end_nt_per_aa, acc_end_ext_min, acc_end_ext)
	int32_t acc_end_aa_min; // floor for the dynamic (query-length-scaled) aa-side cap; see MP_ACC_END_DYN_WIN
	float acc_end_aa_ratio; // aa-side dynamic cap = clamp(qlen * acc_end_aa_ratio, acc_end_aa_min, acc_end_aa_ext)
	int32_t acc_intron_evidence_min; // only MP_ACC_STRICT_MULTI_EXON junctions with an intron at least this long are checked against acc_intron_evidence_ratio
	float acc_intron_evidence_ratio; // MP_ACC_STRICT_MULTI_EXON drops a candidate if any one junction's intron exceeds EITHER its two immediately flanking exons' combined nt OR the whole candidate's total exonic nt, times this ratio
	int32_t acc_intron_abs_max; // MP_ACC_STRICT_MULTI_EXON drops a candidate if any single intron exceeds this nt length regardless of flanking/whole-candidate exonic evidence (<=0 disables); backstops cases where ratio checks are defeated by individually large exons
	float acc_span_ratio; // drop a candidate whose genomic span exceeds this multiple of its aligned query length in nt, once the excess reaches acc_intron_evidence_min (<=0 disables); catches low-purity hits inflated by several individually-passable gaps on compact genomes (see --compact)
	int32_t acc_min_out_score; // drop a candidate whose DP score (PAF AS) is below this (<=0 disables); a poor man's significance cutoff for queries with no true homolog, where the best hit is a negative-score junk alignment
	float acc_min_pos_ratio; // drop a candidate whose positive-substitution fraction 3*n_plus/blen falls below this (<=0 disables); junk alignments run 0.25-0.45 while true homologs sit well above 0.5
	int32_t acc_rescue_np; // exempt a candidate from the two floors above when n_plus >= this (<=0 disables); saves long true hits whose density is diluted by introns/gaps but whose absolute positive evidence is high
	int32_t acc_rescue_score; // exempt a candidate from the pos-ratio floor when dp_score >= this (<=0 disables); with PSSM scoring a high AS is significant even at posr slightly under the floor
	int32_t acc_min_np; // unconditionally drop a candidate with n_plus < this (<=0 disables); kills short high-density grazing junk that passes the score/pos-ratio floors
	const mp_thr_tab_t *thr_tab; // optional per-query DP-score floor table (--thr-table); overrides acc_min_out_score for listed queries
	int32_t thr_fallback; // floor for queries absent from thr_tab (half-bits; <=0 = keep acc_min_out_score)
	int32_t thr_use_ps; // --thr-use-ps: apply acc_min_out_score/thr_tab floor to profile_score (ps) instead of dp_score (AS) when available
	int32_t acc_min_ps; // --acc-min-ps: additional floor on profile_score (ps) applied together with the AS floor/thr_tab (<=0 disables); ps alone is uncalibrated (fully local Viterbi, ~40 half-bits by chance on long windows), so it must sit on top of the AS floor. Only HMM-profile hits carry ps, so the default (60) is a no-op for protein queries
	int32_t acc_ps_rescue_score; // --acc-ps-rescue: exempt a hit from the ps floor when dp_score >= this AND posr >= acc_ps_rescue_posr (<=0 disables); ps is unnormalised so short models fail ps>=60 even when the alignment is strong
	float acc_ps_rescue_posr;
	const mp_cand_tab_t *cand_tab; // --cand-table: external candidate loci replace the sketch/anchor/chain stage; each locus is refined within +-max_ext
	double cand_max_evalue; // --tier/--cand-max-evalue: keep only --cand-table rows (or --prescreen hits) whose E-value is <= this; <=0 keeps every row
	const mp_prescreen_t *prescreen; // --prescreen: built-in ORF index; each HMM batch gets its candidate loci from mp_prescreen_batch() instead of --cand-table
	mp_psopt_t ps; // --prescreen tunables
	int32_t hmm_seed_pos_loss; // HMM queries: a match state offers alternative residue groups whose emission is within this of its best (units of 2*ln p)
	int32_t hmm_seed_win_loss; // HMM queries: total emission loss budget for one k-mer seed across its k positions
	int32_t hmm_seed_max; // HMM queries: at most this many profile-derived seeds per k-window in the coarse (genome-wide) stage; <=0 = consensus only
	int32_t hmm_seed_max2; // same budget for the refine stage (region-local, cheap)
	int32_t acc_refine_min_seg; // MP_ACC_REFINE_PSSM: an ungapped diagonal segment counts as evidence only if >= this (half-bits)
	int32_t acc_refine_xdrop; // MP_ACC_REFINE_PSSM: X-drop for the diagonal extension (half-bits)
	float acc_trim_side_ratio; // MP_ACC_TRIM_INTRON: also cut a junction whose intron exceeds the exonic nt on its smaller side times this (<=0 disables); removes 1-40 aa exons hopped to across 200-1300 nt

	int32_t gpu_device;
	int64_t gpu_min_cells;
	int32_t gpu_prefilter_top_k; // --gpu-prefilter-topk: retain at most this many coarse candidates per query [32]
	int32_t gpu_prefilter_min_per_group; // --gpu-prefilter-diversity: reserve this many survivors per contig/strand group [1]
	int32_t gpu_prefilter_xdrop; // --gpu-prefilter-xdrop: gapless extension X-drop threshold, half-bits [40]
	int32_t go, ge, io, fs; // gap open, extension, intron open, and frame-shift/stop-codon
	int32_t io_end;
	float ie_coef;
	int32_t sp_model;
	int32_t sp_null_bonus, sp_max_bonus;
	float sp_scale;
	int32_t xdrop;
	int32_t end_bonus;
	int32_t asize; // size of the alphabet; always 22 in current implementation
	int32_t gff_delim;
	int32_t max_intron_flank;
	const char *gff_prefix;
	const mp_hmm_t *hmm;
	int8_t mat[484];
} mp_mapopt_t;

typedef struct {
	uint32_t n, m;
	uint64_t *a; // pos<<56 | score<<1 | acceptor
} mp_spsc_t;

typedef struct {
	int64_t off, len;
	char *name;
} mp_ctg_t;

typedef struct {
	int32_t n_ctg, m_ctg;
	int32_t l_name;
	int64_t l_seq, m_seq;
	uint8_t *seq; // TODO: separate this into multiple blocks; low priority
	mp_ctg_t *ctg;
	char *name;
	void *h; // hash table to map sequence name to cid
	mp_spsc_t *spsc; // of size n_ctg*2
	int32_t seq_mmap, name_mmap;
} mp_ntdb_t;

typedef struct {
	mp_idxopt_t opt;
	uint32_t n_block;
	mp_ntdb_t *nt;
	int64_t n_kb, *ki;
	uint32_t *bo, *kb;
	void *idx_mmap;
	size_t idx_mmap_len;
} mp_idx_t;

typedef struct {
	int32_t dp_score, dp_max, dp_max2;
	int32_t n_cigar, m_cigar;
	int32_t blen; // CDS length in alignment
	int32_t n_fs; // number of frameshift events
	int32_t n_stop; // number of in-frame stop codons
	int32_t dist_stop; // distance in bp to the closest stop codon
	int32_t dist_start; // distance in bp the the closest 'M'
	int32_t n_iden, n_plus;
	int32_t profile_score; // CPU HMM Viterbi score, or INT32_MIN for FASTA queries
	int32_t full_dp; // 1 if produced by an unguided full-region DP (anchor-rescue or junc-refine force_full):
	                 // dp_max can be an early local peak far above the real final dp_score; use dp_score for ranking
	uint32_t cigar[];
} mp_extra_t;

typedef struct {
	int64_t vs, ve;
	int32_t qs, qe;
	int16_t type, phase;
	int32_t n_fs, n_stop;
	int32_t score, n_iden, blen;
	char donor[2], acceptor[2];
} mp_feat_t;

typedef struct {
	int32_t off, cnt;
	int32_t id, parent;
	int32_t n_sub, subsc;
	int32_t n_feat, m_feat, n_exon;
	int32_t chn_sc;
	int32_t chn_sc_ungap;
	int32_t rank_sc, rank_sc2;
	uint32_t hash;
	uint32_t vid; // cid<<1 | rev
	int32_t qs, qe;
	int64_t vs, ve;
	uint64_t *a;
	mp_feat_t *feat;
	mp_extra_t *p;
} mp_reg1_t;

struct mp_tbuf_s;
typedef struct mp_tbuf_s mp_tbuf_t;

extern int32_t mp_verbose, mp_dbg_flag;
extern char *ns_tab_nt_i2c, *ns_tab_aa_i2c;
extern uint8_t ns_tab_a2r[22], ns_tab_nt4[256], ns_tab_aa20[256], ns_tab_aa13[256];
extern uint8_t ns_tab_codon[64], ns_tab_codon13[64];

/**
 * Set the translation table and builtin timer
 *
 * Run this before all other ProtLocus routines.
 */
void mp_start(void);

/**
 * Initialize the default parameters for indexing
 *
 * @param io       struct to initialize (out)
 */
void mp_idxopt_init(mp_idxopt_t *io);

/**
 * Initialize the default parameters fr mapping
 *
 * @param mo       struct to initialize (out)
 */
void mp_mapopt_init(mp_mapopt_t *mo);

/**
 * Set frameshift and stop codon penalty
 *
 * This function changes the mp_mapopt_t::fs field and mp_mapopt_t::mat[]
 *
 * @param mo       mapping options (in/out)
 * @param fs       frameshift and stop codon penalty (positive)
 */
void mp_mapopt_set_fs(mp_mapopt_t *mo, int32_t fs);

mp_hmm_t *mp_hmm_read(const char *fn); // first profile in the file
mp_hmm_file_t *mp_hmm_open(const char *fn);
mp_hmm_t *mp_hmm_read1(mp_hmm_file_t *fp); // next profile, or NULL at EOF/error
int mp_hmm_error(const mp_hmm_file_t *fp);
void mp_hmm_close(mp_hmm_file_t *fp);
void mp_hmm_destroy(mp_hmm_t *hmm);
const char *mp_hmm_name(const mp_hmm_t *hmm);
const char *mp_hmm_consensus(const mp_hmm_t *hmm);
int32_t mp_hmm_length(const mp_hmm_t *hmm);
const int16_t *mp_hmm_emissions(const mp_hmm_t *hmm);
const int16_t *mp_hmm_transitions(const mp_hmm_t *hmm);
const int8_t *mp_hmm_pssm(const mp_hmm_t *hmm); // [len][22] half-bit log-odds, nasw residue order; drives the alignment DP in HMM mode
const int8_t *mp_hmm_aa20_to_alpha_table(void); // [22] ns_tab_aa20 index -> mp_hmm_alphabet index (-1 for '*'/'X'); emit[] is stored in the latter order
void mp_hmm_set_stop_sc(mp_hmm_t *hmm, int32_t pen); // stop-codon column of the PSSM (default 23, like the BLOSUM stop row)

mp_thr_tab_t *mp_thr_tab_read(const char *fn); // TSV "query_name<TAB>min_AS"
int32_t mp_thr_tab_get(const mp_thr_tab_t *t, const char *name, int32_t fallback);
void mp_thr_tab_destroy(mp_thr_tab_t *t);

mp_cand_tab_t *mp_cand_tab_read(const char *fn, mp_ntdb_t *nt, double max_evalue); // TSV "query_name<TAB>chrom<TAB>start0<TAB>end0<TAB>strand[<TAB>evalue]"; rows with evalue > max_evalue (>0) are skipped
mp_cand_tab_t *mp_cand_tab_init(void);
void mp_cand_tab_set(mp_cand_tab_t *t, const char *name, int32_t n, mp_cand1_t *a); // takes ownership of a[]; sorts and merges overlapping loci per strand
const mp_cand1_t *mp_cand_tab_get(const mp_cand_tab_t *t, const char *name, int32_t *n);
void mp_cand_tab_destroy(mp_cand_tab_t *t);

void mp_psopt_init(mp_psopt_t *po);
int32_t mp_ps_kmer_thr(const mp_psopt_t *po);
mp_prescreen_t *mp_prescreen_build(const mp_ntdb_t *nt, const mp_psopt_t *po, int32_t n_threads); // six-frame ORF DB + spaced k-mer index
void mp_prescreen_destroy(mp_prescreen_t *ps);
// candidate loci for a batch of profiles (parallel over queries); rows with E > max_evalue (>0) are dropped; dump (optional) gets the TSV rows
mp_cand_tab_t *mp_prescreen_batch(const mp_prescreen_t *ps, const mp_psopt_t *po, double max_evalue, int32_t n, mp_hmm_t **hmm, int32_t n_threads, FILE *dump);
int32_t mp_hmm_dp_score_gpu(const mp_hmm_t *hmm, const uint8_t *nt, int32_t nlen);

/**
 * Set max intron length based on the genome size
 *
 * @param mo       mapping options (in/out)
 * @param gsize    genome size
 */
void mp_mapopt_set_max_intron(mp_mapopt_t *mo, int64_t gsize);

/**
 * Check the integrity of mapping parameters (not complete)
 *
 * @param mo       mapping options
 *
 * @return 0 on success, or negative on failure
 */
int32_t mp_mapopt_check(const mp_mapopt_t *mo);

/**
 * Load or build an index
 *
 * If _fn_ corresponds to a FASTA file, build the index using _io_ and
 * _n_threads_. If _fn_ corresponds to a prebuilt index file, load the index
 * and ignore _io_ and _n_threads_.
 *
 * @param fn         FASTA or index file name
 * @param io         indexing options
 * @param n_threads  number of threads
 *
 * @return the index
 */
mp_idx_t *mp_idx_load(const char *fn, const mp_idxopt_t *io, int32_t n_threads);

/**
 * Deallocate the index
 *
 * @param mi         the index
 */
void mp_idx_destroy(mp_idx_t *mi);

/**
 * Write an index to disk
 *
 * @param fn         index file name
 * @param mi         the index
 *
 * @return 0 on success or negative on failure
 */
int mp_idx_dump(const char *fn, const mp_idx_t *mi);

/**
 * Read an index file
 *
 * @param fn         index file name
 *
 * @return the index, or NULL on failure
 */
mp_idx_t *mp_idx_restore(const char *fn);

/**
 * Load a splice score file
 *
 * @param nt        sequences in mp_idx_t::nt (in/out)
 * @param fn        splice score file name; can be optionally gzip'd
 * @param max_sc    cap splice scores at this parameter
 *
 * @return 0 on success, or negative on failure
 */
int32_t mp_ntseq_read_spsc(mp_ntdb_t *nt, const char *fn, int32_t max_sc);

void mp_set_spsc(const char *fn, mp_idx_t *mi, mp_mapopt_t *mo, int32_t keep_io);

/**
 * Align a protein sequence against the index
 *
 * @param mi        the index
 * @param qlen      protein length
 * @param seq       protein sequence
 * @param n_reg     number of alignments (out)
 * @param b         thread-local buffer; threads shouldn't share buffers (can be NULL)
 * @param opt       mapping parameters
 * @param qname     query name (used for breaking ties)
 *
 * @return list of alignments
 */
mp_reg1_t *mp_map(const mp_idx_t *mi, int qlen, const char *seq, int *n_reg, mp_tbuf_t *b, const mp_mapopt_t *opt, const char *qname);

/**
 * Initialize a thread-local buffer
 *
 * @return the buffer
 */
mp_tbuf_t *mp_tbuf_init(void);

/**
 * Deallocate a thread-local buffer
 *
 * @param b         the buffer
 */
void mp_tbuf_destroy(mp_tbuf_t *b);

// ignore the following for now
void mp_idx_print_stat(const mp_idx_t *mi, int32_t max_occ);
int32_t mp_map_file(const mp_idx_t *idx, const char *fn, const mp_mapopt_t *opt, int n_threads);

#ifdef __cplusplus
}
#endif

#endif
