#ifndef MPPRIV_H
#define MPPRIV_H

#include <stdio.h>
#include "protlocus.h"
#include "nasw.h"
#include "kseq.h"

#define MP_DBG_NO_KALLOC   0x1
#define MP_DBG_QNAME       0x2
#define MP_DBG_NO_REFINE   0x4
#define MP_DBG_MORE_DP     0x8
#define MP_DBG_ANCHOR      0x10
#define MP_DBG_CHAIN       0x20

#define MP_PARENT_UNSET   (-1)
#define MP_PARENT_TMP_PRI (-2)

#define MP_ALIGN_RESCUED 0x1
#define MP_ALIGN_FULL    0x2

typedef struct mp_hmm_batch_s mp_hmm_batch_t;

#ifndef kroundup64
#define kroundup64(x) (--(x), (x)|=(x)>>1, (x)|=(x)>>2, (x)|=(x)>>4, (x)|=(x)>>8, (x)|=(x)>>16, (x)|=(x)>>32, ++(x))
#endif

#ifdef __cplusplus
extern "C" {
#endif

extern int8_t mp_mat_blosum62[484];

struct mp_bseq_file_s;
typedef struct mp_bseq_file_s mp_bseq_file_t;

typedef struct {
	int32_t l_seq;
	char *name, *seq, *comment;
} mp_bseq1_t;

// from bseq.c
mp_bseq_file_t *mp_bseq_open(const char *fn);
void mp_bseq_close(mp_bseq_file_t *fp);
mp_bseq1_t *mp_bseq_read(mp_bseq_file_t *fp, int64_t chunk_size, int with_comment, int *n_);

// from misc.c
char *mp_strdup(const char *src);
void radix_sort_mp64(uint64_t *st, uint64_t *en);
void radix_sort_mp128x(mp128_t *st, mp128_t *en);

// from sys.c
double mp_realtime(void);
double mp_cputime(void);
long mp_peakrss(void);
double mp_percent_cpu(void);

// from ntseq.c
mp_ntdb_t *mp_ntseq_read(const char *fn);
void mp_ntseq_destroy(mp_ntdb_t *db);
void mp_ntseq_dump(FILE *fp, const mp_ntdb_t *nt);
mp_ntdb_t *mp_ntseq_restore(FILE *fp);
int64_t mp_ntseq_get(const mp_ntdb_t *db, int32_t cid, int64_t st, int64_t en, int32_t rev, uint8_t *seq);
int64_t mp_ntseq_get_by_v(const mp_ntdb_t *nt, int32_t vid, int64_t st, int64_t en, uint8_t *seq);
int64_t mp_ntseq_spsc_get(const mp_ntdb_t *db, int32_t cid, int64_t st0, int64_t en0, int32_t rev, uint8_t *sc);
int64_t mp_ntseq_spsc_get_by_v(const mp_ntdb_t *nt, int32_t vid, int64_t st, int64_t en, uint8_t *ss);

// from sketch.c
void mp_sketch_nt4(void *km, const uint8_t *seq, int64_t len, int32_t min_aa_len, int32_t kmer, int32_t mod_bit, int32_t bbit, int64_t boff, mp64_v *a);
void mp_sketch_prot(void *km, const char *seq, int32_t len, int32_t kmer, int32_t mod_bit, mp64_v *a);
void mp_sketch_hmm(void *km, const int16_t *emit, int32_t len, int32_t kmer, int32_t mod_bit, int32_t pos_loss, int32_t win_loss, int32_t max_per_win, mp64_v *a);

// from index.c
int32_t mp_idx_block2pos(const mp_idx_t *mi, uint32_t b);

// from hit.c
int32_t mp_cal_chn_sc_ungap(int32_t n_a, const uint64_t *a, int32_t kmer);
int32_t mp_reg_score(const mp_reg1_t *r);
int32_t mp_reg_try_replace(mp_reg1_t *dst, mp_reg1_t *trial);
void mp_set_rank_score(const mp_mapopt_t *opt, int32_t qlen, mp_reg1_t *r);
int32_t mp_reg_needs_refine(const mp_mapopt_t *opt, const mp_reg1_t *r);
void mp_sort_reg(void *km, int *n_regs, mp_reg1_t *r, int32_t rank_tiebreak);
void mp_set_parent(void *km, float mask_level, int mask_len, int n, mp_reg1_t *r, int sub_diff, int hard_mask_level);
void mp_select_sub(void *km, float pri_ratio, int min_diff, int best_n, int *n_, mp_reg1_t *r);
uint64_t *mp_cal_max_ext(void *km, const mp_ntdb_t *db, int32_t n_reg, mp_reg1_t *reg, const uint64_t *a, int32_t min_ext, int32_t max_ext, int32_t iso_ext);
uint64_t *mp_collate_a(void *km, int32_t n_reg, mp_reg1_t *reg);
void mp_select_multi_exon(int32_t n, mp_reg1_t *r, int32_t single_penalty, int32_t strict);
int32_t mp_reject_spurious_intron(void *km, int32_t n, mp_reg1_t *r, int32_t strict, int32_t min_intron, float ratio, int32_t abs_max, int32_t free_removed);
int32_t mp_reject_low_purity_span(void *km, int32_t n, mp_reg1_t *r, float span_ratio, int32_t min_excess, int32_t free_removed);
int32_t mp_reject_weak_hit(void *km, int32_t n, mp_reg1_t *r, int32_t min_score, float min_pos_ratio, int32_t rescue_np, int32_t rescue_score, int32_t min_np, int32_t free_removed, int32_t use_ps, int32_t min_ps, int32_t ps_rescue_score, float ps_rescue_posr);
// --gpu-prefilter: rank coarse candidates by GPU gapless score and keep at most top_k (per-contig/strand
// diversity floor min_per_vid); scores[i] corresponds to reg[i] before the call. adaptive_k() picks K per
// query (used with --gpu-prefilter-adaptive) from the near-tie count above a per-batch genome calibration.
#define MP_GAPLESS_TIE_PCT 80 // runner-up within 80% of the winner's gapless score counts as a contender
#define MP_GAPLESS_DUP_RICH 0.25f // >=25% of queries with a contender: duplication-rich genome, keep one spare slot
#define MP_GAPLESS_ADAPT_MAX_K 4 // --gpu-prefilter-adaptive never keeps more than this many candidates per query
typedef struct {
	int32_t noise; // median gapless score (half-bits) of non-winning candidates in the batch; 0 = uncalibrated
	int32_t n_query; // queries that contributed
	float dup_frac; // fraction of queries with a near-tie runner-up
} mp_gapless_calib_t;
void mp_gapless_calibrate(int32_t n_query, const int32_t *cand_off, const int32_t *scores, mp_gapless_calib_t *cal);
int32_t mp_gapless_adaptive_k(int32_t n_reg, const int32_t *scores, int32_t qlen, int32_t max_k, const mp_gapless_calib_t *cal);
void mp_select_gapless(void *km, int32_t top_k, int32_t min_per_vid, int32_t *n_reg, mp_reg1_t *reg, const int32_t *scores);

// from chain.c
uint64_t *mp_chain(int32_t max_dist_x, int32_t max_dist_y, int32_t bw, int32_t max_skip, int32_t max_iter, int32_t min_cnt, int32_t min_sc, float chn_coef_log,
				   int32_t is_spliced, int32_t kmer, int32_t bbit, int64_t n, uint64_t *a, int32_t *n_u_, uint64_t **_u, void *km);

mp_reg1_t *mp_reg_gen_from_block(void *km, const mp_idx_t *mi, int32_t n_u, const uint64_t *u, const uint64_t *a, int32_t *n_reg);

// from format.c
void mp_write_output(kstring_t *s, void *km, const mp_idx_t *mi, const mp_bseq1_t *seq, const mp_reg1_t *r, const mp_mapopt_t *opt, int64_t id, int32_t hit_idx);

// from align.c
typedef struct {
	double t_ext, t_gap, t_full;
	uint64_t n_gap_calls, n_cells;
	uint64_t n_ext_calls, n_ext_cells, n_ext_retry; // retry = 2nd io_end DP call when the 1st didn't reach the query boundary
	uint64_t n_full_calls, n_full_cells; // full_dp/rescue whole-window DP calls, counted separately from gap-patch calls above
	uint64_t n_ext_calls_unguarded, n_ext_cells_unguarded; // ext calls/cells NOT covered by the acc_rescue_cells budget (i.e. !rescued && !force_full)
} mp_align_prof_t;

int32_t mp_align(void *km, const mp_mapopt_t *opt, const mp_idx_t *mi, int32_t len, const char *seq, mp_reg1_t *r, int32_t extl0, int32_t extr0, int32_t force_full, mp_align_prof_t *aprof, mp_hmm_batch_t *hb);
// align n candidates of one query together: their end-extension DPs are batched into a few GPU launches
// (CPU otherwise); ext[i] packs extl0<<32|extr0 (NULL: no caps); status[i] receives the mp_align() return value
void mp_align_batch(void *km, const mp_mapopt_t *opt, const mp_idx_t *mi, int32_t len, const char *seq, int32_t n, mp_reg1_t **r, const uint64_t *ext, int32_t force_full, mp_align_prof_t *aprof, int32_t *status, mp_hmm_batch_t *hb);
int32_t mp_hmm_dp_score(const mp_hmm_t *hmm, const uint8_t *nt, int32_t nlen);
int32_t mp_hmm_dp_score_gpu(const mp_hmm_t *hmm, const uint8_t *nt, int32_t nlen);
// score n regions (nt[off[i]..off[i]+len[i]) of one concatenated buffer in a single kernel launch;
// returns 0 when the GPU path is unavailable (score[] untouched)
int mp_hmm_dp_score_gpu_batch(const mp_hmm_t *hmm, int32_t n, const uint8_t *nt, int64_t l_nt, const int32_t *off, const int32_t *len, int32_t *score);
void **mp_hmm_gpu_slot(const mp_hmm_t *hmm, int which); // 0: emissions, 1: transitions
void mp_hmm_gpu_release(mp_hmm_t *hmm);

// from hmm.c: collect the regions of one query and score them together (GPU batch, CPU fallback);
// flush sets r->p->profile_score and recomputes the rank score of every pending region
mp_hmm_batch_t *mp_hmm_batch_init(void *km, const mp_mapopt_t *opt, int32_t qlen); // NULL when opt->hmm is unset
void mp_hmm_batch_add(mp_hmm_batch_t *b, mp_reg1_t *r, const uint8_t *nt, int32_t nlen);
void mp_hmm_batch_flush(mp_hmm_batch_t *b);
void mp_hmm_batch_destroy(mp_hmm_batch_t *b);

static inline float mp_log2(float x) // NB: this doesn't work when x<2
{
	union { float f; uint32_t i; } z = { x };
	float log_2 = ((z.i >> 23) & 255) - 128;
	z.i &= ~(255 << 23);
	z.i += 127 << 23;
	log_2 += (-0.34484843f * z.f + 2.02466578f) * z.f - 0.67487759f;
	return log_2;
}

static inline uint32_t mp_n_bucket(const mp_idxopt_t *io)
{
	return 1U << (io->kmer * MP_BITS_PER_AA - io->mod_bit);
}

#ifdef __cplusplus
}
#endif

#endif
