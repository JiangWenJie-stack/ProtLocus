#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include "mppriv.h"
#include "kthread.h"
#include "kvec-km.h"
#include "kalloc.h"
#include "kseq.h"

typedef struct {
	uint64_t n_query, n_query_aa, n_seed;
	uint64_t n_anchor, n_anchor_rescued, n_coarse_chain;
	uint64_t n_prefilter_candidate, n_prefilter_survivor;
	uint64_t n_prefilter_k1, n_prefilter_k2, n_prefilter_k4, n_prefilter_k8;
	uint64_t n_refine_candidate, n_refined, n_exact_anchor, n_cand_diag_seeded;
	uint64_t n_dp_candidate, n_dp_rescued, n_dp_ext_eligible, n_dp_ext_retry, n_dp_ext_fallback;
	uint64_t n_dp_realign, n_dp_realign_kept, n_output;
	uint64_t n_align_gap_calls, n_align_cells;
	uint64_t n_align_ext_calls, n_align_ext_cells, n_align_ext_retry;
	uint64_t n_align_full_calls, n_align_full_cells;
	uint64_t n_align_ext_calls_unguarded, n_align_ext_cells_unguarded; // ext work not covered by the acc_rescue_cells budget (!rescued && !force_full)
	uint64_t n_refine_kmers, n_refine_pairs, n_refine_pairs_dropped, n_refine_buckets_capped;
	double t_sketch, t_anchor, t_prechain, t_chain, t_refine, t_align, t_output;
	double t_prefilter;
	double t_refine_sketch, t_refine_pair, t_refine_chain;
	double t_align_ext, t_align_gap, t_align_full;
	uint64_t km_max_blocks; // worst per-thread kalloc free-list length seen at a mini-batch boundary (fragmentation proxy)
	size_t km_max_capacity;
} mp_profile_t;

struct mp_tbuf_s {
	void *km;
	int rep_len;
	mp_profile_t profile;
};

static void mp_profile_add(mp_profile_t *dst, const mp_profile_t *src)
{
	dst->n_query += src->n_query, dst->n_query_aa += src->n_query_aa, dst->n_seed += src->n_seed;
	dst->n_anchor += src->n_anchor, dst->n_anchor_rescued += src->n_anchor_rescued, dst->n_coarse_chain += src->n_coarse_chain;
	dst->n_prefilter_candidate += src->n_prefilter_candidate, dst->n_prefilter_survivor += src->n_prefilter_survivor;
	dst->n_prefilter_k1 += src->n_prefilter_k1, dst->n_prefilter_k2 += src->n_prefilter_k2;
	dst->n_prefilter_k4 += src->n_prefilter_k4, dst->n_prefilter_k8 += src->n_prefilter_k8;
	dst->n_refine_candidate += src->n_refine_candidate, dst->n_refined += src->n_refined, dst->n_exact_anchor += src->n_exact_anchor;
	dst->n_cand_diag_seeded += src->n_cand_diag_seeded;
	dst->n_dp_candidate += src->n_dp_candidate, dst->n_dp_rescued += src->n_dp_rescued;
	dst->n_dp_ext_eligible += src->n_dp_ext_eligible, dst->n_dp_ext_retry += src->n_dp_ext_retry;
	dst->n_dp_ext_fallback += src->n_dp_ext_fallback;
	dst->n_dp_realign += src->n_dp_realign, dst->n_dp_realign_kept += src->n_dp_realign_kept, dst->n_output += src->n_output;
	dst->n_align_gap_calls += src->n_align_gap_calls, dst->n_align_cells += src->n_align_cells;
	dst->n_align_ext_calls += src->n_align_ext_calls, dst->n_align_ext_cells += src->n_align_ext_cells, dst->n_align_ext_retry += src->n_align_ext_retry;
	dst->n_align_full_calls += src->n_align_full_calls, dst->n_align_full_cells += src->n_align_full_cells;
	dst->n_align_ext_calls_unguarded += src->n_align_ext_calls_unguarded, dst->n_align_ext_cells_unguarded += src->n_align_ext_cells_unguarded;
	dst->n_refine_kmers += src->n_refine_kmers, dst->n_refine_pairs += src->n_refine_pairs;
	dst->n_refine_pairs_dropped += src->n_refine_pairs_dropped, dst->n_refine_buckets_capped += src->n_refine_buckets_capped;
	dst->t_sketch += src->t_sketch, dst->t_anchor += src->t_anchor, dst->t_prechain += src->t_prechain;
	dst->t_chain += src->t_chain, dst->t_refine += src->t_refine, dst->t_align += src->t_align, dst->t_output += src->t_output;
	dst->t_prefilter += src->t_prefilter;
	dst->t_refine_sketch += src->t_refine_sketch, dst->t_refine_pair += src->t_refine_pair, dst->t_refine_chain += src->t_refine_chain;
	dst->t_align_ext += src->t_align_ext, dst->t_align_gap += src->t_align_gap, dst->t_align_full += src->t_align_full;
	if (src->km_max_blocks > dst->km_max_blocks) dst->km_max_blocks = src->km_max_blocks;
	if (src->km_max_capacity > dst->km_max_capacity) dst->km_max_capacity = src->km_max_capacity;
}

static void mp_profile_add_align(mp_profile_t *p, const mp_align_prof_t *ap)
{
	p->t_align_ext += ap->t_ext, p->t_align_gap += ap->t_gap, p->t_align_full += ap->t_full;
	p->n_align_gap_calls += ap->n_gap_calls, p->n_align_cells += ap->n_cells;
	p->n_align_ext_calls += ap->n_ext_calls, p->n_align_ext_cells += ap->n_ext_cells, p->n_align_ext_retry += ap->n_ext_retry;
	p->n_align_full_calls += ap->n_full_calls, p->n_align_full_cells += ap->n_full_cells;
	p->n_align_ext_calls_unguarded += ap->n_ext_calls_unguarded, p->n_align_ext_cells_unguarded += ap->n_ext_cells_unguarded;
}

mp_tbuf_t *mp_tbuf_init(void)
{
	mp_tbuf_t *b;
	b = Kcalloc(0, mp_tbuf_t, 1);
	if (!(mp_dbg_flag & MP_DBG_NO_KALLOC))
		b->km = km_init();
	return b;
}

void mp_tbuf_destroy(mp_tbuf_t *b)
{
	if (b == 0) return;
	km_destroy(b->km);
	free(b);
}

static inline int32_t mp_pssm_codon_sc(const uint8_t *nt, int64_t p, const int8_t *pssm, int32_t j)
{
	uint8_t aa = nt[p] > 3 || nt[p+1] > 3 || nt[p+2] > 3? ns_tab_aa20['X'] : ns_tab_codon[nt[p]<<4 | nt[p+1]<<2 | nt[p+2]];
	return pssm[(size_t)j * 22 + aa];
}

// best X-drop ungapped PSSM segment on the diagonal through one refine anchor (x = last nt of the k-mer, q = its last aa); half-bits
static int32_t mp_diag_seg(const uint8_t *nt, int64_t l_nt, const int8_t *pssm, int32_t qlen, int32_t kmer, int64_t x, int32_t q, int32_t xdrop, int32_t *aa_e)
{
	int64_t p0 = x + 1 - 3 * kmer;
	int32_t q0 = q + 1 - kmer, j, sc, best, best_j, tot;
	for (j = q0, sc = 0, best = 0, best_j = q0 - 1; j < qlen; ++j) {
		int64_t p = p0 + 3 * (int64_t)(j - q0);
		if (p < 0) continue;
		if (p + 2 >= l_nt) break;
		sc += mp_pssm_codon_sc(nt, p, pssm, j);
		if (sc > best) best = sc, best_j = j;
		else if (best - sc > xdrop) break;
	}
	tot = best, *aa_e = best_j + 1;
	for (j = q0 - 1, sc = 0, best = 0; j >= 0; --j) {
		int64_t p = p0 + 3 * (int64_t)(j - q0);
		if (p < 0) break;
		if (p + 2 >= l_nt) continue;
		sc += mp_pssm_codon_sc(nt, p, pssm, j);
		if (sc > best) best = sc;
		else if (best - sc > xdrop) break;
	}
	return tot + best;
}

// sum of confident (>= min_seg) ungapped PSSM segments over a chain's anchors, each diagonal counted once
static int32_t mp_chain_pssm_ev(const uint8_t *nt, int64_t l_nt, const int8_t *pssm, int32_t qlen, int32_t kmer, int32_t n, const uint64_t *a, int32_t min_seg, int32_t xdrop)
{
	int32_t i, j, m = 0, ev = 0, seg_e[64];
	int64_t seg_d[64];
	for (i = 0; i < n; ++i) {
		int64_t x = a[i]>>32, d;
		int32_t q = (uint32_t)a[i], q0 = q + 1 - kmer, sc, e;
		d = (x + 1 - 3 * kmer) - 3 * (int64_t)q0;
		for (j = 0; j < m; ++j)
			if (seg_d[j] == d && q0 <= seg_e[j]) break;
		if (j < m) continue;
		sc = mp_diag_seg(nt, l_nt, pssm, qlen, kmer, x, q, xdrop, &e);
		if (m < 64) seg_d[m] = d, seg_e[m] = e, ++m;
		if (sc >= min_seg) ev += sc;
	}
	return ev;
}

static void mp_refine_reg(void *km, const mp_idx_t *mi, const mp_mapopt_t *opt, const mp64_v *sd_aa, mp_reg1_t *r, int32_t extl, int32_t extr, const int8_t *pssm, int32_t qlen, mp_profile_t *prof)
{
	const mp_idxopt_t *io = &mi->opt;
	int32_t i, j, k, n_u, max_sc = 0, max_i, max_ev = 0, kmer = opt->kmer2, is_splice = !(opt->flag&MP_F_NO_SPLICE);
	int32_t local = (opt->flag & MP_F_ACCURACY) && (opt->acc_flags & MP_ACC_REFINE_LOCAL);
	int64_t as, ae, l_nt, n_a, ctg_len = mi->nt->ctg[r->vid>>1].len;
	uint8_t *nt;
	uint64_t *a, *u;
	mp64_v sd = {0,0,0};
	double t0 = 0.0;

	as = r->vs > extl? r->vs - extl : 0;
	ae = r->ve + extr < ctg_len? r->ve + extr : ctg_len;
	if (prof) t0 = mp_realtime();
	nt = Kmalloc(km, uint8_t, ae - as);
	l_nt = mp_ntseq_get(mi->nt, r->vid>>1, r->vid&1? ctg_len - ae : as, r->vid&1? ctg_len - as : ae, r->vid&1, nt);
	mp_sketch_nt4(km, nt, l_nt, io->min_aa_len, kmer, 0, 0, 0, &sd);
	if (!pssm) kfree(km, nt), nt = 0; // kept only for the PSSM diagonal rescoring below
	if (prof) prof->n_refine_kmers += (uint64_t)sd.n; // genomic k-mers only, before merging the (fixed, precomputed once per query) protein k-mers below
	for (i = 0; i < sd_aa->n; ++i)
		kv_push(uint64_t, km, sd, sd_aa->a[i] | 1ULL<<31);
	radix_sort_mp64(sd.a, sd.a + sd.n);
	if (prof) prof->t_refine_sketch += mp_realtime() - t0;

	if (prof) t0 = mp_realtime();
	for (k = 0, i = 1, n_a = 0; i <= sd.n; ++i) { // precalculate n_a
		if (i == sd.n || sd.a[k]>>32 != sd.a[i]>>32) {
			int32_t n1, n2;
			for (j = k; j < i; ++j)
				if (sd.a[j]>>31&1) break;
			n1 = j - k, n2 = i - k - n1;
			if (n1 > 0 && n2 > 0) {
				if (n1 * n2 <= opt->max_ava)
					n_a += n1 * n2;
				else if (prof)
					prof->n_refine_pairs_dropped += (uint64_t)n1 * n2, ++prof->n_refine_buckets_capped;
			}
			k = i;
		}
	}
	if (prof) prof->n_refine_pairs += (uint64_t)n_a;
	a = Kmalloc(km, uint64_t, n_a);
	for (k = 0, i = 1, n_a = 0; i <= sd.n; ++i) { // populate a[]
		if (i == sd.n || sd.a[k]>>32 != sd.a[i]>>32) {
			int32_t n1, n2;
			for (j = k; j < i; ++j)
				if (sd.a[j]>>31&1) break;
			n1 = j - k, n2 = i - k - n1;
			if (n1 > 0 && n2 > 0 && n1 * n2 <= opt->max_ava) {
				int32_t i1, i2;
				for (i1 = k; i1 < k + n1; ++i1)
					for (i2 = k + n1; i2 < i; ++i2)
						a[n_a++] = (uint64_t)((uint32_t)sd.a[i1])<<32 | ((uint32_t)sd.a[i2]<<1>>1);
			}
			k = i;
		}
	}
	kfree(km, sd.a);
	if (prof) prof->t_refine_pair += mp_realtime() - t0;

	if (prof) t0 = mp_realtime();
	radix_sort_mp64(a, a + n_a);
	a = mp_chain(opt->max_intron, opt->max_gap, opt->bw, opt->max_chn_max_skip, opt->max_chn_iter, opt->min_chn_cnt, opt->min_chn_sc, opt->chn_coef_log, is_splice, kmer, 0, n_a, a, &n_u, &u, km);
	if (prof) prof->t_refine_chain += mp_realtime() - t0;
	if (n_u == 0) {
		r->cnt = 0, r->off = -1, r->a = 0;
		if (nt) kfree(km, nt);
		return;
	}

	// chain choice: (1) prefer chains touching the coarse locus (REFINE_LOCAL), (2) among those, the most ungapped PSSM
	// evidence (REFINE_PSSM: 5 colinear true anchors beat 16 scattered variant-seed anchors), (3) then chain score
	max_i = -1;
	if (local || pssm) {
		int32_t pass;
		for (pass = 0; pass < 2 && max_i < 0; ++pass) { // pass 0: locus-touching chains only; pass 1: any chain
			for (i = k = 0; i < n_u; ++i) {
				int32_t cnt = (uint32_t)u[i], sc = u[i]>>32, ev = 0;
				int64_t cs = as + (a[k]>>32) + 1 - 3 * kmer, ce = as + (a[k + cnt - 1]>>32) + 1;
				if (local && pass == 0 && !(ce > r->vs && cs < r->ve)) { k += cnt; continue; }
				if (pssm) ev = mp_chain_pssm_ev(nt, l_nt, pssm, qlen, kmer, cnt, a + k, opt->acc_refine_min_seg, opt->acc_refine_xdrop);
				if (max_i < 0 || ev > max_ev || (ev == max_ev && sc > max_sc))
					max_ev = ev, max_sc = sc, max_i = i;
				k += cnt;
			}
			if (!local) break;
		}
	}
	if (max_i < 0) {
		max_sc = u[0]>>32, max_i = 0;
		for (i = 1; i < n_u; ++i)
			if (max_sc < u[i]>>32)
				max_sc = u[i]>>32, max_i = i;
	}
	if (nt) kfree(km, nt);
	for (i = k = 0; i < n_u; ++i) {
		if (i == max_i) break;
		k += (uint32_t)u[i];
	}

	n_a = (int32_t)u[max_i];
	memmove(a, a + k, sizeof(*a) * n_a);
	r->a = a = krelocate(km, a, sizeof(*a) * n_a);
	r->chn_sc = u[max_i]>>32;
	r->rank_sc = pssm? max_ev : 0; // pre-alignment: PSSM evidence of the chosen chain; overwritten once aligned
	r->cnt = n_a, r->off = 0;
	r->qs = (uint32_t)a[0] - (kmer - 1);
	r->qe = (uint32_t)a[n_a-1] + 1;
	r->vs = as + (a[0]>>32) + 1 - 3 * kmer;
	r->ve = as + (a[n_a-1]>>32) + 1;
	for (i = 0; i < n_a; ++i)
		a[i] = ((a[i]>>32) + as - r->vs) << 32 | a[i]<<32>>32;
	r->chn_sc_ungap = mp_cal_chn_sc_ungap(r->cnt, r->a, kmer);
	kfree(km, u);
}

// injected locus whose 5-mer refine chain came back empty (divergent member: consensus k-mers do not hit): seed it with
// one anchor per residue along the prescreen's best ungapped PSSM diagonal so it still reaches the DP stage. Anchor
// layout matches mp_refine_reg(): x = last nt of the k-mer relative to vs, q = its last aa.
static int32_t mp_reg_from_cand_diag(void *km, mp_reg1_t *r, const mp_cand1_t *c, int32_t kmer)
{
	int32_t i, n = c->aqe - c->aqs - (kmer - 1);
	uint64_t *a;
	if (c->aqs < 0 || n <= 0) return 0;
	a = Kmalloc(km, uint64_t, n);
	for (i = 0; i < n; ++i)
		a[i] = (uint64_t)(3 * (i + kmer) - 1) << 32 | (uint32_t)(c->aqs + i + kmer - 1);
	r->a = a, r->cnt = n, r->off = 0;
	r->qs = c->aqs, r->qe = c->aqe;
	r->vs = c->avs, r->ve = c->avs + 3 * (int64_t)(c->aqe - c->aqs);
	r->chn_sc = c->aqe - c->aqs;
	r->rank_sc = c->asc;
	r->chn_sc_ungap = mp_cal_chn_sc_ungap(n, a, kmer);
	return n;
}

static void mp_dbg_chain(const mp_idx_t *mi, int32_t n_reg, const mp_reg1_t *reg, const uint64_t *a, int32_t bbit, const char *label)
{
	int32_t i, k;
	for (i = 0; i < n_reg; ++i) {
		const mp_reg1_t *r = &reg[i];
		for (k = 0; k < r->cnt; ++k) {
			uint64_t ak = a[r->off + k];
			uint64_t off = bbit > 0? (long)((ak>>32) - mi->bo[r->vid]) << bbit : r->vs + (ak>>32);
			fprintf(stderr, "%s\t%d\t%ld\t%s\t%c\t%ld\t%d\n", label, i, (long)(ak>>32), mi->nt->ctg[r->vid>>1].name, "+-"[r->vid&1], (long)off, (uint32_t)ak);
		}
	}
}

// post-alignment candidate dump: label idx ctg strand vs ve qs qe dp_score dp_max rank_sc full_dp n_plus blen n_exon parent profile_score
static void mp_dbg_regs(const mp_idx_t *mi, int32_t n_reg, const mp_reg1_t *reg, const char *label)
{
	int32_t i;
	for (i = 0; i < n_reg; ++i) {
		const mp_reg1_t *r = &reg[i];
		fprintf(stderr, "%s\t%d\t%s\t%c\t%ld\t%ld\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n", label, i, mi->nt->ctg[r->vid>>1].name, "+-"[r->vid&1],
				(long)r->vs, (long)r->ve, r->qs, r->qe, r->p? r->p->dp_score : -1,
				r->p? r->p->dp_max : -1, r->rank_sc, r->p? r->p->full_dp : -1,
				r->p? r->p->n_plus : -1, r->p? r->p->blen : -1, r->n_exon, r->parent,
				r->p && r->p->profile_score != INT32_MIN? r->p->profile_score : -1);
	}
}

static int32_t mp_cal_max_occ(void *km, const mp_idx_t *mi, int32_t n, const uint64_t *a)
{
	int32_t i, n_kmer;
	uint64_t *cnt, q25, q75;
	cnt = Kmalloc(km, uint64_t, n);
	n_kmer = mp_n_bucket(&mi->opt);
	for (i = 0; i < n; ++i) {
		int64_t en = (a[i]>>32) + 1 < n_kmer? mi->ki[(a[i]>>32) + 1] : mi->n_kb;
		cnt[i] = en - mi->ki[a[i]>>32];
	}
	radix_sort_mp64(cnt, cnt + n);
	q25 = cnt[(int64_t)(n * .25 + .499)];
	q75 = cnt[(int64_t)(n * .75 + .499)];
	kfree(km, cnt);
	return (int32_t)(q75 + (q75 - q25) * 1.5 + 10.); // threshold the outliers, the boxplot way (except for +10)
}

static uint32_t mp_hash_qname(const char *qname)
{
	uint32_t hash = 2166136261U;
	if (qname == 0) return hash;
	for (; *qname; ++qname)
		hash = (hash ^ (uint8_t)*qname) * 16777619U;
	return hash;
}

/* Consensus k-mers plus profile-derived alternatives; sorted and deduplicated so the
 * anchor/refine stages never see the same (hash, position) twice. */
static void mp_sketch_query(void *km, const mp_mapopt_t *opt, const char *seq, int32_t qlen, int32_t kmer, int32_t mod_bit, int32_t seed_max, mp64_v *sd)
{
	int64_t i, k;
	mp_sketch_prot(km, seq, qlen, kmer, mod_bit, sd);
	if (opt->hmm && seed_max > 0)
		mp_sketch_hmm(km, mp_hmm_emissions(opt->hmm), mp_hmm_length(opt->hmm), kmer, mod_bit, opt->hmm_seed_pos_loss, opt->hmm_seed_win_loss, seed_max, sd);
	radix_sort_mp64(sd->a, sd->a + sd->n);
	if (opt->hmm) {
		for (i = 1, k = 1; i < sd->n; ++i)
			if (sd->a[i] != sd->a[k-1]) sd->a[k++] = sd->a[i];
		sd->n = k;
	}
}

static uint64_t *mp_collect_anchors(void *km, const mp_idx_t *mi, const mp_mapopt_t *opt, const mp64_v *sd,
		int32_t max_occ, const char *qname, int64_t *n_a_, int64_t *n_rescued_)
{
	int32_t i, *take;
	int64_t k, n_a = 0, n_base = 0, n_rescued = 0, n_kmer = mp_n_bucket(&mi->opt);
	uint64_t *a;
	const int rescue_repeats = (opt->flag & MP_F_ACCURACY) && (opt->acc_flags & MP_ACC_REPEAT_RESCUE) && opt->acc_repeat_cap > 0;

	take = Kcalloc(km, int32_t, sd->n);
	for (i = 0; i < sd->n; ++i) {
		int64_t en = (sd->a[i]>>32) + 1 < n_kmer? mi->ki[(sd->a[i]>>32) + 1] : mi->n_kb;
		int64_t n = en - mi->ki[sd->a[i]>>32];
		if (n <= max_occ) {
			take[i] = n;
			n_base += n;
		} else if (rescue_repeats) {
			take[i] = n < opt->acc_repeat_cap? n : opt->acc_repeat_cap;
			n_rescued += take[i];
		}
	}
	if (n_rescued > opt->acc_anchor_budget) {
		int64_t used = 0, budget = opt->acc_anchor_budget;
		uint32_t start = sd->n > 0? mp_hash_qname(qname) % sd->n : 0;
		for (i = 0; i < sd->n; ++i) {
			int64_t en = (sd->a[i]>>32) + 1 < n_kmer? mi->ki[(sd->a[i]>>32) + 1] : mi->n_kb;
			int64_t n = en - mi->ki[sd->a[i]>>32];
			if (n > max_occ && take[i] > 0)
				take[i] = (int64_t)take[i] * budget / n_rescued, used += take[i];
		}
		for (i = 0; i < sd->n && used < budget; ++i) {
			int32_t j = (start + i) % sd->n;
			int64_t en = (sd->a[j]>>32) + 1 < n_kmer? mi->ki[(sd->a[j]>>32) + 1] : mi->n_kb;
			int64_t n = en - mi->ki[sd->a[j]>>32];
			int32_t cap = n < opt->acc_repeat_cap? n : opt->acc_repeat_cap;
			if (n > max_occ && take[j] < cap)
				++take[j], ++used;
		}
		n_rescued = used;
	}
	n_a = n_base + n_rescued;
	a = Kmalloc(km, uint64_t, n_a);
	for (i = 0, k = 0; i < sd->n; ++i) {
		int32_t j;
		int64_t st = mi->ki[sd->a[i]>>32];
		int64_t en = (sd->a[i]>>32) + 1 < n_kmer? mi->ki[(sd->a[i]>>32) + 1] : mi->n_kb;
		int64_t n = en - st;
		if (take[i] == n) {
			for (j = 0; j < take[i]; ++j)
				a[k++] = (uint64_t)mi->kb[st + j] << 32 | (uint32_t)sd->a[i];
		} else if (take[i] > 0) {
			int64_t midpoint = n / (2 * take[i]);
			for (j = 0; j < take[i]; ++j) {
				int64_t offset = (int64_t)j * n / take[i] + midpoint;
				a[k++] = (uint64_t)mi->kb[st + offset] << 32 | (uint32_t)sd->a[i];
			}
		}
	}
	assert(k == n_a);
	kfree(km, take);
	*n_a_ = n_a;
	*n_rescued_ = n_rescued;
	return a;
}

static uint64_t *mp_chain_by_vid(const mp_idx_t *mi, int32_t max_dist_x, int32_t max_dist_y, int32_t bw,
		int32_t max_skip, int32_t max_iter, int32_t min_cnt, int32_t min_sc, float chn_coef_log,
		int32_t is_spliced, int32_t kmer, int32_t bbit, int64_t n, uint64_t *a, int32_t *n_u_, uint64_t **u_, void *km)
{
	int64_t st, en;
	mp64_v out_a = {0,0,0}, out_u = {0,0,0};

	*n_u_ = 0, *u_ = 0;
	for (st = 0; st < n; st = en) {
		int32_t i, n_sub_u = 0, vid = mp_idx_block2pos(mi, a[st]>>32);
		int64_t n_sub_a = 0;
		uint64_t *sub_a, *sub_u = 0;
		assert(vid >= 0);
		for (en = st + 1; en < n && mp_idx_block2pos(mi, a[en]>>32) == vid; ++en) {}
		sub_a = Kmalloc(km, uint64_t, en - st);
		memcpy(sub_a, &a[st], (en - st) * sizeof(*a));
		sub_a = mp_chain(max_dist_x, max_dist_y, bw, max_skip, max_iter, min_cnt, min_sc, chn_coef_log,
				is_spliced, kmer, bbit, en - st, sub_a, &n_sub_u, &sub_u, km);
		for (i = 0; i < n_sub_u; ++i) {
			n_sub_a += (uint32_t)sub_u[i];
			kv_push(uint64_t, km, out_u, sub_u[i]);
		}
		for (i = 0; i < n_sub_a; ++i)
			kv_push(uint64_t, km, out_a, sub_a[i]);
		kfree(km, sub_u);
		kfree(km, sub_a);
	}
	kfree(km, a);
	*n_u_ = out_u.n, *u_ = out_u.a;
	return out_a.a;
}

static int32_t mp_reg_will_output(void *km, const mp_mapopt_t *opt, int32_t qlen, int32_t n_reg, const mp_reg1_t *reg)
{
	mp_reg1_t *tmp;
	int32_t i, n = 0, best_sc, will_output = 0;
	tmp = Kmalloc(km, mp_reg1_t, n_reg);
	for (i = 0; i < n_reg; ++i)
		if (reg[i].p && reg[i].cnt > 0)
			tmp[n++] = reg[i];
	mp_sort_reg(km, &n, tmp, (opt->flag & MP_F_ACCURACY) && (opt->acc_flags & MP_ACC_RERANK));
	mp_select_multi_exon(n, tmp, opt->io, (opt->flag & MP_F_ACCURACY) && (opt->acc_flags & MP_ACC_STRICT_MULTI_EXON));
	n = mp_reject_spurious_intron(km, n, tmp, (opt->flag & MP_F_ACCURACY) && (opt->acc_flags & MP_ACC_STRICT_MULTI_EXON),
			opt->acc_intron_evidence_min, opt->acc_intron_evidence_ratio, opt->acc_intron_abs_max, 0); // tmp is a shallow scratch copy: never free p/feat here
	if (opt->flag & MP_F_ACCURACY) {
		n = mp_reject_low_purity_span(km, n, tmp, opt->acc_span_ratio, opt->acc_intron_evidence_min, 0);
		n = mp_reject_weak_hit(km, n, tmp, opt->acc_min_out_score, opt->acc_min_pos_ratio, opt->acc_rescue_np, opt->acc_rescue_score, opt->acc_min_np, 0, opt->thr_use_ps, opt->acc_min_ps, opt->acc_ps_rescue_score, opt->acc_ps_rescue_posr);
	}
	best_sc = n > 0? mp_reg_score(&tmp[0]) : -1;
	for (i = 0; i < n && i < opt->out_n; ++i) {
		int32_t sc = mp_reg_score(&tmp[i]);
		if (sc > 0 && sc >= (double)best_sc * opt->out_sim && tmp[i].qe - tmp[i].qs >= (double)qlen * opt->out_cov) {
			will_output = 1;
			break;
		}
	}
	kfree(km, tmp);
	return will_output;
}

// --gpu-prefilter: anchors are sampled evenly along the coarse chain so that most exons of a
// multi-exon gene seed their own gapless segment in the GPU prefilter kernel (see nasw-cuda.cu).
#define MP_GPU_PREFILTER_ANCHORS_PER_CANDIDATE 8

static int32_t mp_gpu_prefilter_pick_anchors(const mp_idx_t *mi, const mp_reg1_t *reg,
		const uint64_t *all_anchors, ns_gpu_gapless_anchor_t picked[MP_GPU_PREFILTER_ANCHORS_PER_CANDIDATE])
{
	int32_t i, first = -1, last = -1, count = 0, n_valid = 0;
	int32_t indexes[MP_GPU_PREFILTER_ANCHORS_PER_CANDIDATE];
	if (reg->cnt <= 0) return 0;
	if (mp_idx_block2pos(mi, all_anchors[reg->off] >> 32) == (int32_t)reg->vid &&
			mp_idx_block2pos(mi, all_anchors[reg->off + reg->cnt - 1] >> 32) == (int32_t)reg->vid) {
		first = reg->off, last = reg->off + reg->cnt - 1;
	} else {
		for (i = reg->off; i < reg->off + reg->cnt; ++i) {
			if (mp_idx_block2pos(mi, all_anchors[i] >> 32) != (int32_t)reg->vid) continue;
			if (first < 0) first = i;
			last = i;
		}
	}
	if (first < 0) return 0;
	n_valid = last - first + 1;
	if (n_valid <= MP_GPU_PREFILTER_ANCHORS_PER_CANDIDATE) {
		for (i = 0; i < n_valid; ++i) indexes[i] = first + i;
		count = n_valid;
	} else {
		for (i = 0; i < MP_GPU_PREFILTER_ANCHORS_PER_CANDIDATE; ++i)
			indexes[i] = first + (int32_t)((int64_t)i * (n_valid - 1) / (MP_GPU_PREFILTER_ANCHORS_PER_CANDIDATE - 1));
		count = MP_GPU_PREFILTER_ANCHORS_PER_CANDIDATE;
	}
	for (i = 0; i < count; ++i) {
		const uint64_t anchor = all_anchors[indexes[i]];
		picked[i].target_block = ((int64_t)(anchor >> 32) - mi->bo[reg->vid]) << mi->opt.bbit;
		picked[i].query_end = (uint32_t)anchor;
		picked[i].reserved = 0;
	}
	return count;
}

// Keeps the top --gpu-prefilter-topk (or an adaptive K, see --gpu-prefilter-adaptive) coarse
// candidates of one query by their GPU gapless X-drop scores (see nasw-cuda.cu).
static void mp_gpu_prefilter_select(void *km, const mp_idx_t *mi, const mp_mapopt_t *opt,
		int32_t qlen, int32_t *n_reg, mp_reg1_t *reg, const int32_t *scores, const mp_gapless_calib_t *cal, mp_profile_t *prof)
{
	int32_t n_before = *n_reg, top_k = opt->gpu_prefilter_top_k;
	if (opt->flag & MP_F_GPU_PREFILTER_ADAPTIVE) {
		int32_t initial_k;
		if (top_k > MP_GAPLESS_ADAPT_MAX_K) top_k = MP_GAPLESS_ADAPT_MAX_K;
		initial_k = mp_gapless_adaptive_k(*n_reg, scores, qlen, top_k, cal);
		mp_select_gapless(km, initial_k, opt->gpu_prefilter_min_per_group, n_reg, reg, scores);
		if (prof) {
			if (initial_k <= 1) ++prof->n_prefilter_k1;
			else if (initial_k <= 2) ++prof->n_prefilter_k2;
			else if (initial_k <= 4) ++prof->n_prefilter_k4;
			else ++prof->n_prefilter_k8;
		}
	} else {
		mp_select_gapless(km, top_k, opt->gpu_prefilter_min_per_group, n_reg, reg, scores);
	}
	mp_sort_reg(km, n_reg, reg, 0);
	mp_set_parent(km, opt->mask_level, opt->mask_len, *n_reg, reg, mi->opt.kmer, 0);
	if (prof) prof->n_prefilter_candidate += n_before, prof->n_prefilter_survivor += *n_reg;
}

// Output of the coarse stage of mp_map() (sketch/anchor/chain/coarse selection), consumed by
// mp_map_finish(). With --gpu-prefilter the pipeline runs the coarse stage for a whole mini-batch,
// scores all candidates in one GPU launch, then finishes (see mp_gpu_prefilter_batch()).
typedef struct {
	uint64_t *a; // coarse anchors, indexed by reg[].off/cnt
	void *a_km; // arena owning `a`: the coarse thread's km, or NULL (malloc) after mp_coarse_detach_anchors()
	mp_reg1_t *reg;
	int32_t n_reg;
	int32_t cand_mode, coarse_n, select_n;
	const mp_cand1_t *cand;
	const int8_t *pssm;
} mp_coarse_map_t;

static void mp_map_coarse(const mp_idx_t *mi, int qlen, const char *seq, mp_tbuf_t *b, const mp_mapopt_t *opt, const char *qname, mp_coarse_map_t *c)
{
	void *km = b->km;
	const mp_idxopt_t *io = &mi->opt;
	mp64_v sd = {0,0,0};
	mp_reg1_t *reg;
	int32_t *n_reg = &c->n_reg;
	int32_t i, n_u, max_occ, is_splice = !(opt->flag&MP_F_NO_SPLICE), do_profile = !!(opt->flag & MP_F_PROFILE);
	int32_t cand_mode = opt->cand_tab != 0;
	int32_t select_n = (opt->flag & MP_F_ACCURACY) && opt->acc_best_n > opt->best_n? opt->acc_best_n : opt->best_n;
	int32_t coarse_n = select_n;
	float select_ratio = (opt->flag & MP_F_ACCURACY) && opt->acc_pri_ratio < opt->pri_ratio? opt->acc_pri_ratio : opt->pri_ratio;
	int64_t k, n_a = 0, n_rescued = 0;
	uint64_t *a, *u;
	const int8_t *pssm = 0;
	double t0 = 0.0;

	memset(c, 0, sizeof(*c));
	c->a_km = km, c->cand_mode = cand_mode, c->select_n = select_n;
	if ((opt->flag & MP_F_ACCURACY) && (opt->acc_flags & MP_ACC_REFINE_PSSM) && opt->hmm && opt->asize == 22 && mp_hmm_length(opt->hmm) == qlen)
		pssm = mp_hmm_pssm(opt->hmm);
	if (pssm && !(mp_dbg_flag & MP_DBG_NO_REFINE) && opt->acc_coarse_n > select_n) coarse_n = opt->acc_coarse_n;
	c->pssm = pssm, c->coarse_n = coarse_n;
	if (cand_mode) { // external candidate loci (--cand-table) replace the genome-wide sketch/anchor/chain stage
		int32_t nc;
		const mp_cand1_t *cd = mp_cand_tab_get(opt->cand_tab, qname, &nc);
		reg = Kcalloc(0, mp_reg1_t, nc);
		for (i = 0; i < nc; ++i) {
			reg[i].vid = cd[i].vid, reg[i].vs = cd[i].vs, reg[i].ve = cd[i].ve;
			reg[i].qs = 0, reg[i].qe = qlen, reg[i].off = -1;
		}
		c->cand = cd, c->reg = reg, c->n_reg = nc, c->a = 0;
		if (do_profile) ++b->profile.n_query, b->profile.n_query_aa += qlen;
		if (mp_dbg_flag & MP_DBG_CHAIN) mp_dbg_regs(mi, nc, reg, "Y0"); // injected prescreen/cand-table loci (no anchors yet)
		return;
	}
	if (do_profile) t0 = mp_realtime();
	mp_sketch_query(km, opt, seq, qlen, io->kmer, io->mod_bit, opt->hmm_seed_max, &sd);
	if (do_profile) {
		b->profile.t_sketch += mp_realtime() - t0;
		++b->profile.n_query, b->profile.n_query_aa += qlen, b->profile.n_seed += sd.n;
		t0 = mp_realtime();
	}

	if (sd.n >= 8) {
		max_occ = mp_cal_max_occ(km, mi, sd.n, sd.a);
		if (max_occ > opt->max_occ) max_occ = opt->max_occ;
	} else max_occ = opt->max_occ;

	a = mp_collect_anchors(km, mi, opt, &sd, max_occ, qname, &n_a, &n_rescued);
	kfree(km, sd.a);
	radix_sort_mp64(a, a + n_a);
	if (do_profile) {
		b->profile.t_anchor += mp_realtime() - t0;
		b->profile.n_anchor += n_a, b->profile.n_anchor_rescued += n_rescued;
	}

	if (mp_dbg_flag & MP_DBG_ANCHOR) {
		for (k = 0; k < n_a; ++k) {
			i = mp_idx_block2pos(mi, a[k]>>32);
			fprintf(stderr, "X\t%ld\t%s\t%c\t%ld\t%d\n", (long)(a[k]>>32), mi->nt->ctg[i>>1].name, "+-"[i&1], (long)((a[k]>>32) - mi->bo[i]) << mi->opt.bbit, (uint32_t)a[k]);
		}
	}

	if (!(opt->flag & MP_F_NO_PRE_CHAIN) && is_splice) {
		int32_t w = 1<<mi->opt.bbit;
		if (do_profile) t0 = mp_realtime();
		if ((opt->flag & MP_F_ACCURACY) && (opt->acc_flags & MP_ACC_VID_CHAIN))
			a = mp_chain_by_vid(mi, w, w, w, opt->max_chn_max_skip, opt->max_chn_iter, 2, 0, opt->chn_coef_log,
					is_splice, mi->opt.kmer, mi->opt.bbit, n_a, a, &n_u, &u, km);
		else
			a = mp_chain(w, w, w, opt->max_chn_max_skip, opt->max_chn_iter, 2, 0, opt->chn_coef_log, is_splice, mi->opt.kmer, mi->opt.bbit, n_a, a, &n_u, &u, km);
		for (i = 0, n_a = 0; i < n_u; ++i) n_a += (uint32_t)u[i];
		kfree(km, u);
		radix_sort_mp64(a, a + n_a);
		if (do_profile) b->profile.t_prechain += mp_realtime() - t0;
	}

	if (do_profile) t0 = mp_realtime();
	if ((opt->flag & MP_F_ACCURACY) && (opt->acc_flags & MP_ACC_VID_CHAIN))
		a = mp_chain_by_vid(mi, opt->max_intron, opt->max_gap, opt->bw, opt->max_chn_max_skip, opt->max_chn_iter, opt->min_chn_cnt, opt->min_chn_sc, opt->chn_coef_log,
				is_splice, mi->opt.kmer, mi->opt.bbit, n_a, a, &n_u, &u, km);
	else
		a = mp_chain(opt->max_intron, opt->max_gap, opt->bw, opt->max_chn_max_skip, opt->max_chn_iter, opt->min_chn_cnt, opt->min_chn_sc, opt->chn_coef_log,
				 is_splice, mi->opt.kmer, mi->opt.bbit, n_a, a, &n_u, &u, km);
	if (do_profile) b->profile.n_coarse_chain += n_u;
	reg = mp_reg_gen_from_block(0, mi, n_u, u, a, n_reg);
	kfree(km, u);
	mp_sort_reg(km, n_reg, reg, 0);
	mp_set_parent(km, opt->mask_level, opt->mask_len, *n_reg, reg, mi->opt.kmer, 0);
	mp_select_sub(km, select_ratio * select_ratio, mi->opt.kmer * 2, coarse_n, n_reg, reg);
	if (do_profile) b->profile.t_chain += mp_realtime() - t0;
	c->a = a, c->reg = reg;
}

// Refine + align stages of mp_map(); takes ownership of c->a / c->reg.
static mp_reg1_t *mp_map_finish(const mp_idx_t *mi, int qlen, const char *seq, int *n_reg, mp_tbuf_t *b, const mp_mapopt_t *opt, mp_coarse_map_t *c)
{
	void *km = b->km, *a_km = c->a_km;
	mp_reg1_t *reg = c->reg;
	uint64_t *a = c->a;
	int32_t i, do_profile = !!(opt->flag & MP_F_PROFILE);
	int32_t cand_mode = c->cand_mode, select_n = c->select_n, coarse_n = c->coarse_n;
	float select_ratio = (opt->flag & MP_F_ACCURACY) && opt->acc_pri_ratio < opt->pri_ratio? opt->acc_pri_ratio : opt->pri_ratio;
	const int8_t *pssm = c->pssm;
	const mp_cand1_t *cand = c->cand;
	double t0 = 0.0;

	*n_reg = c->n_reg;
	c->a = 0, c->reg = 0, c->n_reg = 0;
	if (!cand_mode && (mp_dbg_flag & MP_DBG_CHAIN))
		mp_dbg_chain(mi, *n_reg, reg, a, mi->opt.bbit, "Y1");

	if (cand_mode || !(mp_dbg_flag & MP_DBG_NO_REFINE)) {
		int32_t nr = 0;
		uint64_t *ext = 0;
		mp64_v sd_aa = {0,0,0};
		if (do_profile) {
			t0 = mp_realtime();
			b->profile.n_refine_candidate += *n_reg;
		}
		mp_sketch_query(km, opt, seq, qlen, opt->kmer2, 0, opt->hmm_seed_max2, &sd_aa);
		// injected loci have no anchors, so bound the window by vs/ve (nt!=0 path) instead of a[r->off]; an unbounded
		// +-max_ext window lets the refine chain bleed into a tandem paralog and both loci collapse onto the stronger one
		ext = mp_cal_max_ext(km, cand_mode? mi->nt : 0, *n_reg, reg, a, 100, opt->max_ext, opt->max_ext);
		for (i = 0; i < *n_reg; ++i) {
			int32_t extl = ext? ext[i]>>32 : opt->max_ext, extr = ext? (int32_t)ext[i] : opt->max_ext;
			mp_refine_reg(km, mi, opt, &sd_aa, &reg[i], extl, extr, pssm, qlen, do_profile? &b->profile : 0);
			if (reg[i].cnt == 0 && cand && !(opt->flag & MP_F_NO_CAND_DIAG) && mp_reg_from_cand_diag(km, &reg[i], &cand[i], opt->kmer2) > 0 && do_profile)
				++b->profile.n_cand_diag_seeded;
			if (reg[i].cnt > 0) {
				if (do_profile) b->profile.n_exact_anchor += reg[i].cnt;
				reg[nr++] = reg[i];
			}
		}
		*n_reg = nr;
		if (coarse_n > select_n) { // over-selected: rank by PSSM evidence first so the -N cut below keeps true weak-seed loci over seed-rich junk
			for (i = 0; i < nr; ++i)
				reg[i].chn_sc = reg[i].rank_sc << 10 | (reg[i].chn_sc < 1023? reg[i].chn_sc : 1023);
		}
		if (do_profile) b->profile.n_refined += nr;
		kfree(km, sd_aa.a);
		kfree(km, ext);
		kfree(a_km, a);
		a = mp_collate_a(km, *n_reg, reg);
		a_km = km;
		mp_sort_reg(km, n_reg, reg, 0);
		if (mp_dbg_flag & MP_DBG_CHAIN)
			mp_dbg_chain(mi, *n_reg, reg, a, 0, "Y2");
		mp_set_parent(km, opt->mask_level, opt->mask_len, *n_reg, reg, mi->opt.kmer, 0);
		mp_select_sub(km, select_ratio * select_ratio, mi->opt.kmer * 2, select_n, n_reg, reg);
		if (mp_dbg_flag & MP_DBG_CHAIN) mp_dbg_regs(mi, *n_reg, reg, "Y2s"); // refined loci that survive set_parent+select_sub (== DP candidates)
		if (do_profile) b->profile.t_refine += mp_realtime() - t0;
	}
	if (!(opt->flag & MP_F_NO_ALIGN)) {
		int32_t k, n_align = *n_reg;
		uint64_t *ext = 0, *ext_full = 0;
		mp_reg1_t *align_seed = 0;
		uint8_t *align_flags = 0, *retry_done = 0;
		mp_hmm_batch_t *hb;
		if (do_profile) {
			t0 = mp_realtime();
			b->profile.n_dp_candidate += *n_reg;
		}
		{
			int32_t max_align_ext = opt->max_intron/2;
			int32_t iso_ext = max_align_ext;
			// isolated (no real neighboring candidate) regions otherwise pay a huge, mostly-fruitless
			// intron-scale speculative search (see align.c); cap just that case tighter under --accuracy,
			// where far more candidates reach this stage. Real inter-exon gaps (a genuine neighbor found)
			// still get the full opt->max_intron/2 cap so long introns are never truncated.
			if ((opt->flag & MP_F_ACCURACY) && (opt->acc_flags & MP_ACC_END_EXT_CAP) && opt->acc_end_ext < iso_ext)
				iso_ext = opt->acc_end_ext;
			ext = mp_cal_max_ext(km, mi->nt, *n_reg, reg, a, 100, max_align_ext, iso_ext);
			if (iso_ext < max_align_ext)
				ext_full = mp_cal_max_ext(km, mi->nt, *n_reg, reg, a, 100, max_align_ext, max_align_ext);
		}
		if (ext_full && opt->acc_end_retry > 0) {
			align_flags = Kcalloc(km, uint8_t, n_align);
			retry_done = Kcalloc(km, uint8_t, n_align);
		}
		align_seed = Kmalloc(km, mp_reg1_t, n_align);
		hb = mp_hmm_batch_init(km, opt, qlen);
		// Phase 1: align every candidate; end extensions of all candidates are batched per stage and
		// profile scores are collected in hb and computed in one GPU launch at the flush. Only rank
		// scores depend on them, and nothing below reads a rank score before the flush, so the result
		// is identical to scoring inline.
		{
			mp_align_prof_t ap = {0,0,0,0,0,0,0,0,0,0};
			mp_reg1_t **rp = Kmalloc(km, mp_reg1_t*, n_align);
			int32_t *st = Kmalloc(km, int32_t, n_align);
			for (i = 0; i < n_align; ++i) align_seed[i] = reg[i], rp[i] = &reg[i];
			mp_align_batch(km, opt, mi, qlen, seq, n_align, rp, ext, 0, do_profile? &ap : 0, st, hb);
			for (i = 0; i < n_align; ++i) {
				if (align_flags) align_flags[i] = st[i];
				if (do_profile) b->profile.n_dp_rescued += !!(st[i] & MP_ALIGN_RESCUED);
			}
			if (do_profile) mp_profile_add_align(&b->profile, &ap);
			kfree(km, rp);
			kfree(km, st);
		}
		mp_hmm_batch_flush(hb);
		// Phase 2: junction refinement re-alignments, again batched before the replace decisions
		if ((opt->flag & MP_F_ACCURACY) && (opt->acc_flags & MP_ACC_JUNC_REFINE)) {
			mp_reg1_t *trial = Kmalloc(km, mp_reg1_t, n_align);
			mp_reg1_t **rp = Kmalloc(km, mp_reg1_t*, n_align);
			uint64_t *ext_trial = Kmalloc(km, uint64_t, n_align);
			int32_t *st = Kmalloc(km, int32_t, n_align), n_trial = 0;
			uint8_t *has_trial = Kcalloc(km, uint8_t, n_align);
			mp_align_prof_t ap = {0,0,0,0,0,0,0,0,0,0};
			for (i = 0; i < n_align; ++i) {
				if (!reg[i].p || !mp_reg_needs_refine(opt, &reg[i])) continue;
				trial[i] = align_seed[i];
				trial[i].p = 0, trial[i].feat = 0, trial[i].rank_sc = 0;
				has_trial[i] = 1;
				rp[n_trial] = &trial[i], ext_trial[n_trial] = ext[i], ++n_trial;
			}
			mp_align_batch(km, opt, mi, qlen, seq, n_trial, rp, ext_trial, 1, do_profile? &ap : 0, st, hb);
			if (do_profile) {
				for (i = 0; i < n_trial; ++i)
					if (st[i] & MP_ALIGN_FULL) ++b->profile.n_dp_realign;
				mp_profile_add_align(&b->profile, &ap);
			}
			mp_hmm_batch_flush(hb);
			for (i = 0; i < n_align; ++i)
				if (has_trial[i] && mp_reg_try_replace(&reg[i], &trial[i]) && do_profile)
					++b->profile.n_dp_realign_kept;
			kfree(km, trial);
			kfree(km, rp);
			kfree(km, ext_trial);
			kfree(km, st);
			kfree(km, has_trial);
		}
		mp_hmm_batch_destroy(hb);
		if (align_flags) {
			for (i = 0; i < n_align; ++i)
				if (align_seed[i].cnt > 0 && ((align_flags[i] & MP_ALIGN_RESCUED) || ext[i] != ext_full[i]))
					if (do_profile) ++b->profile.n_dp_ext_eligible;
			if (!mp_reg_will_output(km, opt, qlen, n_align, reg)) {
				int32_t *score_rank = Kmalloc(km, int32_t, n_align);
				int32_t *peak_rank = Kmalloc(km, int32_t, n_align);
				if (do_profile) ++b->profile.n_dp_ext_fallback;
				for (i = 0; i < n_align; ++i) {
					int32_t j;
					score_rank[i] = peak_rank[i] = INT32_MAX;
					if (align_seed[i].cnt <= 0 ||
						!((align_flags[i] & MP_ALIGN_RESCUED) || ext[i] != ext_full[i])) continue;
					score_rank[i] = peak_rank[i] = 1;
					for (j = 0; j < n_align; ++j) {
						int32_t peak_i, peak_j;
						if (j == i || align_seed[j].cnt <= 0 ||
							!((align_flags[j] & MP_ALIGN_RESCUED) || ext[j] != ext_full[j])) continue;
						peak_i = reg[i].p? reg[i].p->dp_max : INT32_MIN;
						peak_j = reg[j].p? reg[j].p->dp_max : INT32_MIN;
						if (mp_reg_score(&reg[j]) > mp_reg_score(&reg[i]) ||
							(mp_reg_score(&reg[j]) == mp_reg_score(&reg[i]) &&
							 (peak_j > peak_i || (peak_j == peak_i && j < i))))
							++score_rank[i];
						if (peak_j > peak_i ||
							(peak_j == peak_i &&
							 (mp_reg_score(&reg[j]) > mp_reg_score(&reg[i]) ||
							  (mp_reg_score(&reg[j]) == mp_reg_score(&reg[i]) && j < i))))
							++peak_rank[i];
					}
				}
				for (k = 0; k < opt->acc_end_retry; ++k) {
					int32_t best = -1, best_rank = INT32_MAX, best_sum = INT32_MAX;
					for (i = 0; i < n_align; ++i) {
						int32_t rank, sum;
						if (retry_done[i] || score_rank[i] == INT32_MAX) continue;
						rank = score_rank[i] < peak_rank[i]? score_rank[i] : peak_rank[i];
						sum = score_rank[i] + peak_rank[i];
						if (best < 0 || rank < best_rank ||
							(rank == best_rank && (sum < best_sum || (sum == best_sum && i < best))))
							best = i, best_rank = rank, best_sum = sum;
					}
					if (best < 0) break;
					retry_done[best] = 1;
					{
						mp_align_prof_t ap = {0,0,0,0,0,0,0,0,0,0};
						mp_reg1_t retry = align_seed[best];
						retry.p = 0, retry.feat = 0, retry.rank_sc = retry.rank_sc2 = 0;
						mp_align(km, opt, mi, qlen, seq, &retry, ext_full[best]>>32, (int32_t)ext_full[best],
							!!(align_flags[best] & MP_ALIGN_RESCUED), do_profile? &ap : 0, 0);
						if (do_profile) {
							++b->profile.n_dp_ext_retry;
							mp_profile_add_align(&b->profile, &ap);
						}
						free(reg[best].p); free(reg[best].feat);
						reg[best] = retry;
						if (mp_reg_will_output(km, opt, qlen, n_align, reg)) break;
					}
				}
				kfree(km, score_rank);
				kfree(km, peak_rank);
			}
		}
		if (align_seed) kfree(km, align_seed);
		if (align_flags) kfree(km, align_flags);
		if (retry_done) kfree(km, retry_done);
		for (i = k = 0; i < n_align; ++i)
			if (reg[i].p) reg[k++] = reg[i];
		*n_reg = k;
		if (ext) kfree(km, ext);
		if (ext_full) kfree(km, ext_full);
		if (mp_dbg_flag & MP_DBG_CHAIN) mp_dbg_regs(mi, *n_reg, reg, "Y3");
		mp_sort_reg(km, n_reg, reg, (opt->flag & MP_F_ACCURACY) && (opt->acc_flags & MP_ACC_RERANK));
		mp_select_multi_exon(*n_reg, reg, opt->io, (opt->flag & MP_F_ACCURACY) && (opt->acc_flags & MP_ACC_STRICT_MULTI_EXON));		*n_reg = mp_reject_spurious_intron(km, *n_reg, reg, (opt->flag & MP_F_ACCURACY) && (opt->acc_flags & MP_ACC_STRICT_MULTI_EXON),
				opt->acc_intron_evidence_min, opt->acc_intron_evidence_ratio, opt->acc_intron_abs_max, 1); // reg is the real owning array: free dropped candidates
		if (mp_dbg_flag & MP_DBG_CHAIN) mp_dbg_regs(mi, *n_reg, reg, "Y3a"); // after spurious-intron filter
		if (opt->flag & MP_F_ACCURACY) {
			*n_reg = mp_reject_low_purity_span(km, *n_reg, reg, opt->acc_span_ratio, opt->acc_intron_evidence_min, 1);
			if (mp_dbg_flag & MP_DBG_CHAIN) mp_dbg_regs(mi, *n_reg, reg, "Y3b"); // after low-purity-span filter
			*n_reg = mp_reject_weak_hit(km, *n_reg, reg, opt->acc_min_out_score, opt->acc_min_pos_ratio, opt->acc_rescue_np, opt->acc_rescue_score, opt->acc_min_np, 1, opt->thr_use_ps, opt->acc_min_ps, opt->acc_ps_rescue_score, opt->acc_ps_rescue_posr);
			if (mp_dbg_flag & MP_DBG_CHAIN) mp_dbg_regs(mi, *n_reg, reg, "Y3c"); // after weak-hit filter
		}
		mp_set_parent(km, opt->mask_level, opt->mask_len, *n_reg, reg, mi->opt.kmer, 0);
		mp_select_sub(km, select_ratio, mi->opt.kmer * 2, select_n, n_reg, reg);
		if (mp_dbg_flag & MP_DBG_CHAIN) mp_dbg_regs(mi, *n_reg, reg, "Y3d"); // after set_parent + select_sub (== what reaches the output loop)
		if (do_profile) b->profile.t_align += mp_realtime() - t0;
	}
	kfree(a_km, a);
	return reg;
}

mp_reg1_t *mp_map(const mp_idx_t *mi, int qlen, const char *seq, int *n_reg, mp_tbuf_t *b, const mp_mapopt_t *opt, const char *qname)
{
	mp_coarse_map_t c;
	mp_map_coarse(mi, qlen, seq, b, opt, qname, &c);
	return mp_map_finish(mi, qlen, seq, n_reg, b, opt, &c);
}

/**************************
 * Multi-threaded mapping *
 **************************/

typedef struct {
	int32_t n_threads;
	int64_t id;
	const mp_mapopt_t *opt;
	mp_bseq_file_t *fp;
	mp_hmm_file_t *hp; // set instead of fp when the query file holds HMMER3 profiles
	FILE *ps_dump; // --ps-dump: candidate table generated by --prescreen
	const mp_idx_t *mi;
	kstring_t str;
	mp_profile_t profile;
	uint64_t gpu_start, gpu_launch_start;
	double wall_start;
	double gpu_h2d_start, gpu_kernel_start, gpu_d2h_start;
} pipeline_t;

typedef struct {
	pipeline_t *p;
    int32_t n_seq;
	mp_bseq1_t *seq;
	mp_hmm_t **hmm; // per-query profile (parallel to seq), or NULL for FASTA input
	mp_cand_tab_t *cand_tab; // --prescreen: candidate loci of this batch's profiles
	int32_t *n_reg;
	mp_reg1_t **reg;
	mp_tbuf_t **buf;
	mp_coarse_map_t *coarse; // --gpu-prefilter: per-query coarse results between the coarse and finish passes
} step_t;

// Read a mini-batch of profiles; each becomes a query whose sequence is the consensus.
static mp_bseq1_t *mp_hmm_read_batch(mp_hmm_file_t *hp, int64_t chunk_size, int32_t stop_sc, mp_hmm_t ***hmm_, int32_t *n_)
{
	int32_t n = 0, m = 0;
	int64_t tot = 0;
	mp_bseq1_t *seq = 0;
	mp_hmm_t **hmm = 0, *h;
	*n_ = 0, *hmm_ = 0;
	while (tot < chunk_size && (h = mp_hmm_read1(hp)) != 0) {
		if (n == m) {
			m = m? m << 1 : 256;
			seq = (mp_bseq1_t*)realloc(seq, (size_t)m * sizeof(*seq));
			hmm = (mp_hmm_t**)realloc(hmm, (size_t)m * sizeof(*hmm));
		}
		mp_hmm_set_stop_sc(h, stop_sc);
		memset(&seq[n], 0, sizeof(seq[n]));
		seq[n].name = mp_strdup(mp_hmm_name(h));
		seq[n].seq = mp_strdup(mp_hmm_consensus(h));
		seq[n].l_seq = mp_hmm_length(h);
		hmm[n++] = h;
		tot += mp_hmm_length(h);
	}
	if (n == 0) { free(seq); free(hmm); return 0; }
	*n_ = n, *hmm_ = hmm;
	return seq;
}

static const mp_mapopt_t *mp_query_opt(const step_t *s, long i, mp_mapopt_t *opt_q)
{
	const mp_mapopt_t *opt = s->p->opt;
	const mp_bseq1_t *seq = &s->seq[i];
	if (!s->hmm && !opt->thr_tab) return opt;
	*opt_q = *opt;
	if (s->hmm) opt_q->hmm = s->hmm[i]; // profile-aware seeding/scoring reads the profile through opt->hmm
	if (s->cand_tab) opt_q->cand_tab = s->cand_tab; // built-in prescreen loci take the --cand-table path in mp_map()
	if (opt->thr_tab) { // per-query DP-score floor (e.g. decoy-calibrated per-family AS) replaces the global --acc-min-score
		int32_t thr = mp_thr_tab_get(opt->thr_tab, seq->name, opt->thr_fallback > 0? opt->thr_fallback : opt->acc_min_out_score);
		if (thr > 0) opt_q->acc_min_out_score = thr;
	}
	return opt_q;
}

static void worker_for(void *_data, long i, int tid) // kt_for() callback
{
    step_t *s = (step_t*)_data;
	mp_bseq1_t *seq = &s->seq[i];
	mp_mapopt_t opt_q;
	const mp_mapopt_t *opt = mp_query_opt(s, i, &opt_q);
	if (mp_dbg_flag & MP_DBG_QNAME)
		fprintf(stderr, "QR\t%s\t%d\t%d\n", s->seq[i].name, s->seq[i].l_seq, tid);
	s->reg[i] = mp_map(s->p->mi, seq->l_seq, seq->seq, &s->n_reg[i], s->buf[tid], opt, seq->name);
}

/*
 * --gpu-prefilter pipeline: coarse-map the whole mini-batch (kt_for), score every coarse candidate
 * of the batch in ONE GPU launch from the pipeline thread, then finish (refine/align) with kt_for.
 * This keeps all CUDA calls single-threaded and amortises H2D/D2H + launch cost over thousands of
 * queries, unlike a per-query launch inside mp_map() which serialises the worker threads on the GPU.
 */

// Compacts the surviving coarse anchors into malloc memory so a different finish thread may free them.
static void mp_coarse_detach_anchors(mp_coarse_map_t *c)
{
	uint64_t *a;
	int64_t n = 0;
	int32_t i;
	if (!c->a) return;
	for (i = 0; i < c->n_reg; ++i)
		if (c->reg[i].cnt > 0) n += c->reg[i].cnt;
	a = n > 0? (uint64_t*)malloc((size_t)n * sizeof(*a)) : 0;
	for (i = 0, n = 0; i < c->n_reg; ++i) {
		if (c->reg[i].cnt > 0)
			memcpy(&a[n], &c->a[c->reg[i].off], (size_t)c->reg[i].cnt * sizeof(*a));
		c->reg[i].off = (int32_t)n;
		n += c->reg[i].cnt > 0? c->reg[i].cnt : 0;
	}
	kfree(c->a_km, c->a);
	c->a = a, c->a_km = 0;
}

static void worker_coarse_for(void *_data, long i, int tid) // kt_for() callback
{
	step_t *s = (step_t*)_data;
	mp_bseq1_t *seq = &s->seq[i];
	mp_mapopt_t opt_q;
	const mp_mapopt_t *opt = mp_query_opt(s, i, &opt_q);
	if (mp_dbg_flag & MP_DBG_QNAME)
		fprintf(stderr, "QR\t%s\t%d\t%d\n", s->seq[i].name, s->seq[i].l_seq, tid);
	mp_map_coarse(s->p->mi, seq->l_seq, seq->seq, s->buf[tid], opt, seq->name, &s->coarse[i]);
	mp_coarse_detach_anchors(&s->coarse[i]);
}

static void worker_finish_for(void *_data, long i, int tid) // kt_for() callback
{
	step_t *s = (step_t*)_data;
	mp_bseq1_t *seq = &s->seq[i];
	mp_mapopt_t opt_q;
	const mp_mapopt_t *opt = mp_query_opt(s, i, &opt_q);
	s->reg[i] = mp_map_finish(s->p->mi, seq->l_seq, seq->seq, &s->n_reg[i], s->buf[tid], opt, &s->coarse[i]);
}

// One GPU gapless launch for every prefilterable query of the batch (>1 coarse candidate, real anchors).
// Runs on the pipeline thread between the coarse and finish kt_for passes. On GPU failure the coarse
// candidate sets are left untouched (a one-time warning is printed).
static void mp_gpu_prefilter_batch(step_t *s)
{
	static int warned = 0;
	pipeline_t *p = s->p;
	const mp_idx_t *mi = p->mi;
	const mp_mapopt_t *opt = p->opt;
	mp_tbuf_t *b = s->buf[0];
	mp_profile_t *prof = (opt->flag & MP_F_PROFILE)? &b->profile : 0;
	ns_gpu_gapless_candidate_t *candidates;
	ns_gpu_gapless_anchor_t *anchors;
	uint8_t *queries;
	int32_t *scores, *cand_off;
	int64_t query_aa = 0, n_cand = 0, n_anchors = 0, query_off = 0;
	int32_t i, j;
	double t0 = prof? mp_realtime() : 0.0;

	for (i = 0; i < s->n_seq; ++i) {
		const mp_coarse_map_t *c = &s->coarse[i];
		if (c->a == 0 || c->n_reg <= 1) continue;
		query_aa += s->seq[i].l_seq;
		n_cand += c->n_reg;
	}
	if (n_cand == 0 || n_cand > INT32_MAX / MP_GPU_PREFILTER_ANCHORS_PER_CANDIDATE) return;
	candidates = (ns_gpu_gapless_candidate_t*)malloc((size_t)n_cand * sizeof(*candidates));
	anchors = (ns_gpu_gapless_anchor_t*)malloc((size_t)n_cand * MP_GPU_PREFILTER_ANCHORS_PER_CANDIDATE * sizeof(*anchors));
	queries = (uint8_t*)malloc((size_t)query_aa);
	scores = (int32_t*)malloc((size_t)n_cand * sizeof(*scores));
	cand_off = (int32_t*)malloc(((size_t)s->n_seq + 1) * sizeof(*cand_off));
	for (i = 0, n_cand = 0; i < s->n_seq; ++i) {
		const mp_coarse_map_t *c = &s->coarse[i];
		const mp_bseq1_t *seq = &s->seq[i];
		cand_off[i] = (int32_t)n_cand;
		if (c->a == 0 || c->n_reg <= 1) continue;
		for (j = 0; j < seq->l_seq; ++j) queries[query_off + j] = ns_tab_aa20[(uint8_t)seq->seq[j]];
		for (j = 0; j < c->n_reg; ++j) {
			const mp_reg1_t *r = &c->reg[j];
			ns_gpu_gapless_candidate_t *cd = &candidates[n_cand + j];
			ns_gpu_gapless_anchor_t picked[MP_GPU_PREFILTER_ANCHORS_PER_CANDIDATE];
			int32_t picked_n = mp_gpu_prefilter_pick_anchors(mi, r, c->a, picked);
			cd->target_offset = mi->nt->ctg[r->vid>>1].off;
			cd->target_length = mi->nt->ctg[r->vid>>1].len;
			cd->query_offset = query_off;
			cd->query_length = seq->l_seq;
			cd->anchor_offset = (int32_t)n_anchors;
			cd->anchor_count = picked_n;
			cd->reverse = r->vid & 1;
			memcpy(&anchors[n_anchors], picked, (size_t)picked_n * sizeof(*picked));
			n_anchors += picked_n;
		}
		n_cand += c->n_reg;
		query_off += seq->l_seq;
	}
	cand_off[s->n_seq] = (int32_t)n_cand;
	if (n_anchors > 0 && ns_gpu_gapless_score_batch(mi->nt->seq, mi->nt->l_seq, queries, query_aa,
			candidates, (int32_t)n_cand, anchors, n_anchors, 1 << mi->opt.bbit, mi->opt.kmer,
			opt->gpu_prefilter_xdrop, opt->mat, opt->asize, ns_tab_codon, scores)) {
		mp_gapless_calib_t cal;
		memset(&cal, 0, sizeof(cal));
		if (opt->flag & MP_F_GPU_PREFILTER_ADAPTIVE) {
			mp_gapless_calibrate(s->n_seq, cand_off, scores, &cal);
			if (mp_verbose >= 3)
				fprintf(stderr, "[M::%s] batch %ld: adaptive-K calibration from %d queries: noise=%d half-bits, dup_frac=%.3f\n",
						__func__, (long)p->id, cal.n_query, cal.noise, cal.dup_frac);
		}
		for (i = 0; i < s->n_seq; ++i) {
			mp_coarse_map_t *c = &s->coarse[i];
			if (cand_off[i + 1] == cand_off[i]) continue;
			mp_gpu_prefilter_select(b->km, mi, opt, s->seq[i].l_seq, &c->n_reg, c->reg, &scores[cand_off[i]], &cal, prof);
		}
	} else if (mp_verbose >= 1 && !warned) {
		warned = 1;
		fprintf(stderr, "[WARNING]\033[1;31m GPU prefilter failed (%s); keeping all CPU candidates for this batch\033[0m\n", ns_gpu_error());
	}
	if (prof) prof->t_prefilter += mp_realtime() - t0;
	free(cand_off); free(scores); free(queries); free(anchors); free(candidates);
}

static void *worker_pipeline(void *shared, int step, void *in)
{
	int32_t i;
    pipeline_t *p = (pipeline_t*)shared;
    if (step == 0) { // step 0: read sequences
        step_t *s;
        s = Kcalloc(0, step_t, 1);
		if (p->hp) s->seq = mp_hmm_read_batch(p->hp, p->opt->mini_batch_size, p->opt->fs, &s->hmm, &s->n_seq);
		else s->seq = mp_bseq_read(p->fp, p->opt->mini_batch_size, 0, &s->n_seq);
		if (s->seq) {
			s->p = p;
			s->buf = Kcalloc(0, mp_tbuf_t*, p->n_threads);
			for (i = 0; i < p->n_threads; ++i)
				s->buf[i] = mp_tbuf_init();
			s->n_reg = Kcalloc(0, int32_t, s->n_seq);
			s->reg = Kcalloc(0, mp_reg1_t*, s->n_seq);
			if (p->opt->flag & MP_F_GPU_PREFILTER) s->coarse = Kcalloc(0, mp_coarse_map_t, s->n_seq);
			return s;
		} else free(s);
    } else if (step == 1) { // step 1: map
		step_t *s = (step_t*)in;
		if (p->opt->prescreen && s->hmm) // serial w.r.t. mapping: the prescreen runs its own kt_for over the batch
			s->cand_tab = mp_prescreen_batch(p->opt->prescreen, &p->opt->ps, p->opt->cand_max_evalue, s->n_seq, s->hmm, p->n_threads, p->ps_dump);
		if (s->coarse) {
			kt_for(p->n_threads, worker_coarse_for, in, s->n_seq);
			mp_gpu_prefilter_batch(s);
			kt_for(p->n_threads, worker_finish_for, in, s->n_seq);
			free(s->coarse);
			s->coarse = 0;
		} else {
			kt_for(p->n_threads, worker_for, in, s->n_seq);
		}
		return in;
    } else if (step == 2) { // step 2: output
        step_t *s = (step_t*)in;
		double t0 = (p->opt->flag & MP_F_PROFILE)? mp_realtime() : 0.0;
		for (i = 0; i < p->n_threads; ++i) {
			if ((p->opt->flag & MP_F_PROFILE) && s->buf[i]->km) { // snapshot free-list fragmentation before the arena is torn down
				km_stat_t st;
				km_stat(s->buf[i]->km, &st);
				if (st.n_blocks > s->buf[i]->profile.km_max_blocks) s->buf[i]->profile.km_max_blocks = st.n_blocks;
				if (st.capacity > s->buf[i]->profile.km_max_capacity) s->buf[i]->profile.km_max_capacity = st.capacity;
			}
			if (p->opt->flag & MP_F_PROFILE) mp_profile_add(&p->profile, &s->buf[i]->profile);
			mp_tbuf_destroy(s->buf[i]);
		}
		free(s->buf);
		for (i = 0; i < s->n_seq; ++i) {
			int32_t best_sc = -1, j, n_out;
			if (s->n_reg[i] > 0)
				best_sc = mp_reg_score(&s->reg[i][0]);
			for (j = 0, n_out = 0; j < s->n_reg[i] && j < p->opt->out_n; ++j) {
				const mp_reg1_t *r = &s->reg[i][j];
				int32_t sc = mp_reg_score(r);
				if (mp_dbg_flag & MP_DBG_CHAIN)
					fprintf(stderr, "Y4\t%s\t%d\t%s\t%c\t%ld\t%ld\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n", s->seq[i].name, j,
							p->mi->nt->ctg[r->vid>>1].name, "+-"[r->vid&1], (long)r->vs, (long)r->ve,
							sc, best_sc, r->p? r->p->dp_score : -1, r->p? r->p->dp_max : -1,
							r->p? r->p->full_dp : -1, r->p? r->p->n_fs : -1, r->p? r->p->n_stop : -1,
							r->p? r->p->dist_start : -1, r->p? r->p->dist_stop : -1);
				if (sc <= 0 || sc < (double)best_sc * p->opt->out_sim) continue;
				if (r->qe - r->qs < (double)s->seq[i].l_seq * p->opt->out_cov) continue;
				mp_write_output(&p->str, 0, p->mi, &s->seq[i], r, p->opt, ++p->id, j + 1);
				fwrite(p->str.s, 1, p->str.l, stdout);
				if (p->opt->flag & MP_F_PROFILE) ++p->profile.n_output;
				++n_out;
			}
			if (n_out == 0) {
				mp_write_output(&p->str, 0, p->mi, &s->seq[i], 0, p->opt, 0, 0);
				fwrite(p->str.s, 1, p->str.l, stdout);
			}
			for (j = 0; j < s->n_reg[i]; ++j) {
				free(s->reg[i][j].feat);
				free(s->reg[i][j].p);
			}
			free(s->reg[i]);
			free(s->seq[i].seq); free(s->seq[i].name);
			if (s->seq[i].comment) free(s->seq[i].comment);
			if (s->hmm) mp_hmm_destroy(s->hmm[i]);
		}
		free(s->reg); free(s->n_reg); free(s->seq); free(s->hmm);
		mp_cand_tab_destroy(s->cand_tab);
		if (p->opt->flag & MP_F_PROFILE) p->profile.t_output += mp_realtime() - t0;
		if (mp_verbose >= 3)
			fprintf(stderr, "[M::%s::%.3f*%.2f] mapped %d sequences\n", __func__, mp_realtime(), mp_cputime() / mp_realtime(), s->n_seq);
		free(s);
	}
    return 0;
}

static int32_t mp_map_hmm_file(const mp_idx_t *idx, const mp_mapopt_t *opt)
{
	mp_bseq1_t query;
	mp_tbuf_t *buf;
	mp_reg1_t *reg;
	kstring_t out = {0, 0, 0};
	int32_t n_reg, i, n_out = 0, best_sc = -1;
	const char *name = mp_hmm_name(opt->hmm), *consensus = mp_hmm_consensus(opt->hmm);
	if (!name || !consensus) return -1;
	memset(&query, 0, sizeof(query));
	query.name = mp_strdup(name);
	query.seq = mp_strdup(consensus);
	query.l_seq = mp_hmm_length(opt->hmm);
	if (!query.name || !query.seq) { free(query.name); free(query.seq); return -1; }
	buf = mp_tbuf_init();
	reg = mp_map(idx, query.l_seq, query.seq, &n_reg, buf, opt, query.name);
	if (opt->flag & MP_F_GFF) puts("##gff-version 3");
	if (n_reg > 0) best_sc = mp_reg_score(&reg[0]);
	for (i = 0; i < n_reg && i < opt->out_n; ++i) {
		int32_t sc = mp_reg_score(&reg[i]);
		if (sc <= 0 || sc < (double)best_sc * opt->out_sim || reg[i].qe - reg[i].qs < (double)query.l_seq * opt->out_cov) continue;
		mp_write_output(&out, 0, idx, &query, &reg[i], opt, 1, i + 1);
		fwrite(out.s, 1, out.l, stdout);
		++n_out;
	}
	if (n_out == 0) {
		mp_write_output(&out, 0, idx, &query, 0, opt, 0, 0);
		fwrite(out.s, 1, out.l, stdout);
	}
	for (i = 0; i < n_reg; ++i) free(reg[i].feat), free(reg[i].p);
	free(reg);
	free(out.s);
	mp_tbuf_destroy(buf);
	free(query.name);
	free(query.seq);
	return 0;
}

static int mp_is_hmm_path(const char *fn)
{
	size_t l = fn? strlen(fn) : 0;
	return l >= 4 && strcmp(fn + l - 4, ".hmm") == 0;
}

int32_t mp_map_file(const mp_idx_t *idx, const char *fn, const mp_mapopt_t *opt, int n_threads)
{
	pipeline_t pl;
	int32_t ret = 0;
	if (opt->hmm) return mp_map_hmm_file(idx, opt); // library API: caller pre-loaded one profile
	memset(&pl, 0, sizeof(pipeline_t));
	if (mp_is_hmm_path(fn)) {
		pl.hp = mp_hmm_open(fn);
		if (pl.hp == 0) return -1;
	} else {
		pl.fp = mp_bseq_open(fn);
		if (pl.fp == 0) return -1;
	}
	pl.opt = opt, pl.mi = idx;
	pl.n_threads = n_threads > 1? n_threads : 1;
	pl.gpu_start = ns_gpu_dispatch_count();
	pl.gpu_launch_start = ns_gpu_launch_count();
	pl.gpu_h2d_start = ns_gpu_h2d_seconds();
	pl.gpu_kernel_start = ns_gpu_kernel_seconds();
	pl.gpu_d2h_start = ns_gpu_d2h_seconds();
	pl.wall_start = mp_realtime();
	if (opt->prescreen && pl.hp && opt->ps.dump_fn) {
		pl.ps_dump = fopen(opt->ps.dump_fn, "w");
		if (pl.ps_dump) fputs("#query_name\tchrom\tstart0\tend0\tstrand\tevalue\n", pl.ps_dump);
		else if (mp_verbose >= 1) fprintf(stderr, "[WARNING]\033[1;31m failed to open --ps-dump file %s\033[0m\n", opt->ps.dump_fn);
	}
	if (opt->flag & MP_F_GFF) puts("##gff-version 3");
	kt_pipeline(2, worker_pipeline, &pl, 3);
	if (pl.hp && mp_hmm_error(pl.hp)) {
		fprintf(stderr, "[M::%s] ERROR parsing HMMER3 profile in %s\n", __func__, fn);
		ret = -1;
	}
	if (opt->flag & MP_F_PROFILE) {
		fprintf(stderr, "[M::profile] query=%llu aa=%llu seed=%llu anchor=%llu anchor_rescued=%llu chain=%llu prefilter_candidate=%llu prefilter_survivor=%llu prefilter_k1=%llu prefilter_k2=%llu prefilter_k4=%llu prefilter_k8=%llu refine_candidate=%llu refined=%llu exact_anchor=%llu cand_diag_seeded=%llu dp_candidate=%llu dp_rescued=%llu dp_ext_eligible=%llu dp_ext_retry=%llu dp_ext_fallback=%llu dp_realign=%llu dp_realign_kept=%llu gpu_dp=%llu gpu_launch=%llu output=%llu align_gap_calls=%llu align_cells=%llu align_ext_calls=%llu align_ext_cells=%llu align_ext_retry=%llu align_full_calls=%llu align_full_cells=%llu align_ext_calls_unguarded=%llu align_ext_cells_unguarded=%llu refine_kmers=%llu refine_pairs=%llu refine_pairs_dropped=%llu refine_buckets_capped=%llu sketch_s=%.6f anchor_s=%.6f prechain_s=%.6f chain_s=%.6f prefilter_s=%.6f refine_s=%.6f align_s=%.6f output_s=%.6f wall_s=%.6f gpu_h2d_s=%.6f gpu_kernel_s=%.6f gpu_d2h_s=%.6f refine_sketch_s=%.6f refine_pair_s=%.6f refine_chain_s=%.6f align_ext_s=%.6f align_gap_s=%.6f align_full_s=%.6f km_max_blocks=%llu km_max_capacity=%llu\n",
				(unsigned long long)pl.profile.n_query, (unsigned long long)pl.profile.n_query_aa,
				(unsigned long long)pl.profile.n_seed, (unsigned long long)pl.profile.n_anchor,
				(unsigned long long)pl.profile.n_anchor_rescued, (unsigned long long)pl.profile.n_coarse_chain,
				(unsigned long long)pl.profile.n_prefilter_candidate, (unsigned long long)pl.profile.n_prefilter_survivor,
				(unsigned long long)pl.profile.n_prefilter_k1, (unsigned long long)pl.profile.n_prefilter_k2,
				(unsigned long long)pl.profile.n_prefilter_k4, (unsigned long long)pl.profile.n_prefilter_k8,
				(unsigned long long)pl.profile.n_refine_candidate, (unsigned long long)pl.profile.n_refined,
				(unsigned long long)pl.profile.n_exact_anchor, (unsigned long long)pl.profile.n_cand_diag_seeded, (unsigned long long)pl.profile.n_dp_candidate,
				(unsigned long long)pl.profile.n_dp_rescued, (unsigned long long)pl.profile.n_dp_ext_eligible,
				(unsigned long long)pl.profile.n_dp_ext_retry, (unsigned long long)pl.profile.n_dp_ext_fallback,
				(unsigned long long)pl.profile.n_dp_realign,
				(unsigned long long)pl.profile.n_dp_realign_kept,
				(unsigned long long)(ns_gpu_dispatch_count() - pl.gpu_start),
				(unsigned long long)(ns_gpu_launch_count() - pl.gpu_launch_start), (unsigned long long)pl.profile.n_output,
				(unsigned long long)pl.profile.n_align_gap_calls, (unsigned long long)pl.profile.n_align_cells,
				(unsigned long long)pl.profile.n_align_ext_calls, (unsigned long long)pl.profile.n_align_ext_cells,
				(unsigned long long)pl.profile.n_align_ext_retry,
				(unsigned long long)pl.profile.n_align_full_calls, (unsigned long long)pl.profile.n_align_full_cells,
				(unsigned long long)pl.profile.n_align_ext_calls_unguarded, (unsigned long long)pl.profile.n_align_ext_cells_unguarded,
				(unsigned long long)pl.profile.n_refine_kmers, (unsigned long long)pl.profile.n_refine_pairs,
				(unsigned long long)pl.profile.n_refine_pairs_dropped, (unsigned long long)pl.profile.n_refine_buckets_capped,
				pl.profile.t_sketch, pl.profile.t_anchor, pl.profile.t_prechain, pl.profile.t_chain,
				pl.profile.t_prefilter, pl.profile.t_refine, pl.profile.t_align, pl.profile.t_output, mp_realtime() - pl.wall_start,
				ns_gpu_h2d_seconds() - pl.gpu_h2d_start, ns_gpu_kernel_seconds() - pl.gpu_kernel_start,
				ns_gpu_d2h_seconds() - pl.gpu_d2h_start,
				pl.profile.t_refine_sketch, pl.profile.t_refine_pair, pl.profile.t_refine_chain,
				pl.profile.t_align_ext, pl.profile.t_align_gap, pl.profile.t_align_full,
				(unsigned long long)pl.profile.km_max_blocks, (unsigned long long)pl.profile.km_max_capacity);
	}
	free(pl.str.s);
	if (pl.ps_dump) fclose(pl.ps_dump);
	if (pl.fp) mp_bseq_close(pl.fp);
	if (pl.hp) mp_hmm_close(pl.hp);
	return ret;
}
