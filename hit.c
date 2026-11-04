#include <stdlib.h>
#include <assert.h>
#include "mppriv.h"
#include "kalloc.h"

static int32_t mp_cal_chn_sc_ungap_approx(const mp_reg1_t *r, const uint64_t *a, int32_t kmer)
{
	int32_t i, x = kmer;
	for (i = 1; i < r->cnt; ++i) {
		const uint64_t a0 = a[r->off + i - 1], a1 = a[r->off + i];
		int32_t dq = (int32_t)a1 - (int32_t)a0;
		x += dq < kmer? dq : kmer;
		if (a1>>32 == a0>>32) x += MP_BLOCK_BONUS;
	}
	return x;
}

int32_t mp_cal_chn_sc_ungap(int32_t n_a, const uint64_t *a, int32_t kmer)
{
	int32_t i, x = kmer;
	for (i = 1; i < n_a; ++i) {
		const uint64_t a0 = a[i - 1], a1 = a[i];
		int32_t dq = (int32_t)a1 - (int32_t)a0, dr3 = (a1>>32) - (a0>>32);
		int32_t dr = dr3 / 3, q = dr3 - dr * 3, dg;
		dg = dq < dr? dq : dr;
		if (dq >= dr && q != 0) --x;
		else x += dg < kmer? dg : kmer;
	}
	return x;
}

/**************************************************
 * --gpu-prefilter: GPU gapless score top-K select *
 **************************************************/

typedef struct {
	int32_t index, gapless_score, chain_score;
	uint32_t vid;
	int64_t vs;
} mp_gapless_rank_t;

static int mp_gapless_rank_cmp(const void *left_, const void *right_)
{
	const mp_gapless_rank_t *left = (const mp_gapless_rank_t*)left_;
	const mp_gapless_rank_t *right = (const mp_gapless_rank_t*)right_;
	if (left->gapless_score != right->gapless_score)
		return left->gapless_score > right->gapless_score? -1 : 1;
	if (left->chain_score != right->chain_score)
		return left->chain_score > right->chain_score? -1 : 1;
	if (left->vid != right->vid) return left->vid < right->vid? -1 : 1;
	if (left->vs != right->vs) return left->vs < right->vs? -1 : 1;
	return left->index - right->index;
}

static void mp_gapless_top_insert(mp_gapless_rank_t *top, int32_t *n_top,
		int32_t capacity, const mp_gapless_rank_t *candidate)
{
	int32_t position;
	for (position = 0; position < *n_top; ++position)
		if (mp_gapless_rank_cmp(candidate, &top[position]) < 0) break;
	if (position >= capacity) return;
	if (*n_top < capacity) ++*n_top;
	memmove(&top[position + 1], &top[position],
			(size_t)(*n_top - position - 1) * sizeof(*top));
	top[position] = *candidate;
}

// The coarse chain score is intron-aware while the gapless score is not; keeping the
// best chain protects multi-exon genes from being displaced by intronless retrocopies.
// Returns -1 when the gapless winner already carries a comparable chain score.
static int32_t mp_gapless_chain_guard(int32_t n_reg, const mp_reg1_t *reg, const int32_t *scores)
{
	int32_t i, best = -1, top = -1;
	for (i = 0; i < n_reg; ++i) {
		if (best < 0 || reg[i].chn_sc > reg[best].chn_sc ||
				(reg[i].chn_sc == reg[best].chn_sc && scores[i] > scores[best]))
			best = i;
		if (top < 0 || scores[i] > scores[top] ||
				(scores[i] == scores[top] && reg[i].chn_sc > reg[top].chn_sc))
			top = i;
	}
	if (best < 0 || best == top) return -1;
	if ((int64_t)reg[top].chn_sc * 10 >= (int64_t)reg[best].chn_sc * 8) return -1;
	return best;
}

static void mp_select_gapless_small(void *km, int32_t limit, int32_t min_per_vid,
		int32_t *n_reg, mp_reg1_t *reg, const int32_t *scores, int32_t guard)
{
	mp_gapless_rank_t top[8];
	uint8_t *selected = Kcalloc(km, uint8_t, *n_reg);
	int32_t i, j, n_top = 0;
	if (min_per_vid > 0) {
		for (i = 0; i < *n_reg; ++i) {
			mp_gapless_rank_t candidate;
			int32_t count = 0, worst_same = -1;
			candidate.index = i;
			candidate.gapless_score = scores[i];
			candidate.chain_score = reg[i].chn_sc;
			candidate.vid = reg[i].vid;
			candidate.vs = reg[i].vs;
			for (j = 0; j < n_top; ++j)
				if (top[j].vid == candidate.vid) ++count, worst_same = j;
			if (count >= min_per_vid) {
				if (mp_gapless_rank_cmp(&candidate, &top[worst_same]) >= 0) continue;
				memmove(&top[worst_same], &top[worst_same + 1],
						(size_t)(n_top - worst_same - 1) * sizeof(*top));
				--n_top;
			}
			mp_gapless_top_insert(top, &n_top, limit, &candidate);
		}
		for (i = 0; i < n_top; ++i) selected[top[i].index] = 1;
	}
	if (n_top < limit) {
		mp_gapless_rank_t fill[8];
		int32_t n_fill = 0, fill_capacity = limit - n_top;
		for (i = 0; i < *n_reg; ++i) {
			mp_gapless_rank_t candidate;
			if (selected[i]) continue;
			candidate.index = i;
			candidate.gapless_score = scores[i];
			candidate.chain_score = reg[i].chn_sc;
			candidate.vid = reg[i].vid;
			candidate.vs = reg[i].vs;
			mp_gapless_top_insert(fill, &n_fill, fill_capacity, &candidate);
		}
		for (i = 0; i < n_fill; ++i) selected[fill[i].index] = 1;
	}
	if (guard >= 0) selected[guard] = 1;
	for (i = j = 0; i < *n_reg; ++i) {
		reg[i].chn_sc_ungap = scores[i];
		if (selected[i]) reg[j++] = reg[i];
	}
	*n_reg = j;
	kfree(km, selected);
}

static int mp_int32_cmp(const void *left_, const void *right_)
{
	int32_t left = *(const int32_t*)left_, right = *(const int32_t*)right_;
	return left < right? -1 : left > right;
}

// Genome calibration from one mini-batch of GPU gapless scores. `noise` is the median score of the
// non-best candidates (coarse chains that survived seeding but are not the query's winner): it rises
// with genome size/repeat content and is what a spurious locus typically scores on this genome.
// `dup_frac` is the fraction of queries whose runner-up is within the near-tie margin of the winner,
// i.e. how paralog/duplication-rich this genome looks for the current query set.
void mp_gapless_calibrate(int32_t n_query, const int32_t *cand_off, const int32_t *scores, mp_gapless_calib_t *cal)
{
	int32_t q, i, n_bg = 0, n_multi = 0, n_scored = 0, n_conf = 0;
	int32_t *bg;
	memset(cal, 0, sizeof(*cal));
	bg = (int32_t*)malloc((size_t)(cand_off[n_query] > 0? cand_off[n_query] : 1) * sizeof(*bg));
	for (q = 0; q < n_query; ++q) {
		int32_t best = INT32_MIN, second = INT32_MIN;
		if (cand_off[q + 1] - cand_off[q] < 2) continue;
		for (i = cand_off[q]; i < cand_off[q + 1]; ++i) {
			if (scores[i] > best) second = best, best = scores[i];
			else if (scores[i] > second) second = scores[i];
		}
		if (best <= 0) continue;
		++n_scored;
		for (i = cand_off[q]; i < cand_off[q + 1]; ++i)
			if (scores[i] > 0 && scores[i] < best) bg[n_bg++] = scores[i];
	}
	if (n_bg >= 64) {
		qsort(bg, (size_t)n_bg, sizeof(*bg), mp_int32_cmp);
		cal->noise = bg[n_bg / 2];
	}
	// Duplication pressure is only meaningful for queries whose winner clearly beats the background;
	// near-ties among noise-level candidates (distant queries) say nothing about paralogs.
	for (q = 0; q < n_query; ++q) {
		int32_t best = INT32_MIN, second = INT32_MIN;
		if (cand_off[q + 1] - cand_off[q] < 2) continue;
		for (i = cand_off[q]; i < cand_off[q + 1]; ++i) {
			if (scores[i] > best) second = best, best = scores[i];
			else if (scores[i] > second) second = scores[i];
		}
		if (best <= 0 || best < 4 * cal->noise) continue;
		++n_conf;
		if (second > 0 && (int64_t)second * 100 >= (int64_t)best * MP_GAPLESS_TIE_PCT) ++n_multi;
	}
	cal->n_query = n_scored;
	cal->dup_frac = n_conf > 0? (float)n_multi / n_conf : 0.0f;
	free(bg);
}

// Per-query K for --gpu-prefilter-adaptive: the number of candidates that are genuinely in
// contention on this genome, rather than a fixed bucket from the winner's margin alone.
//  - the winner is indistinguishable from the genome's background (or weak for its length): the
//    gapless score cannot rank the loci, so hand everything (max_k) to the exact DP;
//  - otherwise keep every candidate above both the near-tie margin and the noise floor, plus one
//    spare slot on duplication-rich genomes where a true paralog may score below the margin.
int32_t mp_gapless_adaptive_k(int32_t n_reg, const int32_t *scores, int32_t qlen, int32_t max_k, const mp_gapless_calib_t *cal)
{
	int32_t i, best = INT32_MIN, target, noise = cal? cal->noise : 0;
	int64_t thr;
	if (n_reg <= 0 || max_k <= 0) return 0;
	for (i = 0; i < n_reg; ++i)
		if (scores[i] > best) best = scores[i];
	if (best <= 0 || best < 2 * noise || (int64_t)best * 4 < qlen) {
		target = max_k;
	} else {
		thr = (int64_t)best * MP_GAPLESS_TIE_PCT / 100;
		if (thr < 2 * noise) thr = 2 * noise;
		for (i = 0, target = 0; i < n_reg; ++i)
			if (scores[i] >= thr) ++target;
		if (cal && cal->dup_frac >= MP_GAPLESS_DUP_RICH) ++target;
	}
	if (target < 1) target = 1;
	if (target > max_k) target = max_k;
	if (target > n_reg) target = n_reg;
	return target;
}

void mp_select_gapless(void *km, int32_t top_k, int32_t min_per_vid,
		int32_t *n_reg, mp_reg1_t *reg, const int32_t *scores)
{
	mp_gapless_rank_t *rank;
	uint8_t *selected;
	uint32_t *group_vid;
	int32_t *group_count;
	int32_t i, j, kept = 0, n_group = 0, limit, guard;
	if (*n_reg <= 0) return;
	limit = *n_reg < top_k? *n_reg : top_k;
	guard = limit < *n_reg? mp_gapless_chain_guard(*n_reg, reg, scores) : -1;
	if (limit <= 8) {
		mp_select_gapless_small(km, limit, min_per_vid, n_reg, reg, scores, guard);
		return;
	}
	rank = Kmalloc(km, mp_gapless_rank_t, *n_reg);
	selected = Kcalloc(km, uint8_t, *n_reg);
	group_vid = Kmalloc(km, uint32_t, limit);
	group_count = Kcalloc(km, int32_t, limit);
	for (i = 0; i < *n_reg; ++i) {
		reg[i].chn_sc_ungap = scores[i];
		rank[i].index = i;
		rank[i].gapless_score = scores[i];
		rank[i].chain_score = reg[i].chn_sc;
		rank[i].vid = reg[i].vid;
		rank[i].vs = reg[i].vs;
	}
	qsort(rank, (size_t)*n_reg, sizeof(*rank), mp_gapless_rank_cmp);
	for (i = 0; i < *n_reg && kept < limit && min_per_vid > 0; ++i) {
		for (j = 0; j < n_group; ++j)
			if (group_vid[j] == rank[i].vid) break;
		if (j == n_group)
			group_vid[n_group++] = rank[i].vid;
		if (group_count[j] < min_per_vid)
			selected[rank[i].index] = 1, ++group_count[j], ++kept;
	}
	for (i = 0; i < *n_reg && kept < limit; ++i)
		if (!selected[rank[i].index]) selected[rank[i].index] = 1, ++kept;
	if (guard >= 0) selected[guard] = 1;
	for (i = j = 0; i < *n_reg; ++i)
		if (selected[i]) reg[j++] = reg[i];
	*n_reg = j;
	kfree(km, group_count);
	kfree(km, group_vid);
	kfree(km, selected);
	kfree(km, rank);
}

mp_reg1_t *mp_reg_gen_from_block(void *km, const mp_idx_t *mi, int32_t n_u, const uint64_t *u, const uint64_t *a, int32_t *n_reg)
{
	int32_t i, k, nr;
	mp_reg1_t *reg;
	reg = Kcalloc(km, mp_reg1_t, n_u);
	for (i = k = nr = 0; i < n_u; ++i) {
		uint32_t n = (uint32_t)u[i];
		int32_t ts, te, is, ie;
		mp_reg1_t *r = &reg[nr++];
		r->off = k, r->cnt = n;
		is = k, ie = k + n - 1;
		ts = mp_idx_block2pos(mi, a[is]>>32);
		te = mp_idx_block2pos(mi, a[ie]>>32);
		assert(ts <= te);
		if (ts == te) { // on the same contig
			r->vid = ts;
		} else { // on different contigs
			int32_t j, js, je;
			for (j = k; j < k + n; ++j)
				if (a[j]>>32 >= mi->bo[ts+1])
					break;
			assert(j < k + n);
			js = j;
			for (j = k + n - 1; j >= js; --j)
				if (a[j]>>32 < mi->bo[te])
					break;
			je = j + 1;
			if (js - k > k + n - je) {
				r->vid = ts, ie = js - 1;
			} else {
				r->vid = te, is = je;
			}
			//fprintf(stderr, "SPLIT: %d<%d %d<=%d<=%d<=%d\n", ts, te, k, js, je, k + n);
		}
		r->vs = ((a[is]>>32) - mi->bo[r->vid]) << mi->opt.bbit;
		r->ve = ((a[ie]>>32) - mi->bo[r->vid] + 1) << mi->opt.bbit;
		r->qs = (uint32_t)a[is];
		r->qe = (uint32_t)a[ie] + 0;
		r->chn_sc = ts == te? u[i]>>32 : (uint32_t)((double)(u[i]>>32) * (ie - is + 1) / n + .499);
		r->chn_sc_ungap = mp_cal_chn_sc_ungap_approx(r, a, mi->opt.kmer);
		k += n;
	}
	*n_reg = nr;
	return reg;
}

uint64_t *mp_collate_a(void *km, int32_t n_reg, mp_reg1_t *reg)
{
	uint64_t *a;
	int64_t n_a;
	int32_t i;
	for (i = 0, n_a = 0; i < n_reg; ++i)
		n_a += reg[i].cnt;
	a = Kmalloc(km, uint64_t, n_a);
	for (i = 0, n_a = 0; i < n_reg; ++i) {
		mp_reg1_t *r = &reg[i];
		r->off = n_a;
		memcpy(&a[n_a], r->a, r->cnt * sizeof(uint64_t));
		kfree(km, r->a);
		r->a = &a[n_a];
		n_a += r->cnt;
	}
	return a;
}

int32_t mp_reg_score(const mp_reg1_t *r)
{
	return r->p? r->rank_sc : r->chn_sc;
}

int32_t mp_reg_try_replace(mp_reg1_t *dst, mp_reg1_t *trial)
{
	int32_t replace;
	assert(dst->p != 0);
	replace = trial->p && (mp_reg_score(trial) > mp_reg_score(dst) ||
			(mp_reg_score(trial) == mp_reg_score(dst) && (trial->p->dp_max > dst->p->dp_max ||
			 (trial->p->dp_max == dst->p->dp_max && trial->p->dp_score > dst->p->dp_score))));
	if (replace) {
		free(dst->p); free(dst->feat);
		*dst = *trial;
	} else {
		free(trial->p); free(trial->feat);
	}
	trial->p = 0, trial->feat = 0;
	return replace;
}

static int32_t mp_nt_eq(char a, char b)
{
	return a == b || a == b + ('a' - 'A');
}

static int32_t mp_is_canonical_junction(const mp_feat_t *left, const mp_feat_t *right)
{
	return (mp_nt_eq(left->donor[0], 'G') && mp_nt_eq(left->donor[1], 'T') &&
			mp_nt_eq(right->acceptor[0], 'A') && mp_nt_eq(right->acceptor[1], 'G')) ||
		(mp_nt_eq(left->donor[0], 'G') && mp_nt_eq(left->donor[1], 'C') &&
			mp_nt_eq(right->acceptor[0], 'A') && mp_nt_eq(right->acceptor[1], 'G')) ||
		(mp_nt_eq(left->donor[0], 'A') && mp_nt_eq(left->donor[1], 'T') &&
			mp_nt_eq(right->acceptor[0], 'A') && mp_nt_eq(right->acceptor[1], 'C'));
}

int32_t mp_reg_needs_refine(const mp_mapopt_t *opt, const mp_reg1_t *r)
{
	int32_t i;
	if (r->p == 0) return 0;
	if (r->p->n_fs > 0 || r->p->n_stop > 0) return 1;
	for (i = 0; i < r->p->n_cigar; ++i)
		if ((r->p->cigar[i]&0xf) == NS_CIGAR_I && (int32_t)(r->p->cigar[i]>>4) <= opt->acc_microexon_max)
			return 1;
	for (i = 0; i + 1 < r->n_feat; ++i) {
		const mp_feat_t *left = &r->feat[i], *right = &r->feat[i + 1];
		if (left->type == MP_FEAT_CDS && right->type == MP_FEAT_CDS && !mp_is_canonical_junction(left, right))
			return 1;
	}
	return 0;
}

void mp_set_rank_score(const mp_mapopt_t *opt, int32_t qlen, mp_reg1_t *r)
{
	int32_t i;
	int64_t score, bonus = 0, penalty = 0, base;
	if (r->p == 0) return;
	// an unguided full-region DP (anchor-rescue or junc-refine) can peak (dp_max) far above
	// its real final dp_score; only dp_score is trustworthy for ranking such a candidate
	base = opt->hmm && r->p->profile_score != INT32_MIN? r->p->profile_score : (r->p->full_dp? r->p->dp_score : r->p->dp_max);
	r->rank_sc = base;
	if (!(opt->flag & MP_F_ACCURACY) || !(opt->acc_flags & MP_ACC_RERANK)) return;
	if (qlen > 0)
		bonus += (int64_t)opt->acc_cov_bonus * (r->qe - r->qs) / qlen;
	if (r->qs == 0 && r->p->dist_start == 0) bonus += opt->acc_end_bonus;
	if (r->qe == qlen && r->p->dist_stop == 0) bonus += opt->acc_end_bonus;
	for (i = 0; i + 1 < r->n_feat; ++i) {
		const mp_feat_t *left = &r->feat[i], *right = &r->feat[i + 1];
		if (left->type != MP_FEAT_CDS || right->type != MP_FEAT_CDS) continue;
		if (mp_nt_eq(left->donor[0], 'G') && mp_nt_eq(left->donor[1], 'T') &&
			mp_nt_eq(right->acceptor[0], 'A') && mp_nt_eq(right->acceptor[1], 'G')) {
			bonus += opt->acc_splice_bonus;
		} else if (mp_is_canonical_junction(left, right)) {
			bonus += (opt->acc_splice_bonus + 1) / 2;
		} else {
			penalty += opt->acc_noncan_penalty;
		}
	}
	penalty += (int64_t)opt->acc_fs_penalty * r->p->n_fs;
	penalty += (int64_t)opt->acc_stop_penalty * r->p->n_stop;
	// cap the composite bonus so rerank can only break near-ties, never outrank a much better raw dp_max
	if (bonus > opt->acc_rerank_margin) bonus = opt->acc_rerank_margin;
	score = base + bonus - penalty;
	if (score < 0) score = 0;
	if (score > INT32_MAX) score = INT32_MAX;
	r->rank_sc = score;
}

static int32_t mp_aux_score_lt(const mp_reg1_t *r, const mp128_t *a, const mp128_t *b)
{
	int32_t score_a = r[(uint32_t)a->y].p->dp_score;
	int32_t score_b = r[(uint32_t)b->y].p->dp_score;
	return score_a != score_b? score_a < score_b : (uint32_t)(a->y>>32) < (uint32_t)(b->y>>32);
}

void mp_sort_reg(void *km, int *n_regs, mp_reg1_t *r, int32_t rank_tiebreak)
{
	int32_t i, j, k, l, n_aux, n = *n_regs, has_cigar = 0, no_cigar = 0;
	mp128_t *aux;
	mp_reg1_t *t;

	if (n <= 1) return;
	aux = (mp128_t*)kmalloc(km, n * 16);
	t = (mp_reg1_t*)kmalloc(km, n * sizeof(mp_reg1_t));
	for (i = n_aux = 0; i < n; ++i) {
		if (r[i].cnt > 0) { // squeeze out elements with cnt==0 (soft deleted)
			int score;
			if (r[i].p) score = mp_reg_score(&r[i]), has_cigar = 1;
			else score = r[i].chn_sc, no_cigar = 1;
			if (rank_tiebreak) {
				aux[n_aux].x = (uint64_t)(uint32_t)score << 32 | ((uint32_t)r[i].p->dp_max ^ 0x80000000U);
				aux[n_aux++].y = (uint64_t)r[i].hash << 32 | (uint32_t)i;
			} else {
				aux[n_aux].x = (uint64_t)score << 32 | r[i].hash;
				aux[n_aux++].y = i;
			}
		} else if (r[i].p) {
			free(r[i].p); free(r[i].feat);
			r[i].p = 0;
		}
	}
	assert(has_cigar + no_cigar == 1);
	radix_sort_mp128x(aux, aux + n_aux);
	if (rank_tiebreak) {
		for (i = 0; i < n_aux; i = j) {
			mp128_t x;
			for (j = i + 1; j < n_aux && aux[j].x == aux[i].x; ++j) {}
			for (k = i + 1; k < j; ++k) {
				x = aux[k];
				for (l = k; l > i && mp_aux_score_lt(r, &x, &aux[l - 1]); --l)
					aux[l] = aux[l - 1];
				aux[l] = x;
			}
		}
	}
	for (i = n_aux - 1; i >= 0; --i)
		t[n_aux - 1 - i] = r[(uint32_t)aux[i].y];
	memcpy(r, t, sizeof(mp_reg1_t) * n_aux);
	*n_regs = n_aux;
	kfree(km, aux);
	kfree(km, t);
}

void mp_set_parent(void *km, float mask_level, int mask_len, int n, mp_reg1_t *r, int sub_diff, int hard_mask_level) // and compute mp_reg1_t::subsc
{
	int i, j, k, *w;
	uint64_t *cov;
	if (n <= 0) return;
	for (i = 0; i < n; ++i) r[i].id = i;
	cov = (uint64_t*)kmalloc(km, n * sizeof(uint64_t));
	w = (int*)kmalloc(km, n * sizeof(int));
	w[0] = 0, r[0].parent = 0;
	for (i = 1, k = 1; i < n; ++i) {
		mp_reg1_t *ri = &r[i];
		int si = ri->qs, ei = ri->qe, n_cov = 0, uncov_len = 0;
		if (hard_mask_level) goto skip_uncov;
		for (j = 0; j < k; ++j) { // traverse existing primary hits to find overlapping hits
			mp_reg1_t *rp = &r[w[j]];
			int sj = rp->qs, ej = rp->qe;
			if (ej <= si || sj >= ei) continue;
			if (sj < si) sj = si;
			if (ej > ei) ej = ei;
			cov[n_cov++] = (uint64_t)sj<<32 | ej;
		}
		if (n_cov == 0) {
			goto set_parent_test; // no overlapping primary hits; then i is a new primary hit
		} else if (n_cov > 0) { // there are overlapping primary hits; find the length not covered by existing primary hits
			int j, x = si;
			radix_sort_mp64(cov, cov + n_cov);
			for (j = 0; j < n_cov; ++j) {
				if ((int)(cov[j]>>32) > x) uncov_len += (cov[j]>>32) - x;
				x = (int32_t)cov[j] > x? (int32_t)cov[j] : x;
			}
			if (ei > x) uncov_len += ei - x;
		}
skip_uncov:
		for (j = 0; j < k; ++j) { // traverse existing primary hits again
			mp_reg1_t *rp = &r[w[j]];
			int sj = rp->qs, ej = rp->qe, min, max, ol;
			if (ej <= si || sj >= ei) continue; // no overlap
			min = ej - sj < ei - si? ej - sj : ei - si;
			max = ej - sj > ei - si? ej - sj : ei - si;
			ol = si < sj? (ei < sj? 0 : ei < ej? ei - sj : ej - sj) : (ej < si? 0 : ej < ei? ej - si : ei - si); // overlap length; TODO: this can be simplified
			if ((float)ol / min - (float)uncov_len / max > mask_level && uncov_len <= mask_len) { // then this is a secondary hit
				int cnt_sub = 0, sci = ri->chn_sc;
				ri->parent = rp->parent;
				rp->subsc = rp->subsc > sci? rp->subsc : sci;
				if (ri->cnt >= rp->cnt) cnt_sub = 1;
				if (rp->p && ri->p && (rp->vid != ri->vid || rp->vs != ri->vs || rp->ve != ri->ve || ol != min)) { // the last condition excludes identical hits after DP
					sci = mp_reg_score(ri);
					rp->p->dp_max2 = rp->p->dp_max2 > ri->p->dp_max? rp->p->dp_max2 : ri->p->dp_max;
					rp->rank_sc2 = rp->rank_sc2 > sci? rp->rank_sc2 : sci;
					if (mp_reg_score(rp) - sci <= sub_diff) cnt_sub = 1;
				}
				if (cnt_sub) ++rp->n_sub;
				break;
			}
		}
set_parent_test:
		if (j == k) w[k++] = i, ri->parent = i, ri->n_sub = 0;
	}
	kfree(km, cov);
	kfree(km, w);
}

void mp_sync_regs(void *km, int n_regs, mp_reg1_t *regs) // keep mp_reg1_t::{id,parent} in sync; also reset id
{
	int *tmp, i, max_id = -1, n_tmp;
	if (n_regs <= 0) return;
	for (i = 0; i < n_regs; ++i) // NB: doesn't work if mp_reg1_t::id is negative
		max_id = max_id > regs[i].id? max_id : regs[i].id;
	n_tmp = max_id + 1;
	tmp = (int*)kmalloc(km, n_tmp * sizeof(int));
	for (i = 0; i < n_tmp; ++i) tmp[i] = -1;
	for (i = 0; i < n_regs; ++i)
		if (regs[i].id >= 0) tmp[regs[i].id] = i;
	for (i = 0; i < n_regs; ++i) {
		mp_reg1_t *r = &regs[i];
		r->id = i;
		if (r->parent == MP_PARENT_TMP_PRI)
			r->parent = i;
		else if (r->parent >= 0 && tmp[r->parent] >= 0)
			r->parent = tmp[r->parent];
		else r->parent = MP_PARENT_UNSET;
	}
	kfree(km, tmp);
}

void mp_select_sub(void *km, float pri_ratio, int min_diff, int best_n, int *n_, mp_reg1_t *r)
{
	if (pri_ratio > 0.0f && *n_ > 0) {
		int i, k, n = *n_, n_2nd = 0, chn_sc_ungap = -1;
		for (i = 0; i < n; ++i) // find the largest chn_sc_ungap
			chn_sc_ungap = chn_sc_ungap > r[i].chn_sc_ungap? chn_sc_ungap : r[i].chn_sc_ungap;
		for (i = k = 0; i < n; ++i) {
			int p = r[i].parent;
			int sci, scp;
			if (p < 0) { // parent was pruned earlier and never resynced; keep this hit as-is
				r[k++] = r[i];
				continue;
			}
			sci = mp_reg_score(&r[i]);
			scp = mp_reg_score(&r[p]);
			if (p == i) { // primary or inversion
				r[k++] = r[i];
			} else if ((sci >= scp * pri_ratio || sci + min_diff >= scp) && n_2nd < best_n) {
				if (!(r[i].qs == r[p].qs && r[i].qe == r[p].qe && r[i].vid == r[p].vid && r[i].vs == r[p].vs && r[i].ve == r[p].ve)) // not identical hits
					r[k++] = r[i], ++n_2nd;
				else if (r[i].p) { free(r[i].p); free(r[i].feat); }
			} else if (r[i].p == 0 && r[p].p == 0 && chn_sc_ungap > 0 && r[i].chn_sc_ungap >= chn_sc_ungap * pri_ratio && n_2nd < best_n) {
				if (!(r[i].qs == r[p].qs && r[i].qe == r[p].qe && r[i].vid == r[p].vid && r[i].vs == r[p].vs && r[i].ve == r[p].ve)) // not identical hits
					r[k++] = r[i], ++n_2nd;
			} else if (r[i].p) { free(r[i].p); free(r[i].feat); }
		}
		if (k != n) mp_sync_regs(km, k, r); // removing hits requires sync()
		*n_ = k;
	}
}

// In strict mode, only let a multi-exon candidate outrank the top single-exon hit when every
// junction looks like a real splice site and the score margin covers all of its introns; this
// stops a single weak/spurious intron from letting a wide, low-purity multi-exon span win over
// a compact, correctly-bounded single-exon hit (the common failure mode on intron-poor genomes,
// e.g. HMM-profile library scans against yeast).
static int32_t mp_multi_exon_all_canonical(const mp_reg1_t *r)
{
	int32_t i;
	for (i = 0; i + 1 < r->n_feat; ++i) {
		const mp_feat_t *left = &r->feat[i], *right = &r->feat[i + 1];
		if (left->type == MP_FEAT_CDS && right->type == MP_FEAT_CDS && !mp_is_canonical_junction(left, right))
			return 0;
	}
	return 1;
}

void mp_select_multi_exon(int32_t n, mp_reg1_t *r, int32_t single_penalty, int32_t strict)
{
	int32_t i, margin;
	mp_reg1_t t;
	if (n < 2 || r[0].n_exon != 1) return;
	for (i = 1; i < n; ++i)
		if (r[i].n_exon >= 2)
			break;
	if (i == n) return;
	if (r[0].p == 0 || r[i].p == 0) return;
	margin = single_penalty;
	if (strict) {
		if (!mp_multi_exon_all_canonical(&r[i])) return;
		margin = single_penalty * (r[i].n_exon - 1);
	}
	if (mp_reg_score(&r[0]) < mp_reg_score(&r[i]) + margin)
		t = r[0], r[0] = r[i], r[i] = t;
}

// In strict mode, drop (not merely deprioritize) a multi-exon candidate as soon as ANY single
// intron is unsupported by ANY of three independent, complementary signals (OR, not a single
// tunable ratio): (1) LOCAL -- wider than the evidence contributed by its own two immediately
// flanking exons alone; catches a spurious junction bridging two otherwise-plausible exons.
// (2) WHOLE-CANDIDATE -- wider than the total exonic evidence summed over every exon in the
// candidate; catches a junction whose own flanking exons look individually large enough to pass
// (1), but the chain as a whole is still mostly intron with only scattered small real anchors.
// (3) ABSOLUTE -- exceeds a hard nt ceiling regardless of any exon evidence; ratio-based checks
// (1) and (2) can both be defeated when every flanking exon is individually large (confirmed via
// RESF1_C/DUF3584 in the yeast gold-standard benchmark: several large, unrelated protein-family
// fragments stitched by a few-kb intron, where local_nt and whole exon_nt both dwarf the gap even
// though the gap is biologically implausible for this genome), so an absolute cap is needed as a
// backstop independent of exon size. A canonical-looking splice site alone is a weak filter
// (GT...AG occurs by chance often enough over several kb of intergenic yeast sequence); this is
// otherwise invisible to mp_select_multi_exon whenever there is no competing single-exon candidate
// to swap in (the common case: this candidate is the query's only output).
int32_t mp_reject_spurious_intron(void *km, int32_t n, mp_reg1_t *r, int32_t strict, int32_t min_intron, float ratio, int32_t abs_max, int32_t free_removed)
{
	int32_t i, k, dropped = 0;
	if (!strict || n <= 0) return n;
	for (i = k = 0; i < n; ++i) {
		mp_reg1_t *ri = &r[i];
		int32_t j, bad = 0;
		int64_t exon_nt = 0;
		if (ri->n_exon < 2 || ri->p == 0 || ri->feat == 0 || ri->n_feat <= 0) { r[k++] = *ri; continue; }
		for (j = 0; j < ri->n_feat; ++j)
			if (ri->feat[j].type == MP_FEAT_CDS) exon_nt += ri->feat[j].ve - ri->feat[j].vs;
		for (j = 0; j + 1 < ri->n_feat; ++j) {
			const mp_feat_t *left = &ri->feat[j], *right = &ri->feat[j + 1];
			int64_t gap, local_nt;
			if (left->type != MP_FEAT_CDS || right->type != MP_FEAT_CDS) continue;
			gap = right->vs - left->ve;
			if (abs_max > 0 && gap > abs_max) { bad = 1; break; }
			if (gap < min_intron) continue;
			local_nt = (left->ve - left->vs) + (right->ve - right->vs);
			if ((double)gap > (double)local_nt * ratio) { bad = 1; break; }
			if ((double)gap > (double)exon_nt * ratio) { bad = 1; break; }
		}
		if (bad) {
			if (free_removed) { free(ri->p); free(ri->feat); }
			++dropped;
			continue;
		}
		r[k++] = *ri;
	}
	(void)dropped; // array is already compacted above; id/parent are meaningless until mp_set_parent() runs, so no mp_sync_regs() here
	return k;
}

// Orthogonal to mp_reject_spurious_intron(): on compact, gene-dense, intron-poor genomes the
// remaining low-purity candidates are hits whose genomic span is many times the aligned query
// length, inflated by several gaps that each individually pass the per-junction checks (few short
// introns plus long low-complexity/tandem-repeat deletion runs). This checks the WHOLE hit at
// once: drop when span > 3*(qe-qs)*span_ratio AND the absolute excess reaches min_excess (so a
// short real intron in a tiny gene can never trip the ratio alone). Disabled when span_ratio<=0.
int32_t mp_reject_low_purity_span(void *km, int32_t n, mp_reg1_t *r, float span_ratio, int32_t min_excess, int32_t free_removed)
{
	int32_t i, k, dropped = 0;
	if (span_ratio <= 0.0f || n <= 0) return n;
	for (i = k = 0; i < n; ++i) {
		mp_reg1_t *ri = &r[i];
		int64_t span, qnt;
		if (ri->p == 0) { r[k++] = *ri; continue; }
		span = ri->ve - ri->vs;
		qnt = 3LL * (ri->qe - ri->qs);
		if (span - qnt >= min_excess && (double)span > (double)qnt * span_ratio) {
			if (free_removed) { free(ri->p); free(ri->feat); }
			++dropped;
			continue;
		}
		r[k++] = *ri;
	}
	(void)dropped; // array is already compacted above; id/parent are meaningless until mp_set_parent() runs, so no mp_sync_regs() here
	return k;
}

// Absolute output-quality floor, standing in for the E-value cutoff of profile-HMM searches:
// with no significance test, a query with no true homolog in the genome still emits its best
// alignment, typically a negative-DP-score, low-positives junk hit that grazes or overlaps a
// real gene and wrecks purity. Drop candidates whose DP score (PAF AS) is below min_score or
// whose positive-substitution fraction 3*n_plus/blen is below min_pos_ratio. True homologs
// score well above both floors; <=0 disables either test. Two np-based refinements (yeast
// Pfam diagnosis): true hits killed by the floors are long alignments whose density is diluted
// by introns/gaps but whose ABSOLUTE positive count stays high (np median 41 vs junk 20), so
// n_plus >= rescue_np exempts a candidate from both floors; conversely junk that sneaks past
// the floors is short high-density grazing (np P90=17, blen~51), so n_plus < min_np drops
// unconditionally. <=0 disables either np rule.
int32_t mp_reject_weak_hit(void *km, int32_t n, mp_reg1_t *r, int32_t min_score, float min_pos_ratio, int32_t rescue_np, int32_t rescue_score, int32_t min_np, int32_t free_removed, int32_t use_ps, int32_t min_ps, int32_t ps_rescue_score, float ps_rescue_posr)
{
	int32_t i, k, dropped = 0;
	if ((min_score <= 0 && min_pos_ratio <= 0.0f && min_np <= 0 && min_ps <= 0) || n <= 0) return n;
	for (i = k = 0; i < n; ++i) {
		mp_reg1_t *ri = &r[i];
		int32_t weak = 0;
		int32_t score;
		if (ri->p == 0) { r[k++] = *ri; continue; }
		// --thr-use-ps: floors read the HMM profile Viterbi score (ps) instead of DP score (AS) when a profile query supplies one
		score = use_ps && ri->p->profile_score != INT32_MIN? ri->p->profile_score : ri->p->dp_score;
		if (min_score > 0 && score < min_score) weak = 1;
		// a score this high is significant on its own; skip the density test (strong hits with posr just under the floor)
		if (!weak && min_pos_ratio > 0.0f && !(rescue_score > 0 && score >= rescue_score) && ri->p->blen > 0 && 3.0 * ri->p->n_plus < (double)ri->p->blen * min_pos_ratio) weak = 1;
		if (weak && rescue_np > 0 && ri->p->n_plus >= rescue_np) weak = 0;
		if (min_np > 0 && ri->p->n_plus < min_np) weak = 1;
		// --acc-min-ps: unconditional ps floor on top of the AS floor (mirrors the offline calibrate_filter/score_sweep ps cut);
		// --acc-ps-rescue: ps is unnormalised, so a strong dense DP alignment (short models) is exempt
		if (min_ps > 0 && ri->p->profile_score != INT32_MIN && ri->p->profile_score < min_ps &&
			!(ps_rescue_score > 0 && ri->p->dp_score >= ps_rescue_score && ri->p->blen > 0 && 3.0 * ri->p->n_plus >= (double)ri->p->blen * ps_rescue_posr))
			weak = 1;
		if (weak) {
			if (free_removed) { free(ri->p); free(ri->feat); }
			++dropped;
			continue;
		}
		r[k++] = *ri;
	}
	(void)dropped; // array is already compacted above; id/parent are meaningless until mp_set_parent() runs, so no mp_sync_regs() here
	return k;
}

uint64_t *mp_cal_max_ext(void *km, const mp_ntdb_t *nt, int32_t n_reg, mp_reg1_t *reg, const uint64_t *a, int32_t min_ext, int32_t max_ext, int32_t iso_ext)
{
	mp128_t *b;
	uint64_t *ext;
	int32_t i;
	if (n_reg <= 0) return 0;
	b = Kmalloc(km, mp128_t, n_reg);
	for (i = 0; i < n_reg; ++i) {
		mp_reg1_t *r = &reg[i];
		b[i].x = nt? r->vs + nt->ctg[r->vid>>1].off + (r->vid&1? nt->ctg[r->vid>>1].len : 0) : a[r->off] >> 32;
		b[i].y = i;
	}
	radix_sort_mp128x(b, b + n_reg);
	ext = Kmalloc(km, uint64_t, n_reg);
	for (i = 0; i < n_reg; ++i) {
		// iso_ext (usually <= max_ext) bounds candidates with no real neighboring region to justify
		// a long speculative search; max_ext still bounds genuine inter-exon gaps between real neighbors
		int32_t left = iso_ext, right = iso_ext, j = b[i].y;
		mp_reg1_t *r = &reg[j];
		if (i > 0) {
			mp_reg1_t *q = &reg[b[i-1].y];
			if (q->vid == r->vid && q->qe >= r->qs) {
				left = r->vs - q->ve < max_ext? r->vs - q->ve : max_ext;
				left = left > min_ext? left : min_ext;
			}
		}
		if (i < n_reg - 1) {
			mp_reg1_t *q = &reg[b[i+1].y];
			if (q->vid == r->vid && r->qe >= q->qs) {
				right = q->vs - r->ve < max_ext? q->vs - r->ve : max_ext;
				right = right > min_ext? right : min_ext;
			}
		}
		ext[j] = (uint64_t)left<<32 | right;
	}
	kfree(km, b);
	return ext;
}
