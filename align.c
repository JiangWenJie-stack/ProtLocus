#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <pthread.h>
#include "mppriv.h"
#include "nasw.h"

static void mp_filter_seed(int32_t cnt, uint64_t *a, int32_t max_aa_dist, int32_t min_cnt, int32_t kmer2, int32_t trim_back)
{
	int32_t i, j;
	for (i = 0; i < cnt; ++i) {
		for (j = i + 1; j < cnt; ++j) {
			int32_t x0, y0, x1, y1;
			x0 = a[j-1]>>32, y0 = (int32_t)a[j-1];
			x1 = a[j]>>32,   y1 = (int32_t)a[j];
			if ((x1 - x0) % 3 != 0 || x1 - x0 > max_aa_dist * 3 || y1 - y0 > max_aa_dist)
				break;
		}
		if (j - i >= min_cnt) { // a tight chain block
			int32_t k, t = (int32_t)a[j-1];
			for (k = j - 2; k >= i; --k)
				if (t - (int32_t)a[k] >= trim_back)
					break;
			t = (int32_t)a[i] + 1 - kmer2;
			for (; i < k; ++i)
				if ((int32_t)a[i] + 1 - t >= trim_back)
					break;
			for (; i <= k; ++i)
				a[i] |= 1ULL<<31;
			i = j - 1;
		}
	}
}

static inline int32_t mp_score_ungapped(int32_t alen, const uint8_t *nseq, const char *aseq, int32_t asize, const int8_t *mat, const int8_t *pssm)
{
	int32_t i, j, score = 0;
	for (i = 0, j = 0; i < alen; i += 3, ++j) {
		uint8_t nt_aa, aa_aa, codon = nseq[i]<<4 | nseq[i+1]<<2 | nseq[i+2];
		nt_aa = nseq[i] > 3 || nseq[i+1] > 3 || nseq[i+2] > 3? ns_tab_aa20['X'] : ns_tab_codon[codon];
		aa_aa = ns_tab_aa20[(uint8_t)aseq[j]];
		score += pssm? pssm[j * asize + nt_aa] : mat[nt_aa * asize + aa_aa];
	}
	return score;
}

typedef struct {
	int32_t n, m;
	uint32_t *c;
} mp_cigar_t;

typedef struct {
	int32_t nlen, alen;
	const uint8_t *nseq, *ss;
	const char *aseq;
	ns_gpu_request_t *gpu_request;
} mp_align_task_t;

static inline void mp_map2ns_opt(const mp_mapopt_t *mo, ns_opt_t *no)
{
	int32_t i;
	ns_opt_init(no);
	no->go = mo->go, no->ge = mo->ge, no->io = mo->io, no->fs = mo->fs, no->xdrop = mo->xdrop, no->sc = mo->mat;
	no->ie_coef = mo->ie_coef;
	no->end_bonus = mo->end_bonus;
	no->sp_null_bonus = mo->sp_null_bonus;
	no->gpu_device = mo->gpu_device, no->gpu_min_cells = mo->gpu_min_cells;
	if (mo->flag & MP_F_GPU) no->flag |= NS_F_GPU;
	ns_opt_set_sp(no, mo->sp_model);
	for (i = 0; i < 6; ++i) no->sp[i] = (int32_t)(no->sp[i] * mo->sp_scale + .499f);
}

static ns_gpu_request_t *mp_align_seq_submit(void *km, const mp_mapopt_t *opt, const ns_opt_t *ns_opt0,
		int32_t nlen, const uint8_t *nseq, int32_t alen, const char *aseq, const uint8_t *ss)
{
	ns_opt_t ns_opt;
	if (!(ns_opt0->flag & NS_F_GPU) || (nlen == alen * 3 && alen <= opt->kmer2) ||
		(int64_t)nlen * alen < ns_opt0->gpu_min_cells)
		return 0;
	ns_opt = *ns_opt0;
	ns_opt.flag |= NS_F_CIGAR;
	return ns_gpu_submit(km, (const char*)nseq, nlen, aseq, alen, &ns_opt, ss);
}

/* mirrors the nt/aa accounting in mp_extra_cal() to catch a corrupt sub-task CIGAR at its source */
static void ns_cigar_span(const uint32_t *cigar, int32_t n_cigar, int32_t *nl_, int32_t *al_)
{
	int32_t k, nl = 0, al = 0;
	for (k = 0; k < n_cigar; ++k) {
		int32_t op = cigar[k]&0xf, len = cigar[k]>>4;
		if (op == NS_CIGAR_M) nl += len * 3, al += len;
		else if (op == NS_CIGAR_I) al += len;
		else if (op == NS_CIGAR_D) nl += len * 3;
		else if (op == NS_CIGAR_F) nl += len;
		else if (op == NS_CIGAR_G) nl += len, ++al;
		else if (op == NS_CIGAR_N || op == NS_CIGAR_U || op == NS_CIGAR_V) nl += len, al += (op != NS_CIGAR_N);
	}
	*nl_ = nl, *al_ = al;
}

static int32_t mp_align_seq(void *km, const mp_mapopt_t *opt, const ns_opt_t *ns_opt0, int32_t nlen,
		const uint8_t *nseq, int32_t alen, const char *aseq, const uint8_t *ss, const int8_t *pssm,
		ns_gpu_request_t *gpu_request, mp_cigar_t *cigar)
{
	int32_t i;
	if (nlen == alen * 3 && alen <= opt->kmer2) {
		cigar->c = ns_push_cigar(km, &cigar->n, &cigar->m, cigar->c, NS_CIGAR_M, alen);
		return mp_score_ungapped(alen, nseq, aseq, opt->asize, opt->mat, pssm);
	} else {
		ns_opt_t ns_opt = *ns_opt0;
		ns_rst_t rst;
		int gpu_success = 0;
		ns_opt.flag |= NS_F_CIGAR;
		ns_opt.pssm = pssm;
		ns_rst_init(&rst);
		if (gpu_request) gpu_success = ns_gpu_wait(gpu_request, &rst);
		if (!gpu_success) {
			// always drop NS_F_GPU here, else ns_global_gs_auto() re-dispatches to the GPU on its own
			ns_opt.flag &= ~NS_F_GPU;
			ns_global_gs_auto(km, (const char*)nseq, nlen, aseq, alen, &ns_opt, ss, &rst);
		}
		if (getenv("NS_GPU_DEBUG_CHECK")) {
			int32_t nl_got, al_got;
			ns_cigar_span(rst.cigar, rst.n_cigar, &nl_got, &al_got);
			if (nl_got != nlen || al_got != alen) {
				fprintf(stderr, "GPU_DEBUG_CHECK: MISMATCH tid=%lu src=%s nlen=%d nl_got=%d alen=%d al_got=%d score=%d n_cigar=%d\n",
						(unsigned long)pthread_self(), gpu_success? "gpu" : "cpu", nlen, nl_got, alen, al_got, rst.score, rst.n_cigar);
				fprintf(stderr, "GPU_DEBUG_CHECK: nseq=");
				for (i = 0; i < nlen; ++i) fputc("ACGTN"[nseq[i] > 4? 4 : nseq[i]], stderr);
				fprintf(stderr, "\nGPU_DEBUG_CHECK: aseq=%.*s\n", alen, aseq);
				fprintf(stderr, "GPU_DEBUG_CHECK: opt flag=%d go=%d ge=%d io=%d fs=%d xdrop=%d end_bonus=%d sp=%d,%d,%d,%d,%d,%d sp_null_bonus=%d ie_coef=%g\n",
						ns_opt.flag, ns_opt.go, ns_opt.ge, ns_opt.io, ns_opt.fs, ns_opt.xdrop, ns_opt.end_bonus,
						ns_opt.sp[0], ns_opt.sp[1], ns_opt.sp[2], ns_opt.sp[3], ns_opt.sp[4], ns_opt.sp[5], ns_opt.sp_null_bonus, ns_opt.ie_coef);
				fprintf(stderr, "GPU_DEBUG_CHECK: ss=");
				if (ss) for (i = 0; i < nlen; ++i) fprintf(stderr, "%02x", ss[i]);
				fputc('\n', stderr);
			}
		}
		//printf("%d\t%d\t%d\t%d\n", nlen, alen, rst.score, rst.n_cigar);
		for (i = 0; i < rst.n_cigar; ++i)
			cigar->c = ns_push_cigar(km, &cigar->n, &cigar->m, cigar->c, rst.cigar[i]&0xf, rst.cigar[i]>>4);
		kfree(km, rst.cigar);
		return rst.score;
	}
}

// pssm: PSSM row of aa[0] (HMM mode), or NULL to score with opt->mat
static void mp_extra_cal(mp_reg1_t *r, const mp_mapopt_t *opt, const uint8_t *nt, int32_t l_nt, const char *aa, int32_t qlen, int32_t has_spsc, const int8_t *pssm)
{
	int32_t k, i, j, l, nl = 0, al = 0, ft, n_intron, has_stop;
	int32_t blen0, n_iden0, phase0, qs0, n_fs0, n_stop0, score0;
	int64_t vs0;
	uint8_t aa_ambi = ns_tab_aa20['X'], aa_stop = ns_tab_aa20['*'];
	char acceptor0[2];
	mp_extra_t *e = r->p;
	mp_feat_t *f;

	has_stop = (r->qe == qlen && r->p->dist_stop == 0); // IMPORTANT: call mp_extra_stop() before this function
	for (k = 0, n_intron = 0; k < r->p->n_cigar; ++k) {
		int32_t op = r->p->cigar[k]&0xf;
		if (op == NS_CIGAR_N || op == NS_CIGAR_U || op == NS_CIGAR_V)
			++n_intron;
	}
	r->n_exon = n_intron + 1;
	r->n_feat = r->n_exon + (!!has_stop);
	r->feat = Kcalloc(0, mp_feat_t, r->n_feat);

	e->blen = e->n_iden = e->n_plus = e->n_fs = e->n_stop = e->dp_max = 0;
	blen0 = n_iden0 = score0 = n_fs0 = n_stop0 = 0, phase0 = 0;
	vs0 = r->vs, qs0 = r->qs;
	acceptor0[0] = acceptor0[1] = 0;
	for (k = 0, ft = 0; k < e->n_cigar; ++k) {
		int32_t op = e->cigar[k]&0xf, len = e->cigar[k]>>4, len3 = len * 3;
		if (op == NS_CIGAR_M) {
			for (i = nl, j = al, l = 0; l < len; ++l, ++j, i += 3) {
				uint8_t nt_aa, aa_aa, codon = nt[i]<<4 | nt[i+1]<<2 | nt[i+2];
				int32_t s;
				nt_aa = nt[i] > 3 || nt[i+1] > 3 || nt[i+2] > 3? aa_ambi : ns_tab_codon[codon];
				aa_aa = ns_tab_aa20[(uint8_t)aa[j]];
				s = pssm? pssm[j * opt->asize + nt_aa] : opt->mat[nt_aa * opt->asize + aa_aa];
				e->n_stop += (nt_aa == aa_stop);
				e->n_iden += (nt_aa == aa_aa);
				e->n_plus += (s > 0);
				e->dp_max += s;
			}
			nl += len3, al += len, e->blen += len3;
		} else if (op == NS_CIGAR_I) {
			e->dp_max -= opt->go + opt->ge * len;
			al += len, e->blen += len3;
		} else if (op == NS_CIGAR_D) {
			for (i = nl, l = 0; l < len; ++l, i += 3) { // we need to count in-frame stop codons in deletions
				uint8_t codon = nt[i]<<4 | nt[i+1]<<2 | nt[i+2];
				uint8_t nt_aa = nt[i] > 3 || nt[i+1] > 3 || nt[i+2] > 3? aa_ambi : ns_tab_codon[codon];
				e->n_stop += (nt_aa == aa_stop);
			}
			e->dp_max -= opt->go + opt->ge * len;
			nl += len3, e->blen += len3;
		} else if (op == NS_CIGAR_F) {
			e->dp_max -= opt->fs;
			nl += len, e->blen += len, e->n_fs++;
		} else if (op == NS_CIGAR_G) {
			e->dp_max -= opt->fs;
			nl += len, ++al, e->blen += 3, e->n_fs++;
		} else if (op == NS_CIGAR_N || op == NS_CIGAR_U || op == NS_CIGAR_V) { // intron
			if (op == NS_CIGAR_U || op == NS_CIGAR_V) { // phase-1 or phase-2 intron
				uint8_t n1, n2, n3, codon, nt_aa, aa_aa;
				int32_t s;
				if (op == NS_CIGAR_U) n1 = nt[nl], n2 = nt[nl + len - 2], n3 = nt[nl + len - 1];
				else n1 = nt[nl], n2 = nt[nl + 1], n3 = nt[nl + len - 1];
				codon = n1<<4 | n2<<2 | n3;
				nt_aa = n1 > 3 || n2 > 3 || n3 > 3? ns_tab_aa20['X'] : ns_tab_codon[codon];
				aa_aa = ns_tab_aa20[(uint8_t)aa[al]];
				s = pssm? pssm[al * opt->asize + nt_aa] : opt->mat[nt_aa * opt->asize + aa_aa];
				//printf("%c:%c, score=%d\n", ns_tab_aa_i2c[nt_aa], ns_tab_aa_i2c[aa_aa], s);
				e->n_stop += (nt_aa == aa_stop);
				e->n_iden += (nt_aa == aa_aa);
				e->n_plus += (s > 0);
				e->dp_max += s;
				e->blen += 3;
			}
			// calculate r->feat
			f = &r->feat[ft++];
			f->type = MP_FEAT_CDS;
			f->vs = vs0, f->qs = qs0, f->qe = r->qs + al, f->n_fs = n_fs0, f->n_stop = n_stop0, f->phase = phase0;
			f->blen = e->blen - blen0, f->n_iden = e->n_iden - n_iden0, f->n_fs = e->n_fs - n_fs0, f->n_stop = e->n_stop - n_stop0, f->score = e->dp_max - score0;
			if (ft > 1) f->acceptor[0] = acceptor0[0], f->acceptor[1] = acceptor0[1];
			if (op == NS_CIGAR_N) {
				f->ve = r->vs + nl;
				vs0 = r->vs + nl + len, phase0 = 0;
			} else if (op == NS_CIGAR_U) {
				f->ve = r->vs + nl + 1;
				vs0 = r->vs + nl + len - 2, phase0 = 2;
			} else if (op == NS_CIGAR_V) {
				f->ve = r->vs + nl + 2;
				vs0 = r->vs + nl + len - 1, phase0 = 1;
			}
			f->donor[0] = f->ve - r->vs     < l_nt? ns_tab_nt_i2c[nt[f->ve - r->vs]]     : '.';
			f->donor[1] = f->ve - r->vs + 1 < l_nt? ns_tab_nt_i2c[nt[f->ve - r->vs + 1]] : '.';
			qs0 = f->qe, n_fs0 = e->n_fs, n_stop0 = e->n_stop, score0 = e->dp_max, blen0 = e->blen, n_iden0 = e->n_iden;
			acceptor0[0] = vs0 - r->vs >= 2? ns_tab_nt_i2c[nt[vs0 - r->vs - 2]] : '.';
			acceptor0[1] = vs0 - r->vs >= 1? ns_tab_nt_i2c[nt[vs0 - r->vs - 1]] : '.';
			// progress length
			nl += len, al += (op != NS_CIGAR_N);
		}
	}
	// update the last exon and possibly stop codon
	f = &r->feat[ft++];
	f->type = MP_FEAT_CDS;
	f->vs = vs0, f->ve = r->vs + nl, f->qs = qs0, f->qe = r->qs + al, f->phase = phase0;
	f->blen = e->blen - blen0, f->n_iden = e->n_iden - n_iden0, f->n_fs = e->n_fs - n_fs0, f->n_stop = e->n_stop - n_stop0, f->score = e->dp_max - score0;
	if (ft > 1) f->acceptor[0] = acceptor0[0], f->acceptor[1] = acceptor0[1];
	if (has_stop) {
		int64_t ve_mRNA = has_stop? r->ve + 3 : r->ve;
		f = &r->feat[ft++];
		f->type = MP_FEAT_STOP;
		f->vs = ve_mRNA - 3, f->ve = ve_mRNA, f->qs = f->qe = r->qe + al, f->phase = 0, f->n_fs = 0;
		f->blen = 3, f->n_iden = 0;
	}
	// check errors: the CIGAR span can legitimately fail to cover the full region (rare DP
	// end-extension edge cases even without --spsc); drop the extra features instead of
	// aborting, matching how the same mismatch is already tolerated in --spsc mode
	if (nl != r->ve - r->vs || al != r->qe - r->qs) {
		if (mp_verbose >= 2)
			fprintf(stderr, "Warning: cigar/region length mismatch (has_spsc=%d, %d!=%d or %d!=%d)\n",
					has_spsc, nl, (int)(r->ve - r->vs), al, r->qe - r->qs);
		free(r->p); free(r->feat);
		r->p = 0, r->feat = 0;
	}
}

static void mp_extra_gen(void *km, mp_reg1_t *r, mp_cigar_t *cigar, int32_t score)
{
	int32_t cap;
	cap = cigar->n + sizeof(mp_extra_t) / 4;
	r->p = Kcalloc(0, mp_extra_t, cap); // allocate globally, not from km
	r->p->dp_score = score;
	r->p->profile_score = INT32_MIN;
	r->p->n_cigar = r->p->m_cigar = cigar->n;
	memcpy(r->p->cigar, cigar->c, cigar->n * sizeof(uint32_t));
	kfree(km, cigar->c);
}

static int32_t mp_extra_stop(const mp_reg1_t *r, const uint8_t *nt, int64_t as, int64_t ae)
{
	int64_t j;
	for (j = r->ve; j + 2 < ae; j += 3) {
		int32_t i = j - as;
		uint8_t codon = nt[i]<<4 | nt[i+1]<<2 | nt[i+2];
		uint8_t aa = nt[i] > 3 || nt[i+1] > 3 || nt[i+2] > 3? ns_tab_aa20['X'] : ns_tab_codon[codon];
		if (aa == 20) return j - r->ve;
	}
	return -1;
}

static int32_t mp_extra_start(const mp_reg1_t *r, const uint8_t *nt, int64_t as, int64_t ae)
{
	int64_t j;
	for (j = r->vs; j >= as && j + 2 < ae; j -= 3) {
		int32_t i = j - as;
		uint8_t codon = nt[i]<<4 | nt[i+1]<<2 | nt[i+2];
		uint8_t aa = nt[i] > 3 || nt[i+1] > 3 || nt[i+2] > 3? ns_tab_aa20['X'] : ns_tab_codon[codon];
		if (aa == 20) break;
		if (aa == 12) return r->vs - j;
	}
	return -1;
}

// nt-side end-extension cap: fixed opt->acc_end_ext, or (with MP_ACC_END_DYN_WIN) scaled with query protein length
static inline int32_t mp_acc_end_ext_cap(const mp_mapopt_t *opt, int32_t len)
{
	int64_t x;
	if (!(opt->acc_flags & MP_ACC_END_DYN_WIN)) return opt->acc_end_ext;
	x = (int64_t)(len * opt->acc_end_nt_per_aa + .499f);
	if (x < opt->acc_end_ext_min) x = opt->acc_end_ext_min;
	if (x > opt->acc_end_ext) x = opt->acc_end_ext;
	return (int32_t)x;
}

// aa-side end-extension cap: fixed opt->acc_end_aa_ext, or (with MP_ACC_END_DYN_WIN) scaled with query protein length
static inline int32_t mp_acc_end_aa_cap(const mp_mapopt_t *opt, int32_t len)
{
	int64_t x;
	if (!(opt->acc_flags & MP_ACC_END_DYN_WIN)) return opt->acc_end_aa_ext;
	x = (int64_t)(len * opt->acc_end_aa_ratio + .499f);
	if (x < opt->acc_end_aa_min) x = opt->acc_end_aa_min;
	if (x > opt->acc_end_aa_ext) x = opt->acc_end_aa_ext;
	return (int32_t)x;
}

/****************************************************************************
 * Alignment of one candidate, split into stages so that the end-extension  *
 * DPs of every candidate of a query can be computed together (one GPU      *
 * launch per stage, see ns_ext_batch_*). Without a batch the stages run    *
 * back to back on the CPU and the result is identical.                     *
 ****************************************************************************/

enum { MP_AS_START = 0, MP_AS_LEFT, MP_AS_LEFT_RETRY, MP_AS_GAP, MP_AS_RIGHT, MP_AS_RIGHT_RETRY, MP_AS_RIGHT_APPLY, MP_AS_FINISH, MP_AS_DONE };

typedef struct {
	void *km;
	const mp_mapopt_t *opt;
	const mp_idx_t *mi;
	int32_t len;
	const char *aa;
	mp_reg1_t *r;
	int32_t extl0, extr0, force_full;
	mp_align_prof_t *aprof;
	mp_hmm_batch_t *hb;
	int32_t stage, ret;
	int32_t i0, ne0, ae0, score, rescued, full_dp, no_end_ext_retry;
	int64_t as, ae, vs0, l_nt;
	uint8_t *nt, *ss;
	ns_opt_t ns_opt0, ns_opt_ext;
	const int8_t *pssm; // HMM mode: row of query position 0; NULL in FASTA mode
	mp_cigar_t cigar;
	int32_t req; // pending ns_ext_batch request; -1 when the last extension ran synchronously
	ns_rst_t rst;
	int32_t nt_len, aa_len;
	int64_t vs1;
	int32_t as1, as1_win, aa_off, aa_win;
	uint64_t ext_cells;
} mp_align_ctx_t;

static void mp_align_ctx_init(mp_align_ctx_t *c, void *km, const mp_mapopt_t *opt, const mp_idx_t *mi, int32_t len, const char *aa, mp_reg1_t *r, int32_t extl0, int32_t extr0, int32_t force_full, mp_align_prof_t *aprof, mp_hmm_batch_t *hb)
{
	memset(c, 0, sizeof(*c));
	c->km = km, c->opt = opt, c->mi = mi, c->len = len, c->aa = aa, c->r = r;
	c->extl0 = extl0, c->extr0 = extr0, c->force_full = force_full, c->aprof = aprof, c->hb = hb;
	c->stage = MP_AS_START, c->req = -1;
}

static inline const int8_t *mp_pssm_at(const mp_align_ctx_t *c, int32_t qoff)
{
	return c->pssm? c->pssm + (size_t)qoff * c->opt->asize : 0;
}

// issue an extension DP: queued in eb (returns 1) or run on the CPU right away (returns 0, result in c->rst)
static int mp_align_ext_issue(mp_align_ctx_t *c, ns_ext_batch_t *eb, const uint8_t *ns, int32_t nl, const char *as, int32_t al, const uint8_t *ss, int32_t qoff)
{
	c->ext_cells = (uint64_t)nl * al;
	c->ns_opt_ext.pssm = mp_pssm_at(c, qoff);
	if (eb) {
		c->req = ns_ext_batch_add(eb, (const char*)ns, nl, as, al, &c->ns_opt_ext, ss);
		return 1;
	} else {
		double t0 = c->aprof? mp_realtime() : 0.0;
		ns_rst_init(&c->rst);
		ns_global_gs16b(c->km, (const char*)ns, nl, as, al, &c->ns_opt_ext, ss, &c->rst);
		if (c->aprof) c->aprof->t_ext += mp_realtime() - t0;
		return 0;
	}
}

static void mp_align_ext_take(mp_align_ctx_t *c, ns_ext_batch_t *eb, int is_retry)
{
	if (c->req >= 0) c->rst = *ns_ext_batch_get(eb, c->req), c->req = -1;
	if (c->aprof) {
		int32_t unguarded = !c->rescued && !c->force_full;
		if (is_retry) ++c->aprof->n_ext_retry;
		else ++c->aprof->n_ext_calls;
		c->aprof->n_ext_cells += c->ext_cells;
		if (unguarded) {
			if (!is_retry) ++c->aprof->n_ext_calls_unguarded;
			c->aprof->n_ext_cells_unguarded += c->ext_cells;
		}
	}
}

// everything before the left extension; returns 0 when the candidate is dropped
static int mp_align_start(mp_align_ctx_t *c)
{
	const mp_mapopt_t *opt = c->opt;
	const mp_idx_t *mi = c->mi;
	mp_reg1_t *r = c->r;
	int32_t i, extl, extr, len = c->len;
	int64_t ctg_len;

	c->full_dp = c->force_full || !!(mp_dbg_flag & MP_DBG_MORE_DP);
	assert(r->cnt > 0);
	if (!c->full_dp) {
		mp_filter_seed(r->cnt, r->a, 6, 3, opt->kmer2, opt->kmer2 + 1);
		for (i = 0; i < r->cnt; ++i)
			if (r->a[i]>>31&1) break;
		if (i == r->cnt) {
			if ((opt->flag & MP_F_ACCURACY) && (opt->acc_flags & MP_ACC_ANCHOR_RESCUE) && opt->acc_rescue_cells > 0)
				c->full_dp = 1, c->rescued = 1, c->i0 = 0;
			else {
				r->cnt = 0;
				return 0;
			}
		} else {
			c->i0 = i;
		}
	}
	// only skip the io_end 2nd-try for no-trusted-anchor (rescued) candidates; keep it for
	// genuine trusted-anchor and force_full/junc-refine candidates, which the data shows can
	// depend on it for correctness
	c->no_end_ext_retry = c->rescued && (opt->flag & MP_F_ACCURACY) && !(opt->acc_flags & MP_ACC_END_EXT_RETRY);

	extl = extr = opt->max_ext;
	if (r->qs >= 10) extl = opt->max_intron/2;
	if (len - r->qe >= 10) extr = opt->max_intron/2;
	if (c->rescued && (opt->flag & MP_F_ACCURACY) && (opt->acc_flags & MP_ACC_END_EXT_CAP)) {
		int32_t end_ext_cap = mp_acc_end_ext_cap(opt, len);
		if (end_ext_cap < extl) extl = end_ext_cap;
		if (end_ext_cap < extr) extr = end_ext_cap;
	}
	if (c->extl0 > 0) extl = extl < c->extl0? extl : c->extl0;
	if (c->extr0 > 0) extr = extr < c->extr0? extr : c->extr0;
	ctg_len = mi->nt->ctg[r->vid>>1].len;
	c->as = r->vs > extl? r->vs - extl : 0;
	c->ae = r->ve + extr < ctg_len? r->ve + extr : ctg_len;
	if ((c->rescued || c->force_full) && (c->ae - c->as) * (int64_t)len > opt->acc_rescue_cells) {
		r->cnt = 0;
		return 0;
	}
	c->nt = Kmalloc(c->km, uint8_t, c->ae - c->as);
	c->l_nt = mp_ntseq_get_by_v(mi->nt, r->vid, c->as, c->ae, c->nt);
	assert(c->l_nt == c->ae - c->as);
	if (mi->nt->spsc) {
		int32_t l_ss;
		c->ss = Kmalloc(c->km, uint8_t, c->ae - c->as);
		l_ss = mp_ntseq_spsc_get_by_v(mi->nt, r->vid, c->as, c->ae, c->ss);
		assert(l_ss == c->l_nt);
	}
	c->vs0 = r->vs;
	mp_map2ns_opt(opt, &c->ns_opt0);
	// HMM mode: the query is the profile consensus, so PSSM rows line up with query positions
	c->pssm = opt->hmm && opt->asize == 22 && mp_hmm_length(opt->hmm) == len? mp_hmm_pssm(opt->hmm) : 0;
	return 1;
}

// full-window DP or gap patching between the trusted anchors; sets r->ve/r->qe
static void mp_align_gap(mp_align_ctx_t *c)
{
	void *km = c->km;
	const mp_mapopt_t *opt = c->opt;
	mp_reg1_t *r = c->r;
	mp_align_prof_t *aprof = c->aprof;
	const uint8_t *nt = c->nt, *ss = c->ss;
	const char *aa = c->aa;
	int64_t as = c->as, vs0 = c->vs0;
	int32_t i, i0 = c->i0, ne0 = c->ne0, ae0 = c->ae0;
	double t0 = 0.0;

	if (c->full_dp) {
		if (aprof) t0 = mp_realtime();
		c->score = mp_align_seq(km, opt, &c->ns_opt0, r->ve - r->vs, &nt[r->vs - as], r->qe - ae0,
				&aa[ae0], ss? &ss[r->vs - as] : 0, mp_pssm_at(c, ae0), 0, &c->cigar);
		if (aprof) aprof->t_full += mp_realtime() - t0, ++aprof->n_full_calls, aprof->n_full_cells += (uint64_t)(r->ve - r->vs) * (r->qe - ae0);
	} else { // patch gaps between anchors
		if (aprof) t0 = mp_realtime();
		mp_align_task_t *tasks = 0;
		int32_t n_tasks = 0;
		if ((c->ns_opt0.flag & NS_F_GPU) && !c->pssm) // the GPU gap kernel scores with the 22x22 matrix only
			tasks = Kmalloc(km, mp_align_task_t, r->cnt - i0);
		if (tasks) {
			for (i = i0; i < r->cnt; ++i) {
				mp_align_task_t *task;
				int32_t ne1, ae1;
				if (!(r->a[i]>>31&1)) continue;
				ne1 = (r->a[i]>>32) + 1, ae1 = ((int32_t)r->a[i]<<1>>1) + 1;
				task = &tasks[n_tasks++];
				task->nlen = ne1 - ne0, task->nseq = &nt[ne0 + vs0 - as];
				task->alen = ae1 - ae0, task->aseq = &aa[ae0];
				task->ss = ss? &ss[ne0 + vs0 - as] : 0;
				task->gpu_request = mp_align_seq_submit(km, opt, &c->ns_opt0, task->nlen,
						task->nseq, task->alen, task->aseq, task->ss);
				i0 = i, ne0 = ne1, ae0 = ae1;
			}
			for (i = 0; i < n_tasks; ++i) {
				mp_align_task_t *task = &tasks[i];
				c->score += mp_align_seq(km, opt, &c->ns_opt0, task->nlen, task->nseq, task->alen,
						task->aseq, task->ss, 0, task->gpu_request, &c->cigar);
			}
			kfree(km, tasks);
		} else {
			for (i = i0; i < r->cnt; ++i) {
				int32_t ne1, ae1;
				if (!(r->a[i]>>31&1)) continue;
				ne1 = (r->a[i]>>32) + 1, ae1 = ((int32_t)r->a[i]<<1>>1) + 1;
				c->score += mp_align_seq(km, opt, &c->ns_opt0, ne1 - ne0, &nt[ne0 + vs0 - as], ae1 - ae0,
						&aa[ae0], ss? &ss[ne0 + vs0 - as] : 0, mp_pssm_at(c, ae0), 0, &c->cigar);
				if (aprof) ++aprof->n_gap_calls, aprof->n_cells += (uint64_t)(ne1 - ne0) * (ae1 - ae0);
				i0 = i, ne0 = ne1, ae0 = ae1;
			}
		}
		if (aprof) aprof->t_gap += mp_realtime() - t0;
		r->ve = ne0 + vs0, r->qe = ae0;
	}
	c->i0 = i0, c->ne0 = ne0, c->ae0 = ae0;
}

// MP_ACC_TRIM_INTRON: a real gene stitched by the chain/extension to a far-away spurious exon
// would otherwise be dropped whole by mp_reject_spurious_intron() (the dominant loss of strong
// homologs on yeast). Same junction test as that filter; cut the CIGAR at every bad junction, keep
// the exon block with the largest DP contribution and rebuild p/feat. Returns 1 if trimmed.
static int mp_align_trim_spurious(mp_align_ctx_t *c)
{
	const mp_mapopt_t *opt = c->opt;
	mp_reg1_t *r = c->r;
	mp_extra_t *e = r->p;
	int32_t j, k, n_cds, n_bad = 0, b_st, best_st = -1, best_en = -1, best_sc = INT32_MIN, best_blen = 0;
	int32_t nl, al, ex, k_st, k_en, nl_st, al_st, nl_en, al_en, full_dp, score;
	int64_t exon_nt = 0, left_nt, vs0 = r->vs;
	int32_t qs0 = r->qs;
	mp_cigar_t cig = {0,0,0};

	if (!(opt->flag & MP_F_ACCURACY) || !(opt->acc_flags & MP_ACC_STRICT_MULTI_EXON) || !(opt->acc_flags & MP_ACC_TRIM_INTRON)) return 0;
	if (e == 0 || r->feat == 0 || r->n_exon < 2) return 0;
	n_cds = r->n_exon; // feat[0..n_exon) are the CDS features in order; an optional STOP feature follows
	for (j = 0; j < n_cds; ++j) exon_nt += r->feat[j].ve - r->feat[j].vs;
	for (j = 0, b_st = 0, left_nt = 0; j < n_cds; ++j) {
		int32_t bad = 0;
		left_nt += r->feat[j].ve - r->feat[j].vs;
		if (j + 1 < n_cds) {
			const mp_feat_t *L = &r->feat[j], *R = &r->feat[j + 1];
			int64_t gap = R->vs - L->ve, local_nt = (L->ve - L->vs) + (R->ve - R->vs);
			int64_t side_nt = left_nt < exon_nt - left_nt? left_nt : exon_nt - left_nt; // exonic nt on the smaller side of the junction
			if (opt->acc_intron_abs_max > 0 && gap > opt->acc_intron_abs_max) bad = 1;
			else if (gap >= opt->acc_intron_evidence_min &&
					 ((double)gap > (double)local_nt * opt->acc_intron_evidence_ratio || (double)gap > (double)exon_nt * opt->acc_intron_evidence_ratio))
				bad = 1;
			// a few aa hopped to across a long intron: the small side alone must justify the intron (yeast purity loss #1)
			else if (opt->acc_trim_side_ratio > 0.0f && (double)gap > (double)side_nt * opt->acc_trim_side_ratio)
				bad = 1;
		}
		if (bad || j + 1 == n_cds) { // exon block [b_st, j] ends here
			int32_t i, sc = 0, bl = 0;
			for (i = b_st; i <= j; ++i) sc += r->feat[i].score, bl += r->feat[i].blen;
			if (sc > best_sc || (sc == best_sc && bl > best_blen))
				best_sc = sc, best_blen = bl, best_st = b_st, best_en = j;
			b_st = j + 1, n_bad += bad;
		}
	}
	if (n_bad == 0) return 0;
	// CIGAR range [k_st, k_en) and nt/aa offsets of exons best_st..best_en; a phase-1/2 intron (U/V)
	// spans one aa, which is dropped together with the junction so both cuts land on codon boundaries
	nl = al = 0, ex = 0;
	k_st = 0, nl_st = al_st = 0;
	k_en = e->n_cigar, nl_en = al_en = -1;
	for (k = 0; k < e->n_cigar; ++k) {
		int32_t op = e->cigar[k]&0xf, len = e->cigar[k]>>4;
		int32_t is_intron = (op == NS_CIGAR_N || op == NS_CIGAR_U || op == NS_CIGAR_V);
		if (is_intron && ex == best_en) { k_en = k, nl_en = nl, al_en = al; break; }
		if (op == NS_CIGAR_M) nl += len * 3, al += len;
		else if (op == NS_CIGAR_I) al += len;
		else if (op == NS_CIGAR_D) nl += len * 3;
		else if (op == NS_CIGAR_F) nl += len;
		else if (op == NS_CIGAR_G) nl += len, ++al;
		else if (is_intron) {
			nl += len, al += (op != NS_CIGAR_N), ++ex;
			if (ex == best_st) k_st = k + 1, nl_st = nl, al_st = al;
		}
	}
	if (nl_en < 0) nl_en = nl, al_en = al;
	if (k_en <= k_st || nl_en <= nl_st || al_en <= al_st) return 0; // degenerate block: leave it to the filter
	if (best_st == 0 && best_en == n_cds - 1) return 0;
	full_dp = e->full_dp;
	score = e->dp_score - (e->dp_max - best_sc); // remove the dropped exons' contribution; the dropped junction penalties stay (conservative)
	cig.n = cig.m = k_en - k_st;
	cig.c = Kmalloc(c->km, uint32_t, cig.n);
	memcpy(cig.c, &e->cigar[k_st], cig.n * sizeof(uint32_t));
	free(r->p); free(r->feat);
	r->p = 0, r->feat = 0;
	r->vs = vs0 + nl_st, r->ve = vs0 + nl_en;
	r->qs = qs0 + al_st, r->qe = qs0 + al_en;
	mp_extra_gen(c->km, r, &cig, score);
	r->p->full_dp = full_dp;
	r->p->dist_stop  = mp_extra_stop(r, c->nt, c->as, c->ae);
	r->p->dist_start = mp_extra_start(r, c->nt, c->as, c->ae);
	mp_extra_cal(r, opt, &c->nt[r->vs - c->as], c->l_nt - (r->vs - c->as), &c->aa[r->qs], c->len, c->mi->nt->spsc != 0, mp_pssm_at(c, r->qs));
	return 1;
}

static void mp_align_finish(mp_align_ctx_t *c)
{
	void *km = c->km;
	const mp_mapopt_t *opt = c->opt;
	mp_reg1_t *r = c->r;
	const uint8_t *nt = c->nt;
	int64_t as = c->as;

	//for (i = 0; i < cigar.n; ++i) printf("%d%c", cigar.c[i]>>4, NS_CIGAR_STR[cigar.c[i]&0xf]); putchar('\n');
	mp_extra_gen(km, r, &c->cigar, c->score);
	r->p->full_dp = c->full_dp;
	r->p->dist_stop  = mp_extra_stop(r, nt, as, c->ae);
	r->p->dist_start = mp_extra_start(r, nt, as, c->ae);
	mp_extra_cal(r, opt, &nt[r->vs - as], c->l_nt - (r->vs - as), &c->aa[r->qs], c->len, c->mi->nt->spsc != 0, mp_pssm_at(c, r->qs));
	mp_align_trim_spurious(c); // may rebuild r->p/r->feat and shrink r->vs/ve/qs/qe
	if (opt->hmm && r->p) {
		if (c->hb) { // scored later, together with the query's other regions, in one GPU launch
			mp_hmm_batch_add(c->hb, r, &nt[r->vs - as], (int32_t)(r->ve - r->vs));
		} else {
			r->p->profile_score = (opt->flag & MP_F_GPU)?
				mp_hmm_dp_score_gpu(opt->hmm, &nt[r->vs - as], (int32_t)(r->ve - r->vs)) : INT32_MIN;
			if (r->p->profile_score == INT32_MIN)
				r->p->profile_score = mp_hmm_dp_score(opt->hmm, &nt[r->vs - as], (int32_t)(r->ve - r->vs));
		}
	}
	mp_set_rank_score(opt, c->len, r);
	if (c->ss) kfree(km, c->ss);
	kfree(km, c->nt);
	c->ret = (c->rescued? MP_ALIGN_RESCUED : 0) | (c->full_dp? MP_ALIGN_FULL : 0);
}

// advance until the candidate is done or an extension DP is waiting in eb
static void mp_align_step(mp_align_ctx_t *c, ns_ext_batch_t *eb)
{
	const mp_mapopt_t *opt = c->opt;
	mp_reg1_t *r = c->r;
	for (;;) {
		switch (c->stage) {
		case MP_AS_START:
			if (!mp_align_start(c)) {
				c->ret = 0, c->stage = MP_AS_DONE;
				return;
			}
			c->vs1 = c->vs0 + (r->a[c->i0]>>32) + 1;
			c->as1 = ((int32_t)r->a[c->i0]<<1>>1) + 1;
			c->as1_win = c->as1, c->aa_off = 0;
			// experimental: cap the (otherwise uncapped) aa-side search extent for rescued candidates
			if (c->rescued && (opt->flag & MP_F_ACCURACY) && (opt->acc_flags & MP_ACC_END_AA_CAP)) {
				int32_t aa_cap = mp_acc_end_aa_cap(opt, c->len);
				if (c->as1 > aa_cap)
					c->aa_off = c->as1 - aa_cap, c->as1_win = aa_cap;
			}
			c->ns_opt_ext = c->ns_opt0;
			c->ns_opt_ext.flag |= NS_F_EXT_LEFT;
			c->stage = MP_AS_LEFT;
			if (mp_align_ext_issue(c, eb, c->nt, c->vs1 - c->as, c->aa + c->aa_off, c->as1_win, c->ss, c->aa_off)) return;
			break;
		case MP_AS_LEFT:
			mp_align_ext_take(c, eb, 0);
			c->nt_len = c->rst.nt_len, c->aa_len = c->rst.aa_len;
			c->stage = MP_AS_GAP;
			if (!c->no_end_ext_retry && c->rst.aa_len != c->as1_win && c->rst.nt_len < opt->max_ext && opt->io > opt->io_end) { // special casing 5'-end exon
				int64_t as_alt = c->vs1 - c->as > opt->acc_end_retry_win? c->vs1 - opt->acc_end_retry_win : c->as;
				c->ns_opt_ext.io = opt->io_end;
				c->stage = MP_AS_LEFT_RETRY;
				if (mp_align_ext_issue(c, eb, &c->nt[as_alt - c->as], c->vs1 - as_alt, c->aa + c->aa_off, c->as1_win, c->ss? &c->ss[as_alt - c->as] : 0, c->aa_off)) return;
			}
			break;
		case MP_AS_LEFT_RETRY:
			mp_align_ext_take(c, eb, 1);
			if (c->rst.aa_len == c->as1_win)
				c->nt_len = c->rst.nt_len, c->aa_len = c->rst.aa_len;
			c->stage = MP_AS_GAP;
			break;
		case MP_AS_GAP:
			r->vs = c->vs1 - c->nt_len;
			r->qs = c->as1 - c->aa_len;
			c->ne0 = r->vs - c->vs0;
			c->ae0 = r->qs;
			mp_align_gap(c);
			c->stage = MP_AS_FINISH;
			if (r->qe < c->len && r->ve < c->ae) { // right extension
				c->aa_win = c->len - r->qe;
				if (c->rescued && (opt->flag & MP_F_ACCURACY) && (opt->acc_flags & MP_ACC_END_AA_CAP)) {
					int32_t aa_cap = mp_acc_end_aa_cap(opt, c->len);
					if (c->aa_win > aa_cap) c->aa_win = aa_cap;
				}
				c->ns_opt_ext = c->ns_opt0;
				c->ns_opt_ext.flag |= NS_F_EXT_RIGHT;
				c->stage = MP_AS_RIGHT;
				if (mp_align_ext_issue(c, eb, &c->nt[r->ve - c->as], c->ae - r->ve, &c->aa[r->qe], c->aa_win, c->ss? &c->ss[r->ve - c->as] : 0, r->qe)) return;
			}
			break;
		case MP_AS_RIGHT:
			mp_align_ext_take(c, eb, 0);
			c->nt_len = c->rst.nt_len, c->aa_len = c->rst.aa_len;
			c->stage = MP_AS_RIGHT_APPLY;
			if (!c->no_end_ext_retry && c->aa_len < c->aa_win && c->nt_len < opt->max_ext && opt->io > opt->io_end) { // special casing 3'-end exon
				int32_t l_ext = c->ae - r->ve < opt->acc_end_retry_win? c->ae - r->ve : opt->acc_end_retry_win;
				c->ns_opt_ext.io = opt->io_end;
				c->stage = MP_AS_RIGHT_RETRY;
				if (mp_align_ext_issue(c, eb, &c->nt[r->ve - c->as], l_ext, &c->aa[r->qe], c->aa_win, c->ss? &c->ss[r->ve - c->as] : 0, r->qe)) return;
			}
			break;
		case MP_AS_RIGHT_RETRY:
			mp_align_ext_take(c, eb, 1);
			if (c->rst.aa_len == c->aa_win)
				c->nt_len = c->rst.nt_len, c->aa_len = c->rst.aa_len;
			c->stage = MP_AS_RIGHT_APPLY;
			break;
		case MP_AS_RIGHT_APPLY: {
			double t0 = c->aprof? mp_realtime() : 0.0;
			c->score += mp_align_seq(c->km, opt, &c->ns_opt0, c->nt_len, &c->nt[r->ve - c->as], c->aa_len, &c->aa[r->qe],
					c->ss? &c->ss[r->ve - c->as] : 0, mp_pssm_at(c, r->qe), 0, &c->cigar);
			r->ve += c->nt_len, r->qe += c->aa_len;
			if (c->aprof) c->aprof->t_ext += mp_realtime() - t0;
			c->stage = MP_AS_FINISH;
			break;
		}
		case MP_AS_FINISH:
			mp_align_finish(c);
			c->stage = MP_AS_DONE;
			return;
		default:
			return;
		}
	}
}

void mp_align_batch(void *km, const mp_mapopt_t *opt, const mp_idx_t *mi, int32_t len, const char *aa, int32_t n, mp_reg1_t **r, const uint64_t *ext, int32_t force_full, mp_align_prof_t *aprof, int32_t *status, mp_hmm_batch_t *hb)
{
	mp_align_ctx_t *ctx;
	ns_ext_batch_t *eb = 0;
	int32_t i, pending;

	if (n <= 0) return;
	ctx = Kmalloc(km, mp_align_ctx_t, n);
	if (opt->flag & MP_F_GPU) eb = ns_ext_batch_init(km);
	for (i = 0; i < n; ++i)
		mp_align_ctx_init(&ctx[i], km, opt, mi, len, aa, r[i], ext? (int32_t)(ext[i]>>32) : 0, ext? (int32_t)ext[i] : 0, force_full, aprof, hb);
	// candidates whose extension is still on the device are skipped; new requests are flushed without waiting,
	// so several flushes are in flight and the device work of one candidate overlaps the CPU work of the others
	for (;;) {
		double t0;
		int32_t progressed = 0;
		for (i = 0, pending = 0; i < n; ++i) {
			if (ctx[i].stage == MP_AS_DONE) continue;
			if (ctx[i].req >= 0 && !ns_ext_batch_ready(eb, ctx[i].req)) { ++pending; continue; }
			mp_align_step(&ctx[i], eb);
			progressed = 1;
			pending += (ctx[i].stage != MP_AS_DONE);
		}
		if (!pending) break;
		t0 = aprof? mp_realtime() : 0.0;
		if (ns_ext_batch_flush(eb) == 0 && !progressed) ns_ext_batch_wait(eb);
		if (aprof) aprof->t_ext += mp_realtime() - t0;
	}
	if (status)
		for (i = 0; i < n; ++i) status[i] = ctx[i].ret;
	ns_ext_batch_destroy(eb);
	kfree(km, ctx);
}

int32_t mp_align(void *km, const mp_mapopt_t *opt, const mp_idx_t *mi, int32_t len, const char *aa, mp_reg1_t *r, int32_t extl0, int32_t extr0, int32_t force_full, mp_align_prof_t *aprof, mp_hmm_batch_t *hb)
{
	uint64_t ext = (uint64_t)(uint32_t)extl0 << 32 | (uint32_t)extr0;
	int32_t status = 0;
	mp_align_batch(km, opt, mi, len, aa, 1, &r, &ext, force_full, aprof, &status, hb);
	return status;
}
