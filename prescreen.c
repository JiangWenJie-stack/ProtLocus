/* Built-in profile prescreen: an MMseqs2-style profile-vs-ORF search implemented natively so HMM-mode whole-library
 * scans no longer need `mmseqs search` + mmseqs2cand.py to produce the --cand-table.
 *
 *   genome --(six frames, stop-free ORFs >= min_orf aa)--> ORF DB --(spaced k-mers, 20-letter alphabet)--> CSR index
 *   profile PSSM --(similar k-mers per window, score >= thr)--> index lookup --(>=2 hits on one (ORF,diagonal))-->
 *   ungapped diagonal score (>= min_ungapped, best per ORF, top max_seqs) --> banded gapped local alignment -->
 *   Karlin-Altschul E-value (ALP BLOSUM62 11/1 parameters, as MMseqs2) --> loci (whole ORF span) with E <= tier cutoff
 *
 * Scores are half-bits throughout (the PSSM from hmm.c is half-bit log-odds), so MMseqs2's 1/8-bit k-mer thresholds
 * are divided by 4 and its alignment-stage constants (gap 11/1, lambda/K) apply unchanged. */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "mppriv.h"
#include "kalloc.h"
#include "kthread.h"
#include "kvec-km.h"

#define PS_AA 20
#define PS_DIAG_OFF (1U<<30)

typedef struct {
	uint32_t vid;
	int32_t aa_len;
	int64_t vs; // oriented nt start; ve = vs + 3*aa_len
	int64_t aa_off; // into mp_prescreen_t::aa
} mp_orf_t;

struct mp_prescreen_s {
	const mp_ntdb_t *nt;
	int32_t kmer, span;
	int8_t pat[8]; // offsets of the informative positions inside the spaced window
	int64_t n_orf;
	mp_orf_t *orf;
	int64_t n_aa;
	uint8_t *aa; // concatenated ORF residues, ns_tab_aa20 index 0..19
	int64_t n_bucket;
	int64_t *off; // CSR bucket boundaries, n_bucket+1
	int64_t n_ent;
	uint64_t *ent; // orf_id<<32 | position of the window start in the ORF
};

/* mmseqs Sequence.h spaced seeds */
static const int8_t ps_pat5[] = {1,1,0,0,1,0,0,0,0,1,0,1};
static const int8_t ps_pat6[] = {1,1,0,1,0,1,0,0,1,1};
static const int8_t ps_pat7[] = {1,1,0,1,0,1,1,0,0,1,1};

void mp_psopt_init(mp_psopt_t *po)
{
	memset(po, 0, sizeof(*po));
	po->kmer = 6;
	po->min_orf = 30;
	po->sens = 7.5f;
	po->kmer_thr = 0;
	po->kmer_max = 65536;
	po->min_ungapped = 15;
	po->max_seqs = 4000;
	po->band = 32;
	po->go = 11, po->ge = 1;
}

// mmseqs Prefiltering::getKmerThreshold() for profile queries without context pseudo-counts, in 1/8 bits -> half-bits
int32_t mp_ps_kmer_thr(const mp_psopt_t *po)
{
	float t;
	if (po->kmer_thr > 0) return po->kmer_thr;
	if (po->kmer == 5) t = 108.8f - po->sens * 4.7f;
	else if (po->kmer == 7) t = 149.15f - po->sens * 6.85f;
	else t = 134.35f - po->sens * 6.15f;
	return (int32_t)floorf((float)(int32_t)t / 4.0f + 0.5f);
}

/**********************
 * ORF DB + k-mer index
 **********************/

typedef struct { int64_t n, m; mp_orf_t *a; } ps_orf_v;
typedef struct { int64_t n, m; uint8_t *a; } ps_aa_v;

typedef struct {
	const mp_ntdb_t *nt;
	int32_t min_orf;
	ps_orf_v *orf;
	ps_aa_v *aa;
} ps_orf_aux_t;

static void ps_add_orf(ps_orf_v *ov, ps_aa_v *av, const uint8_t *seq, uint32_t vid, int64_t st, int32_t n_aa)
{
	int32_t t;
	mp_orf_t o;
	o.vid = vid, o.vs = st, o.aa_len = n_aa, o.aa_off = av->n;
	kv_resize(uint8_t, 0, *av, av->n + n_aa);
	for (t = 0; t < n_aa; ++t) {
		const uint8_t *c = seq + st + 3 * t;
		av->a[av->n++] = ns_tab_codon[c[0]<<4 | c[1]<<2 | c[2]];
	}
	kv_push(mp_orf_t, 0, *ov, o);
}

// same frame walk as mp_sketch_nt4(): stop-free, N-free segments of >= min_orf codons in each of the 3 frames
static void ps_orf_worker(void *data, long j, int tid)
{
	ps_orf_aux_t *aux = (ps_orf_aux_t*)data;
	const mp_ntdb_t *nt = aux->nt;
	uint8_t *seq, codon, p, q;
	int64_t i, l, len, e[3], k[3];
	(void)tid;
	seq = (uint8_t*)malloc(nt->ctg[j>>1].len);
	len = mp_ntseq_get(nt, j>>1, 0, -1, j&1, seq);
	for (p = 0; p < 3; ++p) e[p] = -1, k[p] = 0;
	for (i = 0, p = 1, codon = 0, l = 0; i < len; ++i, ++p) {
		if (p == 3) p = 0;
		if (seq[i] < 4) {
			codon = (codon << 2 | seq[i]) & 0x3f;
			if (++l >= 3) {
				if (ns_tab_codon[codon] >= PS_AA) {
					if (k[p] >= aux->min_orf) ps_add_orf(&aux->orf[j], &aux->aa[j], seq, (uint32_t)j, e[p] + 1 - k[p] * 3, (int32_t)k[p]);
					k[p] = 0, e[p] = -1;
				} else e[p] = i, ++k[p];
			}
		} else {
			for (q = 0; q < 3; ++q) {
				if (k[q] >= aux->min_orf) ps_add_orf(&aux->orf[j], &aux->aa[j], seq, (uint32_t)j, e[q] + 1 - k[q] * 3, (int32_t)k[q]);
				k[q] = 0, e[q] = -1;
			}
			l = 0, codon = 0;
		}
	}
	for (p = 0; p < 3; ++p)
		if (k[p] >= aux->min_orf) ps_add_orf(&aux->orf[j], &aux->aa[j], seq, (uint32_t)j, e[p] + 1 - k[p] * 3, (int32_t)k[p]);
	free(seq);
}

static inline uint32_t ps_code(const mp_prescreen_t *ps, const uint8_t *s)
{
	uint32_t c = 0, mul = 1;
	int32_t t;
	for (t = 0; t < ps->kmer; ++t, mul *= PS_AA) c += s[ps->pat[t]] * mul;
	return c;
}

typedef struct { mp_prescreen_t *ps; int32_t pass; int64_t chunk; } ps_idx_aux_t;

static void ps_idx_worker(void *data, long j, int tid)
{
	ps_idx_aux_t *aux = (ps_idx_aux_t*)data;
	mp_prescreen_t *ps = aux->ps;
	int64_t i, st = j * aux->chunk, en = st + aux->chunk < ps->n_orf? st + aux->chunk : ps->n_orf;
	(void)tid;
	for (i = st; i < en; ++i) {
		const mp_orf_t *o = &ps->orf[i];
		const uint8_t *s = ps->aa + o->aa_off;
		int32_t p;
		for (p = 0; p + ps->span <= o->aa_len; ++p) {
			uint32_t c = ps_code(ps, s + p);
			if (aux->pass == 0) __sync_fetch_and_add(&ps->off[c], 1);
			else ps->ent[__sync_sub_and_fetch(&ps->off[c], 1)] = (uint64_t)i << 32 | (uint32_t)p;
		}
	}
}

mp_prescreen_t *mp_prescreen_build(const mp_ntdb_t *nt, const mp_psopt_t *po, int32_t n_threads)
{
	mp_prescreen_t *ps;
	ps_orf_aux_t aux;
	ps_idx_aux_t iaux;
	const int8_t *pat;
	int32_t i, n_v = nt->n_ctg * 2, plen;
	int64_t j, c;
	double t0 = mp_realtime();

	ps = Kcalloc(0, mp_prescreen_t, 1);
	ps->nt = nt;
	ps->kmer = po->kmer < 5? 5 : po->kmer > 7? 7 : po->kmer;
	if (ps->kmer == 5) pat = ps_pat5, plen = sizeof(ps_pat5);
	else if (ps->kmer == 7) pat = ps_pat7, plen = sizeof(ps_pat7);
	else pat = ps_pat6, plen = sizeof(ps_pat6);
	ps->span = plen;
	for (i = 0, c = 0; i < plen; ++i)
		if (pat[i]) ps->pat[c++] = (int8_t)i;

	memset(&aux, 0, sizeof(aux));
	aux.nt = nt, aux.min_orf = po->min_orf > ps->span? po->min_orf : ps->span;
	aux.orf = Kcalloc(0, ps_orf_v, n_v);
	aux.aa = Kcalloc(0, ps_aa_v, n_v);
	kt_for(n_threads, ps_orf_worker, &aux, n_v);
	for (i = 0; i < n_v; ++i) ps->n_orf += aux.orf[i].n, ps->n_aa += aux.aa[i].n;
	ps->orf = Kmalloc(0, mp_orf_t, ps->n_orf + 1);
	ps->aa = Kmalloc(0, uint8_t, ps->n_aa + 1);
	for (i = 0, ps->n_orf = 0, ps->n_aa = 0; i < n_v; ++i) {
		for (j = 0; j < aux.orf[i].n; ++j) {
			mp_orf_t o = aux.orf[i].a[j];
			o.aa_off += ps->n_aa;
			ps->orf[ps->n_orf++] = o;
		}
		memcpy(ps->aa + ps->n_aa, aux.aa[i].a, aux.aa[i].n);
		ps->n_aa += aux.aa[i].n;
		free(aux.orf[i].a); free(aux.aa[i].a);
	}
	free(aux.orf); free(aux.aa);
	if (mp_verbose >= 3)
		fprintf(stderr, "[M::%s@%.3f*%.2f] %ld ORFs (>=%d aa), %ld residues\n", __func__, mp_realtime(), mp_percent_cpu(), (long)ps->n_orf, aux.min_orf, (long)ps->n_aa);

	for (ps->n_bucket = 1, i = 0; i < ps->kmer; ++i) ps->n_bucket *= PS_AA;
	ps->off = Kcalloc(0, int64_t, ps->n_bucket + 1);
	iaux.ps = ps, iaux.chunk = 4096;
	iaux.pass = 0;
	if (ps->n_orf > 0) kt_for(n_threads, ps_idx_worker, &iaux, (ps->n_orf + iaux.chunk - 1) / iaux.chunk);
	for (c = 0, ps->n_ent = 0; c < ps->n_bucket; ++c) // inclusive prefix sums: off[c] = end of bucket c
		ps->n_ent += ps->off[c], ps->off[c] = ps->n_ent;
	ps->off[ps->n_bucket] = ps->n_ent;
	ps->ent = Kmalloc(0, uint64_t, ps->n_ent + 1);
	iaux.pass = 1; // fills backwards; afterwards off[c] = start of bucket c
	if (ps->n_orf > 0) kt_for(n_threads, ps_idx_worker, &iaux, (ps->n_orf + iaux.chunk - 1) / iaux.chunk);
	if (mp_verbose >= 3)
		fprintf(stderr, "[M::%s@%.3f*%.2f] indexed %ld spaced %d-mers (span %d, thr %d half-bits) in %.1f s; index %.2f GB\n", __func__, mp_realtime(), mp_percent_cpu(),
				(long)ps->n_ent, ps->kmer, ps->span, mp_ps_kmer_thr(po), mp_realtime() - t0,
				((double)ps->n_ent * 8 + (double)(ps->n_bucket + 1) * 8 + (double)ps->n_aa + (double)ps->n_orf * sizeof(mp_orf_t)) / 1e9);
	return ps;
}

void mp_prescreen_destroy(mp_prescreen_t *ps)
{
	if (ps == 0) return;
	free(ps->orf); free(ps->aa); free(ps->off); free(ps->ent); free(ps);
}

/*****************
 * E-value (ALP)  *
 *****************/

// MMseqs2 EvalueComputation defaults for blosum62.out, gap open 11 / extend 1 (lib/alp), scores in half-bits
#define PS_LAMBDA 0.27359865037097330642
#define PS_K      0.044620920658722244834
#define PS_A      1.5938724404943873658
#define PS_B     (-19.959867650284412122)
#define PS_ALPHA  30.455610143099914211
#define PS_BETA  (-622.28684628915891608)
#define PS_SIGMA  29.602444874818868215
#define PS_TAU   (-601.81087985041381216)
#define PS_INV_SQRT_2PI 0.39894228040143267794

static inline double ps_norm_cdf(double x) { return 0.5 * erfc(-x * 0.70710678118654752440); }

// sls_pvalues::get_appr_tail_prob_with_cov_without_errors(): finite-size corrected search area, then E = K * area * exp(-lambda*S)
static double ps_evalue(double S, double m, double n)
{
	const double vthr = 2.0 * PS_ALPHA / PS_LAMBDA, cthr = 2.0 * PS_SIGMA / PS_LAMBDA;
	double l = PS_A * S + PS_B, v = PS_ALPHA * S + PS_BETA, sv, mF, nF, Pm, Pn, p1, p2, cy, area;
	if (v < vthr) v = vthr;
	sv = sqrt(v);
	mF = (m - l) / sv, nF = (n - l) / sv;
	Pm = ps_norm_cdf(mF), Pn = ps_norm_cdf(nF);
	p1 = (m - l) * Pm + sv * PS_INV_SQRT_2PI * exp(-0.5 * mF * mF);
	p2 = (n - l) * Pn + sv * PS_INV_SQRT_2PI * exp(-0.5 * nF * nF);
	cy = PS_SIGMA * S + PS_TAU;
	if (cy < cthr) cy = cthr;
	area = p1 * p2 + cy * Pm * Pn;
	if (area < 1e-300) area = 1e-300;
	return PS_K * area * exp(-PS_LAMBDA * S);
}

/********************
 * Per-query search  *
 ********************/

typedef struct { int64_t orf; int32_t diag, ung, sw, ung_qs, ung_qe; double ev; } ps_cand_t;

typedef struct {
	int32_t m_pos;
	int8_t *sc; // [L*20] per-position emission scores sorted descending
	uint8_t *ai; // [L*20] residue index parallel to sc
	int8_t *pmax; // [L]
	uint64_t *hit; int64_t n_hit, m_hit;
	ps_cand_t *cand; int32_t n_cand, m_cand;
	int32_t *H, *F; int64_t m_dp;
	// stats
	int64_t n_win, n_kmer, n_lookup, n_dbl, n_orf_scored, n_aligned, n_kept;
} ps_scratch_t;

static void ps_scratch_free(ps_scratch_t *s)
{
	free(s->sc); free(s->ai); free(s->pmax); free(s->hit); free(s->cand); free(s->H); free(s->F);
}

static inline void ps_push_hit(ps_scratch_t *s, uint64_t x)
{
	if (s->n_hit == s->m_hit) {
		s->m_hit = s->m_hit? s->m_hit << 1 : 1<<16;
		s->hit = (uint64_t*)realloc(s->hit, s->m_hit * sizeof(uint64_t));
	}
	s->hit[s->n_hit++] = x;
}

// generate all k-mers with sum of per-position scores >= thr over the window at query offset q0 (mmseqs KmerGenerator);
// pushes hits (orf<<32 | diag+PS_DIAG_OFF) straight from the index
static void ps_gen_hits(const mp_prescreen_t *ps, ps_scratch_t *s, int32_t q0, int32_t thr, int32_t cap)
{
	int32_t t, k = ps->kmer, rest[8], idx[8], sum[8], n_gen = 0;
	uint32_t code[8], mul[8];
	const int8_t *sc[8];
	const uint8_t *ai[8];
	for (t = 0; t < k; ++t) {
		int32_t q = q0 + ps->pat[t];
		sc[t] = s->sc + (int64_t)q * PS_AA, ai[t] = s->ai + (int64_t)q * PS_AA;
	}
	for (t = k - 1, rest[k-1] = 0; t > 0; --t) rest[t-1] = rest[t] + sc[t][0];
	for (t = 0, mul[0] = 1; t + 1 < k; ++t) mul[t+1] = mul[t] * PS_AA;
	t = 0, idx[0] = 0, sum[0] = 0, code[0] = 0;
	while (t >= 0) {
		if (idx[t] >= PS_AA || sum[t] + sc[t][idx[t]] + rest[t] < thr) { --t; if (t >= 0) ++idx[t]; continue; }
		if (t + 1 == k) { // leaf: a complete k-mer
			uint32_t c = code[t] + ai[t][idx[t]] * mul[t];
			int64_t e;
			for (e = ps->off[c]; e < ps->off[c+1]; ++e) {
				uint64_t x = ps->ent[e];
				int64_t d = (int64_t)(uint32_t)x - q0 + PS_DIAG_OFF;
				ps_push_hit(s, (x & 0xffffffff00000000ULL) | (uint32_t)d);
			}
			s->n_lookup += ps->off[c+1] - ps->off[c];
			if (++n_gen >= cap) break;
			++idx[t];
		} else {
			sum[t+1] = sum[t] + sc[t][idx[t]];
			code[t+1] = code[t] + ai[t][idx[t]] * mul[t];
			++t, idx[t] = 0;
		}
	}
	s->n_kmer += n_gen;
	++s->n_win;
}

// best ungapped segment on one diagonal; *qs/*qe receive its query interval
static int32_t ps_ungapped(const int8_t *pssm, int32_t L, const uint8_t *orf, int32_t n, int32_t diag, int32_t *qs, int32_t *qe)
{
	int32_t i, i0 = diag < 0? -diag : 0, i1 = n - diag < L? n - diag : L, sc = 0, best = 0, st = i0;
	*qs = *qe = i0;
	for (i = i0; i < i1; ++i) {
		sc += pssm[(int64_t)i * 22 + orf[i + diag]];
		if (sc < 0) sc = 0, st = i + 1;
		else if (sc > best) best = sc, *qs = st, *qe = i + 1;
	}
	return best;
}

// local affine-gap alignment of profile x ORF restricted to |t - i - diag| <= band (band <= 0: full); score only.
// H[t]/F[t] hold row i-1 until column t is processed; columns right of the band are still at their initial 0/-inf
// because the band only ever moves right, and column lo-1 was inside the previous row's band.
static int32_t ps_gapped(ps_scratch_t *s, const int8_t *pssm, int32_t L, const uint8_t *orf, int32_t n, int32_t diag, int32_t band, int32_t go, int32_t ge)
{
	int32_t i, t, best = 0;
	const int32_t NEG = -0x3fffffff;
	if (s->m_dp < n + 2) {
		s->m_dp = n + 2;
		kroundup64(s->m_dp);
		s->H = (int32_t*)realloc(s->H, s->m_dp * sizeof(int32_t));
		s->F = (int32_t*)realloc(s->F, s->m_dp * sizeof(int32_t));
	}
	for (t = 0; t <= n; ++t) s->H[t] = 0, s->F[t] = NEG;
	for (i = 0; i < L; ++i) {
		const int8_t *row = pssm + (int64_t)i * 22;
		int32_t lo = band > 0? i + diag - band : 0, hi = band > 0? i + diag + band : n - 1, e = NEG, h_diag;
		if (lo < 0) lo = 0;
		if (hi > n - 1) hi = n - 1;
		if (lo > hi) continue;
		h_diag = lo > 0? s->H[lo-1] : 0; // H[i-1][lo-1]
		for (t = lo; t <= hi; ++t) {
			int32_t h_up = s->H[t], h, f;
			h = h_diag + row[orf[t]];
			f = s->F[t] - ge > h_up - go? s->F[t] - ge : h_up - go; // vertical gap ending at (i,t)
			s->F[t] = f;
			if (f > h) h = f;
			if (e > h) h = e;
			if (h < 0) h = 0;
			if (h > best) best = h;
			e = e - ge > h - go? e - ge : h - go; // horizontal gap into (i,t+1)
			h_diag = h_up;
			s->H[t] = h;
		}
	}
	return best;
}

static int ps_cand_cmp_ung(const void *a, const void *b)
{
	const ps_cand_t *x = (const ps_cand_t*)a, *y = (const ps_cand_t*)b;
	return x->ung > y->ung? -1 : x->ung < y->ung? 1 : x->orf < y->orf? -1 : x->orf > y->orf;
}

typedef struct {
	const mp_prescreen_t *ps;
	const mp_psopt_t *po;
	double max_evalue;
	int32_t thr;
	mp_hmm_t **hmm;
	ps_scratch_t *scr;
	mp_cand1_t **res;
	int32_t *n_res;
} ps_batch_aux_t;

static void ps_query_worker(void *data, long qi, int tid)
{
	ps_batch_aux_t *aux = (ps_batch_aux_t*)data;
	const mp_prescreen_t *ps = aux->ps;
	const mp_psopt_t *po = aux->po;
	ps_scratch_t *s = &aux->scr[tid];
	const mp_hmm_t *h = aux->hmm[qi];
	const int8_t *pssm = mp_hmm_pssm(h);
	int32_t L = mp_hmm_length(h), i, q, n_out = 0, m_out = 0;
	int64_t k, st;
	mp_cand1_t *out = 0;

	aux->res[qi] = 0, aux->n_res[qi] = 0;
	if (pssm == 0 || L < ps->span) return;
	if (s->m_pos < L) {
		s->m_pos = L < 512? 1024 : L * 2;
		s->sc = (int8_t*)realloc(s->sc, (size_t)s->m_pos * PS_AA);
		s->ai = (uint8_t*)realloc(s->ai, (size_t)s->m_pos * PS_AA);
		s->pmax = (int8_t*)realloc(s->pmax, s->m_pos);
	}
	for (i = 0; i < L; ++i) { // rank the 20 residues of each match state by emission score (insertion sort, 20 items)
		const int8_t *row = pssm + (int64_t)i * 22;
		int8_t *sc = s->sc + (int64_t)i * PS_AA;
		uint8_t *ai = s->ai + (int64_t)i * PS_AA;
		int32_t a, b;
		for (a = 0; a < PS_AA; ++a) {
			int8_t v = row[a];
			for (b = a; b > 0 && sc[b-1] < v; --b) sc[b] = sc[b-1], ai[b] = ai[b-1];
			sc[b] = v, ai[b] = (uint8_t)a;
		}
		s->pmax[i] = sc[0];
	}
	s->n_hit = 0;
	for (q = 0; q + ps->span <= L; ++q)
		ps_gen_hits(ps, s, q, aux->thr, po->kmer_max);
	if (s->n_hit == 0) return;
	radix_sort_mp64(s->hit, s->hit + s->n_hit);

	// double-hit diagonals -> ungapped score -> best diagonal per ORF
	s->n_cand = 0;
	for (k = 0; k < s->n_hit; ) {
		int64_t orf = s->hit[k] >> 32;
		int32_t best = -1, best_diag = 0, best_qs = 0, best_qe = 0;
		while (k < s->n_hit && (int64_t)(s->hit[k] >> 32) == orf) {
			for (st = k; k < s->n_hit && s->hit[k] == s->hit[st]; ++k);
			if (k - st >= 2) {
				const mp_orf_t *o = &ps->orf[orf];
				int32_t diag = (int32_t)((uint32_t)s->hit[st] - PS_DIAG_OFF), ung, qs, qe;
				++s->n_dbl;
				ung = ps_ungapped(pssm, L, ps->aa + o->aa_off, o->aa_len, diag, &qs, &qe);
				if (ung > best) best = ung, best_diag = diag, best_qs = qs, best_qe = qe;
			}
		}
		if (best >= po->min_ungapped) {
			ps_cand_t *c;
			++s->n_orf_scored;
			if (s->n_cand == s->m_cand) {
				s->m_cand = s->m_cand? s->m_cand << 1 : 1024;
				s->cand = (ps_cand_t*)realloc(s->cand, s->m_cand * sizeof(ps_cand_t));
			}
			c = &s->cand[s->n_cand++];
			c->orf = orf, c->diag = best_diag, c->ung = best, c->ung_qs = best_qs, c->ung_qe = best_qe;
		}
	}
	if (s->n_cand > po->max_seqs) {
		qsort(s->cand, s->n_cand, sizeof(ps_cand_t), ps_cand_cmp_ung);
		s->n_cand = po->max_seqs;
	}
	for (i = 0; i < s->n_cand; ++i) {
		ps_cand_t *c = &s->cand[i];
		const mp_orf_t *o = &ps->orf[c->orf];
		++s->n_aligned;
		c->sw = ps_gapped(s, pssm, L, ps->aa + o->aa_off, o->aa_len, c->diag, po->band, po->go, po->ge);
		c->ev = ps_evalue(c->sw, L, (double)ps->n_aa);
		if (aux->max_evalue > 0.0 && c->ev > aux->max_evalue) continue;
		if (n_out == m_out) {
			m_out = m_out? m_out << 1 : 64;
			out = (mp_cand1_t*)realloc(out, m_out * sizeof(mp_cand1_t));
		}
		out[n_out].vid = o->vid, out[n_out].vs = o->vs, out[n_out].ve = o->vs + 3 * (int64_t)o->aa_len, out[n_out].ev = c->ev;
		out[n_out].aqs = c->ung_qs, out[n_out].aqe = c->ung_qe, out[n_out].asc = c->ung, out[n_out].avs = o->vs + 3 * (int64_t)(c->ung_qs + c->diag);
		++n_out;
	}
	s->n_kept += n_out;
	aux->res[qi] = out, aux->n_res[qi] = n_out;
}

mp_cand_tab_t *mp_prescreen_batch(const mp_prescreen_t *ps, const mp_psopt_t *po, double max_evalue, int32_t n, mp_hmm_t **hmm, int32_t n_threads, FILE *dump)
{
	ps_batch_aux_t aux;
	mp_cand_tab_t *t;
	int32_t i;
	int64_t n_win = 0, n_kmer = 0, n_lookup = 0, n_dbl = 0, n_orf = 0, n_aln = 0, n_kept = 0;
	double t0 = mp_realtime();
	if (n_threads < 1) n_threads = 1;
	memset(&aux, 0, sizeof(aux));
	aux.ps = ps, aux.po = po, aux.max_evalue = max_evalue, aux.thr = mp_ps_kmer_thr(po), aux.hmm = hmm;
	aux.scr = Kcalloc(0, ps_scratch_t, n_threads);
	aux.res = Kcalloc(0, mp_cand1_t*, n);
	aux.n_res = Kcalloc(0, int32_t, n);
	kt_for(n_threads, ps_query_worker, &aux, n);
	t = mp_cand_tab_init();
	for (i = 0; i < n; ++i) {
		const char *name = mp_hmm_name(hmm[i]);
		mp_cand_tab_set(t, name, aux.n_res[i], aux.res[i]); // takes ownership, merges overlapping loci
		if (dump) {
			int32_t j, nc;
			const mp_cand1_t *c = mp_cand_tab_get(t, name, &nc);
			for (j = 0; j < nc; ++j) {
				const mp_ctg_t *ctg = &ps->nt->ctg[c[j].vid>>1];
				int64_t st = c[j].vid&1? ctg->len - c[j].ve : c[j].vs, en = c[j].vid&1? ctg->len - c[j].vs : c[j].ve;
				fprintf(dump, "%s\t%s\t%ld\t%ld\t%c\t%.3g\n", name, ctg->name, (long)st, (long)en, "+-"[c[j].vid&1], c[j].ev);
			}
		}
	}
	for (i = 0; i < n_threads; ++i) {
		ps_scratch_t *s = &aux.scr[i];
		n_win += s->n_win, n_kmer += s->n_kmer, n_lookup += s->n_lookup, n_dbl += s->n_dbl, n_orf += s->n_orf_scored, n_aln += s->n_aligned, n_kept += s->n_kept;
		ps_scratch_free(s);
	}
	free(aux.scr); free(aux.res); free(aux.n_res);
	if (mp_verbose >= 3)
		fprintf(stderr, "[M::%s@%.3f*%.2f] %d profiles: %.1f k-mers/pos, %ld index hits, %ld double-hit diagonals, %ld ORFs >= %d, %ld aligned, %ld loci kept (E <= %g) in %.1f s\n",
				__func__, mp_realtime(), mp_percent_cpu(), n, n_win? (double)n_kmer / n_win : 0.0, (long)n_lookup, (long)n_dbl, (long)n_orf, po->min_ungapped,
				(long)n_aln, (long)n_kept, max_evalue, mp_realtime() - t0);
	return t;
}
