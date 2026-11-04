/* protlocus-recheck: native candidate + profile-HMM recheck (ProtLocus 5.0).
 *
 * Reads the PAF produced by an HMM-library protlocus run with --trans (each
 * hit followed by a "##STA\t<peptide>" line carrying the spliced,
 * frameshift-skipped translation of the aligned genome region), rescores
 * every candidate peptide against its own family profile with a full
 * Forward algorithm, converts the bit score to an E-value with the
 * profile's own calibration (the "STATS LOCAL FORWARD tau lambda" line
 * every hmmbuild/hmmcalibrate-era HMMER3 file carries), and filters.
 *
 * The statistical model follows the published HMMER3 plan-7 architecture
 * (Eddy, PLoS Comput Biol 2011): multihit local profile configuration with
 * occupancy-weighted uniform entry, target-length model N/C/J loops,
 * Forward in log space, null1 geometric length model, a posterior-weighted
 * null2 biased-composition correction (the ByExpectation method, computed
 * from a full Backward pass; disable with -n), and an exponential tail
 * P-value exp(-lambda*(bits-tau)) capped at 1. This is an independent
 * re-implementation; no HMMER code or runtime is used. Differences from
 * hmmsearch's reported numbers: we score and bias-correct the full peptide
 * (sequence-level, treating the candidate as one envelope) rather than
 * per-domain envelopes; the decoy-FDR machinery is the ground-truth
 * calibration check.
 *
 * E-value scale: E = Z * P with -Z defaulting to the number of unique
 * candidate peptides; pass -Z <six-frame ORF count> to put the numbers on
 * exactly the same yardstick as a genome-wide six-frame hmmsearch.
 */
#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>
#include "kthread.h"
#include "ketopt.h"

#define RC_VERSION "0.1.0"
#define RC_LN2 0.69314718055994531
#define RC_NULL2_OMEGA (1.0 / 256.0) /* prior weight of the biased-composition null (HMMER omega) */
#define RC_ASIZE 20   /* canonical amino acids, HMMER column order */
#define RC_XIDX  20   /* degenerate/unknown residue */
#define RC_NCODE 21

/* HMMER match-emission column order and p7_AminoFrequencies background */
static const char rc_alphabet[] = "ACDEFGHIKLMNPQRSTVWY";
static const double rc_bg[RC_ASIZE] = {
	0.0787945, 0.0151600, 0.0535222, 0.0668298, 0.0397062, 0.0695071, 0.0229198, 0.0590092, 0.0594422, 0.0963728,
	0.0237718, 0.0414386, 0.0482904, 0.0395639, 0.0540978, 0.0683364, 0.0540687, 0.0673417, 0.0114135, 0.0304133
};

static uint8_t rc_aa2idx[256];

static void rc_init_tables(void)
{
	int i;
	for (i = 0; i < 256; ++i) rc_aa2idx[i] = RC_XIDX;
	for (i = 0; i < RC_ASIZE; ++i) {
		rc_aa2idx[(uint8_t)rc_alphabet[i]] = (uint8_t)i;
		rc_aa2idx[(uint8_t)tolower(rc_alphabet[i])] = (uint8_t)i;
	}
}

/**************
 * Profile data
 **************/

typedef struct {
	char *name;
	int32_t M;
	float *msc;    /* [(k-1)*RC_NCODE + x], k=1..M: match log-odds (nats); x==RC_XIDX = bg-weighted expectation */
	float *tsc;    /* [(k)*8 + t]: transitions in nats; see enum below; slot k holds transitions out of node k (into k+1) plus B->M_{k+1} */
	float ftau, flambda;  /* STATS LOCAL FORWARD, bit-score exponential tail */
	int has_stats;
} rc_prof_t;

enum { T_MM, T_IM, T_DM, T_BM, T_MI, T_II, T_MD, T_DD }; /* tsc[k*8+..]: into node k+1 (MM/IM/DM/BM, MD/DD) or within node k (MI/II) */

#define NEG_INF (-1e30f)

static double parse_nlogp(const char *s) /* HMMER stores -ln(p); '*' means p=0 */
{
	if (s[0] == '*' && s[1] == 0) return NEG_INF;
	return -strtod(s, 0);
}

static char *rc_strdup(const char *s)
{
	size_t n = strlen(s) + 1;
	char *p = (char*)malloc(n);
	if (p) memcpy(p, s, n);
	return p;
}

static int rc_iequal(const char *a, const char *b)
{
	for (; *a && *b; ++a, ++b)
		if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) return 0;
	return *a == 0 && *b == 0;
}

static int rc_split(char *line, char **tok, int max_tok)
{
	int n = 0;
	char *p = line, *q;
	while (n < max_tok) {
		while (*p && isspace((unsigned char)*p)) ++p;
		if (*p == 0) break;
		q = p;
		while (*q && !isspace((unsigned char)*q)) ++q;
		tok[n++] = p;
		if (*q == 0) break;
		*q = 0, p = q + 1;
	}
	return n;
}

/* Read one profile from a HMMER3 ASCII file; returns 0 on success, -1 on EOF, -2 on parse error.
 * If skip_set is non-NULL and the NAME is not wanted, the body is skipped and *skipped is set. */
static int rc_read_prof(FILE *fp, rc_prof_t *pr, int (*want)(void*, const char*), void *wopq, int *skipped)
{
	char line[65536], *tok[64];
	char name[256];
	int32_t M = -1, k = 0, state = 0, n;
	double *lmat = 0, *lt = 0; /* -ln p from file: lmat[(k-1)*20+x], lt[k*7+t], k=0..M */
	int has_name = 0, wanted = 1;
	float ftau = 0.f, flambda = 0.f;
	int has_stats = 0;

	*skipped = 0;
	memset(pr, 0, sizeof(*pr));
	while (fgets(line, sizeof(line), fp)) {
		if (state == 0) {
			if (strncmp(line, "HMMER3", 6) == 0) { state = 1; continue; }
			if (line[0] == 0 || line[0] == '\n') continue;
			return -2;
		}
		if (state == 1) { /* header section */
			n = rc_split(line, tok, 64);
			if (n == 0) continue;
			if (strcmp(tok[0], "NAME") == 0 && n >= 2) {
				snprintf(name, sizeof(name), "%s", tok[1]), has_name = 1;
				if (want && !want(wopq, name)) wanted = 0;
			} else if (strcmp(tok[0], "LENG") == 0 && n >= 2) {
				M = atoi(tok[1]);
			} else if (strcmp(tok[0], "ALPH") == 0 && n >= 2) {
				if (!rc_iequal(tok[1], "amino")) return -2;
			} else if (strcmp(tok[0], "STATS") == 0 && n >= 5) {
				if (strcmp(tok[1], "LOCAL") == 0 && strcmp(tok[2], "FORWARD") == 0)
					ftau = (float)atof(tok[3]), flambda = (float)atof(tok[4]), has_stats = 1;
			} else if (strcmp(tok[0], "HMM") == 0) {
				if (!has_name || M <= 0) return -2;
				if (!fgets(line, sizeof(line), fp)) return -2; /* "m->m m->i ..." header */
				if (!wanted) { /* skip body up to '//' */
					while (fgets(line, sizeof(line), fp))
						if (line[0] == '/' && line[1] == '/') { *skipped = 1; return 0; }
					return -2;
				}
				lmat = (double*)malloc(sizeof(double) * (size_t)M * RC_ASIZE);
				lt = (double*)malloc(sizeof(double) * (size_t)(M + 1) * 7);
				if (!lmat || !lt) { free(lmat); free(lt); return -2; }
				state = 2, k = 0; /* expect COMPO or node-0 insert line next */
				continue;
			}
			continue;
		}
		if (state == 2) { /* model body */
			n = rc_split(line, tok, 64);
			if (n == 0) continue;
			if (strcmp(tok[0], "//") == 0) break;
			if (strcmp(tok[0], "COMPO") == 0) continue;
			if (k == 0 && n == 7) { /* node 0 transition line (insert-0 emission line has 20 fields) */
				int t;
				for (t = 0; t < 7; ++t) lt[t] = parse_nlogp(tok[t]);
				k = 1;
				continue;
			}
			if (n == RC_ASIZE) continue; /* insert emission line: hardwired to bg (score 0), ignore */
			if (k >= 1 && k <= M && n >= 1 + RC_ASIZE && atoi(tok[0]) == k) { /* match emission line */
				int x, t;
				for (x = 0; x < RC_ASIZE; ++x) lmat[(k-1)*RC_ASIZE + x] = parse_nlogp(tok[1 + x]);
				/* skip the insert emission line, then read the transition line */
				if (!fgets(line, sizeof(line), fp)) goto err;
				if (!fgets(line, sizeof(line), fp)) goto err;
				n = rc_split(line, tok, 64);
				if (n < 7) goto err;
				for (t = 0; t < 7; ++t) lt[k*7 + t] = parse_nlogp(tok[t]);
				++k;
				continue;
			}
			goto err;
		}
	}
	if (state == 0) return -1; /* clean EOF */
	if (state != 2 || k != M + 1) goto err;

	/* build the multihit-local search profile (p7_ProfileConfig equivalent) */
	{
		double *occ = (double*)malloc(sizeof(double) * (size_t)(M + 1));
		double Z = 0.0;
		int x;
		if (!occ) goto err;
		pr->name = rc_strdup(name);
		pr->M = M;
		pr->ftau = ftau, pr->flambda = flambda, pr->has_stats = has_stats;
		pr->msc = (float*)malloc(sizeof(float) * (size_t)M * RC_NCODE);
		pr->tsc = (float*)malloc(sizeof(float) * (size_t)M * 8);
		if (!pr->name || !pr->msc || !pr->tsc) { free(occ); goto err; }
		/* match log-odds + degenerate-X expectation */
		for (k = 1; k <= M; ++k) {
			double ex = 0.0;
			for (x = 0; x < RC_ASIZE; ++x) {
				double lp = lmat[(k-1)*RC_ASIZE + x];
				float sc = lp <= NEG_INF? NEG_INF : (float)(lp - log(rc_bg[x]));
				pr->msc[(k-1)*RC_NCODE + x] = sc;
				ex += rc_bg[x] * (sc <= NEG_INF? -1e4 : sc);
			}
			pr->msc[(k-1)*RC_NCODE + RC_XIDX] = (float)ex;
		}
		/* occupancy (p7_hmm_CalculateOccupancy) in probability space */
		occ[0] = 0.0;
		occ[1] = exp(lt[0]) + exp(lt[1]); /* t0: m->m + m->i (1 - B->D1) */
		for (k = 2; k <= M; ++k) {
			double tmm = exp(lt[(k-1)*7 + 0]), tmi = exp(lt[(k-1)*7 + 1]), tdm = exp(lt[(k-1)*7 + 5]);
			occ[k] = occ[k-1] * (tmm + tmi) + (1.0 - occ[k-1]) * tdm;
		}
		for (k = 1; k <= M; ++k) Z += occ[k] * (double)(M - k + 1);
		/* transitions; slot k-1 holds entries into node k */
		for (k = 1; k <= M; ++k) {
			float *tp = pr->tsc + (size_t)(k-1)*8;
			tp[T_BM] = occ[k] > 0.0 && Z > 0.0? (float)log(occ[k] / Z) : NEG_INF;
			if (k == 1) { /* no node-0 core transitions in the search profile */
				tp[T_MM] = tp[T_IM] = tp[T_DM] = NEG_INF;
			} else {
				tp[T_MM] = (float)lt[(k-1)*7 + 0];
				tp[T_IM] = (float)lt[(k-1)*7 + 3];
				tp[T_DM] = (float)lt[(k-1)*7 + 5];
			}
			/* within node k (used for I_k and D_{k+1}); I_M impossible */
			tp[T_MI] = k < M? (float)lt[k*7 + 1] : NEG_INF;
			tp[T_II] = k < M? (float)lt[k*7 + 4] : NEG_INF;
			tp[T_MD] = k < M? (float)lt[k*7 + 2] : NEG_INF;
			tp[T_DD] = k < M? (float)lt[k*7 + 6] : NEG_INF;
		}
		free(occ);
	}
	free(lmat); free(lt);
	return 0;
err:
	free(lmat); free(lt);
	free(pr->name); free(pr->msc); free(pr->tsc);
	memset(pr, 0, sizeof(*pr));
	return -2;
}

/**************
 * Forward
 **************/

static inline float rc_logsum(float a, float b)
{
	float mx = a > b? a : b, mn = a > b? b : a;
	if (mn <= NEG_INF || mx - mn > 50.0f) return mx;
	return mx + log1pf(expf(mn - mx));
}

typedef struct { float *buf; size_t cap; } rc_scratch_t;

static float *rc_scr(rc_scratch_t *sc, size_t need)
{
	if (sc->cap < need) {
		sc->cap = need + (need >> 2) + 1024;
		sc->buf = (float*)realloc(sc->buf, sc->cap * sizeof(float));
	}
	return sc->buf;
}

#define RC_NULL2_MAXCELLS 33554432 /* skip the bias correction above 32M DP cells (rare huge pairs) */

/* Full-sequence Forward (+ optional Backward/posterior null2 bias correction,
 * the p7_GNull2_ByExpectation method), multihit local, length model L = lseq.
 * Sets *bits_out (null1- and bias-corrected) and *bias_bits_out. */
static void rc_score(const rc_prof_t *pr, const uint8_t *dsq, int32_t L, int do_null2,
                     rc_scratch_t *sc, float *bits_out, float *bias_bits_out)
{
	int32_t M = pr->M, W = M + 1, i, k, x;
	float pmove = (2.0f + 1.0f) / ((float)L + 2.0f + 1.0f); /* multihit: nj=1 */
	float lmove = logf(pmove), lloop = logf(1.0f - pmove);
	const float lEmove = -(float)RC_LN2, lEloop = -(float)RC_LN2; /* E->C, E->J */
	size_t cells = (size_t)(L + 1) * W;
	float *FM, *FI, *BM, *BI, *Df, *xNf, *xJf, *xCf, *xBf, *buf;
	float tot, bias = 0.0f;

	if (do_null2 && cells > RC_NULL2_MAXCELLS) do_null2 = 0;
	buf = rc_scr(sc, cells * (do_null2? 4 : 2) + 2 * W + 4 * (size_t)(L + 1));
	FM = buf; FI = FM + cells;
	BM = do_null2? FI + cells : 0; BI = do_null2? BM + cells : 0;
	Df = (do_null2? BI + cells : FI + cells);           /* two D row buffers */
	xNf = Df + 2 * W; xJf = xNf + (L + 1); xCf = xJf + (L + 1); xBf = xCf + (L + 1);

	/* ---- Forward ---- */
	for (k = 0; k <= M; ++k) FM[k] = FI[k] = Df[k] = NEG_INF;
	xNf[0] = 0.0f, xBf[0] = lmove, xJf[0] = xCf[0] = NEG_INF;
	for (i = 1; i <= L; ++i) {
		const float *Mp = FM + (size_t)(i-1)*W, *Ip = FI + (size_t)(i-1)*W;
		float *Mr = FM + (size_t)i*W, *Ir = FI + (size_t)i*W;
		float *Dp = Df + (i & 1? 0 : W), *Dr = Df + (i & 1? W : 0);
		uint8_t xr = dsq[i-1];
		float e = NEG_INF;
		Mr[0] = Ir[0] = Dr[0] = NEG_INF;
		for (k = 1; k <= M; ++k) {
			const float *tp = pr->tsc + (size_t)(k-1)*8;
			float m = rc_logsum(rc_logsum(Mp[k-1] + tp[T_MM], Ip[k-1] + tp[T_IM]),
			                    rc_logsum(xBf[i-1] + tp[T_BM], Dp[k-1] + tp[T_DM]));
			Mr[k] = m + pr->msc[(size_t)(k-1)*RC_NCODE + xr];
			Ir[k] = k < M? rc_logsum(Mp[k] + tp[T_MI], Ip[k] + tp[T_II]) : NEG_INF;
			Dr[k] = k >= 2? rc_logsum(Mr[k-1] + pr->tsc[(size_t)(k-2)*8 + T_MD],
			                          Dr[k-1] + pr->tsc[(size_t)(k-2)*8 + T_DD]) : NEG_INF;
			e = rc_logsum(e, rc_logsum(Mr[k], Dr[k])); /* local exits, prob 1 */
		}
		xJf[i] = rc_logsum(xJf[i-1] + lloop, e + lEloop);
		xCf[i] = rc_logsum(xCf[i-1] + lloop, e + lEmove);
		xNf[i] = xNf[i-1] + lloop;
		xBf[i] = rc_logsum(xNf[i] + lmove, xJf[i] + lmove);
	}
	tot = xCf[L] + lmove; /* C->T: Forward lod score, nats */

	/* ---- Backward + posterior null2 (p7_GNull2_ByExpectation) ---- */
	if (do_null2) {
		float bE, bJ, bC, bN, bB, bJn, bCn, bNn;
		float *lum, *lui, uN, uJ, uC, logL = logf((float)L);
		double n2sc = 0.0, a;
		float lnodds[RC_NCODE];
		/* row L */
		{
			float *Mr = BM + (size_t)L*W, *Ir = BI + (size_t)L*W, *Dr = Df;
			bC = lmove; bE = bC + lEmove; bJ = bN = bB = NEG_INF;
			Mr[M] = Dr[M] = bE; Ir[M] = Ir[0] = NEG_INF;
			for (k = M-1; k >= 1; --k) {
				Mr[k] = rc_logsum(bE, Dr[k+1] + pr->tsc[(size_t)(k-1)*8 + T_MD]);
				Dr[k] = rc_logsum(bE, Dr[k+1] + pr->tsc[(size_t)(k-1)*8 + T_DD]);
				Ir[k] = NEG_INF;
			}
			Mr[0] = Dr[0] = NEG_INF;
		}
		xBf[L] = bB; /* reuse xBf/xNf/xJf/xCf? keep fwd values; store bwd specials per row in scalars + arrays below */
		/* we need per-row backward N/J/C for usage; reuse three trailing slices of Df? allocate on the fly */
		{
			float *bNr = xBf; /* overwrite xBf (fwd B no longer needed) with backward N */
			float *bJr = (float*)malloc(sizeof(float) * (size_t)(L + 1) * 2);
			float *bCr = bJr + (L + 1);
			bNr[L] = bN, bJr[L] = bJ, bCr[L] = bC;
			for (i = L-1; i >= 1; --i) {
				const float *Mn = BM + (size_t)(i+1)*W, *In = BI + (size_t)(i+1)*W;
				float *Mr = BM + (size_t)i*W, *Ir = BI + (size_t)i*W;
				float *Dr = Df; /* backward D depends only on the current row (descending k) */
				uint8_t xn = dsq[i]; /* residue i+1 */
				bB = NEG_INF;
				for (k = 1; k <= M; ++k)
					bB = rc_logsum(bB, Mn[k] + pr->tsc[(size_t)(k-1)*8 + T_BM]
					                        + pr->msc[(size_t)(k-1)*RC_NCODE + xn]);
				bJn = bJr[i+1], bCn = bCr[i+1], bNn = bNr[i+1];
				bJ = rc_logsum(bJn + lloop, bB + lmove);
				bC = bCn + lloop;
				bE = rc_logsum(bJ + lEloop, bC + lEmove);
				bN = rc_logsum(bNn + lloop, bB + lmove);
				bNr[i] = bN, bJr[i] = bJ, bCr[i] = bC;
				{
					float *DD = Dr;
					DD[M] = bE; Mr[M] = bE; Ir[M] = NEG_INF;
					for (k = M-1; k >= 1; --k) {
						const float *tk = pr->tsc + (size_t)k*8;       /* out of node k into k+1 */
						const float *tk1 = pr->tsc + (size_t)(k-1)*8;  /* within node k */
						float mn1 = Mn[k+1] + pr->msc[(size_t)k*RC_NCODE + xn];
						Mr[k] = rc_logsum(rc_logsum(mn1 + tk[T_MM], In[k] + tk1[T_MI]),
						                  rc_logsum(bE, DD[k+1] + tk1[T_MD]));
						Ir[k] = rc_logsum(mn1 + tk[T_IM], In[k] + tk1[T_II]);
						DD[k] = rc_logsum(mn1 + tk[T_DM], rc_logsum(DD[k+1] + tk1[T_DD], bE));
					}
					Mr[0] = Ir[0] = DD[0] = NEG_INF;
				}
			}
			/* usage in log space */
			lum = (float*)malloc(sizeof(float) * (size_t)W * 2);
			lui = lum + W;
			for (k = 0; k <= M; ++k) lum[k] = lui[k] = NEG_INF;
			for (i = 1; i <= L; ++i) {
				const float *fm = FM + (size_t)i*W, *fi = FI + (size_t)i*W;
				const float *bm = BM + (size_t)i*W, *bi = BI + (size_t)i*W;
				for (k = 1; k <= M; ++k) {
					lum[k] = rc_logsum(lum[k], fm[k] + bm[k]);
					if (k < M) lui[k] = rc_logsum(lui[k], fi[k] + bi[k]);
				}
			}
			uN = uJ = uC = NEG_INF;
			for (i = 1; i <= L; ++i) {
				uN = rc_logsum(uN, xNf[i-1] + bNr[i] + lloop);
				uJ = rc_logsum(uJ, xJf[i-1] + bJr[i] + lloop);
				uC = rc_logsum(uC, xCf[i-1] + bCr[i] + lloop);
			}
			free(bJr);
#ifdef RC_DEBUG
			{ /* consistency: Backward total (bN at row 0) must equal the Forward score */
				const float *M1 = BM + (size_t)1*W;
				float bB0 = NEG_INF, bN0;
				for (k = 1; k <= M; ++k)
					bB0 = rc_logsum(bB0, M1[k] + pr->tsc[(size_t)(k-1)*8 + T_BM]
					                          + pr->msc[(size_t)(k-1)*RC_NCODE + dsq[0]]);
				bN0 = rc_logsum(bNr[1] + lloop, bB0 + lmove);
				fprintf(stderr, "[debug] fwd=%.4f bwd=%.4f (nats)\n", tot, bN0);
			}
#endif
			{
				float xfac = rc_logsum(rc_logsum(uN, uJ), uC) - tot - logL;
				float li = NEG_INF;
				for (k = 1; k < M; ++k) li = rc_logsum(li, lui[k] - tot - logL); /* insert log-odds = 0 */
				for (x = 0; x < RC_ASIZE; ++x) {
					float v = rc_logsum(xfac, li);
					for (k = 1; k <= M; ++k)
						v = rc_logsum(v, (lum[k] - tot - logL) + pr->msc[(size_t)(k-1)*RC_NCODE + x]);
					lnodds[x] = v;
				}
				{ /* degenerate X: average of odds ratios */
					double s = 0.0;
					for (x = 0; x < RC_ASIZE; ++x) s += exp((double)lnodds[x]);
					lnodds[RC_XIDX] = (float)log(s / RC_ASIZE);
				}
				for (i = 0; i < L; ++i) n2sc += lnodds[dsq[i]];
				a = log(RC_NULL2_OMEGA) + n2sc;
				bias = (float)(a > 0.0? a + log1p(exp(-a)) : log1p(exp(a)));
			}
			free(lum);
		}
	}
	{
		float p1 = (float)L / (float)(L + 1);
		float null1 = (float)L * logf(p1) + logf(1.0f - p1);
		*bits_out = (tot - null1 - bias) / (float)RC_LN2;
		*bias_bits_out = bias / (float)RC_LN2;
	}
}

/**************
 * Candidates
 **************/

typedef struct {
	char *raw;        /* PAF line (no newline) */
	char *family;     /* points into raw copy */
	int32_t pep_id;   /* unique-peptide index, -1 if no ##STA */
	int32_t n_stop;
} rc_cand_t;

typedef struct {
	char *seq;        /* peptide, '*' kept as-is (scored as X) */
	int32_t len;
	int32_t prof_id;  /* -1 if family not found in the library */
	float bits, evalue, bias_bits;
	int done;
} rc_pep_t;

typedef struct {
	rc_cand_t *cand; int64_t n_cand, m_cand;
	rc_pep_t *pep; int64_t n_pep, m_pep;
	rc_prof_t *prof; int32_t n_prof;
	double Z;
	int no_null2;
	rc_scratch_t *scratch; /* per thread */
} rc_data_t;

static int rc_is_paf(char **f, int nf)
{
	char *e;
	if (nf < 12) return 0;
	strtol(f[1], &e, 10); if (*e) return 0;
	strtol(f[2], &e, 10); if (*e) return 0;
	strtol(f[7], &e, 10); if (*e) return 0;
	return f[4][1] == 0 && (f[4][0] == '+' || f[4][0] == '-' || f[4][0] == '*');
}

/* simple string-key hash (FNV-1a + open addressing) */
typedef struct { char **key; int32_t *val; uint32_t size, used; } rc_hash_t;

static uint32_t rc_fnv(const char *s, const char *s2)
{
	uint32_t h = 2166136261u;
	for (; *s; ++s) h = (h ^ (uint8_t)*s) * 16777619u;
	if (s2) { h = (h ^ 1u) * 16777619u; for (; *s2; ++s2) h = (h ^ (uint8_t)*s2) * 16777619u; }
	return h;
}

static void rc_hash_grow(rc_hash_t *h)
{
	uint32_t nsize = h->size? h->size << 1 : 4096, i;
	char **nk = (char**)calloc(nsize, sizeof(char*));
	int32_t *nv = (int32_t*)malloc(nsize * sizeof(int32_t));
	for (i = 0; i < h->size; ++i) {
		if (!h->key[i]) continue;
		uint32_t j = rc_fnv(h->key[i], 0) & (nsize - 1);
		while (nk[j]) j = (j + 1) & (nsize - 1);
		nk[j] = h->key[i], nv[j] = h->val[i];
	}
	free(h->key); free(h->val);
	h->key = nk, h->val = nv, h->size = nsize;
}

/* returns slot value; inserts (strdup'd key2 appended) with val if absent; *absent set */
static int32_t rc_hash_put(rc_hash_t *h, const char *k1, const char *k2, int32_t val, int *absent)
{
	uint32_t i;
	if (h->used * 4 >= h->size * 3) rc_hash_grow(h);
	i = rc_fnv(k1, k2) & (h->size - 1);
	for (;;) {
		if (!h->key[i]) {
			size_t l1 = strlen(k1), l2 = k2? strlen(k2) : 0;
			char *s = (char*)malloc(l1 + l2 + 2);
			memcpy(s, k1, l1); s[l1] = k2? 1 : 0;
			if (k2) memcpy(s + l1 + 1, k2, l2 + 1);
			h->key[i] = s, h->val[i] = val, ++h->used;
			*absent = 1;
			return val;
		} else {
			const char *s = h->key[i];
			size_t l1 = strlen(k1);
			if (strncmp(s, k1, l1) == 0 && (uint8_t)s[l1] == (k2? 1 : 0) &&
			    (!k2 || strcmp(s + l1 + 1, k2) == 0)) { *absent = 0; return h->val[i]; }
			i = (i + 1) & (h->size - 1);
		}
	}
}

static int rc_fam_wanted(void *opq, const char *nm)
{
	rc_hash_t *hh = (rc_hash_t*)opq;
	uint32_t j;
	size_t ln = strlen(nm);
	if (hh->size == 0) return 0;
	j = rc_fnv(nm, 0) & (hh->size - 1);
	while (hh->key[j]) {
		if (strncmp(hh->key[j], nm, ln) == 0 && hh->key[j][ln] == 0) return 1;
		j = (j + 1) & (hh->size - 1);
	}
	return 0;
}

static void worker(void *data, long i, int tid)
{
	rc_data_t *d = (rc_data_t*)data;
	rc_pep_t *p = &d->pep[i];
	const rc_prof_t *pr;
	uint8_t *dsq;
	int32_t j;
	float bits, bias = 0.0f;
	if (p->prof_id < 0 || p->len == 0) { p->done = 0; return; }
	pr = &d->prof[p->prof_id];
	dsq = (uint8_t*)malloc((size_t)p->len);
	for (j = 0; j < p->len; ++j) dsq[j] = rc_aa2idx[(uint8_t)p->seq[j]];
	rc_score(pr, dsq, p->len, !d->no_null2, &d->scratch[tid], &bits, &bias);
	free(dsq);
	p->bits = bits;
	p->bias_bits = bias; /* already in bits */
	if (pr->has_stats) {
		double pv = bits >= pr->ftau? exp(-(double)pr->flambda * (bits - pr->ftau)) : 1.0;
		double e = pv * d->Z;
		p->evalue = (float)(e < d->Z? e : d->Z);
	} else p->evalue = NAN;
	p->done = 1;
}

static void usage(FILE *fp)
{
	fprintf(fp, "Usage: protlocus-recheck [options] <profiles.hmm> <cand.paf>\n");
	fprintf(fp, "  cand.paf: protlocus HMM-library run with --trans (PAF + ##STA lines); '-'=stdin, .gz ok\n");
	fprintf(fp, "Options:\n");
	fprintf(fp, "  -e FLOAT   max Forward E-value to pass [1e-5]\n");
	fprintf(fp, "  -Z FLOAT   effective database size for E-values [unique candidate peptides]\n");
	fprintf(fp, "  -t INT     threads [8]\n");
	fprintf(fp, "  -T FILE    write per-candidate TSV\n");
	fprintf(fp, "  -a         keep failing lines, tagged hp:i:0\n");
	fprintf(fp, "  -n         disable the biased-composition (null2-style) correction\n");
	fprintf(fp, "  -v         print version and exit\n");
	fprintf(fp, "Output: passing PAF lines with fB:f:<bits> fE:f:<E-value> appended\n");
}

int main(int argc, char *argv[])
{
	ketopt_t o = KETOPT_INIT;
	double max_e = 1e-5, Z = -1.0;
	int c, n_threads = 8, keep_fail = 0, no_null2 = 0;
	const char *tsv_fn = 0;
	rc_data_t d;
	rc_hash_t fam_need = {0}, pep_dedup = {0}, prof_by_name = {0};
	gzFile fp;
	FILE *hfp;
	char *line = 0;
	size_t m_line = 0;
	int64_t i, n_pass = 0, n_noprof = 0, n_nopep = 0;

	while ((c = ketopt(&o, argc, argv, 1, "e:Z:t:T:anv", 0)) >= 0) {
		if (c == 'e') max_e = atof(o.arg);
		else if (c == 'Z') Z = atof(o.arg);
		else if (c == 't') n_threads = atoi(o.arg);
		else if (c == 'T') tsv_fn = o.arg;
		else if (c == 'a') keep_fail = 1;
		else if (c == 'n') no_null2 = 1;
		else if (c == 'v') { puts(RC_VERSION); return 0; }
		else { usage(stderr); return 1; }
	}
	if (argc - o.ind < 2) { usage(stderr); return 1; }
	rc_init_tables();
	memset(&d, 0, sizeof(d));

	/* pass 1: read candidates, dedup peptides per family */
	fp = strcmp(argv[o.ind+1], "-") == 0? gzdopen(0, "r") : gzopen(argv[o.ind+1], "r");
	if (!fp) { fprintf(stderr, "[recheck] ERROR: cannot open %s\n", argv[o.ind+1]); return 1; }
	{
		char buf[262144];
		rc_cand_t *pending = 0;
		line = (char*)malloc(m_line = 1 << 20);
		while (gzgets(fp, buf, sizeof(buf))) {
			size_t l = strlen(buf), cur = l;
			if (l + 1 > m_line) line = (char*)realloc(line, m_line = (l + 1) * 2);
			memcpy(line, buf, l + 1);
			while (l && line[l-1] != '\n' && gzgets(fp, buf, sizeof(buf))) { /* long line */
				l = strlen(buf);
				if (cur + l + 1 > m_line) line = (char*)realloc(line, m_line = (cur + l + 1) * 2);
				memcpy(line + cur, buf, l + 1), cur += l;
			}
			l = cur;
			while (l && (line[l-1] == '\n' || line[l-1] == '\r')) line[--l] = 0;
			if (strncmp(line, "##PAF\t", 6) == 0) memmove(line, line + 6, l - 5), l -= 6;
			if (strncmp(line, "##STA\t", 6) == 0) {
				if (pending) {
					const char *pep = line + 6;
					int absent, j;
					int32_t id;
					for (j = 0; pep[j]; ++j) if (pep[j] == '*') ++pending->n_stop;
					id = rc_hash_put(&pep_dedup, pending->family, pep, (int32_t)d.n_pep, &absent);
					if (absent) {
						if (d.n_pep == d.m_pep) {
							d.m_pep = d.m_pep? d.m_pep << 1 : 1024;
							d.pep = (rc_pep_t*)realloc(d.pep, d.m_pep * sizeof(rc_pep_t));
						}
						memset(&d.pep[d.n_pep], 0, sizeof(rc_pep_t));
						d.pep[d.n_pep].seq = rc_strdup(pep);
						d.pep[d.n_pep].len = (int32_t)strlen(pep);
						d.pep[d.n_pep].prof_id = -1;
						++d.n_pep;
						rc_hash_put(&fam_need, pending->family, 0, 0, &absent);
					}
					pending->pep_id = id;
					pending = 0;
				}
				continue;
			}
			if (line[0] == '#' || line[0] == 0) continue;
			{
				char *copy = rc_strdup(line), *copy2 = rc_strdup(line), *f[16];
				int nf = rc_split(copy2, f, 13);
				pending = 0;
				if (!rc_is_paf(f, nf) || f[5][0] == '*' || f[4][0] == '*') { free(copy); free(copy2); continue; }
				if (d.n_cand == d.m_cand) {
					d.m_cand = d.m_cand? d.m_cand << 1 : 4096;
					d.cand = (rc_cand_t*)realloc(d.cand, d.m_cand * sizeof(rc_cand_t));
				}
				d.cand[d.n_cand].raw = copy;
				d.cand[d.n_cand].family = rc_strdup(f[0]);
				d.cand[d.n_cand].pep_id = -1;
				d.cand[d.n_cand].n_stop = 0;
				pending = &d.cand[d.n_cand++];
				free(copy2);
			}
		}
	}
	gzclose(fp);
	fprintf(stderr, "[recheck] %lld candidates, %lld unique peptides, %u families\n",
	        (long long)d.n_cand, (long long)d.n_pep, fam_need.used);
	if (d.n_pep == 0) {
		fprintf(stderr, "[recheck] ERROR: no ##STA peptides; was protlocus run with --trans?\n");
		return 1;
	}
	d.Z = Z > 0.0? Z : (double)d.n_pep;
	d.no_null2 = no_null2;

	/* pass 2: load only the needed profiles */
	hfp = fopen(argv[o.ind], "r");
	if (!hfp) { fprintf(stderr, "[recheck] ERROR: cannot open %s\n", argv[o.ind]); return 1; }
	{
		int32_t m_prof = 0;
		for (;;) {
			rc_prof_t pr;
			int skipped, ret;
			ret = rc_read_prof(hfp, &pr, rc_fam_wanted, &fam_need, &skipped);
			if (ret == -1) break;
			if (ret == -2) { fprintf(stderr, "[recheck] ERROR: malformed profile in %s\n", argv[o.ind]); return 1; }
			if (skipped) continue;
			if (d.n_prof == m_prof) {
				m_prof = m_prof? m_prof << 1 : 1024;
				d.prof = (rc_prof_t*)realloc(d.prof, m_prof * sizeof(rc_prof_t));
			}
			d.prof[d.n_prof] = pr;
			{
				int absent;
				rc_hash_put(&prof_by_name, pr.name, 0, d.n_prof, &absent);
			}
			++d.n_prof;
		}
	}
	fclose(hfp);
	fprintf(stderr, "[recheck] loaded %d needed profiles from %s\n", d.n_prof, argv[o.ind]);

	/* bind peptides to profiles (via their first candidate's family) */
	for (i = 0; i < d.n_cand; ++i) {
		rc_cand_t *cd = &d.cand[i];
		if (cd->pep_id >= 0 && d.pep[cd->pep_id].prof_id < 0) {
			int absent;
			int32_t v = rc_hash_put(&prof_by_name, cd->family, 0, -1, &absent);
			if (!absent && v >= 0) d.pep[cd->pep_id].prof_id = v;
		}
	}
	for (i = 0; i < d.n_pep; ++i) if (d.pep[i].prof_id < 0) ++n_noprof;
	if (n_noprof)
		fprintf(stderr, "[recheck] WARNING: %lld peptides belong to families absent from the library\n", (long long)n_noprof);

	/* score */
	d.scratch = (rc_scratch_t*)calloc(n_threads, sizeof(rc_scratch_t));
	kt_for(n_threads, worker, &d, d.n_pep);

	/* output */
	{
		FILE *tsv = tsv_fn? fopen(tsv_fn, "w") : 0;
		if (tsv) fprintf(tsv, "cand_idx\tfamily\tpep_len\tn_stop\tfwd_bits\tbias_bits\tfwd_E\tpass\n");
		for (i = 0; i < d.n_cand; ++i) {
			rc_cand_t *cd = &d.cand[i];
			rc_pep_t *p = cd->pep_id >= 0? &d.pep[cd->pep_id] : 0;
			int ok = p && p->done && !isnan(p->evalue) && p->evalue <= max_e;
			if (!p || cd->pep_id < 0) ++n_nopep;
			if (ok) {
				++n_pass;
				printf("%s\tfB:f:%.2f\tfE:f:%.3g\n", cd->raw, p->bits, p->evalue);
			} else if (keep_fail) {
				if (p && p->done) printf("%s\tfB:f:%.2f\tfE:f:%.3g\thp:i:0\n", cd->raw, p->bits, p->evalue);
				else printf("%s\thp:i:0\n", cd->raw);
			}
			if (tsv) {
				if (p && p->done)
					fprintf(tsv, "%lld\t%s\t%d\t%d\t%.2f\t%.2f\t%.3g\t%d\n", (long long)i, cd->family,
					        p->len, cd->n_stop, p->bits, p->bias_bits, p->evalue, ok);
				else
					fprintf(tsv, "%lld\t%s\t.\t.\t.\t.\t.\t0\n", (long long)i, cd->family);
			}
		}
		if (tsv) fclose(tsv);
	}
	fprintf(stderr, "[recheck] passed %lld/%lld (E<=%g, Z=%.0f); %lld without peptide/profile\n",
	        (long long)n_pass, (long long)d.n_cand, max_e, d.Z, (long long)(n_nopep));
	return 0;
}
