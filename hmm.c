#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "protlocus.h"
#include "mppriv.h"

extern uint8_t ns_tab_codon[64];
extern uint8_t ns_tab_aa20[256];

struct mp_hmm_s {
	char *name;
	char *consensus;
	int32_t len;
	int16_t *emit; /* [profile position][amino-acid index] */
	int16_t *trans; /* m->m,m->i,m->d,i->m,i->i,d->m,d->d */
	int8_t *pssm; /* [profile position][nasw residue index, 22 columns]: half-bit log-odds vs the HMMER null model */
	void *gpu[2]; /* device-side emit/trans buffers owned by hmm-gpu.c */
};

#define MP_HMM_ASIZE 22
#define MP_HMM_STOP_SC_DEFAULT 23 /* mp_mapopt_init() fs; ns_set_stop_sc() uses the same value for the BLOSUM stop row */

/* HMMER3 match-emission column order and the p7 amino-acid background (p7_AminoFrequencies) */
static const char mp_hmm_alphabet[] = "ACDEFGHIKLMNPQRSTVWY";
static const double mp_hmm_bg[20] = {
	0.0787945, 0.0151600, 0.0535222, 0.0668298, 0.0397062, 0.0695071, 0.0229198, 0.0590092, 0.0594422, 0.0963728,
	0.0237718, 0.0414386, 0.0482904, 0.0395639, 0.0540978, 0.0683364, 0.0540687, 0.0673417, 0.0114135, 0.0304133
};

struct mp_hmm_file_s {
	FILE *fp;
	int err;
};

static char *dup_text(const char *s)
{
	size_t n = strlen(s);
	char *p = (char*)malloc(n + 1);
	if (p) memcpy(p, s, n + 1);
	return p;
}

static int parse_score(const char *s)
{
	char *end;
	double x;
	if (strcmp(s, "*") == 0) return -32768;
	errno = 0;
	x = strtod(s, &end);
	if (errno || end == s || *end) return -32768;
	x = -2.0 * x; /* HMMER stores negative log probabilities. */
	if (x < -32767.) x = -32767.;
	if (x > 32767.) x = 32767.;
	return (int)lrint(x);
}

static int split(char *line, char **tok, int max_tok)
{
	int n = 0;
	char *p = strtok(line, " \t\r\n");
	while (p && n < max_tok) tok[n++] = p, p = strtok(0, " \t\r\n");
	return n;
}

/* half-bit log-odds of a HMMER match emission (-ln p) against the background */
static int8_t pssm_score(const char *s, double bg)
{
	char *end;
	double x, lo;
	if (strcmp(s, "*") == 0) return -64;
	errno = 0;
	x = strtod(s, &end);
	if (errno || end == s || *end) return -64;
	lo = 2.0 * (-x - log(bg)) / log(2.0);
	if (lo < -64.) lo = -64.;
	if (lo > 64.) lo = 64.;
	return (int8_t)lrint(lo);
}

static int is_position_line(const char *line)
{
	char *end;
	(void)strtol(line, &end, 10);
	return end != line && isspace((unsigned char)*end);
}

mp_hmm_file_t *mp_hmm_open(const char *fn)
{
	mp_hmm_file_t *f;
	FILE *fp = fopen(fn, "r");
	if (!fp) return 0;
	f = (mp_hmm_file_t*)calloc(1, sizeof(*f));
	if (!f) { fclose(fp); return 0; }
	f->fp = fp;
	return f;
}

void mp_hmm_close(mp_hmm_file_t *f)
{
	if (!f) return;
	if (f->fp) fclose(f->fp);
	free(f);
}

int mp_hmm_error(const mp_hmm_file_t *f) { return f? f->err : 1; }

/* Parse the next HMMER3 profile (up to and including its "//" terminator).
 * Returns NULL at EOF; on a malformed record also sets the file error flag. */
mp_hmm_t *mp_hmm_read1(mp_hmm_file_t *f)
{
	FILE *fp;
	char line[8192], *tok[64];
	mp_hmm_t *h = 0;
	int in_hmm = 0, have_header = 0, pos = 0;
	if (!f || !f->fp || f->err) return 0;
	fp = f->fp;
	h = (mp_hmm_t*)calloc(1, sizeof(*h));
	if (!h) { f->err = 1; return 0; }
	while (fgets(line, sizeof(line), fp)) {
		char copy[8192];
		char *data = line;
		int n, i;
		while (*data && isspace((unsigned char)*data)) ++data;
		if (strncmp(line, "NAME", 4) == 0) {
			char name[4096];
			if (sscanf(line + 4, "%4095s", name) == 1) h->name = dup_text(name);
		} else if (strncmp(line, "LENG", 4) == 0) {
			if (sscanf(line + 4, "%d", &h->len) != 1 || h->len <= 0) goto fail;
			free(h->emit);
			h->emit = (int16_t*)calloc((size_t)h->len * 20, sizeof(*h->emit));
			free(h->trans);
			h->trans = (int16_t*)calloc((size_t)h->len * 7, sizeof(*h->trans));
			free(h->pssm);
			h->pssm = (int8_t*)calloc((size_t)h->len * MP_HMM_ASIZE, sizeof(*h->pssm));
			if (!h->emit || !h->trans || !h->pssm) goto fail;
		} else if (strncmp(line, "HMM", 3) == 0 && isspace((unsigned char)line[3])) {
			in_hmm = 1, have_header = 1;
		} else if (in_hmm && line[0] == '/') {
			break;
		} else if (in_hmm && is_position_line(data)) {
			if (*data == '0') {
				if (!fgets(line, sizeof(line), fp) || !fgets(line, sizeof(line), fp)) break;
				continue;
			}
			if (pos >= h->len) goto fail;
			strncpy(copy, data, sizeof(copy) - 1), copy[sizeof(copy) - 1] = 0;
			n = split(copy, tok, 64);
			if (n < 21) goto fail;
			{
				int best = -32768;
				int8_t *row = h->pssm + (size_t)pos * MP_HMM_ASIZE;
				for (i = 0; i < 20; ++i) {
					h->emit[pos * 20 + i] = (int16_t)parse_score(tok[i + 1]);
					if (h->emit[pos * 20 + i] > best) best = h->emit[pos * 20 + i];
					row[ns_tab_aa20[(uint8_t)mp_hmm_alphabet[i]]] = pssm_score(tok[i + 1], mp_hmm_bg[i]);
				}
				row[ns_tab_aa20['*']] = -MP_HMM_STOP_SC_DEFAULT;
				row[ns_tab_aa20['X']] = -1;
				/* Put the most likely residue at +4. This removes the arbitrary
				 * HMMER log-probability offset while retaining per-position contrast. */
				for (i = 0; i < 20; ++i) h->emit[pos * 20 + i] = (int16_t)(h->emit[pos * 20 + i] - best + 4);
			}
			++pos;
			/* Each match state is followed by insert emissions and transitions. */
			if (!fgets(line, sizeof(line), fp) || !fgets(line, sizeof(line), fp)) break;
			strncpy(copy, line, sizeof(copy) - 1), copy[sizeof(copy) - 1] = 0;
			n = split(copy, tok, 16);
			if (n >= 7)
				for (i = 0; i < 7; ++i) h->trans[(pos - 1) * 7 + i] = (int16_t)parse_score(tok[i]);
		}
	}
	if (!have_header && !h->emit && pos == 0) { /* clean EOF, no further record */
		mp_hmm_destroy(h);
		return 0;
	}
	if (!have_header || !h->emit || pos != h->len) goto fail;
	if (!h->name) h->name = dup_text("hmm_query");
	h->consensus = (char*)malloc((size_t)h->len + 1);
	if (!h->consensus) goto fail;
	for (pos = 0; pos < h->len; ++pos) {
		int i, best = 0;
		for (i = 1; i < 20; ++i)
			if (h->emit[pos * 20 + i] > h->emit[pos * 20 + best]) best = i;
		h->consensus[pos] = "ACDEFGHIKLMNPQRSTVWY"[best];
	}
	h->consensus[h->len] = 0;
	return h;
fail:
	f->err = 1;
	mp_hmm_destroy(h);
	return 0;
}

mp_hmm_t *mp_hmm_read(const char *fn)
{
	mp_hmm_t *h;
	mp_hmm_file_t *f = mp_hmm_open(fn);
	if (!f) return 0;
	h = mp_hmm_read1(f);
	mp_hmm_close(f);
	return h;
}

void mp_hmm_destroy(mp_hmm_t *h)
{
	if (!h) return;
	mp_hmm_gpu_release(h);
	free(h->name), free(h->consensus), free(h->emit), free(h->trans), free(h->pssm), free(h);
}

void mp_hmm_set_stop_sc(mp_hmm_t *h, int32_t pen)
{
	int32_t pos;
	int8_t sc = pen > 127? -127 : (int8_t)-pen;
	if (!h || !h->pssm) return;
	for (pos = 0; pos < h->len; ++pos)
		h->pssm[(size_t)pos * MP_HMM_ASIZE + ns_tab_aa20['*']] = sc;
}

void **mp_hmm_gpu_slot(const mp_hmm_t *h, int which) { return h? (void**)&h->gpu[which & 1] : 0; }

const char *mp_hmm_name(const mp_hmm_t *h) { return h? h->name : 0; }
const char *mp_hmm_consensus(const mp_hmm_t *h) { return h? h->consensus : 0; }
int32_t mp_hmm_length(const mp_hmm_t *h) { return h? h->len : 0; }
const int16_t *mp_hmm_emissions(const mp_hmm_t *h) { return h? h->emit : 0; }
const int16_t *mp_hmm_transitions(const mp_hmm_t *h) { return h? h->trans : 0; }
const int8_t *mp_hmm_pssm(const mp_hmm_t *h) { return h? h->pssm : 0; }

/* ns_tab_codon()/ns_tab_aa20[] index residues in ns_tab_aa_i2c order (ARNDCQEGHILKMFPSTWYV...),
 * but emit[]/h->consensus are stored/read in mp_hmm_alphabet (HMMER column) order, so any codon-derived
 * aa index must go through this table before indexing emit[]. */
static int8_t mp_hmm_aa20_to_alpha[22];
static int mp_hmm_aa20_to_alpha_ready = 0;

static void mp_hmm_init_aa20_to_alpha(void)
{
	int i;
	for (i = 0; i < 22; ++i) mp_hmm_aa20_to_alpha[i] = -1;
	for (i = 0; i < 20; ++i) mp_hmm_aa20_to_alpha[ns_tab_aa20[(uint8_t)mp_hmm_alphabet[i]]] = (int8_t)i;
	mp_hmm_aa20_to_alpha_ready = 1;
}

const int8_t *mp_hmm_aa20_to_alpha_table(void) // used by hmm-gpu.c to build a remapped codon table
{
	if (!mp_hmm_aa20_to_alpha_ready) mp_hmm_init_aa20_to_alpha();
	return mp_hmm_aa20_to_alpha;
}

int32_t mp_hmm_dp_score(const mp_hmm_t *h, const uint8_t *nt, int32_t nlen)
{
	int32_t *pm, *pi, *pd, *cm, *ci, *cd;
	int32_t i, j, tlen, best = 0;
	const int16_t *emit, *trans;
	if (!h || !nt || nlen < 3 || h->len <= 0) return 0;
	if (!mp_hmm_aa20_to_alpha_ready) mp_hmm_init_aa20_to_alpha();
	tlen = nlen / 3;
	pm = (int32_t*)calloc((size_t)h->len + 1, sizeof(*pm));
	pi = (int32_t*)calloc((size_t)h->len + 1, sizeof(*pi));
	pd = (int32_t*)calloc((size_t)h->len + 1, sizeof(*pd));
	cm = (int32_t*)calloc((size_t)h->len + 1, sizeof(*cm));
	ci = (int32_t*)calloc((size_t)h->len + 1, sizeof(*ci));
	cd = (int32_t*)calloc((size_t)h->len + 1, sizeof(*cd));
	if (!pm || !pi || !pd || !cm || !ci || !cd) goto done;
	emit = h->emit;
	trans = h->trans;
	for (i = 0; i < tlen; ++i) {
		int aa0 = (nt[i * 3] > 3 || nt[i * 3 + 1] > 3 || nt[i * 3 + 2] > 3)? 20 : ns_tab_codon[nt[i * 3] << 4 | nt[i * 3 + 1] << 2 | nt[i * 3 + 2]];
		int aa = aa0 < 20? mp_hmm_aa20_to_alpha[aa0] : -1; // -1 = stop/ambiguous, no emission
		// no per-row clearing: every cell j>=1 is overwritten below and cell 0 stays 0 from calloc
		for (j = 1; j <= h->len; ++j) {
			int32_t from = pm[j - 1] + trans[(j - 1) * 7];
			if (pi[j - 1] + trans[(j - 1) * 7 + 3] > from) from = pi[j - 1] + trans[(j - 1) * 7 + 3];
			if (pd[j - 1] + trans[(j - 1) * 7 + 5] > from) from = pd[j - 1] + trans[(j - 1) * 7 + 5];
			cm[j] = from + (aa >= 0? emit[(j - 1) * 20 + aa] : 0);
			if (cm[j] < 0) cm[j] = 0;
			ci[j] = pm[j] + trans[(j - 1) * 7 + 1] > pi[j] + trans[(j - 1) * 7 + 4]? pm[j] + trans[(j - 1) * 7 + 1] : pi[j] + trans[(j - 1) * 7 + 4];
			if (ci[j] < 0) ci[j] = 0;
			cd[j] = pm[j - 1] + trans[(j - 1) * 7 + 2] > pd[j - 1] + trans[(j - 1) * 7 + 6]? pm[j - 1] + trans[(j - 1) * 7 + 2] : pd[j - 1] + trans[(j - 1) * 7 + 6];
			if (cd[j] < 0) cd[j] = 0;
			if (cm[j] > best) best = cm[j];
			if (ci[j] > best) best = ci[j];
		}
		{ int32_t *x = pm; pm = cm; cm = x; x = pi; pi = ci; ci = x; x = pd; pd = cd; cd = x; }
	}
done:
	free(pm); free(pi); free(pd); free(cm); free(ci); free(cd);
	return best;
}

/*******************************************
 * Deferred (batched) profile scoring      *
 *******************************************/

struct mp_hmm_batch_s {
	void *km;
	const mp_mapopt_t *opt;
	int32_t qlen;
	int32_t n, m;
	mp_reg1_t **reg;
	int32_t *off, *len;
	int64_t l_nt, m_nt;
	uint8_t *nt; // regions concatenated, ready for a single device upload
};

#define MP_HMM_BATCH_MAX_NT (1LL<<28)

mp_hmm_batch_t *mp_hmm_batch_init(void *km, const mp_mapopt_t *opt, int32_t qlen)
{
	mp_hmm_batch_t *b;
	if (!opt || !opt->hmm) return 0;
	b = Kcalloc(km, mp_hmm_batch_t, 1);
	b->km = km, b->opt = opt, b->qlen = qlen;
	return b;
}

void mp_hmm_batch_flush(mp_hmm_batch_t *b)
{
	int32_t i, *score;
	if (!b || b->n == 0) return;
	score = Kmalloc(b->km, int32_t, b->n);
	if (!(b->opt->flag & MP_F_GPU) || !mp_hmm_dp_score_gpu_batch(b->opt->hmm, b->n, b->nt, b->l_nt, b->off, b->len, score))
		for (i = 0; i < b->n; ++i) score[i] = INT32_MIN;
	for (i = 0; i < b->n; ++i) {
		mp_reg1_t *r = b->reg[i];
		if (score[i] == INT32_MIN)
			score[i] = mp_hmm_dp_score(b->opt->hmm, &b->nt[b->off[i]], b->len[i]);
		r->p->profile_score = score[i];
		mp_set_rank_score(b->opt, b->qlen, r);
	}
	kfree(b->km, score);
	b->n = 0, b->l_nt = 0;
}

void mp_hmm_batch_add(mp_hmm_batch_t *b, mp_reg1_t *r, const uint8_t *nt, int32_t nlen)
{
	if (!b || !r || !r->p) return;
	if (nlen < 0) nlen = 0;
	if (b->n > 0 && b->l_nt + nlen > MP_HMM_BATCH_MAX_NT) mp_hmm_batch_flush(b);
	if (b->n == b->m) {
		b->m = b->m? b->m << 1 : 16;
		b->reg = Krealloc(b->km, mp_reg1_t*, b->reg, b->m);
		b->off = Krealloc(b->km, int32_t, b->off, b->m);
		b->len = Krealloc(b->km, int32_t, b->len, b->m);
	}
	if (b->l_nt + nlen > b->m_nt) {
		b->m_nt = b->l_nt + nlen;
		kroundup64(b->m_nt);
		b->nt = Krealloc(b->km, uint8_t, b->nt, b->m_nt);
	}
	b->reg[b->n] = r, b->off[b->n] = (int32_t)b->l_nt, b->len[b->n] = nlen;
	if (nlen > 0) memcpy(&b->nt[b->l_nt], nt, (size_t)nlen);
	b->l_nt += nlen;
	++b->n;
}

void mp_hmm_batch_destroy(mp_hmm_batch_t *b)
{
	if (!b) return;
	mp_hmm_batch_flush(b);
	kfree(b->km, b->reg); kfree(b->km, b->off); kfree(b->km, b->len); kfree(b->km, b->nt);
	kfree(b->km, b);
}
