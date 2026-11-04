#include <assert.h>
#include <stdio.h>
#include "mppriv.h"
#include "kalloc.h"
#include "kvec-km.h"

static inline uint32_t mp_hash32_mask(uint32_t key, uint32_t mask)
{
	key  = (key + ~(key << 15)) & mask;
	key ^=  (key >> 10);
	key  =  (key + (key <<  3)) & mask;
	key ^=  (key >> 6);
	key  = (key + ~(key << 11)) & mask;
	key ^=  (key >> 16);
	return key;
}

void mp_sketch_prot(void *km, const char *seq, int32_t len, int32_t kmer, int32_t mod_bit, mp64_v *a)
{
	int32_t i, l = 0;
	uint32_t x = 0, y, mask_k = (1U<<kmer*MP_BITS_PER_AA) - 1, mask_mod = (1U<<mod_bit) - 1;
	a->n = 0;
	for (i = 0; i < len; ++i) {
		int32_t c = ns_tab_aa13[(uint8_t)seq[i]];
		if (c < 14) {
			#if MP_BITS_PER_AA == 4
			x = (x<<MP_BITS_PER_AA | c) & mask_k;
			#else
			x = (x<<MP_BITS_PER_AA | c>>1) & mask_k;
			#endif
			if (++l >= kmer) {
				y = mp_hash32_mask(x, mask_k);
				if ((y&mask_mod) == 0)
					kv_push(uint64_t, km, *a, (uint64_t)(y>>mod_bit)<<32 | i);
			}
		} else x = 0, l = 0;
	}
}

#define MP_HMM_MAX_ALT 4

/* Profile-aware seeds: for every k-window of match states, enumerate reduced-alphabet k-mers whose
 * summed emission loss against the per-position best group (units of 2*ln p) stays within win_loss;
 * each position offers at most MP_HMM_MAX_ALT groups within pos_loss of its best. Appends to a[]
 * without resetting it; the caller must sort and deduplicate. */
void mp_sketch_hmm(void *km, const int16_t *emit, int32_t len, int32_t kmer, int32_t mod_bit, int32_t pos_loss, int32_t win_loss, int32_t max_per_win, mp64_v *a)
{
	static const char aa[] = "ACDEFGHIKLMNPQRSTVWY";
	int32_t i, j, g;
	uint8_t *alt_code, *n_alt;
	int16_t *alt_loss;
	uint32_t mask_k = (1U<<kmer*MP_BITS_PER_AA) - 1, mask_mod = (1U<<mod_bit) - 1;

	if (!emit || len < kmer || max_per_win <= 0) return;
	alt_code = Kmalloc(km, uint8_t, (size_t)len * MP_HMM_MAX_ALT);
	alt_loss = Kmalloc(km, int16_t, (size_t)len * MP_HMM_MAX_ALT);
	n_alt = Kmalloc(km, uint8_t, len);
	for (i = 0; i < len; ++i) {
		int32_t gs[14], best = INT16_MIN, n = 0;
		for (g = 0; g < 14; ++g) gs[g] = INT16_MIN;
		for (j = 0; j < 20; ++j) {
			g = ns_tab_aa13[(uint8_t)aa[j]];
			if (emit[i * 20 + j] > gs[g]) gs[g] = emit[i * 20 + j];
		}
		for (g = 0; g < 14; ++g) if (gs[g] > best) best = gs[g];
		for (g = 0; g < 14; ++g) { // insertion-sort groups by loss, keep the closest MP_HMM_MAX_ALT
			int32_t loss = best - gs[g], p;
			if (gs[g] == INT16_MIN || loss > pos_loss) continue;
			if (n == MP_HMM_MAX_ALT && loss >= alt_loss[i * MP_HMM_MAX_ALT + n - 1]) continue;
			if (n < MP_HMM_MAX_ALT) ++n;
			for (p = n - 1; p > 0 && alt_loss[i * MP_HMM_MAX_ALT + p - 1] > loss; --p)
				alt_loss[i * MP_HMM_MAX_ALT + p] = alt_loss[i * MP_HMM_MAX_ALT + p - 1], alt_code[i * MP_HMM_MAX_ALT + p] = alt_code[i * MP_HMM_MAX_ALT + p - 1];
			alt_loss[i * MP_HMM_MAX_ALT + p] = loss, alt_code[i * MP_HMM_MAX_ALT + p] = g;
		}
		n_alt[i] = n;
	}
	for (i = kmer - 1; i < len; ++i) {
		int32_t st = i - kmer + 1, d = 0, n_emit = 0;
		int32_t idx[16], acc[17]; // per-level alternative index and cumulative loss; kmer <= 8 in practice
		if (kmer > 16) break;
		idx[0] = 0, acc[0] = 0;
		while (d >= 0) {
			int32_t p = st + d, k = idx[d];
			if (k >= n_alt[p] || acc[d] + alt_loss[p * MP_HMM_MAX_ALT + k] > win_loss) {
				if (--d >= 0) ++idx[d];
				continue;
			}
			acc[d + 1] = acc[d] + alt_loss[p * MP_HMM_MAX_ALT + k];
			if (d + 1 == kmer) {
				uint32_t x = 0, y;
				for (j = 0; j < kmer; ++j) {
					#if MP_BITS_PER_AA == 4
					x = (x<<MP_BITS_PER_AA | alt_code[(st + j) * MP_HMM_MAX_ALT + idx[j]]) & mask_k;
					#else
					x = (x<<MP_BITS_PER_AA | alt_code[(st + j) * MP_HMM_MAX_ALT + idx[j]]>>1) & mask_k;
					#endif
				}
				y = mp_hash32_mask(x, mask_k);
				if ((y&mask_mod) == 0)
					kv_push(uint64_t, km, *a, (uint64_t)(y>>mod_bit)<<32 | i);
				if (++n_emit >= max_per_win) break;
				++idx[d];
			} else ++d, idx[d] = 0;
		}
	}
	kfree(km, alt_code), kfree(km, alt_loss), kfree(km, n_alt);
}

void mp_sketch_clean_orf(void *km, const uint8_t *seq, int64_t st, int64_t en, int32_t kmer, int32_t mod_bit, int32_t bbit, int64_t boff, mp64_v *a)
{
	int64_t i;
	int32_t l;
	uint32_t x, y, mask_k = (1U<<kmer*MP_BITS_PER_AA) - 1, mask_mod = (1U<<mod_bit) - 1;
	for (i = st, l = 0, x = 0; i < en; i += 3) {
		uint8_t codon = seq[i]<<4 | seq[i+1]<<2 | seq[i+2];
		assert(seq[i] < 4 && seq[i+1] < 4 && seq[i+2] < 4);
		#if MP_BITS_PER_AA == 4
		x = (x<<MP_BITS_PER_AA | ns_tab_codon13[codon]) & mask_k;
		#else
		x = (x<<MP_BITS_PER_AA | ns_tab_codon13[codon]>>1) & mask_k;
		#endif
		if (++l >= kmer) {
			y = mp_hash32_mask(x, mask_k);
			assert(y <= mask_k);
			if ((y&mask_mod) == 0)
				kv_push(uint64_t, km, *a, (uint64_t)(y>>mod_bit)<<32 | (((i+2)>>bbit) + boff));
		}
	}
}

void mp_sketch_nt4(void *km, const uint8_t *seq, int64_t len, int32_t min_aa_len, int32_t kmer, int32_t mod_bit, int32_t bbit, int64_t boff, mp64_v *a)
{
	uint8_t codon, p, q;
	int64_t i, j, l, e[3], k[3];
	kv_resize(uint64_t, km, *a, len>>2);
	a->n = 0;
	for (p = 0; p < 3; ++p)
		e[p] = -1, k[p] = 0;
	for (i = 0, p = 1, codon = 0, l = 0; i < len; ++i, ++p) {
		if (p == 3) p = 0;
		if (seq[i] < 4) {
			codon = (codon << 2 | seq[i]) & 0x3f;
			if (++l >= 3) {
				uint8_t aa = ns_tab_codon[codon];
				if (aa >= 20) {
					if (k[p] >= min_aa_len)
						mp_sketch_clean_orf(km, seq, e[p] + 1 - k[p] * 3, e[p] + 1, kmer, mod_bit, bbit, boff, a);
					k[p] = 0, e[p] = -1;
				} else e[p] = i, ++k[p];
			}
		} else {
			for (q = 0; q < 3; ++q) {
				if (k[q] >= min_aa_len)
					mp_sketch_clean_orf(km, seq, e[q] + 1 - k[q] * 3, e[q] + 1, kmer, mod_bit, bbit, boff, a);
				k[q] = 0, e[q] = -1;
			}
			l = 0, codon = 0;
		}
	}
	for (p = 0; p < 3; ++p)
		if (k[p] >= min_aa_len)
			mp_sketch_clean_orf(km, seq, e[p] + 1 - k[p] * 3, e[p] + 1, kmer, mod_bit, bbit, boff, a);
	if (a->n <= 1) return; // the following loop doesn't work when a->n == 0
	radix_sort_mp64(a->a, a->a + a->n);
	for (i = 1, j = 0; i < a->n; ++i) // squeeze out identical entries
		if (a->a[j] != a->a[i])
			a->a[++j] = a->a[i];
	a->n = ++j;
}
