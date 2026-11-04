/*
  The MIT License

  Copyright (c) 2022-     Dana-Farber Cancer Institute

  Permission is hereby granted, free of charge, to any person obtaining
  a copy of this software and associated documentation files (the
  "Software"), to deal in the Software without restriction, including
  without limitation the rights to use, copy, modify, merge, publish,
  distribute, sublicense, and/or sell copies of the Software, and to
  permit persons to whom the Software is furnished to do so, subject to
  the following conditions:

  The above copyright notice and this permission notice shall be
  included in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
  BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
  ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
*/
#ifndef NASW_H
#define NASW_H

#include <stdint.h>
#include <string.h>
#include "kalloc.h"

#define NS_CIGAR_M	0
#define NS_CIGAR_I	1
#define NS_CIGAR_D	2
#define NS_CIGAR_N	3  // phase-0 intron
// the following CIGAR operators are unique to nasw:
#define NS_CIGAR_F	10 // frameshift deletion
#define NS_CIGAR_G	11 // frameshift match
#define NS_CIGAR_U	12 // phase-1 intron
#define NS_CIGAR_V	13 // phase-2 intron
#define NS_CIGAR_E	14 // empty placeholder (not used)

#define NS_CIGAR_STR   "MIDNSHP=XBFGUVE"

#define NS_F_CIGAR      0x1
#define NS_F_EXT_LEFT   0x2
#define NS_F_EXT_RIGHT  0x4
#define NS_F_GPU        0x8

#define NS_S_NONE       0
#define NS_S_GENERIC    1
#define NS_S_MAMMAL     2

#define NS_SPSC_OFFSET 64

extern char *ns_tab_nt_i2c, *ns_tab_aa_i2c;
extern uint8_t ns_tab_a2r[22], ns_tab_nt4[256], ns_tab_aa20[256], ns_tab_aa13[256];
extern uint8_t ns_tab_codon[64], ns_tab_codon13[64];
extern int8_t ns_mat_blosum62[484];

typedef struct {
	int32_t flag;
	int32_t go, ge, io, fs; // gap open, extension, intron open, frameshift
	int32_t xdrop, end_bonus; // xdrop for extension, and bonus for reaching end of proteins
	int32_t asize; // alphabet size; always 22 in the current implementation
	int32_t gpu_device;
	int64_t gpu_min_cells;
	int32_t sp[6]; // 0:pos3, 1:GC-AC, 2:AT-AC, 3:other, 4:pos0, 5:poly-Y
	int32_t sp_null_bonus; // it is negative
	float ie_coef;
	const int8_t *sc; // 22x22 scoring matrix
	const int8_t *pssm; // optional al x asize position-specific scores for as[0..al-1]; NULL: use sc[]. CPU only.
	uint8_t *nt4, *aa20, *codon; // char-to-int convertion table and codon table
} ns_opt_t;

typedef struct {
	int32_t n_cigar, m_cigar; // number of cigar operators; max number
	int32_t nt_len, aa_len; // length of nt/aa sequence in the alignment
	int32_t score; // alignment score
	uint32_t *cigar; // CIGAR; allocated from the "km" memory block in ns_global_gs16()
} ns_rst_t;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Generate codon and conversion tables.
 *
 * This function should be called before invoking other nasw functions. It is
 * not thread safe.
 *
 * @param codon_type    NCBI genetic code
 *
 * @return 0 if tables were initialized; <0 if the translation table is not defined
 */
int ns_make_tables(int codon_type);

/**
 * Initialize alignment parameters
 *
 * @param opt      options to initialize
 */
void ns_opt_init(ns_opt_t *opt);

/**
 * Aligning a protein sequence to a nucleotide sequence with splicing and frameshift
 *
 * This function implements striped dynamic programming (DP) using SSE2 on
 * x86_64 or NEON on ARM. On AVX2-capable x86_64 CPUs, 16-bit extension DP is
 * dispatched to a 256-bit kernel automatically. It supports global alignment
 * and left or right extension. In the extension mode, only extension ends and
 * alignment score are computed. Users need to call the function again to get
 * CIGAR.
 *
 * This function only works when the max score is not larger than 32767. If the
 * max score is larger, the 32-bit version ns_global_gs32() should be used
 * instead. ns_global_gs32() is twice as slow and uses twice as much memory and
 * it does not support extension alignment or end_bonus yet.
 *
 * The initial conditions of the DP are not technically correct. This issue may
 * affect gap openings at the beginning of the input sequences. The rest of
 * alignment should be ok.
 *
 * @param km        kalloc allocator; NULL to use the system malloc()
 * @param ns        the nucleotide sequence; can be normal char string or 0/1/2/3/4 encoding
 * @param nl        length of the nucleotide sequence
 * @param as        the protein sequence
 * @param al        the length of the protein sequence
 * @param opt       alignment settings and parameters
 * @param rst       (out) the result
 */
void ns_global_gs16(void *km, const char *ns, int32_t nl, const char *as, int32_t al, const ns_opt_t *opt, ns_rst_t *r);
void ns_global_gs32(void *km, const char *ns, int32_t nl, const char *as, int32_t al, const ns_opt_t *opt, ns_rst_t *r);

void ns_global_gs16b(void *km, const char *ns, int32_t nl, const char *as, int32_t al, const ns_opt_t *opt, const uint8_t *ss, ns_rst_t *r);
void ns_global_gs32b(void *km, const char *ns, int32_t nl, const char *as, int32_t al, const ns_opt_t *opt, const uint8_t *ss, ns_rst_t *r);
void ns_global_gs_auto(void *km, const char *ns, int32_t nl, const char *as, int32_t al, const ns_opt_t *opt, const uint8_t *ss, ns_rst_t *r);
typedef struct ns_gpu_request_s ns_gpu_request_t;
int ns_gpu_init(int32_t device);
const char *ns_gpu_error(void);
const char *ns_gpu_name(void);
uint64_t ns_gpu_dispatch_count(void);
uint64_t ns_gpu_launch_count(void);
double ns_gpu_h2d_seconds(void);
double ns_gpu_kernel_seconds(void);
double ns_gpu_d2h_seconds(void);
/* A successful submission must be consumed exactly once; km must remain valid through ns_gpu_wait(). */
ns_gpu_request_t *ns_gpu_submit(void *km, const char *ns, int32_t nl, const char *as, int32_t al, const ns_opt_t *opt, const uint8_t *ss);
int ns_gpu_wait(ns_gpu_request_t *request, ns_rst_t *r);
int ns_gpu_global(void *km, const char *ns, int32_t nl, const char *as, int32_t al, const ns_opt_t *opt, const uint8_t *ss, ns_rst_t *r);

// GPU gapless X-drop candidate prefilter (--gpu-prefilter; CUDA backend only, see nasw-cuda.cu). One
// candidate is one coarse-chain locus; up to 8 anchors seed independent gapless exon segments that are
// summed after removing query overlap (see nasw-cuda.cu ns_gapless_xdrop_batch for the exact scoring).
typedef struct {
	int64_t target_offset, target_length; // contig offset/length in mp_ntdb_t::seq (4-bit packed), oriented per `reverse`
	int64_t query_offset; // start of this candidate's query in the caller's packed `queries` buffer
	int32_t query_length;
	int32_t anchor_offset, anchor_count; // into the caller's `anchors` array
	int32_t reverse;
} ns_gpu_gapless_candidate_t;
typedef struct {
	int64_t target_block; // oriented nt position of the anchor's last codon base, relative to target_offset
	int32_t query_end; // last query aa position (0-based) covered by the anchor k-mer
	int32_t reserved;
} ns_gpu_gapless_anchor_t;
typedef struct {
	uint64_t batches, candidates, anchors, seed_positions;
} ns_gpu_gapless_profile_t;
void ns_gpu_gapless_profile(ns_gpu_gapless_profile_t *profile);
int ns_gpu_gapless_supported(void); // 1 only when built with 'make cuda=1'; OpenCL never implemented this kernel
void ns_gpu_gapless_reset_target(void); // forces the next score_batch call to re-upload the genome buffer
// packed_nt: mp_ntdb_t::seq (4-bit packed, 2 bases/byte); queries: aa20-coded protein residues, one buffer
// shared by every candidate (candidate::query_offset indexes into it); scores[i] receives candidate i's
// gapless score, or INT32_MIN if no anchor produced a usable seed. Returns 0 when the GPU path failed/unavailable.
int ns_gpu_gapless_score_batch(const uint8_t *packed_nt, int64_t packed_nt_bases,
		const uint8_t *queries, int64_t query_aa,
		const ns_gpu_gapless_candidate_t *candidates, int32_t n_candidates,
		const ns_gpu_gapless_anchor_t *anchors, int64_t n_anchors,
		int32_t block_width, int32_t kmer, int32_t xdrop,
		const int8_t *score_matrix, int32_t asize, const uint8_t *codon,
		int32_t *scores);

void ns_opt_set_sp(ns_opt_t *opt, int32_t model);

// Building blocks of the 16-bit extension DP, exposed so a device kernel can reproduce it bit-for-bit.
// ns_ext_prep() returns one km block holding nas[nl], aas[al], donor[nl+1] and acceptor[nl+1] (left-extension
// inputs come back reversed exactly as the CPU kernel sees them); ns_ext_stripe_width() is the SIMD lane
// count the CPU kernel would use (the padded columns it implies can affect the row maximum).
uint8_t *ns_ext_prep(void *km, const char *ns, int32_t nl, const char *as, int32_t al, const ns_opt_t *opt, const uint8_t *ss, uint8_t **aas, int8_t **donor, int8_t **acceptor);
int32_t ns_ext_stripe_width(const ns_opt_t *opt, int32_t al);
int32_t ns_ext_ie_penalty(const ns_opt_t *opt, int32_t d);

// Batched extension DP (ns_global_gs16b with NS_F_EXT_LEFT/RIGHT, no CIGAR). Requests are collected and
// handed to the device together at flush time (input preparation and DP both run on the device; CPU
// fallback for anything the device can't take). ns_ext_batch_flush() does not wait: results are fetched
// per request with ns_ext_batch_get(), which blocks until that request's flush has completed;
// ns_ext_batch_ready() polls without blocking and ns_ext_batch_wait() sleeps until some outstanding flush
// completes, so a caller can keep several flushes in flight. Input pointers must stay valid until the
// request's result has been fetched. Results equal ns_global_gs16b() exactly.
typedef struct ns_ext_batch_s ns_ext_batch_t;
ns_ext_batch_t *ns_ext_batch_init(void *km); // NULL when no GPU backend is available
int32_t ns_ext_batch_add(ns_ext_batch_t *b, const char *ns, int32_t nl, const char *as, int32_t al, const ns_opt_t *opt, const uint8_t *ss);
int32_t ns_ext_batch_flush(ns_ext_batch_t *b); // returns the number of requests submitted
int ns_ext_batch_ready(ns_ext_batch_t *b, int32_t id);
void ns_ext_batch_wait(ns_ext_batch_t *b);
const ns_rst_t *ns_ext_batch_get(ns_ext_batch_t *b, int32_t id); // valid until a flush after every result of that generation was fetched
void ns_ext_batch_destroy(ns_ext_batch_t *b);
uint64_t ns_ext_gpu_launch_count(void);
uint64_t ns_ext_gpu_request_count(void);

// Non-SIMD implementation with transposed DP matrix. For debugging only. Not intended for general developers.
void ns_splice_s1(void *km, const char *ns, int32_t nl, const char *as, int32_t al, const ns_opt_t *opt, ns_rst_t *r);

void ns_set_stop_sc(int32_t asize, int8_t *mat, int8_t score);

static inline uint32_t *ns_push_cigar(void *km, int32_t *n_cigar, int32_t *m_cigar, uint32_t *cigar, uint32_t op, int32_t len)
{
	if (*n_cigar == 0 || op != (cigar[(*n_cigar) - 1]&0xf) || op == NS_CIGAR_F || op == NS_CIGAR_G) {
		if (*n_cigar == *m_cigar) {
			(*m_cigar) += ((*m_cigar)>>1) + 8;
			cigar = Krealloc(km, uint32_t, cigar, *m_cigar);
		}
		cigar[(*n_cigar)++] = len<<4 | op;
	} else cigar[(*n_cigar)-1] += len<<4;
	return cigar;
}

static inline void ns_rst_init(ns_rst_t *r)
{
	memset(r, 0, sizeof(*r));
}

#ifdef __cplusplus
}
#endif

#endif
