#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "nasw.h"

static void test_cigar_equivalence(void)
{
	const char nt[] = "ATGAAAACTTGG";
	const char aa[] = "MKQTW";
	int32_t i, has_insertion = 0;
	ns_opt_t opt;
	ns_rst_t result16, result32;

	ns_opt_init(&opt);
	opt.flag |= NS_F_CIGAR;
	opt.go = 3, opt.ge = 1, opt.io = 100, opt.fs = 100;
	ns_opt_set_sp(&opt, NS_S_NONE);
	ns_rst_init(&result16);
	ns_rst_init(&result32);
	ns_global_gs16b(0, nt, (int32_t)strlen(nt), aa, (int32_t)strlen(aa), &opt, 0, &result16);
	ns_global_gs32b(0, nt, (int32_t)strlen(nt), aa, (int32_t)strlen(aa), &opt, 0, &result32);
	assert(result16.score == result32.score);
	assert(result16.n_cigar == result32.n_cigar);
	assert(memcmp(result16.cigar, result32.cigar, result16.n_cigar * sizeof(uint32_t)) == 0);
	for (i = 0; i < result16.n_cigar; ++i)
		has_insertion |= (result16.cigar[i]&0xf) == NS_CIGAR_I;
	assert(has_insertion);
	free(result16.cigar);
	free(result32.cigar);
}

static void test_extension_exact(int32_t flag, int32_t aa_len)
{
	char nt[19 * 3 + 1], aa[19 + 1];
	int32_t i, match_score;
	ns_opt_t opt;
	ns_rst_t result;

	assert(aa_len <= 19);
	for (i = 0; i < aa_len; ++i) {
		nt[i * 3] = 'T', nt[i * 3 + 1] = 'G', nt[i * 3 + 2] = 'G';
		aa[i] = 'W';
	}
	nt[aa_len * 3] = aa[aa_len] = 0;
	ns_opt_init(&opt);
	opt.flag |= flag;
	ns_rst_init(&result);
	ns_global_gs16b(0, nt, aa_len * 3, aa, aa_len, &opt, 0, &result);
	match_score = ns_mat_blosum62[ns_tab_aa20['W'] * opt.asize + ns_tab_aa20['W']];
	assert(result.score == aa_len * match_score + opt.end_bonus);
	assert(result.nt_len == aa_len * 3);
	assert(result.aa_len == aa_len);
}

int main(void)
{
	const int32_t aa_len = 3000;
	int32_t i;
	char *nt = (char*)malloc((size_t)aa_len * 3);
	char *aa = (char*)malloc((size_t)aa_len);
	ns_opt_t opt;
	ns_rst_t result;

	assert(nt != 0 && aa != 0);
	assert(ns_make_tables(1) == 0);
	test_cigar_equivalence();
	test_extension_exact(NS_F_EXT_LEFT, 7);
	test_extension_exact(NS_F_EXT_RIGHT, 7);
	test_extension_exact(NS_F_EXT_LEFT, 19);
	test_extension_exact(NS_F_EXT_RIGHT, 19);
	for (i = 0; i < aa_len; ++i) {
		nt[i * 3] = 'T';
		nt[i * 3 + 1] = 'G';
		nt[i * 3 + 2] = 'G';
		aa[i] = 'W';
	}
	ns_opt_init(&opt);
	opt.flag |= NS_F_CIGAR;
	ns_rst_init(&result);
	ns_global_gs_auto(0, nt, aa_len * 3, aa, aa_len, &opt, 0, &result);
	assert(result.score == aa_len * ns_mat_blosum62[ns_tab_aa20['W'] * 22 + ns_tab_aa20['W']]);
	assert(result.score > INT16_MAX);
	assert(result.n_cigar == 1);
	assert((result.cigar[0] & 0xf) == NS_CIGAR_M);
	assert((result.cigar[0] >> 4) == aa_len);
	free(result.cigar);
	free(aa);
	free(nt);
	return 0;
}