#define _POSIX_C_SOURCE 200809L
#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "nasw.h"
#include "protlocus.h"
#include "mppriv.h"

#define GPU_THREADS 32
#define GPU_ASYNC_REQUESTS 129

typedef struct {
	const char *nt, *aa;
	int32_t nt_len, aa_len;
	const ns_opt_t *opt;
	ns_rst_t result;
	int success;
} gpu_thread_t;

static pthread_mutex_t start_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t start_condition = PTHREAD_COND_INITIALIZER;
static int ready_threads, start_threads;

static void print_cigar(const char *label, const uint32_t *cigar, int32_t n_cigar)
{
	static const char ops[] = "MIDNSHP=XBFGUVE";
	int32_t i;
	fprintf(stderr, "%s:", label);
	for (i = 0; i < n_cigar; ++i)
		fprintf(stderr, " %d%c", cigar[i]>>4, ops[cigar[i]&0xf]);
	fputc('\n', stderr);
}

static void assert_same_result(const ns_rst_t *expected, const ns_rst_t *actual)
{
	assert(actual->nt_len == expected->nt_len);
	assert(actual->aa_len == expected->aa_len);
	assert(actual->score == expected->score);
	assert(actual->n_cigar == expected->n_cigar);
	if (memcmp(actual->cigar, expected->cigar, (size_t)expected->n_cigar * sizeof(*expected->cigar)) != 0) {
		print_cigar("expected", expected->cigar, expected->n_cigar);
		print_cigar("actual  ", actual->cigar, actual->n_cigar);
	}
	assert(memcmp(actual->cigar, expected->cigar,
			(size_t)expected->n_cigar * sizeof(*expected->cigar)) == 0);
}

static void test_gpu_tile_boundaries(void)
{
	static const struct {
		int32_t nt_len, aa_len;
	} cases[] = {
		{ 3, 1 }, { 9, 3 }, { 12, 4 }, { 15, 5 }, { 18, 6 },
		{ 21, 7 }, { 24, 8 }, { 27, 9 },
		{ 45, 15 }, { 48, 16 }, { 51, 17 }
	};
	static const char codon[] = "TGG";
	char nt[51], aa[17];
	ns_opt_t opt;
	int32_t case_index, position;

	ns_opt_init(&opt);
	opt.flag |= NS_F_CIGAR;
	for (case_index = 0; case_index < (int32_t)(sizeof(cases) / sizeof(cases[0])); ++case_index) {
		ns_rst_t expected, actual;
		int32_t nt_len = cases[case_index].nt_len, aa_len = cases[case_index].aa_len;
		int gpu_ok;
		for (position = 0; position < nt_len; ++position)
			nt[position] = codon[position % 3];
		memset(aa, 'W', aa_len);
		ns_rst_init(&expected);
		ns_rst_init(&actual);
		ns_global_gs16b(0, nt, nt_len, aa, aa_len, &opt, 0, &expected);
		gpu_ok = ns_gpu_global(0, nt, nt_len, aa, aa_len, &opt, 0, &actual);
		if (!gpu_ok)
			fprintf(stderr, "GPU call failed at case_index=%d nt_len=%d aa_len=%d: %s\n", case_index, nt_len, aa_len, ns_gpu_error());
		assert(gpu_ok);
		if (expected.score != actual.score || expected.n_cigar != actual.n_cigar)
			fprintf(stderr, "GPU mismatch at case_index=%d nt_len=%d aa_len=%d: score=%d/%d cigar=%d/%d\n",
				case_index, nt_len, aa_len, expected.score, actual.score, expected.n_cigar, actual.n_cigar);
		fprintf(stderr, "checking case_index=%d nt_len=%d aa_len=%d\n", case_index, nt_len, aa_len);
		assert_same_result(&expected, &actual);
		free(expected.cigar);
		free(actual.cigar);
	}
}

static void test_32bit_fallback(void)
{
	const int32_t aa_len = 300;
	char *nt = (char*)malloc((size_t)aa_len * 3);
	char *aa = (char*)malloc((size_t)aa_len);
	int8_t matrix[484];
	int32_t i, tryptophan = ns_tab_aa20['W'];
	ns_opt_t opt;
	ns_rst_t expected, actual;
	uint64_t dispatches;

	assert(nt != 0 && aa != 0);
	for (i = 0; i < aa_len; ++i) {
		nt[i * 3] = 'T', nt[i * 3 + 1] = 'G', nt[i * 3 + 2] = 'G';
		aa[i] = 'W';
	}
	memcpy(matrix, ns_mat_blosum62, sizeof(matrix));
	matrix[tryptophan * 22 + tryptophan] = 127;
	ns_opt_init(&opt);
	opt.flag |= NS_F_CIGAR | NS_F_GPU;
	opt.gpu_min_cells = 0;
	opt.sc = matrix;
	ns_rst_init(&expected);
	ns_rst_init(&actual);
	ns_global_gs32b(0, nt, aa_len * 3, aa, aa_len, &opt, 0, &expected);
	dispatches = ns_gpu_dispatch_count();
	ns_global_gs_auto(0, nt, aa_len * 3, aa, aa_len, &opt, 0, &actual);
	assert(ns_gpu_dispatch_count() == dispatches);
	assert_same_result(&expected, &actual);
	free(expected.cigar);
	free(actual.cigar);
	free(aa);
	free(nt);
}

static void *run_gpu_alignment(void *data)
{
	gpu_thread_t *thread = (gpu_thread_t*)data;
	pthread_mutex_lock(&start_mutex);
	++ready_threads;
	pthread_cond_broadcast(&start_condition);
	while (!start_threads)
		pthread_cond_wait(&start_condition, &start_mutex);
	pthread_mutex_unlock(&start_mutex);
	ns_rst_init(&thread->result);
	thread->success = ns_gpu_global(0, thread->nt, thread->nt_len,
			thread->aa, thread->aa_len, thread->opt, 0, &thread->result);
	return 0;
}

static void test_gpu_batch(void)
{
	const char nt[] = "ATGAAAACTTGG";
	const char aa[] = "MKQTW";
	pthread_t threads[GPU_THREADS];
	gpu_thread_t data[GPU_THREADS];
	ns_opt_t opt;
	ns_rst_t expected;
	uint64_t dispatches, launches;
	int32_t i;

	ns_opt_init(&opt);
	opt.flag |= NS_F_CIGAR;
	opt.go = 3, opt.ge = 1, opt.io = 100, opt.fs = 100;
	ns_opt_set_sp(&opt, NS_S_NONE);
	ns_rst_init(&expected);
	ns_global_gs16b(0, nt, (int32_t)strlen(nt), aa, (int32_t)strlen(aa), &opt, 0, &expected);

	memset(data, 0, sizeof(data));
	for (i = 0; i < GPU_THREADS; ++i) {
		data[i].nt = nt, data[i].aa = aa;
		data[i].nt_len = (int32_t)strlen(nt), data[i].aa_len = (int32_t)strlen(aa);
		data[i].opt = &opt;
		assert(pthread_create(&threads[i], 0, run_gpu_alignment, &data[i]) == 0);
	}
	pthread_mutex_lock(&start_mutex);
	while (ready_threads != GPU_THREADS)
		pthread_cond_wait(&start_condition, &start_mutex);
	dispatches = ns_gpu_dispatch_count();
	launches = ns_gpu_launch_count();
	start_threads = 1;
	pthread_cond_broadcast(&start_condition);
	pthread_mutex_unlock(&start_mutex);

	for (i = 0; i < GPU_THREADS; ++i) {
		assert(pthread_join(threads[i], 0) == 0);
		assert(data[i].success);
		assert_same_result(&expected, &data[i].result);
		free(data[i].result.cigar);
	}
	assert(ns_gpu_dispatch_count() - dispatches == GPU_THREADS);
	assert(ns_gpu_launch_count() > launches);
	assert(ns_gpu_launch_count() - launches < GPU_THREADS);
	free(expected.cigar);
}

static void test_gpu_async(void)
{
	const char nt[] = "ATGAAAACTTGG";
	const char aa[] = "MKQTW";
	ns_gpu_request_t *requests[GPU_ASYNC_REQUESTS];
	ns_opt_t opt;
	ns_rst_t expected;
	uint64_t dispatches, launches;
	int32_t i;

	ns_opt_init(&opt);
	opt.flag |= NS_F_CIGAR;
	opt.go = 3, opt.ge = 1, opt.io = 100, opt.fs = 100;
	ns_opt_set_sp(&opt, NS_S_NONE);
	ns_rst_init(&expected);
	ns_global_gs16b(0, nt, (int32_t)strlen(nt), aa, (int32_t)strlen(aa), &opt, 0, &expected);
	dispatches = ns_gpu_dispatch_count();
	launches = ns_gpu_launch_count();
	for (i = 0; i < GPU_ASYNC_REQUESTS; ++i) {
		requests[i] = ns_gpu_submit(0, nt, (int32_t)strlen(nt), aa, (int32_t)strlen(aa), &opt, 0);
		assert(requests[i] != 0);
	}
	for (i = 0; i < GPU_ASYNC_REQUESTS; ++i) {
		ns_rst_t actual;
		ns_rst_init(&actual);
		assert(ns_gpu_wait(requests[i], &actual));
		assert_same_result(&expected, &actual);
		free(actual.cigar);
	}
	assert(ns_gpu_dispatch_count() - dispatches == GPU_ASYNC_REQUESTS);
	assert(ns_gpu_launch_count() - launches >= 2);
	free(expected.cigar);
}

// Two tiny HMMER3-style profiles in one file: exercises the streaming reader and
// checks the OpenCL profile DP agrees with the CPU Viterbi on random DNA.
static void write_profile(FILE *fp, const char *name, int len, unsigned seed)
{
	int i, j;
	fprintf(fp, "HMMER3/f [test]\nNAME  %s\nLENG  %d\nALPH  amino\nHMM          A        C        D        E        F        G        H        I        K        L        M        N        P        Q        R        S        T        V        W        Y\n", name, len);
	fprintf(fp, "            m->m     m->i     m->d     i->m     i->i     d->m     d->d\n");
	fprintf(fp, "  COMPO   2.5 2.5 2.5 2.5 2.5 2.5 2.5 2.5 2.5 2.5 2.5 2.5 2.5 2.5 2.5 2.5 2.5 2.5 2.5 2.5\n");
	fprintf(fp, "          2.5 2.5 2.5 2.5 2.5 2.5 2.5 2.5 2.5 2.5 2.5 2.5 2.5 2.5 2.5 2.5 2.5 2.5 2.5 2.5\n");
	fprintf(fp, "          0.02 4.5 5.0 0.6 0.8 0.0 *\n");
	for (i = 1; i <= len; ++i) {
		fprintf(fp, "%7d", i);
		for (j = 0; j < 20; ++j) {
			seed = seed * 1103515245u + 12345u;
			fprintf(fp, " %.3f", 0.5 + (seed >> 16) % 4000 / 1000.0);
		}
		fprintf(fp, " %d - - -\n", i);
		fprintf(fp, "          2.5 2.5 2.5 2.5 2.5 2.5 2.5 2.5 2.5 2.5 2.5 2.5 2.5 2.5 2.5 2.5 2.5 2.5 2.5 2.5\n");
		fprintf(fp, "          0.02 4.5 5.0 0.6 0.8 0.5 1.0\n");
	}
	fprintf(fp, "//\n");
}

static void test_hmm_gpu_matches_cpu(void)
{
	char path[] = "/tmp/protlocus_test_XXXXXX";
	int fd = mkstemp(path), i, n = 0;
	FILE *fp;
	mp_hmm_file_t *hf;
	mp_hmm_t *h;
	uint8_t nt[3000];
	unsigned seed = 7;
	assert(fd >= 0);
	fp = fdopen(fd, "w");
	assert(fp);
	write_profile(fp, "tiny_a", 37, 1);
	write_profile(fp, "tiny_b", 300, 2);
	fclose(fp);
	for (i = 0; i < (int)sizeof(nt); ++i) seed = seed * 1103515245u + 12345u, nt[i] = (seed >> 16) & 3;
	hf = mp_hmm_open(path);
	assert(hf);
	while ((h = mp_hmm_read1(hf)) != 0) {
		int32_t cpu = mp_hmm_dp_score(h, nt, (int32_t)sizeof(nt));
		int32_t gpu = mp_hmm_dp_score_gpu(h, nt, (int32_t)sizeof(nt));
		assert(mp_hmm_length(h) == (n == 0? 37 : 300));
		assert(strcmp(mp_hmm_name(h), n == 0? "tiny_a" : "tiny_b") == 0);
		if (gpu == INT32_MIN) fprintf(stderr, "[SKIP] HMM profile DP fell back to CPU for %s\n", mp_hmm_name(h));
		else assert(gpu == cpu);
		// second call reuses the cached device buffers
		assert(mp_hmm_dp_score_gpu(h, nt, (int32_t)sizeof(nt)) == gpu);
		if (gpu != INT32_MIN) { // many regions of varying length in one launch, incl. an empty one
			int32_t off[5], len[5], score[5], k;
			for (k = 0, off[0] = 0; k < 5; ++k) {
				len[k] = k == 3? 0 : 3 * (61 * (k + 1) + (k == 4? 2 : 0));
				if (k) off[k] = off[k - 1] + len[k - 1];
			}
			len[4] += 2;
			assert(off[4] + len[4] <= (int32_t)sizeof(nt));
			assert(mp_hmm_dp_score_gpu_batch(h, 5, nt, off[4] + len[4], off, len, score));
			for (k = 0; k < 5; ++k)
				assert(score[k] == mp_hmm_dp_score(h, &nt[off[k]], len[k]));
		}
		mp_hmm_destroy(h);
		++n;
	}
	assert(n == 2);
	assert(!mp_hmm_error(hf));
	mp_hmm_close(hf);
	unlink(path);
}

// Batched end-extension DP: many left/right requests of assorted sizes (SSE and AVX2 stripe widths,
// with/without splice-score tracks, both intron penalties) in one flush must equal ns_global_gs16b().
static void test_ext_gpu_matches_cpu(void)
{
	enum { N_REQ = 42, MAX_NT = 30000, MAX_AA = 700 };
	static const int32_t aa_lens[] = { 1, 3, 7, 15, 16, 17, 31, 33, 64, 100, 129, 257, 300, 511, 640 };
	char *nt = (char*)malloc(MAX_NT), *aa = (char*)malloc(MAX_AA);
	uint8_t *ss = (uint8_t*)malloc(MAX_NT);
	struct { const char *ns, *as; const uint8_t *ss; int32_t nl, al; ns_opt_t opt; } req[N_REQ];
	ns_rst_t expected[N_REQ];
	ns_ext_batch_t *b;
	unsigned seed = 11;
	uint64_t n_gpu;
	int32_t i, k, grp, n_dev, id[N_REQ];

	assert(nt && aa && ss);
	setenv("NS_EXT_GPU_STRICT", "1", 1);
	for (i = 0; i < MAX_NT; ++i) seed = seed * 1103515245u + 12345u, nt[i] = "ACGT"[(seed >> 16) & 3];
	for (i = 0; i < MAX_AA; ++i) seed = seed * 1103515245u + 12345u, aa[i] = "ACDEFGHIKLMNPQRSTVWY"[(seed >> 16) % 20];
	for (i = 0; i < MAX_NT; ++i) { // a mix of unset, donor and acceptor scores like a real --spsc track
		seed = seed * 1103515245u + 12345u;
		ss[i] = (seed >> 16) % 5 == 0? (uint8_t)((((seed >> 20) % 40) + 44) << 1 | ((seed >> 27) & 1)) : 0xff;
	}
	// plant a translated copy of aa[] so extensions actually find something to align to
	for (k = 0; k < 400 && k * 3 + 2 < MAX_NT; ++k) {
		static const char *codon[20] = { "GCT","TGT","GAT","GAA","TTT","GGT","CAT","ATT","AAA","TTA","ATG","AAT","CCT","CAA","CGT","TCT","ACT","GTT","TGG","TAT" };
		const char *c = codon[(int)(strchr("ACDEFGHIKLMNPQRSTVWY", aa[k]) - "ACDEFGHIKLMNPQRSTVWY")];
		int32_t p = 100 + k * 3 + (k > 150? 90 : 0); // one 90bp "intron" in the middle
		if (p + 2 < MAX_NT && ((seed = seed * 1103515245u + 12345u) >> 16) % 8) nt[p] = c[0], nt[p+1] = c[1], nt[p+2] = c[2];
	}
	b = ns_ext_batch_init(0);
	if (!b) { fprintf(stderr, "[SKIP] extension GPU batch: backend unavailable\n"); goto done; }
	// xdrop/end_bonus are run-wide constants in protlocus, so they only change between flush groups;
	// io (io vs io_end retry), direction and the splice-score track vary per request as in production
	for (grp = 0; grp < 3; ++grp) {
		int32_t i0 = grp * N_REQ / 3, i1 = (grp + 1) * N_REQ / 3;
		n_gpu = ns_ext_gpu_request_count();
		for (i = i0; i < i1; ++i) {
			ns_opt_t *o = &req[i].opt;
			seed = seed * 1103515245u + 12345u;
			ns_opt_init(o);
			ns_opt_set_sp(o, NS_S_GENERIC);
			o->flag |= (i & 1)? NS_F_EXT_LEFT : NS_F_EXT_RIGHT;
			o->io = (i % 3 == 0)? 20 : o->io;
			o->end_bonus = grp == 0? 5 : 0;
			o->xdrop = grp == 1? 30 : grp == 2? 1000000 : 100; // group 2 never x-drops: long scans span several device chunks
			req[i].al = aa_lens[i % (int32_t)(sizeof(aa_lens) / sizeof(aa_lens[0]))];
			req[i].nl = 3 + (int32_t)((seed >> 8) % (MAX_NT - 10));
			if (i % 6 == 0) req[i].nl = 3 + (i / 6); // tiny windows
			req[i].ns = nt + ((seed >> 4) % 7), req[i].as = aa + ((seed >> 12) % 5);
			req[i].ss = (i % 3 == 1)? ss + ((seed >> 4) % 7) : 0;
			ns_rst_init(&expected[i]);
			ns_global_gs16b(0, req[i].ns, req[i].nl, req[i].as, req[i].al, o, req[i].ss, &expected[i]);
			id[i] = ns_ext_batch_add(b, req[i].ns, req[i].nl, req[i].as, req[i].al, o, req[i].ss);
		}
		ns_ext_batch_flush(b);
		for (i = i0, n_dev = 0; i < i1; ++i) n_dev += req[i].nl >= 3; // nl < 3 is CPU-only by design
		assert(ns_ext_gpu_request_count() - n_gpu == (uint64_t)n_dev);
		for (i = i0; i < i1; ++i) {
			const ns_rst_t *got = ns_ext_batch_get(b, id[i]);
			if (got->nt_len != expected[i].nt_len || got->aa_len != expected[i].aa_len || got->score != expected[i].score)
				fprintf(stderr, "extension mismatch at %d (%s nl=%d al=%d io=%d ss=%d): nt_len %d/%d aa_len %d/%d score %d/%d\n", i,
					(req[i].opt.flag & NS_F_EXT_LEFT)? "left" : "right", req[i].nl, req[i].al, req[i].opt.io, req[i].ss != 0,
					expected[i].nt_len, got->nt_len, expected[i].aa_len, got->aa_len, expected[i].score, got->score);
			assert(got->nt_len == expected[i].nt_len && got->aa_len == expected[i].aa_len && got->score == expected[i].score);
		}
	}
	// results survive adding the next round; a second flush recomputes only the new request
	id[0] = ns_ext_batch_add(b, req[1].ns, req[1].nl, req[1].as, req[1].al, &req[1].opt, req[1].ss);
	assert(ns_ext_batch_get(b, id[N_REQ - 1])->score == expected[N_REQ - 1].score);
	assert(ns_ext_batch_flush(b) == 1);
	assert(ns_ext_batch_get(b, id[0])->nt_len == expected[1].nt_len && ns_ext_batch_get(b, id[0])->aa_len == expected[1].aa_len);
	// several flushes in flight: none is waited for at flush time; wait() returns once one has completed
	for (grp = 0; grp < 3; ++grp) {
		int32_t i0 = grp * N_REQ / 3, i1 = (grp + 1) * N_REQ / 3;
		for (i = i0; i < i1; ++i) id[i] = ns_ext_batch_add(b, req[i].ns, req[i].nl, req[i].as, req[i].al, &req[i].opt, req[i].ss);
		assert(ns_ext_batch_flush(b) == i1 - i0);
	}
	assert(ns_ext_batch_flush(b) == 0);
	for (grp = 0; grp < 3; ++grp) {
		int32_t i0 = grp * N_REQ / 3, i1 = (grp + 1) * N_REQ / 3;
		while (!ns_ext_batch_ready(b, id[i0])) ns_ext_batch_wait(b);
		for (i = i0; i < i1; ++i) {
			const ns_rst_t *got;
			assert(ns_ext_batch_ready(b, id[i]));
			got = ns_ext_batch_get(b, id[i]);
			assert(got->nt_len == expected[i].nt_len && got->aa_len == expected[i].aa_len && got->score == expected[i].score);
		}
	}
	ns_ext_batch_wait(b); // nothing outstanding: returns at once
	ns_ext_batch_destroy(b);
done:
	unsetenv("NS_EXT_GPU_STRICT");
	free(nt); free(aa); free(ss);
}

int main(void)
{
	const char *device_text = getenv("NS_GPU_TEST_DEVICE");
	int32_t device = device_text? atoi(device_text) : 0;

	assert(ns_make_tables(1) == 0);
	test_32bit_fallback();
	if (!ns_gpu_init(device)) {
		fprintf(stderr, "[%s] GPU tests: %s\n", getenv("NS_GPU_TEST_REQUIRED")? "FAIL" : "SKIP", ns_gpu_error());
		return getenv("NS_GPU_TEST_REQUIRED")? 1 : 0;
	}
	test_gpu_tile_boundaries();
	test_gpu_batch();
	test_gpu_async();
	test_hmm_gpu_matches_cpu();
	test_ext_gpu_matches_cpu();
	return 0;
}