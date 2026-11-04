#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "nasw.h"

#ifndef NS_CUDA
int ns_gpu_init(int32_t device)
{
	(void)device;
	return 0;
}

const char *ns_gpu_error(void)
{
	return "CUDA backend not compiled; rebuild with 'make cuda=1'";
}

const char *ns_gpu_name(void)
{
	return "none";
}

uint64_t ns_gpu_dispatch_count(void)
{
	return 0;
}

uint64_t ns_gpu_launch_count(void)
{
	return 0;
}

double ns_gpu_h2d_seconds(void) { return 0.0; }
double ns_gpu_kernel_seconds(void) { return 0.0; }
double ns_gpu_d2h_seconds(void) { return 0.0; }

ns_gpu_request_t *ns_gpu_submit(void *km, const char *ns, int32_t nl, const char *as, int32_t al,
		const ns_opt_t *opt, const uint8_t *ss)
{
	(void)km, (void)ns, (void)nl, (void)as, (void)al, (void)opt, (void)ss;
	return 0;
}

int ns_gpu_wait(ns_gpu_request_t *request, ns_rst_t *r)
{
	(void)request, (void)r;
	return 0;
}

int ns_gpu_global(void *km, const char *ns, int32_t nl, const char *as, int32_t al,
		const ns_opt_t *opt, const uint8_t *ss, ns_rst_t *r)
{
	(void)km, (void)ns, (void)nl, (void)as, (void)al, (void)opt, (void)ss, (void)r;
	return 0;
}

void ns_gpu_gapless_profile(ns_gpu_gapless_profile_t *profile)
{
	memset(profile, 0, sizeof(*profile));
}

int ns_gpu_gapless_supported(void) { return 0; }

void ns_gpu_gapless_reset_target(void) {}

int ns_gpu_gapless_score_batch(const uint8_t *packed_nt, int64_t packed_nt_bases,
		const uint8_t *queries, int64_t query_aa,
		const ns_gpu_gapless_candidate_t *candidates, int32_t n_candidates,
		const ns_gpu_gapless_anchor_t *anchors, int64_t n_anchors,
		int32_t block_width, int32_t kmer, int32_t xdrop,
		const int8_t *score_matrix, int32_t asize, const uint8_t *codon,
		int32_t *scores)
{
	(void)packed_nt, (void)packed_nt_bases, (void)queries, (void)query_aa;
	(void)candidates, (void)n_candidates, (void)anchors, (void)n_anchors;
	(void)block_width, (void)kmer, (void)xdrop, (void)score_matrix, (void)asize, (void)codon, (void)scores;
	return 0;
}

#else

#include <pthread.h>
#include <cuda_runtime.h>

#ifndef NS_GPU_TILE_SIZE
#define NS_GPU_TILE_SIZE 4
#endif
#if NS_GPU_TILE_SIZE < 3
#error "NS_GPU_TILE_SIZE must be at least 3"
#endif

#define TILE_SIZE NS_GPU_TILE_SIZE
#define H_RIGHT 0
#define H_BOTTOM (TILE_SIZE)
#define H_CORNER (4 * TILE_SIZE)
#define I_RIGHT (4 * TILE_SIZE + 3)
#define D_BOTTOM (5 * TILE_SIZE + 3)
#define A_BOTTOM (8 * TILE_SIZE + 3)
#define B_BOTTOM (9 * TILE_SIZE + 3)
#define C_BOTTOM (10 * TILE_SIZE + 3)
#define FRONTIER_STRIDE (11 * TILE_SIZE + 3)
#define NEG (-32768)

#ifndef NS_GPU_LOCAL_SIZE
#define NS_GPU_LOCAL_SIZE 256
#endif

/* ------------------------------------------------------------------ */
/* device helpers, mirroring the OpenCL kernel bit-for-bit             */
/* ------------------------------------------------------------------ */

__device__ __forceinline__ int32_t ns_add(int32_t a, int32_t b)
{
	int32_t value = a + b;
	return value < -32768 ? -32768 : (value > 32767 ? 32767 : value);
}

__device__ __forceinline__ int32_t ns_sub(int32_t a, int32_t b)
{
	int32_t value = a - b;
	return value < -32768 ? -32768 : (value > 32767 ? 32767 : value);
}

__device__ __forceinline__ int32_t ns_min_i32(int32_t a, int32_t b) { return a < b ? a : b; }
__device__ __forceinline__ int32_t ns_max_i32(int32_t a, int32_t b) { return a > b ? a : b; }

__device__ __forceinline__ int32_t ns_edge_h(int32_t i, int32_t j, int32_t fs)
{
	if (j == -1) {
		if (i == -1) return 0;
		if (i == 0 || i == 1) return -fs;
	}
	return NEG;
}

/* one warp lane always exchanges a full FRONTIER_STRIDE column at a time */
__device__ __forceinline__ short ns_shfl_short(short v, int32_t src_lane)
{
	src_lane = src_lane < 0 ? 0 : (src_lane > 31 ? 31 : src_lane);
	return (short)__shfl_sync(0xffffffffu, (int)v, src_lane, 32);
}

/*
 * Per-tile recurrence shared by both kernels. top_boundary/left_boundary are
 * read-only inputs (from shared memory or from a warp-shuffled register
 * array); output_boundary is where this tile's outgoing boundary is written.
 */
__device__ __forceinline__ void ns_process_tile(
		const uint8_t *nas, const uint8_t *aas, const int8_t *donor, const int8_t *acceptor,
		const int8_t *score, int32_t asize, int32_t go, int32_t ge, int32_t io, int32_t fs,
		int32_t al, int32_t nl, int32_t tile_row, int32_t tile_column, int32_t tile_rows, int32_t tile_columns,
		const short *top_boundary, const short *left_boundary, short *output_boundary,
		uint16_t *trace, int32_t *result, int32_t job)
{
	short h_rows[4][TILE_SIZE + 1];
	short d_rows[3][TILE_SIZE];
	short a_state[TILE_SIZE], b_state[TILE_SIZE], c_state[TILE_SIZE];
	const int32_t first_i = 2 + tile_row * TILE_SIZE;
	const int32_t tile_height = ns_min_i32(TILE_SIZE, nl - first_i);
	const int32_t first_j = tile_column * TILE_SIZE;
	const int32_t tile_width = ns_min_i32(TILE_SIZE, al - first_j);

	for (int32_t halo = 0; halo < 3; ++halo) {
		const int32_t halo_i = first_i - 3 + halo;
		if (tile_row == 0 || tile_column == 0)
			h_rows[halo][0] = (short)ns_edge_h(halo_i, first_j - 1, fs);
		else
			h_rows[halo][0] = top_boundary[H_CORNER + halo];
		for (int32_t column = 0; column < tile_width; ++column) {
			if (tile_row == 0) {
				h_rows[halo][column + 1] = (short)ns_edge_h(halo_i, first_j + column, fs);
				d_rows[halo][column] = (short)NEG;
			} else {
				h_rows[halo][column + 1] = top_boundary[H_BOTTOM + halo * TILE_SIZE + column];
				d_rows[halo][column] = top_boundary[D_BOTTOM + halo * TILE_SIZE + column];
			}
		}
	}
	for (int32_t column = 0; column < tile_width; ++column) {
		if (tile_row == 0) {
			a_state[column] = (short)NEG;
			b_state[column] = (short)NEG;
			c_state[column] = (short)NEG;
		} else {
			a_state[column] = top_boundary[A_BOTTOM + column];
			b_state[column] = top_boundary[B_BOTTOM + column];
			c_state[column] = top_boundary[C_BOTTOM + column];
		}
	}
	for (int32_t row = 0; row < tile_height; ++row) {
		const int32_t i = first_i + row;
		const int32_t current_h = (row + 3) & 3;
		const int32_t h_m1 = (row + 2) & 3;
		const int32_t h_m2 = (row + 1) & 3;
		const int32_t h_m3 = row & 3;
		const int32_t current_d = row % 3;
		int32_t insertion = NEG;
		if (tile_column == 0)
			h_rows[current_h][0] = (short)ns_edge_h(i, -1, fs);
		else {
			h_rows[current_h][0] = left_boundary[H_RIGHT + row];
			insertion = left_boundary[I_RIGHT + row];
		}
		for (int32_t column = 0; column < tile_width; ++column) {
			const int32_t j = first_j + column;
			const int64_t cell = (int64_t)(i - 2) * al + j;
			int32_t bits = 0, path = 0, candidate, open_score, extension_score;
			int32_t deletion, intron_a, intron_b, intron_c;
			int32_t best = ns_add(h_rows[h_m3][column], (int32_t)score[(int32_t)nas[i] * asize + (int32_t)aas[j]]);
			open_score = ns_sub(h_rows[current_h][column], go);
			extension_score = insertion;
			if (extension_score > open_score) bits |= 1 << 4;
			insertion = ns_sub(ns_max_i32(open_score, extension_score), ge);
			if (insertion > best) best = insertion, path = 1;
			open_score = ns_sub(h_rows[h_m3][column + 1], go);
			extension_score = d_rows[current_d][column];
			if (extension_score > open_score) bits |= 1 << 5;
			deletion = ns_sub(ns_max_i32(open_score, extension_score), nas[i] == 20 ? fs : ge);
			if (deletion > best) best = deletion, path = 2;
			open_score = ns_sub(ns_sub(h_rows[h_m1][column + 1], io), donor[i - 1]);
			extension_score = a_state[column];
			if (extension_score > open_score) bits |= 1 << 6;
			intron_a = ns_max_i32(open_score, extension_score);
			candidate = ns_sub(intron_a, acceptor[i]);
			if (candidate > best) best = candidate, path = 3;
			open_score = ns_sub(ns_sub(h_rows[h_m1][column], io), donor[i]);
			extension_score = b_state[column];
			if (extension_score > open_score) bits |= 1 << 7;
			intron_b = ns_max_i32(open_score, extension_score);
			candidate = ns_sub(intron_b, acceptor[i - 2]);
			if (candidate > best) best = candidate, path = 4;
			open_score = ns_sub(ns_sub(h_rows[h_m1][column], io), donor[i + 1]);
			extension_score = c_state[column];
			if (extension_score > open_score) bits |= 1 << 8;
			intron_c = ns_max_i32(open_score, extension_score);
			candidate = ns_sub(intron_c, acceptor[i - 1]);
			if (candidate > best) best = candidate, path = 5;
			candidate = ns_sub(h_rows[h_m1][column], fs);
			if (candidate > best) best = candidate, path = 6;
			candidate = ns_sub(h_rows[h_m2][column], fs);
			if (candidate > best) best = candidate, path = 7;
			candidate = ns_sub(h_rows[h_m1][column + 1], fs);
			if (candidate > best) best = candidate, path = 8;
			candidate = ns_sub(h_rows[h_m2][column + 1], fs);
			if (candidate > best) best = candidate, path = 9;
			h_rows[current_h][column + 1] = (short)best;
			d_rows[current_d][column] = (short)deletion;
			a_state[column] = (short)intron_a;
			b_state[column] = (short)intron_b;
			c_state[column] = (short)intron_c;
			trace[cell] = (uint16_t)(bits | path);
		}
		output_boundary[H_RIGHT + row] = h_rows[current_h][tile_width];
		output_boundary[I_RIGHT + row] = (short)insertion;
		if (row + 3 >= tile_height) {
			const int32_t bottom_row = row + 3 - tile_height;
			output_boundary[H_CORNER + bottom_row] = h_rows[current_h][0];
			for (int32_t column = 0; column < tile_width; ++column) {
				output_boundary[H_BOTTOM + bottom_row * TILE_SIZE + column] = h_rows[current_h][column + 1];
				output_boundary[D_BOTTOM + bottom_row * TILE_SIZE + column] = d_rows[current_d][column];
			}
		}
	}
	for (int32_t column = 0; column < tile_width; ++column) {
		output_boundary[A_BOTTOM + column] = a_state[column];
		output_boundary[B_BOTTOM + column] = b_state[column];
		output_boundary[C_BOTTOM + column] = c_state[column];
	}
	if (tile_row == tile_rows - 1 && tile_column == tile_columns - 1) {
		const int32_t last_h = (tile_height + 2) & 3;
		result[job] = h_rows[last_h][tile_width];
	}
}

/* ------------------------------------------------------------------ */
/* shared-memory kernel: mechanical translation of the OpenCL kernel   */
/* used whenever parallel_tiles > 32 (one block per job)               */
/* ------------------------------------------------------------------ */
extern __shared__ short ns_frontier_shared[];

__global__ void ns_splice_dp_shared(
		const uint8_t *all_nas, const uint8_t *all_aas,
		const int8_t *all_donor, const int8_t *all_acceptor,
		const int8_t *all_score, const uint64_t *offsets,
		const int32_t *dimensions, const int32_t *parameters,
		uint16_t *all_trace, int32_t *result)
{
	const int32_t job = blockIdx.x;
	const int32_t lane = threadIdx.x;
	const int32_t lanes = blockDim.x;
	const uint64_t *job_offsets = offsets + (int64_t)job * 6;
	const uint8_t *nas = all_nas + job_offsets[0];
	const uint8_t *aas = all_aas + job_offsets[1];
	const int8_t *donor = all_donor + job_offsets[2];
	const int8_t *acceptor = all_acceptor + job_offsets[3];
	const int8_t *score = all_score + job_offsets[4];
	uint16_t *trace = all_trace + job_offsets[5];
	const int32_t nl = dimensions[job * 3];
	const int32_t al = dimensions[job * 3 + 1];
	const int32_t asize = dimensions[job * 3 + 2];
	const int32_t go = parameters[job * 5];
	const int32_t ge = parameters[job * 5 + 1];
	const int32_t io = parameters[job * 5 + 2];
	const int32_t fs = parameters[job * 5 + 3];
	const int32_t matrix_rows = nl - 2;
	const int32_t tile_rows = (matrix_rows + TILE_SIZE - 1) / TILE_SIZE;
	const int32_t tile_columns = (al + TILE_SIZE - 1) / TILE_SIZE;
	const int32_t frontier_plane_size = tile_columns * FRONTIER_STRIDE;

	for (int32_t tile_diagonal = 0; tile_diagonal < tile_rows + tile_columns - 1; ++tile_diagonal) {
		const int32_t first_tile_row = ns_max_i32(0, tile_diagonal - tile_columns + 1);
		const int32_t last_tile_row = ns_min_i32(tile_rows - 1, tile_diagonal);
		for (int32_t tile_row = first_tile_row + lane; tile_row <= last_tile_row; tile_row += lanes) {
			const int32_t tile_column = tile_diagonal - tile_row;
			short *input_frontier = ns_frontier_shared + ((tile_diagonal + 1) & 1) * frontier_plane_size;
			short *output_frontier = ns_frontier_shared + (tile_diagonal & 1) * frontier_plane_size;
			short *top_boundary = input_frontier + tile_column * FRONTIER_STRIDE;
			short *left_boundary = input_frontier + ns_max_i32(0, tile_column - 1) * FRONTIER_STRIDE;
			short *output_boundary = output_frontier + tile_column * FRONTIER_STRIDE;
			ns_process_tile(nas, aas, donor, acceptor, score, asize, go, ge, io, fs, al, nl,
					tile_row, tile_column, tile_rows, tile_columns,
					top_boundary, left_boundary, output_boundary, trace, result, job);
		}
		__syncthreads();
	}
}

/* ------------------------------------------------------------------ */
/* warp kernel: fixed 32 lanes, boundary exchange via __shfl_sync      */
/* instead of shared memory + barriers. Used when parallel_tiles<=32,  */
/* which covers the vast majority of real jobs (one workgroup/warp).   */
/*                                                                     */
/* For a given anti-diagonal d, a tile's top/left neighbours were both */
/* produced at diagonal d-1. Let step(d) = (d >= tile_columns) ? 1 : 0.*/
/* Then, for the lane L handling this diagonal's tile:                 */
/*   srcLane_top  = L - 1 + step(d)                                    */
/*   srcLane_left = L + step(d)                                        */
/* (derivation: tile_row(L,d) = first_tile_row(d) + L, tile_column(L,d)*/
/* = d - tile_row(L,d); matching indices against diagonal d-1's lane   */
/* assignment yields the two formulas above.)                          */
/* ------------------------------------------------------------------ */
__global__ void ns_splice_dp_warp(
		const uint8_t *all_nas, const uint8_t *all_aas,
		const int8_t *all_donor, const int8_t *all_acceptor,
		const int8_t *all_score, const uint64_t *offsets,
		const int32_t *dimensions, const int32_t *parameters,
		uint16_t *all_trace, int32_t *result)
{
	const int32_t job = blockIdx.x;
	const int32_t lane = threadIdx.x;
	const uint64_t *job_offsets = offsets + (int64_t)job * 6;
	const uint8_t *nas = all_nas + job_offsets[0];
	const uint8_t *aas = all_aas + job_offsets[1];
	const int8_t *donor = all_donor + job_offsets[2];
	const int8_t *acceptor = all_acceptor + job_offsets[3];
	const int8_t *score = all_score + job_offsets[4];
	uint16_t *trace = all_trace + job_offsets[5];
	const int32_t nl = dimensions[job * 3];
	const int32_t al = dimensions[job * 3 + 1];
	const int32_t asize = dimensions[job * 3 + 2];
	const int32_t go = parameters[job * 5];
	const int32_t ge = parameters[job * 5 + 1];
	const int32_t io = parameters[job * 5 + 2];
	const int32_t fs = parameters[job * 5 + 3];
	const int32_t matrix_rows = nl - 2;
	const int32_t tile_rows = (matrix_rows + TILE_SIZE - 1) / TILE_SIZE;
	const int32_t tile_columns = (al + TILE_SIZE - 1) / TILE_SIZE;
	const int32_t n_diagonals = tile_rows + tile_columns - 1;

	short my_boundary[FRONTIER_STRIDE];
#pragma unroll
	for (int32_t k = 0; k < FRONTIER_STRIDE; ++k) my_boundary[k] = 0;

	for (int32_t tile_diagonal = 0; tile_diagonal < n_diagonals; ++tile_diagonal) {
		const int32_t first_tile_row = ns_max_i32(0, tile_diagonal - tile_columns + 1);
		const int32_t last_tile_row = ns_min_i32(tile_rows - 1, tile_diagonal);
		const int32_t tile_row = first_tile_row + lane;
		const bool active = tile_row <= last_tile_row;
		const int32_t tile_column = tile_diagonal - tile_row;
		const int32_t step = (tile_diagonal >= tile_columns) ? 1 : 0;
		const int32_t src_top = lane - 1 + step;
		const int32_t src_left = lane + step;

		short top_boundary[FRONTIER_STRIDE], left_boundary[FRONTIER_STRIDE], output_boundary[FRONTIER_STRIDE];
#pragma unroll
		for (int32_t k = 0; k < FRONTIER_STRIDE; ++k) {
			top_boundary[k] = ns_shfl_short(my_boundary[k], src_top);
			left_boundary[k] = ns_shfl_short(my_boundary[k], src_left);
		}
		if (active) {
			ns_process_tile(nas, aas, donor, acceptor, score, asize, go, ge, io, fs, al, nl,
					tile_row, tile_column, tile_rows, tile_columns,
					top_boundary, left_boundary, output_boundary, trace, result, job);
#pragma unroll
			for (int32_t k = 0; k < FRONTIER_STRIDE; ++k) my_boundary[k] = output_boundary[k];
		}
	}
}

/* ------------------------------------------------------------------ */
/* host side                                                           */
/* ------------------------------------------------------------------ */

struct ns_gpu_request_s {
	void *km;
	uint8_t *prepared, *aas;
	int8_t *donor, *acceptor;
	uint16_t *trace;
	int8_t *score_matrix;
	int32_t nl, al, asize, go, ge, io, fs, narrow, parallel_tiles, score;
	size_t trace_bytes, state_bytes;
	int done, success;
	struct ns_gpu_request_s *next;
};

#ifndef NS_GPU_MAX_BATCH
#define NS_GPU_MAX_BATCH 128
#endif
#ifndef NS_GPU_BATCH_WAIT_NS
#define NS_GPU_BATCH_WAIT_NS 2000000L
#endif

typedef struct ns_gpu_batch_s {
	ns_gpu_request_t *requests[NS_GPU_MAX_BATCH];
	int count;
	struct ns_gpu_batch_s *next;
} ns_gpu_batch_t;

static int ns_gpu_dispatch_batch(ns_gpu_request_t **requests, int count);
static void *ns_gpu_collect_batches(void *unused);
static void *ns_gpu_execute_batches(void *unused);

#ifndef NS_GPU_NUM_BUCKETS
#define NS_GPU_NUM_BUCKETS 8
#endif

/*
 * Classifies a request by required thread count so one batch never mixes a
 * small job with a much larger outlier: the whole batch launches with a
 * single blockDim sized for its largest member, which would otherwise waste
 * most warps on every small job sharing that launch.
 */
static int ns_gpu_bucket_for(int32_t parallel_tiles)
{
	int32_t threshold = 32;
	int bucket = 0;
	while (parallel_tiles > threshold && bucket < NS_GPU_NUM_BUCKETS - 1) {
		threshold <<= 1;
		++bucket;
	}
	return bucket;
}

static pthread_mutex_t ns_gpu_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t ns_gpu_condition = PTHREAD_COND_INITIALIZER;
static pthread_t ns_gpu_collector_thread, ns_gpu_executor_thread;
static ns_gpu_request_t *ns_gpu_bucket_head[NS_GPU_NUM_BUCKETS], *ns_gpu_bucket_tail[NS_GPU_NUM_BUCKETS];
static int ns_gpu_bucket_size[NS_GPU_NUM_BUCKETS];
static ns_gpu_batch_t *ns_gpu_batch_head, *ns_gpu_batch_tail;
static int ns_gpu_queue_size, ns_gpu_batches_active, ns_gpu_worker_started;
static int ns_gpu_ready;
static int ns_gpu_device_index;
static size_t ns_gpu_shared_mem_limit;
static int32_t ns_gpu_max_threads = NS_GPU_LOCAL_SIZE;
static uint64_t ns_gpu_dispatches;
static uint64_t ns_gpu_launches;
/* accumulated phase time in nanoseconds, written only by the single executor thread */
static uint64_t ns_gpu_h2d_ns, ns_gpu_kernel_ns, ns_gpu_d2h_ns;
static char ns_gpu_message[4096] = "GPU backend has not been initialized";
static char ns_gpu_device_name[256] = "none";

/*
 * Persistent grow-only device/host buffer pool for the batch executor thread.
 * cudaMalloc/cudaFree implicitly synchronize the device, so reallocating on
 * every batch (thousands of launches per run) dominates wall time; these
 * buffers are only ever grown to the largest request seen and reused as-is
 * for every smaller batch. Only the single executor thread touches them, so
 * no locking is needed.
 */
typedef struct { void *ptr; size_t cap; } ns_gpu_devbuf_t;
static ns_gpu_devbuf_t ns_gpu_buf_nas, ns_gpu_buf_aas, ns_gpu_buf_donor, ns_gpu_buf_acceptor,
		ns_gpu_buf_score, ns_gpu_buf_offsets, ns_gpu_buf_dimensions, ns_gpu_buf_parameters,
		ns_gpu_buf_trace, ns_gpu_buf_result;
static uint16_t *ns_gpu_trace_host_buf;
static size_t ns_gpu_trace_host_cap;

static cudaError_t ns_gpu_devbuf_ensure(ns_gpu_devbuf_t *buf, size_t need)
{
	cudaError_t status;
	if (need == 0 || buf->cap >= need) return cudaSuccess;
	if (buf->ptr) cudaFree(buf->ptr);
	buf->ptr = 0, buf->cap = 0;
	if ((status = cudaMalloc(&buf->ptr, need)) != cudaSuccess) return status;
	buf->cap = need;
	return cudaSuccess;
}

static int ns_gpu_state_size(int32_t nl, int32_t al, size_t *bytes, int32_t *parallel_tiles)
{
	size_t tile_rows = ((size_t)nl - 2 + NS_GPU_TILE_SIZE - 1) / NS_GPU_TILE_SIZE;
	size_t tile_columns = ((size_t)al + NS_GPU_TILE_SIZE - 1) / NS_GPU_TILE_SIZE;
	if (tile_columns > SIZE_MAX / (2 * FRONTIER_STRIDE * sizeof(int16_t))) return 0;
	*bytes = 2 * tile_columns * FRONTIER_STRIDE * sizeof(int16_t);
	*parallel_tiles = (int32_t)(tile_rows < tile_columns ? tile_rows : tile_columns);
	return 1;
}

static void ns_gpu_set_error(const char *operation, cudaError_t status)
{
	snprintf(ns_gpu_message, sizeof(ns_gpu_message), "%s failed: %s", operation, cudaGetErrorString(status));
}

int ns_gpu_init(int32_t requested_device)
{
	int device_count = 0;
	cudaError_t status;
	struct cudaDeviceProp prop;
	int thread_status;

	pthread_mutex_lock(&ns_gpu_mutex);
	if (ns_gpu_ready) {
		pthread_mutex_unlock(&ns_gpu_mutex);
		return 1;
	}
	status = cudaGetDeviceCount(&device_count);
	if (status != cudaSuccess) {
		ns_gpu_set_error("cudaGetDeviceCount", status);
		goto fail;
	}
	if (requested_device < 0 || requested_device >= device_count) {
		snprintf(ns_gpu_message, sizeof(ns_gpu_message), "CUDA GPU %d does not exist (%d available)",
				requested_device, device_count);
		goto fail;
	}
	status = cudaSetDevice(requested_device);
	if (status != cudaSuccess) {
		ns_gpu_set_error("cudaSetDevice", status);
		goto fail;
	}
	status = cudaGetDeviceProperties(&prop, requested_device);
	if (status != cudaSuccess) {
		ns_gpu_set_error("cudaGetDeviceProperties", status);
		goto fail;
	}
	strncpy(ns_gpu_device_name, prop.name, sizeof(ns_gpu_device_name) - 1);
	ns_gpu_device_name[sizeof(ns_gpu_device_name) - 1] = 0;
	ns_gpu_shared_mem_limit = prop.sharedMemPerBlockOptin ? (size_t)prop.sharedMemPerBlockOptin : (size_t)prop.sharedMemPerBlock;
	ns_gpu_max_threads = prop.maxThreadsPerBlock < NS_GPU_LOCAL_SIZE ? prop.maxThreadsPerBlock : NS_GPU_LOCAL_SIZE;
	ns_gpu_device_index = requested_device;
	strcpy(ns_gpu_message, "none");
	ns_gpu_ready = 1;
	thread_status = pthread_create(&ns_gpu_executor_thread, 0, ns_gpu_execute_batches, 0);
	if (thread_status != 0) {
		snprintf(ns_gpu_message, sizeof(ns_gpu_message), "failed to start GPU executor: %s", strerror(thread_status));
		ns_gpu_ready = 0;
		goto fail;
	}
	pthread_detach(ns_gpu_executor_thread);
	thread_status = pthread_create(&ns_gpu_collector_thread, 0, ns_gpu_collect_batches, 0);
	if (thread_status != 0) {
		snprintf(ns_gpu_message, sizeof(ns_gpu_message), "failed to start GPU batch collector: %s", strerror(thread_status));
		ns_gpu_ready = 0;
		goto fail;
	}
	pthread_detach(ns_gpu_collector_thread);
	ns_gpu_worker_started = 1;
	pthread_mutex_unlock(&ns_gpu_mutex);
	return 1;

fail:
	pthread_mutex_unlock(&ns_gpu_mutex);
	return 0;
}

const char *ns_gpu_error(void)
{
	return ns_gpu_message;
}

const char *ns_gpu_name(void)
{
	return ns_gpu_device_name;
}

uint64_t ns_gpu_dispatch_count(void)
{
	return __sync_fetch_and_add(&ns_gpu_dispatches, 0);
}

uint64_t ns_gpu_launch_count(void)
{
	return __sync_fetch_and_add(&ns_gpu_launches, 0);
}

double ns_gpu_h2d_seconds(void)
{
	return (double)__sync_fetch_and_add(&ns_gpu_h2d_ns, 0) * 1e-9;
}

double ns_gpu_kernel_seconds(void)
{
	return (double)__sync_fetch_and_add(&ns_gpu_kernel_ns, 0) * 1e-9;
}

double ns_gpu_d2h_seconds(void)
{
	return (double)__sync_fetch_and_add(&ns_gpu_d2h_ns, 0) * 1e-9;
}

static uint8_t *ns_gpu_prep_seq(void *km, const char *ns, int32_t nl, const char *as, int32_t al,
		const ns_opt_t *opt, const uint8_t *ss, uint8_t **aas_, int8_t **donor_, int8_t **acceptor_)
{
	int32_t i, j, run;
	uint8_t codon, *nas, *aas;
	int8_t *donor, *acceptor;
	nas = Kmalloc(km, uint8_t, nl + al + (nl + 1) * 2);
	*aas_ = aas = nas + nl;
	*donor_ = donor = (int8_t*)aas + al, *acceptor_ = acceptor = donor + nl + 1;
	for (j = 0; j < al; ++j) aas[j] = opt->aa20[(uint8_t)as[j]];
	for (i = 0; i < nl; ++i) nas[i] = opt->nt4[(uint8_t)ns[i]];
	for (i = 0; i < nl + 1; ++i) donor[i] = acceptor[i] = opt->sp[3];
	for (i = 0; i < nl - 3; ++i) {
		int32_t type = 3;
		if (nas[i+1] == 2 && nas[i+2] == 3)
			type = i + 3 < nl && (nas[i+3] == 0 || nas[i+3] == 2)? (nas[i] == 2? -1 : 4) : 0;
		else if (nas[i+1] == 2 && nas[i+2] == 1 && nas[i] == 2) type = 1;
		else if (nas[i+1] == 0 && nas[i+2] == 3) type = 2;
		donor[i] = type < 0? 0 : opt->sp[type];
	}
	for (i = 1; i < nl; ++i) {
		int32_t type = 3, penalty = 0;
		if (nas[i-1] == 0 && nas[i] == 2) {
			type = i >= 2 && (nas[i-2] == 1 || nas[i-2] == 3)? -1 : 0;
			for (j = i - 4; j >= 0 && j > i - 7; --j)
				if (nas[j] != 1 && nas[j] != 3) penalty += opt->sp[5];
		} else if (nas[i-1] == 0 && nas[i] == 1) type = 2;
		acceptor[i] = type < 0? 0 : opt->sp[type];
		if (type == -1 || type == 0) acceptor[i] += penalty;
	}
	if (ss) {
		int32_t max_spsc = (opt->io + 1) / 2 - 1;
		for (i = 1; i < nl; ++i) {
			int32_t spsc = (int8_t)(ss[i]>>1) - (int8_t)NS_SPSC_OFFSET;
			if (spsc > max_spsc) spsc = max_spsc;
			if (ss[i] == 0xff) {
				donor[i-1] -= opt->sp_null_bonus;
				acceptor[i-1] -= opt->sp_null_bonus;
			} else if (ss[i]&1) acceptor[i-1] -= spsc;
			else donor[i-1] -= spsc;
		}
	}
	memset(nas, opt->aa20['X'], nl);
	for (i = run = 0, codon = 0; i < nl; ++i) {
		uint8_t base = opt->nt4[(uint8_t)ns[i]];
		if (base < 4) {
			codon = (codon << 2 | base) & 0x3f;
			if (++run >= 3) nas[i] = opt->codon[codon];
		} else codon = 0, run = 0;
	}
	return nas;
}

static void ns_gpu_fix_tiny_uv(int32_t n_cigar, uint32_t *cigar)
{
	int32_t i;
	for (i = 0; i < n_cigar; ++i) {
		int32_t op = cigar[i]&0xf;
		if ((op == NS_CIGAR_U || op == NS_CIGAR_V) && cigar[i]>>4 < 3)
			cigar[i] = cigar[i]>>4<<4 | NS_CIGAR_G;
	}
}

static void ns_gpu_backtrack(void *km, const uint16_t *trace, int32_t nl, int32_t al, ns_rst_t *r)
{
	int32_t i = nl - 1, j = al - 1, last = 0, index;
	uint32_t temporary;
	while (i >= 2 && j >= 0) {
		int32_t value = trace[(int64_t)(i - 2) * al + j];
		int32_t state, extension;
		if (value>>9&1) value = 1|value>>4<<4;
		state = last == 0? value&0xf : last;
		extension = state >= 1 && state <= 5? value>>(state+3)&1 : 0;
		if (state == 0) r->cigar = ns_push_cigar(km, &r->n_cigar, &r->m_cigar, r->cigar, NS_CIGAR_M, 1), i -= 3, --j;
		else if (state == 1) r->cigar = ns_push_cigar(km, &r->n_cigar, &r->m_cigar, r->cigar, NS_CIGAR_I, 1), --j;
		else if (state == 2) r->cigar = ns_push_cigar(km, &r->n_cigar, &r->m_cigar, r->cigar, NS_CIGAR_D, 1), i -= 3;
		else if (state == 3) r->cigar = ns_push_cigar(km, &r->n_cigar, &r->m_cigar, r->cigar, NS_CIGAR_N, 1), --i;
		else if (state == 4) { r->cigar = ns_push_cigar(km, &r->n_cigar, &r->m_cigar, r->cigar, NS_CIGAR_U, 1), --i; if (!extension) --j; }
		else if (state == 5) { r->cigar = ns_push_cigar(km, &r->n_cigar, &r->m_cigar, r->cigar, NS_CIGAR_V, 1), --i; if (!extension) --j; }
		else if (state == 6) r->cigar = ns_push_cigar(km, &r->n_cigar, &r->m_cigar, r->cigar, NS_CIGAR_F, 1), --i;
		else if (state == 7) r->cigar = ns_push_cigar(km, &r->n_cigar, &r->m_cigar, r->cigar, NS_CIGAR_F, 2), i -= 2;
		else if (state == 8) r->cigar = ns_push_cigar(km, &r->n_cigar, &r->m_cigar, r->cigar, NS_CIGAR_G, 1), --i, --j;
		else if (state == 9) r->cigar = ns_push_cigar(km, &r->n_cigar, &r->m_cigar, r->cigar, NS_CIGAR_G, 2), i -= 2, --j;
		last = state >= 1 && state <= 5 && extension? state : 0;
	}
	if (j > 0) r->cigar = ns_push_cigar(km, &r->n_cigar, &r->m_cigar, r->cigar, NS_CIGAR_I, j);
	if (i >= 0) { // D length counts codons, like the per-step pushes above
		int32_t length = (i+1) / 3, tail = (i+1) % 3;
		if (length > 0) r->cigar = ns_push_cigar(km, &r->n_cigar, &r->m_cigar, r->cigar, NS_CIGAR_D, length);
		if (tail != 0) r->cigar = ns_push_cigar(km, &r->n_cigar, &r->m_cigar, r->cigar, NS_CIGAR_F, tail);
	}
	for (index = 0; index < r->n_cigar>>1; ++index)
		temporary = r->cigar[index], r->cigar[index] = r->cigar[r->n_cigar - 1 - index], r->cigar[r->n_cigar - 1 - index] = temporary;
	ns_gpu_fix_tiny_uv(r->n_cigar, r->cigar);
}

/* allocates device buffers, launches the appropriate kernel, and copies results back */
static int ns_gpu_run_job(const uint8_t *prepared, int32_t nl, const uint8_t *aas, int32_t al,
		const int8_t *donor, const int8_t *acceptor, const int8_t *score_matrix, int32_t asize,
		int32_t go, int32_t ge, int32_t io, int32_t fs, int32_t parallel_tiles, size_t state_bytes,
		uint16_t *trace, size_t trace_bytes, int32_t *score_out)
{
	uint8_t *d_nas = 0, *d_aas = 0;
	int8_t *d_donor = 0, *d_acceptor = 0, *d_score = 0;
	uint64_t offsets[6] = {0, 0, 0, 0, 0, 0};
	uint64_t *d_offsets = 0;
	int32_t dimensions[3], parameters[5];
	int32_t *d_dimensions = 0, *d_parameters = 0, *d_result = 0;
	uint16_t *d_trace = 0;
	cudaError_t status = cudaSuccess;
	int success = 0;

	dimensions[0] = nl, dimensions[1] = al, dimensions[2] = asize;
	parameters[0] = go, parameters[1] = ge, parameters[2] = io, parameters[3] = fs, parameters[4] = 1;

	if ((status = cudaMalloc(&d_nas, nl)) != cudaSuccess) goto cuda_error;
	if ((status = cudaMalloc(&d_aas, al)) != cudaSuccess) goto cuda_error;
	if ((status = cudaMalloc(&d_donor, nl + 1)) != cudaSuccess) goto cuda_error;
	if ((status = cudaMalloc(&d_acceptor, nl + 1)) != cudaSuccess) goto cuda_error;
	if ((status = cudaMalloc(&d_score, (size_t)asize * asize)) != cudaSuccess) goto cuda_error;
	if ((status = cudaMalloc(&d_offsets, sizeof(offsets))) != cudaSuccess) goto cuda_error;
	if ((status = cudaMalloc(&d_dimensions, sizeof(dimensions))) != cudaSuccess) goto cuda_error;
	if ((status = cudaMalloc(&d_parameters, sizeof(parameters))) != cudaSuccess) goto cuda_error;
	if ((status = cudaMalloc(&d_trace, trace_bytes)) != cudaSuccess) goto cuda_error;
	if ((status = cudaMalloc(&d_result, sizeof(int32_t))) != cudaSuccess) goto cuda_error;

	if ((status = cudaMemcpy(d_nas, prepared, nl, cudaMemcpyHostToDevice)) != cudaSuccess) goto cuda_error;
	if ((status = cudaMemcpy(d_aas, aas, al, cudaMemcpyHostToDevice)) != cudaSuccess) goto cuda_error;
	if ((status = cudaMemcpy(d_donor, donor, nl + 1, cudaMemcpyHostToDevice)) != cudaSuccess) goto cuda_error;
	if ((status = cudaMemcpy(d_acceptor, acceptor, nl + 1, cudaMemcpyHostToDevice)) != cudaSuccess) goto cuda_error;
	if ((status = cudaMemcpy(d_score, score_matrix, (size_t)asize * asize, cudaMemcpyHostToDevice)) != cudaSuccess) goto cuda_error;
	if ((status = cudaMemcpy(d_offsets, offsets, sizeof(offsets), cudaMemcpyHostToDevice)) != cudaSuccess) goto cuda_error;
	if ((status = cudaMemcpy(d_dimensions, dimensions, sizeof(dimensions), cudaMemcpyHostToDevice)) != cudaSuccess) goto cuda_error;
	if ((status = cudaMemcpy(d_parameters, parameters, sizeof(parameters), cudaMemcpyHostToDevice)) != cudaSuccess) goto cuda_error;

	if (parallel_tiles <= 32) {
		ns_splice_dp_warp<<<1, 32>>>(d_nas, d_aas, d_donor, d_acceptor, d_score, d_offsets,
				d_dimensions, d_parameters, d_trace, d_result);
	} else {
		size_t local_size = 32;
		while (local_size < (size_t)parallel_tiles && local_size < (size_t)ns_gpu_max_threads) local_size <<= 1;
		if (state_bytes > 49152)
			cudaFuncSetAttribute(ns_splice_dp_shared, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)state_bytes);
		ns_splice_dp_shared<<<1, (unsigned)local_size, state_bytes>>>(d_nas, d_aas, d_donor, d_acceptor, d_score,
				d_offsets, d_dimensions, d_parameters, d_trace, d_result);
	}
	if ((status = cudaGetLastError()) != cudaSuccess) goto cuda_error;
	if ((status = cudaMemcpy(trace, d_trace, trace_bytes, cudaMemcpyDeviceToHost)) != cudaSuccess) goto cuda_error;
	if ((status = cudaMemcpy(score_out, d_result, sizeof(int32_t), cudaMemcpyDeviceToHost)) != cudaSuccess) goto cuda_error;
	success = 1;
	goto release;

cuda_error:
	ns_gpu_set_error("GPU alignment", status);
release:
	if (d_nas) cudaFree(d_nas);
	if (d_aas) cudaFree(d_aas);
	if (d_donor) cudaFree(d_donor);
	if (d_acceptor) cudaFree(d_acceptor);
	if (d_score) cudaFree(d_score);
	if (d_offsets) cudaFree(d_offsets);
	if (d_dimensions) cudaFree(d_dimensions);
	if (d_parameters) cudaFree(d_parameters);
	if (d_trace) cudaFree(d_trace);
	if (d_result) cudaFree(d_result);
	return success;
}

static int ns_gpu_global_direct(void *km, const char *ns, int32_t nl, const char *as, int32_t al,
		const ns_opt_t *opt, const uint8_t *ss, ns_rst_t *r)
{
	uint8_t *prepared = 0, *aas = 0;
	int8_t *donor = 0, *acceptor = 0;
	uint16_t *trace = 0;
	int32_t narrow, parallel_tiles, score = 0, max_sub = 0;
	size_t state_bytes, trace_bytes;
	int64_t cells;
	int success = 0, i;

	if (nl < 3 || al < 1 || opt->asize < 1) return 0;
	cells = (int64_t)(nl - 2) * al;
	if (cells <= 0 || (uint64_t)cells > SIZE_MAX / sizeof(uint16_t)) return 0;
	if (!ns_gpu_state_size(nl, al, &state_bytes, &parallel_tiles)) return 0;
	trace_bytes = (size_t)cells * sizeof(uint16_t);
	for (i = 0; i < opt->asize * opt->asize; ++i)
		if (max_sub < opt->sc[i]) max_sub = opt->sc[i];
	narrow = (int64_t)al * max_sub + opt->end_bonus <= 30000;
	if (!narrow || state_bytes > ns_gpu_shared_mem_limit) goto cleanup;
	if (!ns_gpu_ready) goto cleanup;

	prepared = ns_gpu_prep_seq(km, ns, nl, as, al, opt, ss, &aas, &donor, &acceptor);
	trace = Kmalloc(km, uint16_t, cells);
	if (prepared == 0 || trace == 0) goto cleanup;

	success = ns_gpu_run_job(prepared, nl, aas, al, donor, acceptor, opt->sc, opt->asize,
			opt->go, opt->ge, opt->io, opt->fs, parallel_tiles, state_bytes, trace, trace_bytes, &score);
	if (success) {
		__sync_fetch_and_add(&ns_gpu_dispatches, 1);
		__sync_fetch_and_add(&ns_gpu_launches, 1);
		r->nt_len = nl, r->aa_len = al, r->score = score;
		ns_gpu_backtrack(km, trace, nl, al, r);
	}
cleanup:
	if (trace) kfree(km, trace);
	if (prepared) kfree(km, prepared);
	return success;
}

/*
 * Batches concurrently-submitted requests into a single kernel launch: one
 * CUDA block per job (kernels already index everything by blockIdx.x), a
 * single set of packed host->device buffers, and one bulk readback split
 * back out per request. Mirrors the OpenCL backend's batching scheme.
 */
static int ns_gpu_dispatch_batch(ns_gpu_request_t **requests, int count)
{
	uint8_t *d_nas = 0, *d_aas = 0;
	int8_t *d_donor = 0, *d_acceptor = 0, *d_score = 0;
	uint64_t *d_offsets = 0;
	int32_t *d_dimensions = 0, *d_parameters = 0, *d_result = 0;
	uint16_t *d_trace = 0;
	uint8_t *nas_data = 0, *aas_data = 0;
	int8_t *donor_data = 0, *acceptor_data = 0, *score_data = 0;
	uint64_t *offsets = 0;
	int32_t *dimensions = 0, *parameters = 0, *scores = 0;
	size_t total_nas = 0, total_aas = 0, total_sites = 0, total_scores = 0, total_trace = 0;
	size_t nas_offset = 0, aas_offset = 0, site_offset = 0, score_offset = 0, trace_offset = 0;
	size_t state_bytes = 0, trace_elem_offset = 0;
	int32_t max_parallel_tiles = 1;
	cudaError_t status = cudaSuccess;
	struct timespec ts0, ts1;
	int success = 0, i;

	for (i = 0; i < count; ++i) {
		ns_gpu_request_t *request = requests[i];
		size_t score_bytes = (size_t)request->asize * request->asize;
		total_nas += (size_t)request->nl;
		total_aas += (size_t)request->al;
		total_sites += (size_t)request->nl + 1;
		total_scores += score_bytes;
		total_trace += request->trace_bytes;
		if (state_bytes < request->state_bytes) state_bytes = request->state_bytes;
		if (max_parallel_tiles < request->parallel_tiles) max_parallel_tiles = request->parallel_tiles;
	}
	nas_data = (uint8_t*)malloc(total_nas);
	aas_data = (uint8_t*)malloc(total_aas);
	donor_data = (int8_t*)malloc(total_sites);
	acceptor_data = (int8_t*)malloc(total_sites);
	score_data = (int8_t*)malloc(total_scores);
	offsets = (uint64_t*)malloc((size_t)count * 6 * sizeof(*offsets));
	dimensions = (int32_t*)malloc((size_t)count * 3 * sizeof(*dimensions));
	parameters = (int32_t*)malloc((size_t)count * 5 * sizeof(*parameters));
	scores = (int32_t*)malloc((size_t)count * sizeof(*scores));
	if (!nas_data || !aas_data || !donor_data || !acceptor_data || !score_data ||
			!offsets || !dimensions || !parameters || !scores) {
		snprintf(ns_gpu_message, sizeof(ns_gpu_message), "failed to allocate GPU batch buffers");
		goto cleanup;
	}
	for (i = 0; i < count; ++i) {
		ns_gpu_request_t *request = requests[i];
		size_t score_bytes = (size_t)request->asize * request->asize;
		offsets[i * 6] = nas_offset;
		offsets[i * 6 + 1] = aas_offset;
		offsets[i * 6 + 2] = site_offset;
		offsets[i * 6 + 3] = site_offset;
		offsets[i * 6 + 4] = score_offset;
		offsets[i * 6 + 5] = trace_offset / sizeof(uint16_t);
		dimensions[i * 3] = request->nl;
		dimensions[i * 3 + 1] = request->al;
		dimensions[i * 3 + 2] = request->asize;
		parameters[i * 5] = request->go;
		parameters[i * 5 + 1] = request->ge;
		parameters[i * 5 + 2] = request->io;
		parameters[i * 5 + 3] = request->fs;
		parameters[i * 5 + 4] = 1;
		memcpy(nas_data + nas_offset, request->prepared, (size_t)request->nl);
		memcpy(aas_data + aas_offset, request->aas, (size_t)request->al);
		memcpy(donor_data + site_offset, request->donor, (size_t)request->nl + 1);
		memcpy(acceptor_data + site_offset, request->acceptor, (size_t)request->nl + 1);
		memcpy(score_data + score_offset, request->score_matrix, score_bytes);
		nas_offset += (size_t)request->nl;
		aas_offset += (size_t)request->al;
		site_offset += (size_t)request->nl + 1;
		score_offset += score_bytes;
		trace_offset += request->trace_bytes;
	}

	if ((status = ns_gpu_devbuf_ensure(&ns_gpu_buf_nas, total_nas)) != cudaSuccess) goto cuda_error;
	if ((status = ns_gpu_devbuf_ensure(&ns_gpu_buf_aas, total_aas)) != cudaSuccess) goto cuda_error;
	if ((status = ns_gpu_devbuf_ensure(&ns_gpu_buf_donor, total_sites)) != cudaSuccess) goto cuda_error;
	if ((status = ns_gpu_devbuf_ensure(&ns_gpu_buf_acceptor, total_sites)) != cudaSuccess) goto cuda_error;
	if ((status = ns_gpu_devbuf_ensure(&ns_gpu_buf_score, total_scores)) != cudaSuccess) goto cuda_error;
	if ((status = ns_gpu_devbuf_ensure(&ns_gpu_buf_offsets, (size_t)count * 6 * sizeof(*offsets))) != cudaSuccess) goto cuda_error;
	if ((status = ns_gpu_devbuf_ensure(&ns_gpu_buf_dimensions, (size_t)count * 3 * sizeof(*dimensions))) != cudaSuccess) goto cuda_error;
	if ((status = ns_gpu_devbuf_ensure(&ns_gpu_buf_parameters, (size_t)count * 5 * sizeof(*parameters))) != cudaSuccess) goto cuda_error;
	if ((status = ns_gpu_devbuf_ensure(&ns_gpu_buf_trace, total_trace)) != cudaSuccess) goto cuda_error;
	if ((status = ns_gpu_devbuf_ensure(&ns_gpu_buf_result, (size_t)count * sizeof(int32_t))) != cudaSuccess) goto cuda_error;
	d_nas = (uint8_t*)ns_gpu_buf_nas.ptr;
	d_aas = (uint8_t*)ns_gpu_buf_aas.ptr;
	d_donor = (int8_t*)ns_gpu_buf_donor.ptr;
	d_acceptor = (int8_t*)ns_gpu_buf_acceptor.ptr;
	d_score = (int8_t*)ns_gpu_buf_score.ptr;
	d_offsets = (uint64_t*)ns_gpu_buf_offsets.ptr;
	d_dimensions = (int32_t*)ns_gpu_buf_dimensions.ptr;
	d_parameters = (int32_t*)ns_gpu_buf_parameters.ptr;
	d_trace = (uint16_t*)ns_gpu_buf_trace.ptr;
	d_result = (int32_t*)ns_gpu_buf_result.ptr;

	clock_gettime(CLOCK_MONOTONIC, &ts0);
	if ((status = cudaMemcpy(d_nas, nas_data, total_nas, cudaMemcpyHostToDevice)) != cudaSuccess) goto cuda_error;
	if ((status = cudaMemcpy(d_aas, aas_data, total_aas, cudaMemcpyHostToDevice)) != cudaSuccess) goto cuda_error;
	if ((status = cudaMemcpy(d_donor, donor_data, total_sites, cudaMemcpyHostToDevice)) != cudaSuccess) goto cuda_error;
	if ((status = cudaMemcpy(d_acceptor, acceptor_data, total_sites, cudaMemcpyHostToDevice)) != cudaSuccess) goto cuda_error;
	if ((status = cudaMemcpy(d_score, score_data, total_scores, cudaMemcpyHostToDevice)) != cudaSuccess) goto cuda_error;
	if ((status = cudaMemcpy(d_offsets, offsets, (size_t)count * 6 * sizeof(*offsets), cudaMemcpyHostToDevice)) != cudaSuccess) goto cuda_error;
	if ((status = cudaMemcpy(d_dimensions, dimensions, (size_t)count * 3 * sizeof(*dimensions), cudaMemcpyHostToDevice)) != cudaSuccess) goto cuda_error;
	if ((status = cudaMemcpy(d_parameters, parameters, (size_t)count * 5 * sizeof(*parameters), cudaMemcpyHostToDevice)) != cudaSuccess) goto cuda_error;
	clock_gettime(CLOCK_MONOTONIC, &ts1);
	__sync_fetch_and_add(&ns_gpu_h2d_ns, (uint64_t)((ts1.tv_sec - ts0.tv_sec) * 1000000000LL + (ts1.tv_nsec - ts0.tv_nsec)));

	clock_gettime(CLOCK_MONOTONIC, &ts0);
	if (max_parallel_tiles <= 32) {
		ns_splice_dp_warp<<<count, 32>>>(d_nas, d_aas, d_donor, d_acceptor, d_score, d_offsets,
				d_dimensions, d_parameters, d_trace, d_result);
	} else {
		size_t local_size = 32;
		while (local_size < (size_t)max_parallel_tiles && local_size < (size_t)ns_gpu_max_threads) local_size <<= 1;
		if (state_bytes > 49152)
			cudaFuncSetAttribute(ns_splice_dp_shared, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)state_bytes);
		ns_splice_dp_shared<<<(unsigned)count, (unsigned)local_size, state_bytes>>>(d_nas, d_aas, d_donor, d_acceptor,
				d_score, d_offsets, d_dimensions, d_parameters, d_trace, d_result);
	}
	if ((status = cudaGetLastError()) != cudaSuccess) goto cuda_error;
	if ((status = cudaDeviceSynchronize()) != cudaSuccess) goto cuda_error; /* isolate kernel time from D2H time below */
	clock_gettime(CLOCK_MONOTONIC, &ts1);
	__sync_fetch_and_add(&ns_gpu_kernel_ns, (uint64_t)((ts1.tv_sec - ts0.tv_sec) * 1000000000LL + (ts1.tv_nsec - ts0.tv_nsec)));

	if (ns_gpu_trace_host_cap < total_trace) {
		uint16_t *grown = (uint16_t*)realloc(ns_gpu_trace_host_buf, total_trace);
		if (!grown) {
			snprintf(ns_gpu_message, sizeof(ns_gpu_message), "failed to allocate GPU trace staging buffer");
			goto cleanup;
		}
		ns_gpu_trace_host_buf = grown;
		ns_gpu_trace_host_cap = total_trace;
	}
	clock_gettime(CLOCK_MONOTONIC, &ts0);
	if ((status = cudaMemcpy(ns_gpu_trace_host_buf, d_trace, total_trace, cudaMemcpyDeviceToHost)) != cudaSuccess) goto cuda_error;
	if ((status = cudaMemcpy(scores, d_result, (size_t)count * sizeof(int32_t), cudaMemcpyDeviceToHost)) != cudaSuccess) goto cuda_error;
	clock_gettime(CLOCK_MONOTONIC, &ts1);
	__sync_fetch_and_add(&ns_gpu_d2h_ns, (uint64_t)((ts1.tv_sec - ts0.tv_sec) * 1000000000LL + (ts1.tv_nsec - ts0.tv_nsec)));
	for (i = 0; i < count; ++i) {
		size_t elems = requests[i]->trace_bytes / sizeof(uint16_t);
		memcpy(requests[i]->trace, ns_gpu_trace_host_buf + trace_elem_offset, requests[i]->trace_bytes);
		trace_elem_offset += elems;
	}
	for (i = 0; i < count; ++i) requests[i]->score = scores[i];
	__sync_fetch_and_add(&ns_gpu_dispatches, count);
	__sync_fetch_and_add(&ns_gpu_launches, 1);
	success = 1;
	goto cleanup;

cuda_error:
	ns_gpu_set_error("GPU batch alignment", status);
cleanup:
	/* device buffers are persistent (see ns_gpu_devbuf_ensure) and intentionally not freed here */
	free(scores); free(parameters); free(dimensions); free(offsets);
	free(score_data); free(acceptor_data); free(donor_data); free(aas_data); free(nas_data);
	return success;
}

/* collects up to NS_GPU_MAX_BATCH pending requests, waiting up to NS_GPU_BATCH_WAIT_NS for more to arrive */
static void *ns_gpu_collect_batches(void *unused)
{
	(void)unused;
	for (;;) {
		ns_gpu_batch_t *batch;
		struct timespec deadline;
		int status, chosen, b;
		pthread_mutex_lock(&ns_gpu_mutex);
		while (ns_gpu_queue_size == 0 || ns_gpu_batches_active >= 1)
			pthread_cond_wait(&ns_gpu_condition, &ns_gpu_mutex);
		/* batch the fullest bucket first so a launch never mixes wildly different job sizes */
		chosen = -1;
		for (b = 0; b < NS_GPU_NUM_BUCKETS; ++b)
			if (ns_gpu_bucket_size[b] > 0 && (chosen < 0 || ns_gpu_bucket_size[b] > ns_gpu_bucket_size[chosen]))
				chosen = b;
		clock_gettime(CLOCK_REALTIME, &deadline);
		deadline.tv_nsec += NS_GPU_BATCH_WAIT_NS;
		if (deadline.tv_nsec >= 1000000000L)
			++deadline.tv_sec, deadline.tv_nsec -= 1000000000L;
		while (ns_gpu_bucket_size[chosen] < NS_GPU_MAX_BATCH) {
			status = pthread_cond_timedwait(&ns_gpu_condition, &ns_gpu_mutex, &deadline);
			if (status == ETIMEDOUT) break;
		}
		batch = (ns_gpu_batch_t*)malloc(sizeof(*batch));
		if (batch == 0) {
			ns_gpu_request_t *request = ns_gpu_bucket_head[chosen];
			ns_gpu_bucket_head[chosen] = request->next;
			if (--ns_gpu_bucket_size[chosen] == 0) ns_gpu_bucket_tail[chosen] = 0;
			--ns_gpu_queue_size;
			request->success = 0;
			request->done = 1;
			snprintf(ns_gpu_message, sizeof(ns_gpu_message), "failed to allocate GPU batch");
			pthread_cond_broadcast(&ns_gpu_condition);
			pthread_mutex_unlock(&ns_gpu_mutex);
			continue;
		}
		memset(batch, 0, sizeof(*batch));
		while (batch->count < NS_GPU_MAX_BATCH && ns_gpu_bucket_head[chosen]) {
			batch->requests[batch->count++] = ns_gpu_bucket_head[chosen];
			ns_gpu_bucket_head[chosen] = ns_gpu_bucket_head[chosen]->next;
			--ns_gpu_bucket_size[chosen];
			--ns_gpu_queue_size;
		}
		if (ns_gpu_bucket_head[chosen] == 0) ns_gpu_bucket_tail[chosen] = 0;
		if (ns_gpu_batch_tail) ns_gpu_batch_tail->next = batch;
		else ns_gpu_batch_head = batch;
		ns_gpu_batch_tail = batch;
		++ns_gpu_batches_active;
		pthread_cond_broadcast(&ns_gpu_condition);
		pthread_mutex_unlock(&ns_gpu_mutex);
	}
	return 0;
}

static void *ns_gpu_execute_batches(void *unused)
{
	(void)unused;
	cudaSetDevice(ns_gpu_device_index); /* CUDA context is per-thread; select the requested device here too */
	for (;;) {
		ns_gpu_batch_t *batch;
		int i, success;
		pthread_mutex_lock(&ns_gpu_mutex);
		while (ns_gpu_batch_head == 0)
			pthread_cond_wait(&ns_gpu_condition, &ns_gpu_mutex);
		batch = ns_gpu_batch_head;
		ns_gpu_batch_head = batch->next;
		if (ns_gpu_batch_head == 0) ns_gpu_batch_tail = 0;
		pthread_mutex_unlock(&ns_gpu_mutex);

		success = ns_gpu_dispatch_batch(batch->requests, batch->count);
		pthread_mutex_lock(&ns_gpu_mutex);
		for (i = 0; i < batch->count; ++i) {
			batch->requests[i]->success = success;
			batch->requests[i]->done = 1;
		}
		--ns_gpu_batches_active;
		pthread_cond_broadcast(&ns_gpu_condition);
		pthread_mutex_unlock(&ns_gpu_mutex);
		free(batch);
	}
	return 0;
}

ns_gpu_request_t *ns_gpu_submit(void *km, const char *ns, int32_t nl, const char *as, int32_t al,
		const ns_opt_t *opt, const uint8_t *ss)
{
	ns_gpu_request_t *request;
	int32_t i, max_sub = 0;
	int64_t cells;

	if (!ns_gpu_worker_started) return 0;
	if (nl < 3 || al < 1 || opt->asize < 1) return 0;
	cells = (int64_t)(nl - 2) * al;
	if (cells <= 0 || (uint64_t)cells > SIZE_MAX / sizeof(uint16_t)) return 0;
	for (i = 0; i < opt->asize * opt->asize; ++i)
		if (max_sub < opt->sc[i]) max_sub = opt->sc[i];
	request = (ns_gpu_request_t*)calloc(1, sizeof(*request));
	if (request == 0) return 0;
	request->km = km;
	request->narrow = (int64_t)al * max_sub + opt->end_bonus <= 30000;
	if (!request->narrow || !ns_gpu_state_size(nl, al, &request->state_bytes, &request->parallel_tiles) ||
			request->state_bytes > ns_gpu_shared_mem_limit) goto fail;
	request->prepared = ns_gpu_prep_seq(km, ns, nl, as, al, opt, ss, &request->aas, &request->donor, &request->acceptor);
	request->trace = Kmalloc(km, uint16_t, cells);
	request->score_matrix = Kmalloc(km, int8_t, (size_t)opt->asize * opt->asize);
	if (!request->prepared || !request->trace || !request->score_matrix) goto fail;
	memcpy(request->score_matrix, opt->sc, (size_t)opt->asize * opt->asize);
	request->nl = nl, request->al = al, request->asize = opt->asize;
	request->go = opt->go, request->ge = opt->ge, request->io = opt->io, request->fs = opt->fs;
	request->trace_bytes = (size_t)cells * sizeof(uint16_t);

	{
		int bucket = ns_gpu_bucket_for(request->parallel_tiles);
		pthread_mutex_lock(&ns_gpu_mutex);
		if (ns_gpu_bucket_tail[bucket]) ns_gpu_bucket_tail[bucket]->next = request;
		else ns_gpu_bucket_head[bucket] = request;
		ns_gpu_bucket_tail[bucket] = request;
		++ns_gpu_bucket_size[bucket];
		++ns_gpu_queue_size;
		pthread_cond_broadcast(&ns_gpu_condition);
		pthread_mutex_unlock(&ns_gpu_mutex);
	}
	return request;

fail:
	if (request->score_matrix) kfree(km, request->score_matrix);
	if (request->trace) kfree(km, request->trace);
	if (request->prepared) kfree(km, request->prepared);
	free(request);
	return 0;
}

int ns_gpu_wait(ns_gpu_request_t *request, ns_rst_t *r)
{
	int success;
	if (request == 0) return 0;
	pthread_mutex_lock(&ns_gpu_mutex);
	while (!request->done)
		pthread_cond_wait(&ns_gpu_condition, &ns_gpu_mutex);
	success = request->success;
	pthread_mutex_unlock(&ns_gpu_mutex);
	if (success) {
		r->nt_len = request->nl, r->aa_len = request->al, r->score = request->score;
		ns_gpu_backtrack(request->km, request->trace, request->nl, request->al, r);
	}
	if (request->score_matrix) kfree(request->km, request->score_matrix);
	if (request->trace) kfree(request->km, request->trace);
	if (request->prepared) kfree(request->km, request->prepared);
	free(request);
	return success;
}

int ns_gpu_global(void *km, const char *ns, int32_t nl, const char *as, int32_t al,
		const ns_opt_t *opt, const uint8_t *ss, ns_rst_t *r)
{
	ns_gpu_request_t *request;
	if (!ns_gpu_worker_started)
		return ns_gpu_global_direct(km, ns, nl, as, al, opt, ss, r);
	request = ns_gpu_submit(km, ns, nl, as, al, opt, ss);
	return request? ns_gpu_wait(request, r) : 0;
}

/* ------------------------------------------------------------------ */
/* --gpu-prefilter: GPU gapless X-drop candidate prefilter kernel      */
/* ------------------------------------------------------------------ */

__device__ __forceinline__ uint8_t ns_gapless_base(const uint8_t *packed_nt,
		int64_t target_offset, int64_t target_length, int32_t reverse, int64_t position)
{
	int64_t absolute;
	uint8_t base;
	if (position < 0 || position >= target_length) return 4;
	absolute = target_offset + (reverse? target_length - 1 - position : position);
	base = packed_nt[absolute >> 1] >> ((absolute & 1) * 4) & 0xf;
	return reverse && base < 4? 3 - base : base;
}

__device__ __forceinline__ uint8_t ns_gapless_translate(const uint8_t *packed_nt,
		const uint8_t *codon, int64_t target_offset, int64_t target_length,
		int32_t reverse, int64_t codon_end)
{
	const uint8_t b0 = ns_gapless_base(packed_nt, target_offset, target_length, reverse, codon_end - 2);
	const uint8_t b1 = ns_gapless_base(packed_nt, target_offset, target_length, reverse, codon_end - 1);
	const uint8_t b2 = ns_gapless_base(packed_nt, target_offset, target_length, reverse, codon_end);
	return b0 < 4 && b1 < 4 && b2 < 4? codon[(b0 << 4) | (b1 << 2) | b2] : 20;
}

#ifndef NS_GAPLESS_MAX_ANCHORS
#define NS_GAPLESS_MAX_ANCHORS 32
#endif

/*
 * Splice-aware gapless prefilter score. Each anchor seeds its own gapless
 * X-drop segment (roughly one exon); segments are then summed after removing
 * query overlap, so a multi-exon gene is no longer outscored by an intronless
 * processed pseudogene whose single diagonal spans the whole protein.
 */
__global__ void ns_gapless_xdrop_batch(const uint8_t *packed_nt, const uint8_t *queries,
		const ns_gpu_gapless_candidate_t *candidates, const ns_gpu_gapless_anchor_t *anchors,
		int32_t n_candidates, int32_t block_width, int32_t kmer, int32_t xdrop,
		const int8_t *score_matrix, int32_t asize, const uint8_t *codon, int32_t *scores)
{
	const int32_t lane = threadIdx.x & 31;
	const int32_t candidate_index = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
	// per-lane slot: lane i owns the best seed of anchor i (anchor_count <= 32)
	int32_t my_seed = INT32_MIN, my_query_end = -1;
	int64_t my_target_end = INT64_MAX;
	int32_t my_qs = 0, my_qe = -1, my_score = INT32_MIN;
	int32_t anchor_count;
	if (candidate_index >= n_candidates) return;
	const ns_gpu_gapless_candidate_t candidate = candidates[candidate_index];
	anchor_count = candidate.anchor_count < NS_GAPLESS_MAX_ANCHORS? candidate.anchor_count : NS_GAPLESS_MAX_ANCHORS;
	for (int32_t anchor_index = 0; anchor_index < anchor_count; ++anchor_index) {
		const ns_gpu_gapless_anchor_t anchor = anchors[candidate.anchor_offset + anchor_index];
		int32_t best_seed = INT32_MIN, best_query_end = -1;
		int64_t best_target_end = INT64_MAX;
		if (anchor.query_end >= kmer - 1 && anchor.query_end < candidate.query_length) {
			for (int32_t delta = lane; delta < block_width; delta += 32) {
				const int64_t target_end = anchor.target_block + delta;
				int32_t seed_score = 0;
				if (target_end - (int64_t)(kmer - 1) * 3 - 2 < 0 || target_end >= candidate.target_length)
					continue;
				for (int32_t seed_index = 0; seed_index < kmer; ++seed_index) {
					const int64_t codon_end = target_end - (int64_t)(kmer - 1 - seed_index) * 3;
					const uint8_t target_aa = ns_gapless_translate(packed_nt, codon,
							candidate.target_offset, candidate.target_length, candidate.reverse, codon_end);
					const uint8_t query_aa = queries[candidate.query_offset + anchor.query_end - kmer + 1 + seed_index];
					seed_score += score_matrix[(int32_t)target_aa * asize + query_aa];
				}
				if (seed_score > best_seed || (seed_score == best_seed && target_end < best_target_end))
					best_seed = seed_score, best_target_end = target_end, best_query_end = anchor.query_end;
			}
		}
		for (int32_t offset = 16; offset > 0; offset >>= 1) {
			const int32_t other_score = __shfl_down_sync(0xffffffffu, best_seed, offset);
			const int64_t other_target = __shfl_down_sync(0xffffffffu, best_target_end, offset);
			const int32_t other_query = __shfl_down_sync(0xffffffffu, best_query_end, offset);
			if (other_score > best_seed || (other_score == best_seed && other_target < best_target_end))
				best_seed = other_score, best_target_end = other_target, best_query_end = other_query;
		}
		best_seed = __shfl_sync(0xffffffffu, best_seed, 0);
		best_target_end = __shfl_sync(0xffffffffu, best_target_end, 0);
		best_query_end = __shfl_sync(0xffffffffu, best_query_end, 0);
		if (lane == anchor_index)
			my_seed = best_seed, my_target_end = best_target_end, my_query_end = best_query_end;
	}
	// lanes 0..anchor_count-1 extend their own anchor in parallel
	if (lane < anchor_count && my_query_end >= 0) {
		int32_t left_running = 0, left_best = 0, left_len = 0;
		int32_t right_running = 0, right_best = 0, right_len = 0;
		for (int32_t query_pos = my_query_end - kmer, step = 1; query_pos >= 0; --query_pos, ++step) {
			const int64_t target_end = my_target_end - (int64_t)(kmer - 1 + step) * 3;
			uint8_t target_aa;
			if (target_end - 2 < 0) break;
			target_aa = ns_gapless_translate(packed_nt, codon, candidate.target_offset,
					candidate.target_length, candidate.reverse, target_end);
			left_running += score_matrix[(int32_t)target_aa * asize + queries[candidate.query_offset + query_pos]];
			if (left_running > left_best) left_best = left_running, left_len = step;
			if (left_best - left_running > xdrop) break;
		}
		for (int32_t query_pos = my_query_end + 1, step = 1;
				query_pos < candidate.query_length; ++query_pos, ++step) {
			const int64_t target_end = my_target_end + (int64_t)step * 3;
			uint8_t target_aa;
			if (target_end >= candidate.target_length) break;
			target_aa = ns_gapless_translate(packed_nt, codon, candidate.target_offset,
					candidate.target_length, candidate.reverse, target_end);
			right_running += score_matrix[(int32_t)target_aa * asize + queries[candidate.query_offset + query_pos]];
			if (right_running > right_best) right_best = right_running, right_len = step;
			if (right_best - right_running > xdrop) break;
		}
		my_score = my_seed + left_best + right_best;
		my_qs = my_query_end - (kmer - 1) - left_len;
		my_qe = my_query_end + right_len;
	}
	__syncwarp();
	{
		int32_t seg_qs[NS_GAPLESS_MAX_ANCHORS], seg_qe[NS_GAPLESS_MAX_ANCHORS], seg_sc[NS_GAPLESS_MAX_ANCHORS];
		int32_t n_seg = 0, total = 0, any = 0;
		// the shuffles are warp-uniform; only lane 0 keeps the gathered segments
		for (int32_t i = 0; i < anchor_count; ++i) {
			const int32_t sc = __shfl_sync(0xffffffffu, my_score, i);
			const int32_t qs = __shfl_sync(0xffffffffu, my_qs, i);
			const int32_t qe = __shfl_sync(0xffffffffu, my_qe, i);
			int32_t j;
			if (lane != 0 || sc == INT32_MIN) continue;
			any = 1;
			for (j = n_seg; j > 0 && seg_sc[j - 1] < sc; --j)
				seg_qs[j] = seg_qs[j - 1], seg_qe[j] = seg_qe[j - 1], seg_sc[j] = seg_sc[j - 1];
			seg_qs[j] = qs, seg_qe[j] = qe, seg_sc[j] = sc, ++n_seg;
		}
		if (lane != 0) return;
		// greedy by score: a segment contributes only its query span not already covered;
		// beyond the best segment, only segments above the X-drop noise scale are summed
		// so that chains of many weak anchors do not outscore one genuine exon
		for (int32_t i = 0; i < n_seg; ++i) {
			const int32_t len = seg_qe[i] - seg_qs[i] + 1;
			int32_t overlap = 0;
			if (i > 0 && seg_sc[i] < xdrop) break;
			for (int32_t j = 0; j < i; ++j) {
				const int32_t s = seg_qs[i] > seg_qs[j]? seg_qs[i] : seg_qs[j];
				const int32_t e = seg_qe[i] < seg_qe[j]? seg_qe[i] : seg_qe[j];
				if (seg_sc[j] != INT32_MIN && e >= s) overlap += e - s + 1;
			}
			if (overlap >= len) { seg_sc[i] = INT32_MIN; continue; } // same diagonal as a better segment
			if (overlap > 0) total += (int32_t)((int64_t)seg_sc[i] * (len - overlap) / len);
			else total += seg_sc[i];
		}
		scores[candidate_index] = any? total : INT32_MIN;
	}
}

enum {
	NS_GAPLESS_BUF_TARGET,
	NS_GAPLESS_BUF_QUERY,
	NS_GAPLESS_BUF_CANDIDATE,
	NS_GAPLESS_BUF_ANCHOR,
	NS_GAPLESS_BUF_SCORE_MATRIX,
	NS_GAPLESS_BUF_CODON,
	NS_GAPLESS_BUF_RESULT,
	NS_GAPLESS_BUF_COUNT
};
static ns_gpu_devbuf_t ns_gpu_gapless_buf[NS_GAPLESS_BUF_COUNT];
static const uint8_t *ns_gpu_gapless_target_source;
static int64_t ns_gpu_gapless_target_bases;
static uint64_t ns_gpu_gapless_batches, ns_gpu_gapless_candidates;
static uint64_t ns_gpu_gapless_anchors, ns_gpu_gapless_seed_positions;
// mp_gpu_prefilter() runs inside kt_for() worker threads; the buffers above are shared, so
// ensure/memcpy/launch/copy-back must be serialized (ns_gpu_mutex only covers the DP job queue)
static pthread_mutex_t ns_gpu_gapless_mutex = PTHREAD_MUTEX_INITIALIZER;

void ns_gpu_gapless_profile(ns_gpu_gapless_profile_t *profile)
{
	profile->batches = __sync_fetch_and_add(&ns_gpu_gapless_batches, 0);
	profile->candidates = __sync_fetch_and_add(&ns_gpu_gapless_candidates, 0);
	profile->anchors = __sync_fetch_and_add(&ns_gpu_gapless_anchors, 0);
	profile->seed_positions = __sync_fetch_and_add(&ns_gpu_gapless_seed_positions, 0);
}

int ns_gpu_gapless_supported(void) { return 1; }

void ns_gpu_gapless_reset_target(void)
{
	pthread_mutex_lock(&ns_gpu_gapless_mutex);
	ns_gpu_gapless_target_source = 0;
	ns_gpu_gapless_target_bases = 0;
	pthread_mutex_unlock(&ns_gpu_gapless_mutex);
}

int ns_gpu_gapless_score_batch(const uint8_t *packed_nt, int64_t packed_nt_bases,
		const uint8_t *queries, int64_t query_aa,
		const ns_gpu_gapless_candidate_t *candidates, int32_t n_candidates,
		const ns_gpu_gapless_anchor_t *anchors, int64_t n_anchors,
		int32_t block_width, int32_t kmer, int32_t xdrop,
		const int8_t *score_matrix, int32_t asize, const uint8_t *codon,
		int32_t *scores)
{
	size_t sizes[NS_GAPLESS_BUF_COUNT];
	cudaError_t status = cudaSuccess;
	struct timespec ts0, ts1;
	const int32_t warps_per_block = 8;
	if (!ns_gpu_worker_started || !packed_nt || packed_nt_bases < 1 || !queries || query_aa < 1 ||
			!candidates || n_candidates < 1 || !anchors || n_anchors < 1 || block_width < 1 ||
			kmer < 1 || xdrop < 0 || !score_matrix || asize < 1 || !codon || !scores)
		return 0;
	sizes[NS_GAPLESS_BUF_TARGET] = ((size_t)packed_nt_bases + 1) >> 1;
	sizes[NS_GAPLESS_BUF_QUERY] = (size_t)query_aa;
	sizes[NS_GAPLESS_BUF_CANDIDATE] = (size_t)n_candidates * sizeof(*candidates);
	sizes[NS_GAPLESS_BUF_ANCHOR] = (size_t)n_anchors * sizeof(*anchors);
	sizes[NS_GAPLESS_BUF_SCORE_MATRIX] = (size_t)asize * asize;
	sizes[NS_GAPLESS_BUF_CODON] = 64;
	sizes[NS_GAPLESS_BUF_RESULT] = (size_t)n_candidates * sizeof(*scores);
	pthread_mutex_lock(&ns_gpu_gapless_mutex);
	for (int32_t i = 0; i < NS_GAPLESS_BUF_COUNT; ++i)
		if ((status = ns_gpu_devbuf_ensure(&ns_gpu_gapless_buf[i], sizes[i])) != cudaSuccess)
			goto cuda_error;
	clock_gettime(CLOCK_MONOTONIC, &ts0);
	if (ns_gpu_gapless_target_source != packed_nt || ns_gpu_gapless_target_bases != packed_nt_bases) {
		if ((status = cudaMemcpy(ns_gpu_gapless_buf[NS_GAPLESS_BUF_TARGET].ptr, packed_nt,
				sizes[NS_GAPLESS_BUF_TARGET], cudaMemcpyHostToDevice)) != cudaSuccess) goto cuda_error;
		ns_gpu_gapless_target_source = packed_nt;
		ns_gpu_gapless_target_bases = packed_nt_bases;
	}
	if ((status = cudaMemcpy(ns_gpu_gapless_buf[NS_GAPLESS_BUF_QUERY].ptr, queries,
			sizes[NS_GAPLESS_BUF_QUERY], cudaMemcpyHostToDevice)) != cudaSuccess ||
			(status = cudaMemcpy(ns_gpu_gapless_buf[NS_GAPLESS_BUF_CANDIDATE].ptr, candidates,
			sizes[NS_GAPLESS_BUF_CANDIDATE], cudaMemcpyHostToDevice)) != cudaSuccess ||
			(status = cudaMemcpy(ns_gpu_gapless_buf[NS_GAPLESS_BUF_ANCHOR].ptr, anchors,
			sizes[NS_GAPLESS_BUF_ANCHOR], cudaMemcpyHostToDevice)) != cudaSuccess ||
			(status = cudaMemcpy(ns_gpu_gapless_buf[NS_GAPLESS_BUF_SCORE_MATRIX].ptr, score_matrix,
			sizes[NS_GAPLESS_BUF_SCORE_MATRIX], cudaMemcpyHostToDevice)) != cudaSuccess ||
			(status = cudaMemcpy(ns_gpu_gapless_buf[NS_GAPLESS_BUF_CODON].ptr, codon,
			sizes[NS_GAPLESS_BUF_CODON], cudaMemcpyHostToDevice)) != cudaSuccess)
		goto cuda_error;
	clock_gettime(CLOCK_MONOTONIC, &ts1);
	__sync_fetch_and_add(&ns_gpu_h2d_ns, (uint64_t)((ts1.tv_sec - ts0.tv_sec) * 1000000000LL + (ts1.tv_nsec - ts0.tv_nsec)));
	clock_gettime(CLOCK_MONOTONIC, &ts0);
	ns_gapless_xdrop_batch<<<(unsigned)((n_candidates + warps_per_block - 1) / warps_per_block),
			warps_per_block * 32>>>(
			(const uint8_t*)ns_gpu_gapless_buf[NS_GAPLESS_BUF_TARGET].ptr,
			(const uint8_t*)ns_gpu_gapless_buf[NS_GAPLESS_BUF_QUERY].ptr,
			(const ns_gpu_gapless_candidate_t*)ns_gpu_gapless_buf[NS_GAPLESS_BUF_CANDIDATE].ptr,
			(const ns_gpu_gapless_anchor_t*)ns_gpu_gapless_buf[NS_GAPLESS_BUF_ANCHOR].ptr,
			n_candidates, block_width, kmer, xdrop,
			(const int8_t*)ns_gpu_gapless_buf[NS_GAPLESS_BUF_SCORE_MATRIX].ptr, asize,
			(const uint8_t*)ns_gpu_gapless_buf[NS_GAPLESS_BUF_CODON].ptr,
			(int32_t*)ns_gpu_gapless_buf[NS_GAPLESS_BUF_RESULT].ptr);
	if ((status = cudaGetLastError()) != cudaSuccess ||
			(status = cudaDeviceSynchronize()) != cudaSuccess) goto cuda_error;
	clock_gettime(CLOCK_MONOTONIC, &ts1);
	__sync_fetch_and_add(&ns_gpu_kernel_ns, (uint64_t)((ts1.tv_sec - ts0.tv_sec) * 1000000000LL + (ts1.tv_nsec - ts0.tv_nsec)));
	clock_gettime(CLOCK_MONOTONIC, &ts0);
	if ((status = cudaMemcpy(scores, ns_gpu_gapless_buf[NS_GAPLESS_BUF_RESULT].ptr,
			sizes[NS_GAPLESS_BUF_RESULT], cudaMemcpyDeviceToHost)) != cudaSuccess) goto cuda_error;
	clock_gettime(CLOCK_MONOTONIC, &ts1);
	__sync_fetch_and_add(&ns_gpu_d2h_ns, (uint64_t)((ts1.tv_sec - ts0.tv_sec) * 1000000000LL + (ts1.tv_nsec - ts0.tv_nsec)));
	__sync_fetch_and_add(&ns_gpu_gapless_batches, 1);
	__sync_fetch_and_add(&ns_gpu_gapless_candidates, (uint64_t)n_candidates);
	__sync_fetch_and_add(&ns_gpu_gapless_anchors, (uint64_t)n_anchors);
	__sync_fetch_and_add(&ns_gpu_gapless_seed_positions, (uint64_t)n_anchors * block_width);
	pthread_mutex_unlock(&ns_gpu_gapless_mutex);
	return 1;

cuda_error:
	ns_gpu_set_error("gapless GPU batch", status);
	pthread_mutex_unlock(&ns_gpu_gapless_mutex);
	return 0;
}

#endif
