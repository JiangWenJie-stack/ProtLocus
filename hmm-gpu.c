#include <stdint.h>
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "protlocus.h"
#include "mppriv.h"

#ifdef NS_OPENCL

// MP_HMM_GPU_DEBUG: blocking time mapping threads spend in the profile DP launches, printed at exit
static pthread_mutex_t mp_hmm_gpu_stat_mutex = PTHREAD_MUTEX_INITIALIZER;
static double mp_hmm_gpu_t_wait;
static uint64_t mp_hmm_gpu_n_launch, mp_hmm_gpu_n_reg, mp_hmm_gpu_nt;
static int mp_hmm_gpu_debug = -1;

static double mp_hmm_gpu_now(void)
{
	struct timeval tv;
	gettimeofday(&tv, 0);
	return tv.tv_sec + tv.tv_usec * 1e-6;
}

static void mp_hmm_gpu_print_stats(void)
{
	fprintf(stderr, "[mp_hmm_gpu] launches=%llu regions=%llu nt_MB=%.1f wait_s=%.3f\n",
		(unsigned long long)mp_hmm_gpu_n_launch, (unsigned long long)mp_hmm_gpu_n_reg, mp_hmm_gpu_nt / 1048576.0, mp_hmm_gpu_t_wait);
}

static int mp_hmm_gpu_debug_on(void)
{
	if (mp_hmm_gpu_debug < 0) {
		mp_hmm_gpu_debug = getenv("MP_HMM_GPU_DEBUG") != 0;
		if (mp_hmm_gpu_debug) atexit(mp_hmm_gpu_print_stats);
	}
	return mp_hmm_gpu_debug;
}

typedef int32_t cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef cl_ulong cl_bitfield;
typedef cl_bitfield cl_device_type;
typedef cl_bitfield cl_mem_flags;
typedef struct _cl_platform_id *cl_platform_id;
typedef struct _cl_device_id *cl_device_id;
typedef struct _cl_context *cl_context;
typedef struct _cl_command_queue *cl_command_queue;
typedef struct _cl_mem *cl_mem;
typedef struct _cl_program *cl_program;
typedef struct _cl_kernel *cl_kernel;

#define CL_SUCCESS 0
#define CL_TRUE 1
#define CL_DEVICE_TYPE_GPU (1ULL << 2)
#define CL_MEM_READ_ONLY (1ULL << 2)
#define CL_MEM_WRITE_ONLY (1ULL << 1)
#define CL_MEM_COPY_HOST_PTR (1ULL << 5)
#define CL_DEVICE_LOCAL_MEM_SIZE 0x1023
#define CL_DEVICE_MAX_WORK_GROUP_SIZE 0x1004
#define CL_PROGRAM_BUILD_LOG 0x1183

extern cl_int clGetPlatformIDs(cl_uint, cl_platform_id *, cl_uint *);
extern cl_int clGetDeviceIDs(cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *);
extern cl_int clGetDeviceInfo(cl_device_id, cl_uint, size_t, void *, size_t *);
extern cl_context clCreateContext(const intptr_t *, cl_uint, const cl_device_id *, void (*)(const char *, const void *, size_t, void *), void *, cl_int *);
extern cl_command_queue clCreateCommandQueue(cl_context, cl_device_id, cl_bitfield, cl_int *);
extern cl_program clCreateProgramWithSource(cl_context, cl_uint, const char **, const size_t *, cl_int *);
extern cl_int clBuildProgram(cl_program, cl_uint, const cl_device_id *, const char *, void (*)(cl_program, void *), void *);
extern cl_int clGetProgramBuildInfo(cl_program, cl_device_id, cl_uint, size_t, void *, size_t *);
extern cl_kernel clCreateKernel(cl_program, const char *, cl_int *);
extern cl_mem clCreateBuffer(cl_context, cl_mem_flags, size_t, void *, cl_int *);
extern cl_int clSetKernelArg(cl_kernel, cl_uint, size_t, const void *);
extern cl_int clEnqueueNDRangeKernel(cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *, const size_t *, cl_uint, const void *, void *);
extern cl_int clEnqueueReadBuffer(cl_command_queue, cl_mem, cl_uint, size_t, size_t, void *, cl_uint, const void *, void *);
extern cl_int clFinish(cl_command_queue);
extern cl_int clReleaseMemObject(cl_mem);
extern cl_int clReleaseKernel(cl_kernel);
extern cl_int clReleaseProgram(cl_program);
extern cl_int clReleaseCommandQueue(cl_command_queue);
extern cl_int clReleaseContext(cl_context);

extern uint8_t ns_tab_codon[64];

/* codon -> mp_hmm_alphabet-order aa index (or 20 = stop/no-emission), so the GPU kernel's
 * emit[(j-1)*20+aa] lookup matches how mp_hmm_read1 stores emit[] (see hmm.c). */
static const uint8_t *mp_hmm_gpu_codon_alpha(void)
{
	static uint8_t tab[64];
	static int ready = 0;
	if (!ready) {
		const int8_t *a2a = mp_hmm_aa20_to_alpha_table();
		int c;
		for (c = 0; c < 64; ++c) {
			int base = ns_tab_codon[c];
			tab[c] = (uint8_t)(base < 20 && a2a[base] >= 0? a2a[base] : 20);
		}
		ready = 1;
	}
	return tab;
}

/* One work-group per candidate region (group g scores nt[off[g] .. off[g]+len[g])).
 * Lanes stripe over profile positions; each lane keeps its own running maximum so
 * a row costs a single barrier, and the maxima are reduced once at the end. */
static const char mp_hmm_gpu_source[] =
"inline int mx(int a, int b) { return a > b ? a : b; }\n"
"inline int addv(int a, int b) { int x = a + b; return x < 0 ? 0 : x; }\n"
"__kernel void mp_hmm_dp(__global const uchar *nt, __global const int *off,\n"
"    __global const int *len, int m,\n"
"    __global const short *emit, __global const short *trans,\n"
"    __global const uchar *codon, __local int *pm, __local int *pi,\n"
"    __local int *pd, __local int *cm, __local int *ci, __local int *cd,\n"
"    __global int *answer) {\n"
"  int g = (int)get_group_id(0);\n"
"  int lane = (int)get_local_id(0), lanes = (int)get_local_size(0);\n"
"  int nlen = len[g];\n"
"  nt += off[g];\n"
"  for (int j = lane; j <= m; j += lanes) pm[j] = pi[j] = pd[j] = 0;\n"
"  barrier(CLK_LOCAL_MEM_FENCE);\n"
"  int best = 0;\n"
"  for (int i = 0; i < nlen / 3; ++i) {\n"
"    int aa = (nt[i * 3] > 3 || nt[i * 3 + 1] > 3 || nt[i * 3 + 2] > 3) ? 20 :\n"
"      (int)codon[(int)nt[i * 3] * 16 + (int)nt[i * 3 + 1] * 4 + (int)nt[i * 3 + 2]];\n"
"    if (lane == 0) cm[0] = ci[0] = cd[0] = 0;\n"
"    for (int j = lane + 1; j <= m; j += lanes) {\n"
"      int k = (j - 1) * 7, from = mx(pm[j - 1] + trans[k],\n"
"        mx(pi[j - 1] + trans[k + 3], pd[j - 1] + trans[k + 5]));\n"
"      int vm = addv(from, aa < 20 ? emit[(j - 1) * 20 + aa] : 0);\n"
"      int vi = mx(0, mx(pm[j] + trans[k + 1], pi[j] + trans[k + 4]));\n"
"      int vd = mx(0, mx(pm[j - 1] + trans[k + 2], pd[j - 1] + trans[k + 6]));\n"
"      cm[j] = vm; ci[j] = vi; cd[j] = vd;\n"
"      best = mx(best, mx(vm, vi));\n"
"    }\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    { __local int *x = pm; pm = cm; cm = x; x = pi; pi = ci; ci = x; x = pd; pd = cd; cd = x; }\n"
"  }\n"
"  pm[lane] = best;\n"
"  barrier(CLK_LOCAL_MEM_FENCE);\n"
"  if (lane == 0) {\n"
"    int b = 0;\n"
"    for (int l = 0; l < lanes; ++l) b = mx(b, pm[l]);\n"
"    answer[g] = b;\n"
"  }\n"
"}\n";

typedef struct {
	cl_command_queue queue;
	cl_kernel kernel;
} mp_hmm_gpu_ctx_t;

static pthread_mutex_t mp_hmm_gpu_init_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t mp_hmm_gpu_cache_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t mp_hmm_gpu_pool_mutex = PTHREAD_MUTEX_INITIALIZER;
static cl_context mp_hmm_gpu_context;
static cl_device_id mp_hmm_gpu_device;
static cl_program mp_hmm_gpu_program;
static cl_mem mp_hmm_gpu_codon;
static size_t mp_hmm_gpu_max_group;
static cl_ulong mp_hmm_gpu_local_mem;
static mp_hmm_gpu_ctx_t *mp_hmm_gpu_pool;
static int mp_hmm_gpu_pool_n, mp_hmm_gpu_pool_m;

static int mp_hmm_gpu_prepare(void)
{
	cl_platform_id platforms[8];
	cl_device_id devices[8];
	cl_uint nplatforms = 0, ndevices = 0;
	cl_int status;

	pthread_mutex_lock(&mp_hmm_gpu_init_mutex);
	if (mp_hmm_gpu_program) {
		pthread_mutex_unlock(&mp_hmm_gpu_init_mutex);
		return 1;
	}
	if (clGetPlatformIDs(8, platforms, &nplatforms) != CL_SUCCESS || nplatforms == 0)
		goto fail;
	for (cl_uint i = 0; i < nplatforms && ndevices == 0; ++i) {
		cl_uint count = 0;
		if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 8, devices, &count) == CL_SUCCESS && count)
			ndevices = count;
	}
	if (!ndevices) goto fail;
	mp_hmm_gpu_context = clCreateContext(0, 1, devices, 0, 0, &status);
	if (status != CL_SUCCESS) goto fail;
	mp_hmm_gpu_program = clCreateProgramWithSource(mp_hmm_gpu_context, 1,
		&(const char *){ mp_hmm_gpu_source }, 0, &status);
	if (status != CL_SUCCESS || clBuildProgram(mp_hmm_gpu_program, 1, devices, 0, 0, 0) != CL_SUCCESS)
		goto fail_release;
	mp_hmm_gpu_codon = clCreateBuffer(mp_hmm_gpu_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 64, mp_hmm_gpu_codon_alpha(), &status);
	if (status != CL_SUCCESS) goto fail_release;
	mp_hmm_gpu_device = devices[0];
	clGetDeviceInfo(mp_hmm_gpu_device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(mp_hmm_gpu_max_group), &mp_hmm_gpu_max_group, 0);
	clGetDeviceInfo(mp_hmm_gpu_device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(mp_hmm_gpu_local_mem), &mp_hmm_gpu_local_mem, 0);
	pthread_mutex_unlock(&mp_hmm_gpu_init_mutex);
	return 1;
fail_release:
	if (mp_hmm_gpu_codon) clReleaseMemObject(mp_hmm_gpu_codon);
	mp_hmm_gpu_codon = 0;
	if (mp_hmm_gpu_program) clReleaseProgram(mp_hmm_gpu_program);
	mp_hmm_gpu_program = 0;
	clReleaseContext(mp_hmm_gpu_context);
	mp_hmm_gpu_context = 0;
fail:
	pthread_mutex_unlock(&mp_hmm_gpu_init_mutex);
	return 0;
}

/* Command queue + kernel pairs are pooled so concurrent mapping threads never
 * share a kernel object (clSetKernelArg is not thread-safe) yet also never pay
 * per-call creation cost. The pool only grows to the peak concurrency. */
static int mp_hmm_gpu_ctx_get(mp_hmm_gpu_ctx_t *ctx)
{
	cl_int status;
	pthread_mutex_lock(&mp_hmm_gpu_pool_mutex);
	if (mp_hmm_gpu_pool_n > 0) {
		*ctx = mp_hmm_gpu_pool[--mp_hmm_gpu_pool_n];
		pthread_mutex_unlock(&mp_hmm_gpu_pool_mutex);
		return 1;
	}
	pthread_mutex_unlock(&mp_hmm_gpu_pool_mutex);
	ctx->queue = clCreateCommandQueue(mp_hmm_gpu_context, mp_hmm_gpu_device, 0, &status);
	if (status != CL_SUCCESS) return 0;
	ctx->kernel = clCreateKernel(mp_hmm_gpu_program, "mp_hmm_dp", &status);
	if (status != CL_SUCCESS) {
		clReleaseCommandQueue(ctx->queue);
		return 0;
	}
	return 1;
}

static void mp_hmm_gpu_ctx_put(mp_hmm_gpu_ctx_t *ctx)
{
	pthread_mutex_lock(&mp_hmm_gpu_pool_mutex);
	if (mp_hmm_gpu_pool_n == mp_hmm_gpu_pool_m) {
		int m = mp_hmm_gpu_pool_m? mp_hmm_gpu_pool_m << 1 : 16;
		mp_hmm_gpu_ctx_t *p = (mp_hmm_gpu_ctx_t*)realloc(mp_hmm_gpu_pool, (size_t)m * sizeof(*p));
		if (!p) {
			pthread_mutex_unlock(&mp_hmm_gpu_pool_mutex);
			clReleaseKernel(ctx->kernel);
			clReleaseCommandQueue(ctx->queue);
			return;
		}
		mp_hmm_gpu_pool = p, mp_hmm_gpu_pool_m = m;
	}
	mp_hmm_gpu_pool[mp_hmm_gpu_pool_n++] = *ctx;
	pthread_mutex_unlock(&mp_hmm_gpu_pool_mutex);
}

/* Emission/transition tables are uploaded once per profile and reused for every
 * candidate region scored against it. */
static int mp_hmm_gpu_profile_bufs(const mp_hmm_t *h, int32_t m, cl_mem *emit_buf, cl_mem *trans_buf)
{
	void **slot_e = mp_hmm_gpu_slot(h, 0), **slot_t = mp_hmm_gpu_slot(h, 1);
	cl_int status;
	pthread_mutex_lock(&mp_hmm_gpu_cache_mutex);
	if (!*slot_e) {
		cl_mem e, t;
		e = clCreateBuffer(mp_hmm_gpu_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (size_t)m * 20 * sizeof(int16_t), (void *)mp_hmm_emissions(h), &status);
		if (status != CL_SUCCESS) goto fail;
		t = clCreateBuffer(mp_hmm_gpu_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (size_t)m * 7 * sizeof(int16_t), (void *)mp_hmm_transitions(h), &status);
		if (status != CL_SUCCESS) { clReleaseMemObject(e); goto fail; }
		*slot_e = e, *slot_t = t;
	}
	*emit_buf = (cl_mem)*slot_e, *trans_buf = (cl_mem)*slot_t;
	pthread_mutex_unlock(&mp_hmm_gpu_cache_mutex);
	return 1;
fail:
	pthread_mutex_unlock(&mp_hmm_gpu_cache_mutex);
	return 0;
}

void mp_hmm_gpu_release(mp_hmm_t *h)
{
	void **slot_e = mp_hmm_gpu_slot(h, 0), **slot_t = mp_hmm_gpu_slot(h, 1);
	if (!slot_e) return;
	if (*slot_e) clReleaseMemObject((cl_mem)*slot_e), *slot_e = 0;
	if (*slot_t) clReleaseMemObject((cl_mem)*slot_t), *slot_t = 0;
}

int mp_hmm_dp_score_gpu_batch(const mp_hmm_t *h, int32_t n, const uint8_t *nt, int64_t l_nt, const int32_t *off, const int32_t *len, int32_t *score)
{
	mp_hmm_gpu_ctx_t ctx = {0, 0};
	cl_mem nt_buf = 0, off_buf = 0, len_buf = 0, emit_buf = 0, trans_buf = 0, answer_buf = 0;
	cl_int status = CL_SUCCESS;
	int32_t m, lanes, ok = 0;
	size_t local_bytes, global_size, local_size;
	int dbg = mp_hmm_gpu_debug_on();
	double t0 = dbg? mp_hmm_gpu_now() : 0;

	if (!h || n <= 0 || !nt || l_nt <= 0 || (m = mp_hmm_length(h)) <= 0) return 0;
	if (m > 1023) return 0;
	// blocking per-query launches on a separate cl_context contend with the ext pipeline; opt-in only
	if (!getenv("MP_HMM_GPU")) return 0;
	if (!mp_hmm_gpu_prepare()) return 0;
	lanes = 1;
	while (lanes < m + 1) lanes <<= 1;
	if ((size_t)lanes > mp_hmm_gpu_max_group || (cl_ulong)lanes * 6 * sizeof(int32_t) > mp_hmm_gpu_local_mem) return 0;
	local_size = (size_t)lanes;
	global_size = local_size * (size_t)n;
	if (!mp_hmm_gpu_profile_bufs(h, m, &emit_buf, &trans_buf)) return 0;
	if (!mp_hmm_gpu_ctx_get(&ctx)) return 0;
	nt_buf = clCreateBuffer(mp_hmm_gpu_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (size_t)l_nt, (void *)nt, &status);
	if (status != CL_SUCCESS) goto done;
	off_buf = clCreateBuffer(mp_hmm_gpu_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (size_t)n * sizeof(int32_t), (void *)off, &status);
	if (status != CL_SUCCESS) goto done;
	len_buf = clCreateBuffer(mp_hmm_gpu_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (size_t)n * sizeof(int32_t), (void *)len, &status);
	if (status != CL_SUCCESS) goto done;
	answer_buf = clCreateBuffer(mp_hmm_gpu_context, CL_MEM_WRITE_ONLY, (size_t)n * sizeof(int32_t), 0, &status);
	if (status != CL_SUCCESS) goto done;
	{
		cl_uint arg = 0;
#define ARG(value) do { status = clSetKernelArg(ctx.kernel, arg++, sizeof(value), &(value)); if (status != CL_SUCCESS) goto done; } while (0)
		ARG(nt_buf); ARG(off_buf); ARG(len_buf); ARG(m); ARG(emit_buf); ARG(trans_buf);
		ARG(mp_hmm_gpu_codon);
		local_bytes = (size_t)lanes * sizeof(int32_t); // lanes >= m+1; also holds the final per-lane maxima
		status = clSetKernelArg(ctx.kernel, arg++, local_bytes, 0); if (status != CL_SUCCESS) goto done;
		status = clSetKernelArg(ctx.kernel, arg++, local_bytes, 0); if (status != CL_SUCCESS) goto done;
		status = clSetKernelArg(ctx.kernel, arg++, local_bytes, 0); if (status != CL_SUCCESS) goto done;
		status = clSetKernelArg(ctx.kernel, arg++, local_bytes, 0); if (status != CL_SUCCESS) goto done;
		status = clSetKernelArg(ctx.kernel, arg++, local_bytes, 0); if (status != CL_SUCCESS) goto done;
		status = clSetKernelArg(ctx.kernel, arg++, local_bytes, 0); if (status != CL_SUCCESS) goto done;
		ARG(answer_buf);
#undef ARG
	}
	status = clEnqueueNDRangeKernel(ctx.queue, ctx.kernel, 1, 0, &global_size, &local_size, 0, 0, 0);
	if (status != CL_SUCCESS) goto done;
	if (clEnqueueReadBuffer(ctx.queue, answer_buf, CL_TRUE, 0, (size_t)n * sizeof(int32_t), score, 0, 0, 0) == CL_SUCCESS)
		ok = 1;

done:
	if (answer_buf) clReleaseMemObject(answer_buf);
	if (len_buf) clReleaseMemObject(len_buf);
	if (off_buf) clReleaseMemObject(off_buf);
	if (nt_buf) clReleaseMemObject(nt_buf);
	mp_hmm_gpu_ctx_put(&ctx);
	if (dbg) {
		pthread_mutex_lock(&mp_hmm_gpu_stat_mutex);
		mp_hmm_gpu_t_wait += mp_hmm_gpu_now() - t0, ++mp_hmm_gpu_n_launch, mp_hmm_gpu_n_reg += n, mp_hmm_gpu_nt += l_nt;
		pthread_mutex_unlock(&mp_hmm_gpu_stat_mutex);
	}
	return ok;
}

int32_t mp_hmm_dp_score_gpu(const mp_hmm_t *h, const uint8_t *nt, int32_t nlen)
{
	int32_t off = 0, score;
	if (nlen < 3) return INT32_MIN;
	if (!mp_hmm_dp_score_gpu_batch(h, 1, nt, nlen, &off, &nlen, &score)) return INT32_MIN;
	return score;
}

#else

int mp_hmm_dp_score_gpu_batch(const mp_hmm_t *h, int32_t n, const uint8_t *nt, int64_t l_nt, const int32_t *off, const int32_t *len, int32_t *score)
{
	(void)h; (void)n; (void)nt; (void)l_nt; (void)off; (void)len; (void)score;
	return 0;
}

int32_t mp_hmm_dp_score_gpu(const mp_hmm_t *hmm, const uint8_t *nt, int32_t nlen)
{
	(void)hmm;
	(void)nt;
	(void)nlen;
	return INT32_MIN;
}

void mp_hmm_gpu_release(mp_hmm_t *hmm)
{
	(void)hmm;
}

#endif
