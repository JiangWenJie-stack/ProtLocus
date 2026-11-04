/* Batched end-extension DP (ns_global_gs16b in NS_F_EXT_LEFT/RIGHT mode) on OpenCL.
 *
 * The CPU kernel is a striped 16-bit SIMD DP with saturating arithmetic, an
 * ie_coef*log2 length penalty on the row maximum, x-drop termination and a
 * "first column reaching the maximum" rule for aa_len. The device kernel below
 * reproduces it bit-for-bit: one work-group per request, one lane per block of
 * columns, rows processed sequentially. The horizontal (insertion) recurrence is
 * solved with a prefix max-scan instead of lazy-F; both are exact. Padded columns
 * (up to the CPU stripe width) are kept because they feed the row maximum.
 *
 * Input preparation (codon translation, donor/acceptor scoring, splice-score
 * track) also runs on the device: submitters only copy the raw strings. A flush
 * hands its requests to the executor thread and returns; results are fetched
 * (and waited for) per request, so a mapping thread can keep several flushes in
 * flight and overlap its own work with the device. The executor drives several
 * command queues ("pipelines", NS_EXT_GPU_QUEUES) so that launches of different
 * generations overlap on the device and a short launch is not held behind a
 * long one. */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <sys/time.h>
#include <pthread.h>
#include "nasw.h"
#include "kalloc.h"

#define NS_EXT_OUT_INTS 4   // nt_len, aa_len, score, ok

typedef struct {
	const char *ns, *as;
	const uint8_t *ss;
	int32_t nl, al, npad;
	ns_opt_t opt;
	ns_rst_t rst;
	int32_t gpu; // 1: eligible for the device
	int32_t dev_idx; // position in the (sorted) device request list
} ns_ext_req_t;

/* One flush of one mapping thread (a "generation"): its requests, the raw device inputs and the
 * device results. The executor admits jobs from every thread into shared device arenas and
 * advances all live requests by NS_EXT_CHUNK_ROWS rows per launch, so the device always holds the
 * work of many queries at once and no job waits behind another job's longest request. */
typedef struct ns_ext_job_s {
	int32_t base, n;      // request ids [base, base+n) of this generation
	ns_ext_req_t *req;    // n requests
	ns_ext_req_t **dev;   // the nd device-eligible ones
	int32_t nd, max_nl;
	int64_t l_nas, l_aas, l_sp;
	uint8_t *rns, *rss, *ras; // raw nt strings, splice-score track (only where present), proteins
	int has_ss;
	const ns_opt_t *o0;
	int32_t *out;         // nd * NS_EXT_OUT_INTS, in dev[] order
	int submitted, done, ok, finalized;
	int released;         // the executor no longer references the job (arena slot freed); set after done
	int32_t n_got;
	// executor-side bookkeeping
	int64_t a_nas, a_aas, a_sp, a_state, a_req; // arena offsets (elements)
	int64_t state_len;
	int32_t n_live;
	int pipe;             // pipeline whose launches carry this job
	struct ns_ext_job_s *next;  // executor queues (submission, pending, live)
	struct ns_ext_job_s *gnext; // the submitting batch's generation list
} ns_ext_job_t;

struct ns_ext_batch_s {
	void *km;
	int32_t n, m, next_id;
	ns_ext_req_t *req;    // requests added since the last flush
	ns_ext_job_t *gens;   // flushed generations, oldest first
};

static uint64_t ns_ext_launches, ns_ext_requests;
static int ns_ext_strict = -1; // NS_EXT_GPU_STRICT: abort instead of silently falling back to the CPU
static int ns_ext_warp; // kernel built with NS_EXT_WARP (warp-synchronous path for 32-lane sub-groups)

// NS_EXT_GPU_DEBUG: phase timings and volumes, printed at exit
static double ns_ext_t_prep, ns_ext_t_upload, ns_ext_t_dev, ns_ext_t_cpu, ns_ext_t_wait;
static uint64_t ns_ext_bytes, ns_ext_cells, ns_ext_flushes, ns_ext_cpu_reqs, ns_ext_wgs, ns_ext_wgs_warp;
static pthread_mutex_t ns_ext_stat_mutex = PTHREAD_MUTEX_INITIALIZER;
static int ns_ext_debug = -1;

static pthread_mutex_t ns_ext_q_mutex = PTHREAD_MUTEX_INITIALIZER;
#ifdef NS_OPENCL
static pthread_cond_t ns_ext_q_cond = PTHREAD_COND_INITIALIZER;    // executor: jobs arrived or a launch completed
#endif
static pthread_cond_t ns_ext_done_cond = PTHREAD_COND_INITIALIZER; // submitters: a job completed or was released

static double ns_ext_now(void)
{
	struct timeval tv;
	gettimeofday(&tv, 0);
	return tv.tv_sec + tv.tv_usec * 1e-6;
}

static void ns_ext_print_stats(void)
{
	fprintf(stderr, "[ns_ext_gpu] flushes=%llu launches=%llu wgs=%llu warp_wgs=%llu(%s) gpu_reqs=%llu cpu_reqs=%llu upload_MB=%.1f cells=%.3g prep_s=%.3f upload_s=%.3f device_s=%.3f cpu_s=%.3f wait_s=%.3f\n",
		(unsigned long long)ns_ext_flushes, (unsigned long long)ns_ext_launches, (unsigned long long)ns_ext_wgs, (unsigned long long)ns_ext_wgs_warp, ns_ext_warp? "on" : "off",
		(unsigned long long)ns_ext_requests, (unsigned long long)ns_ext_cpu_reqs,
		ns_ext_bytes / 1048576.0, (double)ns_ext_cells, ns_ext_t_prep, ns_ext_t_upload, ns_ext_t_dev, ns_ext_t_cpu, ns_ext_t_wait);
}

static int ns_ext_debug_on(void)
{
	if (ns_ext_debug < 0) {
		const char *s = getenv("NS_EXT_GPU_DEBUG");
		ns_ext_debug = s? (atoi(s) > 3? 3 : atoi(s) < 1? 1 : atoi(s)) : 0; // 1 stats, 2 per-iteration log, 3 also verify device-side input preparation
		if (ns_ext_debug) atexit(ns_ext_print_stats);
	}
	return ns_ext_debug;
}

uint64_t ns_ext_gpu_launch_count(void) { return __sync_fetch_and_add(&ns_ext_launches, 0); }
uint64_t ns_ext_gpu_request_count(void) { return __sync_fetch_and_add(&ns_ext_requests, 0); }

static void ns_ext_cpu(void *km, ns_ext_req_t *q)
{
	ns_rst_init(&q->rst);
	ns_global_gs16b(km, q->ns, q->nl, q->as, q->al, &q->opt, q->ss, &q->rst);
}

static void ns_ext_fallback(void *km, ns_ext_req_t *q, const char *why)
{
	if (ns_ext_strict < 0) ns_ext_strict = getenv("NS_EXT_GPU_STRICT") != 0;
	if (ns_ext_strict) {
		fprintf(stderr, "[ns_ext_gpu] GPU-eligible request fell back to the CPU (%s): nl=%d al=%d npad=%d\n", why, q->nl, q->al, q->npad);
		abort();
	}
	ns_ext_cpu(km, q);
}

#ifdef NS_OPENCL

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
typedef struct _cl_event *cl_event;

#define CL_SUCCESS 0
#define CL_TRUE 1
#define CL_FALSE 0
#define CL_COMPLETE 0
#define CL_EVENT_COMMAND_EXECUTION_STATUS 0x11D3
#define CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE (1ULL << 0)
#define CL_DEVICE_TYPE_GPU (1ULL << 2)
#define CL_MEM_READ_ONLY (1ULL << 2)
#define CL_MEM_WRITE_ONLY (1ULL << 1)
#define CL_MEM_READ_WRITE (1ULL << 0)
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
extern cl_int clEnqueueWriteBuffer(cl_command_queue, cl_mem, cl_uint, size_t, size_t, const void *, cl_uint, const void *, void *);
extern cl_int clFinish(cl_command_queue);
extern cl_int clFlush(cl_command_queue);
extern cl_int clEnqueueBarrierWithWaitList(cl_command_queue, cl_uint, const cl_event *, cl_event *);
extern cl_int clSetEventCallback(cl_event, cl_int, void (*)(cl_event, cl_int, void *), void *);
extern cl_int clGetEventInfo(cl_event, cl_uint, size_t, void *, size_t *);
extern cl_int clReleaseEvent(cl_event);
extern cl_int clReleaseMemObject(cl_mem);
extern cl_int clReleaseKernel(cl_kernel);
extern cl_int clReleaseProgram(cl_program);
extern cl_int clReleaseCommandQueue(cl_command_queue);
extern cl_int clReleaseContext(cl_context);

#define NS_EXT_REQ_INTS 16  // 0 nas_off, 1 nl, 2 aas_off, 3 al, 4 npad, 5 io, 6 sp_off, 7 state_off(shorts), 8 row, 9 status, 10..13 result, 14 flags (1 left, 2 has ss), 15 unused
#define NS_EXT_STATE_INTS 4 // max_sc, max_sc_log, max_i, first_j
#define NS_EXT_MAX_LANES 256 // lanes per work-group: npad / NS_EXT_C columns-per-lane, so npad <= 256 * NS_EXT_C
#define NS_EXT_MIN_LANES 32
#define NS_EXT_WG 256       // lanes per work-group; several sub-groups (requests) share one
#define NS_EXT_PREP_WG 256  // work-group size of the input-preparation kernel (one work-group per request)
#define NS_EXT_TAB_BYTES 576 // nt4[256] + aa20[256] + codon[64]

#include "nasw-ext-kernel.h"

typedef struct { int32_t lanes, remain, idx; } ns_ext_slot_t;

/* One command queue with at most one launch in flight. Within a pipeline the uploads, the
 * preparation kernel, the DP kernel and the readback are ordered; pipelines overlap on the device. */
typedef struct {
	cl_command_queue q;
	cl_mem list, plist;   // DP work-group list, preparation request list
	cl_event ev;          // completion of the launch in flight
	int idx, busy, fired; // fired: set by the event callback under ns_ext_q_mutex
	uint64_t seq;         // launches issued so far
	int32_t *plist_h; size_t m_plist; int32_t n_plist; // requests admitted since the last launch, to be prepared
	int32_t *list_h; size_t m_list;
	ns_ext_slot_t *slot; size_t m_slot;
	int32_t n_jobs, n_req, n_wg; // of the launch in flight
	int32_t max_rows, max_lanes; // debug: longest request (rows this launch) and largest sub-group
	double t0;
} ns_ext_pipe_t;

#ifndef NS_EXT_GPU_QUEUES_DEFAULT
#define NS_EXT_GPU_QUEUES_DEFAULT 8 // 2000 profiles -t16: 4 queues 115 s, 8 110 s, 16 107 s; each queue holds a 16 MB list buffer
#endif
#define NS_EXT_GPU_QUEUES_MAX 16

#ifndef NS_EXT_CHUNK_ROWS
#define NS_EXT_CHUNK_ROWS 8192 // rows advanced per launch; bounds how long any job waits for a straggler
#endif
#if NS_EXT_CHUNK_ROWS % NS_EXT_RW
#error NS_EXT_CHUNK_ROWS must be a multiple of NS_EXT_RW
#endif
#ifndef NS_EXT_ARENA_SEQ_BYTES
#define NS_EXT_ARENA_SEQ_BYTES (512u << 20) // each of raw nt / raw ss / nas / donor / acceptor
#endif
#ifndef NS_EXT_ARENA_AAS_BYTES
#define NS_EXT_ARENA_AAS_BYTES (16u << 20)
#endif
#ifndef NS_EXT_ARENA_REQ_MAX
#define NS_EXT_ARENA_REQ_MAX (1u << 18) // live requests
#endif
#ifndef NS_EXT_ARENA_STATE_SHORTS
#define NS_EXT_ARENA_STATE_SHORTS (256u << 20) // 512 MB of saved lane registers
#endif
#ifndef NS_EXT_MAX_JOBS
#define NS_EXT_MAX_JOBS 1024
#endif

typedef struct { int64_t cap, head, tail; int n_live; } ns_ext_ring_t;

enum { NS_EXT_B_NAS = 0, NS_EXT_B_AAS, NS_EXT_B_DONOR, NS_EXT_B_ACC, NS_EXT_B_REQ, NS_EXT_B_STATE, NS_EXT_B_MAT, NS_EXT_B_PEN,
	NS_EXT_B_RNS, NS_EXT_B_RSS, NS_EXT_B_RAS, NS_EXT_B_TAB, NS_EXT_NBUF };

static pthread_mutex_t ns_ext_init_mutex = PTHREAD_MUTEX_INITIALIZER;
static cl_context ns_ext_context;
static cl_device_id ns_ext_device;
static cl_program ns_ext_program;
static ns_ext_pipe_t *ns_ext_pipes;
static int ns_ext_n_pipes;
static cl_kernel ns_ext_kernel, ns_ext_prep_kernel;
static cl_mem ns_ext_buf[NS_EXT_NBUF];
static size_t ns_ext_max_group;
static cl_ulong ns_ext_local_mem;
static int32_t ns_ext_max_npad; // largest padded column count the work-group size allows
static int ns_ext_init_failed;
static pthread_t ns_ext_executor;
static ns_ext_job_t *ns_ext_q_head, *ns_ext_q_tail;
static int ns_ext_q_n;

static size_t ns_ext_local_shorts(int32_t lanes) { return (size_t)2 * (2 * NS_EXT_RW + 3) * lanes; }
static size_t ns_ext_local_ints(int32_t lanes) { return (size_t)2 * NS_EXT_RW * (lanes + lanes / 32) + lanes / 32; }

static void *ns_ext_executor_main(void *unused);

static void ns_ext_pipes_free(void)
{
	int k;
	if (!ns_ext_pipes) return;
	for (k = 0; k < ns_ext_n_pipes; ++k) {
		ns_ext_pipe_t *p = &ns_ext_pipes[k];
		if (p->plist) clReleaseMemObject(p->plist);
		if (p->list) clReleaseMemObject(p->list);
		if (p->q) clReleaseCommandQueue(p->q);
	}
	free(ns_ext_pipes);
	ns_ext_pipes = 0, ns_ext_n_pipes = 0;
}

static int ns_ext_gpu_prepare(void)
{
	cl_platform_id platforms[8];
	cl_device_id devices[8];
	cl_uint nplatforms = 0, ndevices = 0;
	cl_int status;
	const char *src = ns_ext_gpu_source;
	int k;

	pthread_mutex_lock(&ns_ext_init_mutex);
	if (ns_ext_program) { pthread_mutex_unlock(&ns_ext_init_mutex); return 1; }
	if (ns_ext_init_failed) { pthread_mutex_unlock(&ns_ext_init_mutex); return 0; }
	if (clGetPlatformIDs(8, platforms, &nplatforms) != CL_SUCCESS || nplatforms == 0) goto fail;
	for (cl_uint i = 0; i < nplatforms && ndevices == 0; ++i) {
		cl_uint count = 0;
		if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 8, devices, &count) == CL_SUCCESS && count)
			ndevices = count;
	}
	if (!ndevices) goto fail;
	ns_ext_context = clCreateContext(0, 1, devices, 0, 0, &status);
	if (status != CL_SUCCESS) goto fail;
	ns_ext_program = clCreateProgramWithSource(ns_ext_context, 1, &src, 0, &status);
	if (status != CL_SUCCESS) goto fail_release;
	ns_ext_warp = !getenv("NS_EXT_GPU_NO_WARP");
	if (ns_ext_warp && clBuildProgram(ns_ext_program, 1, devices, "-DNS_EXT_WARP", 0, 0) != CL_SUCCESS) { // inline PTX: NVIDIA only
		ns_ext_warp = 0;
		clReleaseProgram(ns_ext_program);
		ns_ext_program = clCreateProgramWithSource(ns_ext_context, 1, &src, 0, &status);
		if (status != CL_SUCCESS) goto fail_release;
	}
	if (!ns_ext_warp && clBuildProgram(ns_ext_program, 1, devices, 0, 0, 0) != CL_SUCCESS) {
		if (getenv("NS_EXT_GPU_DEBUG")) {
			size_t len = 0;
			char *log;
			clGetProgramBuildInfo(ns_ext_program, devices[0], CL_PROGRAM_BUILD_LOG, 0, 0, &len);
			log = (char*)calloc(len + 1, 1);
			if (log) {
				clGetProgramBuildInfo(ns_ext_program, devices[0], CL_PROGRAM_BUILD_LOG, len, log, 0);
				fprintf(stderr, "[ns_ext_gpu] build failed:\n%s\n", log);
				free(log);
			}
		}
		goto fail_release;
	}
	ns_ext_device = devices[0];
	clGetDeviceInfo(ns_ext_device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(ns_ext_max_group), &ns_ext_max_group, 0);
	clGetDeviceInfo(ns_ext_device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(ns_ext_local_mem), &ns_ext_local_mem, 0);
	if (ns_ext_max_group < NS_EXT_MIN_LANES) goto fail_release;
	if ((size_t)NS_EXT_WG > ns_ext_max_group) goto fail_release;
	if ((int64_t)ns_ext_local_shorts(NS_EXT_WG) * 2 + (int64_t)ns_ext_local_ints(NS_EXT_WG) * 4 + 484 > (int64_t)ns_ext_local_mem) goto fail_release;
	ns_ext_max_npad = NS_EXT_WG * NS_EXT_C;
	{
		const char *s = getenv("NS_EXT_GPU_QUEUES");
		ns_ext_n_pipes = s? atoi(s) : NS_EXT_GPU_QUEUES_DEFAULT;
		if (ns_ext_n_pipes < 1) ns_ext_n_pipes = 1;
		if (ns_ext_n_pipes > NS_EXT_GPU_QUEUES_MAX) ns_ext_n_pipes = NS_EXT_GPU_QUEUES_MAX;
	}
	ns_ext_pipes = (ns_ext_pipe_t*)calloc(ns_ext_n_pipes, sizeof(ns_ext_pipe_t));
	for (k = 0; k < ns_ext_n_pipes; ++k) { // in-order queues: one pipeline's uploads -> preparation -> DP -> readback depend on each other
		ns_ext_pipe_t *p = &ns_ext_pipes[k];
		p->idx = k;
		p->q = clCreateCommandQueue(ns_ext_context, ns_ext_device, 0, &status);
		if (status != CL_SUCCESS) goto fail_queue;
		p->list = clCreateBuffer(ns_ext_context, CL_MEM_READ_ONLY, (size_t)NS_EXT_ARENA_REQ_MAX * (NS_EXT_WG / 32) * 2 * sizeof(int32_t), 0, &status);
		if (status != CL_SUCCESS) goto fail_queue;
		p->plist = clCreateBuffer(ns_ext_context, CL_MEM_READ_ONLY, (size_t)NS_EXT_ARENA_REQ_MAX * sizeof(int32_t), 0, &status);
		if (status != CL_SUCCESS) goto fail_queue;
	}
	ns_ext_kernel = clCreateKernel(ns_ext_program, "ns_ext_dp", &status);
	if (status != CL_SUCCESS) goto fail_queue;
	ns_ext_prep_kernel = clCreateKernel(ns_ext_program, "ns_ext_prep", &status);
	if (status != CL_SUCCESS) goto fail_bufs;
	{
		size_t sizes[NS_EXT_NBUF];
		sizes[NS_EXT_B_NAS] = NS_EXT_ARENA_SEQ_BYTES, sizes[NS_EXT_B_DONOR] = NS_EXT_ARENA_SEQ_BYTES + 1, sizes[NS_EXT_B_ACC] = NS_EXT_ARENA_SEQ_BYTES + 1;
		sizes[NS_EXT_B_RNS] = NS_EXT_ARENA_SEQ_BYTES, sizes[NS_EXT_B_RSS] = NS_EXT_ARENA_SEQ_BYTES;
		sizes[NS_EXT_B_AAS] = NS_EXT_ARENA_AAS_BYTES, sizes[NS_EXT_B_RAS] = NS_EXT_ARENA_AAS_BYTES;
		sizes[NS_EXT_B_REQ] = (size_t)NS_EXT_ARENA_REQ_MAX * NS_EXT_REQ_INTS * sizeof(int32_t);
		sizes[NS_EXT_B_STATE] = (size_t)NS_EXT_ARENA_STATE_SHORTS * sizeof(int16_t);
		sizes[NS_EXT_B_MAT] = 22 * 22, sizes[NS_EXT_B_TAB] = NS_EXT_TAB_BYTES, sizes[NS_EXT_B_PEN] = 0; // pen grows with the longest request seen
		for (k = 0; k < NS_EXT_NBUF; ++k) {
			if (sizes[k] == 0) continue;
			ns_ext_buf[k] = clCreateBuffer(ns_ext_context, CL_MEM_READ_WRITE, sizes[k], 0, &status);
			if (status != CL_SUCCESS) goto fail_bufs;
		}
	}
	if (pthread_create(&ns_ext_executor, 0, ns_ext_executor_main, 0) != 0) goto fail_bufs;
	pthread_mutex_unlock(&ns_ext_init_mutex);
	return 1;
fail_bufs:
	for (k = 0; k < NS_EXT_NBUF; ++k)
		if (ns_ext_buf[k]) clReleaseMemObject(ns_ext_buf[k]), ns_ext_buf[k] = 0;
	if (ns_ext_prep_kernel) clReleaseKernel(ns_ext_prep_kernel);
	ns_ext_prep_kernel = 0;
	clReleaseKernel(ns_ext_kernel); ns_ext_kernel = 0;
fail_queue:
	ns_ext_pipes_free();
fail_release:
	if (ns_ext_program) clReleaseProgram(ns_ext_program);
	ns_ext_program = 0;
	clReleaseContext(ns_ext_context);
	ns_ext_context = 0;
fail:
	ns_ext_init_failed = 1;
	pthread_mutex_unlock(&ns_ext_init_mutex);
	return 0;
}

// NS_EXT_C columns per lane; the work-group is the smallest power of two >= npad/C, at least a warp
static int32_t ns_ext_lanes_for(int32_t npad)
{
	int32_t lanes = NS_EXT_MIN_LANES, nb = (npad + NS_EXT_C - 1) / NS_EXT_C;
	while (lanes < nb) lanes <<= 1;
	return lanes;
}

static int ns_ext_opt_same(const ns_opt_t *a, const ns_opt_t *b)
{
	// tables are compared by content: every mapping thread carries its own copy of the options
	return a->go == b->go && a->ge == b->ge && a->fs == b->fs && a->xdrop == b->xdrop &&
		a->end_bonus == b->end_bonus && a->asize == b->asize && a->ie_coef == b->ie_coef &&
		a->sp_null_bonus == b->sp_null_bonus && memcmp(a->sp, b->sp, sizeof(a->sp)) == 0 &&
		(a->sc == b->sc || memcmp(a->sc, b->sc, (size_t)a->asize * a->asize) == 0) &&
		(a->nt4 == b->nt4 || memcmp(a->nt4, b->nt4, 256) == 0) &&
		(a->aa20 == b->aa20 || memcmp(a->aa20, b->aa20, 256) == 0) &&
		(a->codon == b->codon || memcmp(a->codon, b->codon, 64) == 0);
}

/* Ring arena: jobs are admitted in FIFO order and released in FIFO order (a job that finishes
 * early keeps its slot until every older job is done, which is what keeps the layout trivial). */
static int64_t ns_ext_ring_alloc(ns_ext_ring_t *r, int64_t size)
{
	if (size == 0) return r->tail;
	if (r->n_live == 0) r->head = r->tail = 0;
	if (r->tail >= r->head) {
		if (r->tail + size <= r->cap) { int64_t o = r->tail; r->tail += size; return o; }
		if (size < r->head) { r->tail = size; return 0; } // wrap
		return -1;
	}
	if (r->tail + size < r->head) { int64_t o = r->tail; r->tail += size; return o; }
	return -1;
}

typedef struct {
	ns_ext_ring_t nas, aas, sp, state, req;
	ns_ext_job_t *live_head, *live_tail; // admitted jobs, oldest first
	int n_live;
	ns_opt_t opt; int has_opt;
	int8_t mat[484]; // opt.sc points here: the submitter's matrix lives only as long as its job
	uint8_t tab[NS_EXT_TAB_BYTES];
	int32_t *pen, m_pen; size_t pen_cap;
	int32_t *reqv;                  // host mirror of the request records; full arena size, never moved (pending uploads read from it)
	ns_ext_pipe_t *pipe; int n_pipe, n_busy;
	double t_busy;                  // when n_busy last left zero
	int failed;
} ns_ext_exec_t;

static int ns_ext_upload(cl_command_queue q, cl_mem buf, int64_t off, int64_t size, const void *data, size_t elem)
{
	if (size == 0) return 1;
	return clEnqueueWriteBuffer(q, buf, CL_FALSE, (size_t)off * elem, (size_t)size * elem, data, 0, 0, 0) == CL_SUCCESS;
}

// try to place a new job into the arenas and upload its raw inputs on pipeline p; 0 if it doesn't fit right now
static int ns_ext_admit(ns_ext_exec_t *x, ns_ext_pipe_t *p, ns_ext_job_t *jb)
{
	int64_t o_nas, o_aas, o_sp, o_state, o_req, state_len = 0;
	int32_t k;
	ns_ext_ring_t save_nas = x->nas, save_aas = x->aas, save_sp = x->sp, save_state = x->state, save_req = x->req;
	for (k = 0; k < jb->nd; ++k) state_len += (int64_t)9 * jb->dev[k]->npad + 2 * NS_EXT_STATE_INTS;
	o_nas = ns_ext_ring_alloc(&x->nas, jb->l_nas);
	o_aas = ns_ext_ring_alloc(&x->aas, jb->l_aas);
	o_sp = ns_ext_ring_alloc(&x->sp, jb->l_sp);
	o_state = ns_ext_ring_alloc(&x->state, state_len);
	o_req = ns_ext_ring_alloc(&x->req, jb->nd);
	if (o_nas < 0 || o_aas < 0 || o_sp < 0 || o_state < 0 || o_req < 0) {
		x->nas = save_nas, x->aas = save_aas, x->sp = save_sp, x->state = save_state, x->req = save_req;
		return 0;
	}
	++x->nas.n_live, ++x->aas.n_live, ++x->sp.n_live, ++x->state.n_live, ++x->req.n_live;
	jb->a_nas = o_nas, jb->a_aas = o_aas, jb->a_sp = o_sp, jb->a_state = o_state, jb->a_req = o_req, jb->state_len = state_len;
	jb->pipe = p->idx;
	if ((size_t)(p->n_plist + jb->nd) > p->m_plist) {
		p->m_plist = (size_t)(p->n_plist + jb->nd) * 2;
		p->plist_h = (int32_t*)realloc(p->plist_h, p->m_plist * sizeof(int32_t));
	}
	{
		int64_t p_nas = o_nas, p_aas = o_aas, p_sp = o_sp, p_state = o_state;
		for (k = 0; k < jb->nd; ++k) {
			const ns_ext_req_t *q = jb->dev[k];
			int32_t *rq = &x->reqv[(o_req + k) * NS_EXT_REQ_INTS];
			memset(rq, 0, NS_EXT_REQ_INTS * sizeof(int32_t));
			rq[0] = (int32_t)p_nas, rq[1] = q->nl, rq[2] = (int32_t)p_aas, rq[3] = q->al, rq[4] = q->npad, rq[5] = q->opt.io;
			rq[6] = (int32_t)p_sp, rq[7] = (int32_t)p_state, rq[8] = 2, rq[9] = 0;
			rq[14] = ((q->opt.flag & NS_F_EXT_LEFT)? 1 : 0) | (q->ss? 2 : 0);
			p->plist_h[p->n_plist++] = (int32_t)(o_req + k);
			p_nas += q->nl, p_aas += q->al, p_sp += q->nl + 1, p_state += (int64_t)9 * q->npad + 2 * NS_EXT_STATE_INTS;
		}
	}
	if (jb->max_nl > x->m_pen) { // ie_coef is a run-wide constant, so the penalty table just grows
		cl_int status;
		x->m_pen = jb->max_nl + (jb->max_nl >> 1);
		x->pen = (int32_t*)realloc(x->pen, (size_t)x->m_pen * sizeof(int32_t));
		for (k = 0; k < x->m_pen; ++k) x->pen[k] = ns_ext_ie_penalty(jb->o0, k);
		if (ns_ext_buf[NS_EXT_B_PEN]) clReleaseMemObject(ns_ext_buf[NS_EXT_B_PEN]); // launches in flight keep the old object alive until they finish
		ns_ext_buf[NS_EXT_B_PEN] = clCreateBuffer(ns_ext_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (size_t)x->m_pen * sizeof(int32_t), x->pen, &status);
		if (status != CL_SUCCESS) { ns_ext_buf[NS_EXT_B_PEN] = 0; x->failed = 1; return 1; }
	}
	if (!ns_ext_upload(p->q, ns_ext_buf[NS_EXT_B_RNS], o_nas, jb->l_nas, jb->rns, 1) ||
		(jb->has_ss && !ns_ext_upload(p->q, ns_ext_buf[NS_EXT_B_RSS], o_nas, jb->l_nas, jb->rss, 1)) ||
		!ns_ext_upload(p->q, ns_ext_buf[NS_EXT_B_RAS], o_aas, jb->l_aas, jb->ras, 1) ||
		!ns_ext_upload(p->q, ns_ext_buf[NS_EXT_B_REQ], o_req * NS_EXT_REQ_INTS, (int64_t)jb->nd * NS_EXT_REQ_INTS, &x->reqv[o_req * NS_EXT_REQ_INTS], sizeof(int32_t)))
		x->failed = 1;
	jb->n_live = jb->nd;
	jb->next = 0;
	if (x->live_tail) x->live_tail->next = jb; else x->live_head = jb;
	x->live_tail = jb;
	++x->n_live;
	__sync_fetch_and_add(&ns_ext_bytes, jb->l_nas * (1 + jb->has_ss) + jb->l_aas);
	return 1;
}

// device-side input preparation for the requests admitted to pipeline p since its last launch
static int ns_ext_prepare_inputs(ns_ext_exec_t *x, ns_ext_pipe_t *p)
{
	cl_uint arg = 0;
	cl_int status, sp[6], sp_null = x->opt.sp_null_bonus;
	size_t global_size = (size_t)p->n_plist * NS_EXT_PREP_WG, local_size = NS_EXT_PREP_WG;
	int k;
	if (p->n_plist == 0) return 1;
	for (k = 0; k < 6; ++k) sp[k] = x->opt.sp[k];
	if (!ns_ext_upload(p->q, p->plist, 0, p->n_plist, p->plist_h, sizeof(int32_t))) return 0;
#define ARG(value) do { status = clSetKernelArg(ns_ext_prep_kernel, arg++, sizeof(value), &(value)); if (status != CL_SUCCESS) return 0; } while (0)
	ARG(ns_ext_buf[NS_EXT_B_RNS]); ARG(ns_ext_buf[NS_EXT_B_RSS]); ARG(ns_ext_buf[NS_EXT_B_RAS]);
	ARG(ns_ext_buf[NS_EXT_B_NAS]); ARG(ns_ext_buf[NS_EXT_B_AAS]); ARG(ns_ext_buf[NS_EXT_B_DONOR]); ARG(ns_ext_buf[NS_EXT_B_ACC]);
	ARG(ns_ext_buf[NS_EXT_B_REQ]); ARG(p->plist); ARG(ns_ext_buf[NS_EXT_B_TAB]);
	ARG(sp[0]); ARG(sp[1]); ARG(sp[2]); ARG(sp[3]); ARG(sp[4]); ARG(sp[5]); ARG(sp_null);
#undef ARG
	status = clEnqueueNDRangeKernel(p->q, ns_ext_prep_kernel, 1, 0, &global_size, &local_size, 0, 0, 0);
	p->n_plist = 0;
	return status == CL_SUCCESS;
}

// NS_EXT_GPU_DEBUG=3: read the device-prepared inputs of a job back and compare them with ns_ext_prep()
static void ns_ext_check_prep(const ns_ext_exec_t *x, ns_ext_pipe_t *p, const ns_ext_job_t *jb)
{
	int32_t k;
	clFinish(p->q);
	for (k = 0; k < jb->nd; ++k) {
		const ns_ext_req_t *q = jb->dev[k];
		const int32_t *rq = &x->reqv[(jb->a_req + k) * NS_EXT_REQ_INTS];
		uint8_t *nas, *aas, *d_nas, *d_aas;
		int8_t *donor, *acc, *d_donor, *d_acc;
		int32_t i, bad = -1;
		const char *what = 0;
		nas = ns_ext_prep(0, q->ns, q->nl, q->as, q->al, &q->opt, q->ss, &aas, &donor, &acc);
		d_nas = (uint8_t*)malloc(q->nl + q->al + 2 * (q->nl + 1));
		d_aas = d_nas + q->nl, d_donor = (int8_t*)d_aas + q->al, d_acc = d_donor + q->nl + 1;
		clEnqueueReadBuffer(p->q, ns_ext_buf[NS_EXT_B_NAS], CL_TRUE, rq[0], q->nl, d_nas, 0, 0, 0);
		clEnqueueReadBuffer(p->q, ns_ext_buf[NS_EXT_B_AAS], CL_TRUE, rq[2], q->al, d_aas, 0, 0, 0);
		clEnqueueReadBuffer(p->q, ns_ext_buf[NS_EXT_B_DONOR], CL_TRUE, rq[6], q->nl + 1, d_donor, 0, 0, 0);
		clEnqueueReadBuffer(p->q, ns_ext_buf[NS_EXT_B_ACC], CL_TRUE, rq[6], q->nl + 1, d_acc, 0, 0, 0);
		for (i = 0; i < q->nl && bad < 0; ++i) if (nas[i] != d_nas[i]) bad = i, what = "nas";
		for (i = 0; i < q->al && bad < 0; ++i) if (aas[i] != d_aas[i]) bad = i, what = "aas";
		for (i = 0; i <= q->nl && bad < 0; ++i) if (donor[i] != d_donor[i]) bad = i, what = "donor";
		for (i = 0; i <= q->nl && bad < 0; ++i) if (acc[i] != d_acc[i]) bad = i, what = "acceptor";
		if (bad >= 0)
			fprintf(stderr, "[ns_ext_gpu] prep mismatch (%s nl=%d al=%d ss=%d io=%d): %s[%d] host=%d device=%d\n", (q->opt.flag & NS_F_EXT_LEFT)? "left" : "right",
				q->nl, q->al, q->ss != 0, q->opt.io, what, bad,
				strcmp(what, "nas") == 0? nas[bad] : strcmp(what, "aas") == 0? aas[bad] : strcmp(what, "donor") == 0? donor[bad] : acc[bad],
				strcmp(what, "nas") == 0? d_nas[bad] : strcmp(what, "aas") == 0? d_aas[bad] : strcmp(what, "donor") == 0? d_donor[bad] : d_acc[bad]);
		free(d_nas);
		kfree(0, nas);
	}
}

// same sub-group size first, then the most rows left: requests sharing a work-group step in lockstep
static int ns_ext_slot_cmp(const void *a, const void *b)
{
	const ns_ext_slot_t *x = (const ns_ext_slot_t*)a, *y = (const ns_ext_slot_t*)b;
	if (x->lanes != y->lanes) return x->lanes > y->lanes? -1 : 1;
	if (x->remain != y->remain) return x->remain > y->remain? -1 : 1;
	return x->idx < y->idx? -1 : x->idx > y->idx;
}

static int32_t ns_ext_pipe_live(const ns_ext_exec_t *x, int pipe)
{
	const ns_ext_job_t *jb;
	int32_t n = 0;
	for (jb = x->live_head; jb; jb = jb->next)
		if (jb->pipe == pipe) n += jb->n_live;
	return n;
}

// runs on a driver thread; the sequence number guards against a callback of an already harvested launch
static void ns_ext_pipe_fired(cl_event ev, cl_int status, void *data)
{
	uintptr_t v = (uintptr_t)data;
	ns_ext_pipe_t *p = &ns_ext_pipes[v & 0xff];
	(void)ev; (void)status;
	pthread_mutex_lock(&ns_ext_q_mutex);
	if (p->busy && p->seq == (uint64_t)(v >> 8)) p->fired = 1;
	pthread_cond_broadcast(&ns_ext_q_cond);
	pthread_mutex_unlock(&ns_ext_q_mutex);
}

// one launch on pipeline p over its live, unfinished requests, several per work-group, followed by an
// asynchronous readback of the records; nothing is enqueued when the pipeline holds no live request
static int ns_ext_launch(ns_ext_exec_t *x, ns_ext_pipe_t *p)
{
	ns_ext_job_t *jb;
	int32_t n = 0, k, nwg = 0, nj = 0, *list;
	ns_ext_slot_t *slot;
	cl_int status;
	for (jb = x->live_head; jb; jb = jb->next) if (jb->pipe == p->idx) n += jb->n_live;
	if (n == 0) return 1;
	if ((size_t)n > p->m_slot) p->m_slot = (size_t)n * 2, p->slot = (ns_ext_slot_t*)realloc(p->slot, p->m_slot * sizeof(ns_ext_slot_t));
	slot = p->slot;
	for (jb = x->live_head, n = 0; jb; jb = jb->next) {
		if (jb->pipe != p->idx || jb->n_live == 0) continue;
		for (k = 0; k < jb->nd; ++k) {
			const int32_t *rq = &x->reqv[(jb->a_req + k) * NS_EXT_REQ_INTS];
			if (rq[9]) continue;
			slot[n].lanes = ns_ext_lanes_for(rq[4]), slot[n].remain = rq[1] - rq[8], slot[n].idx = (int32_t)(jb->a_req + k), ++n;
		}
	}
	qsort(slot, n, sizeof(ns_ext_slot_t), ns_ext_slot_cmp);
	p->max_rows = 0, p->max_lanes = 0;
	for (k = 0; k < n; ++k) {
		int32_t r = slot[k].remain < NS_EXT_CHUNK_ROWS? slot[k].remain : NS_EXT_CHUNK_ROWS;
		if (r > p->max_rows) p->max_rows = r;
		if (slot[k].lanes > p->max_lanes) p->max_lanes = slot[k].lanes;
	}
	if ((size_t)(n + NS_EXT_WG / 32) * 2 * (NS_EXT_WG / 32) > p->m_list) {
		p->m_list = (size_t)(n + NS_EXT_WG / 32) * 2 * (NS_EXT_WG / 32) * 2;
		p->list_h = (int32_t*)realloc(p->list_h, p->m_list * sizeof(int32_t));
	}
	list = p->list_h;
	for (k = 0; k < n; ) { // pack one sub-group size per work-group
		int32_t lanes = slot[k].lanes, per = NS_EXT_WG / lanes, s, q;
		int32_t *e = &list[nwg * (NS_EXT_WG / 32) * 2];
		for (s = 0; s < NS_EXT_WG / 32; ++s) e[s * 2] = -1, e[s * 2 + 1] = 0;
		for (q = 0; q < per && k < n && slot[k].lanes == lanes; ++q, ++k)
			for (s = 0; s < lanes / 32; ++s)
				e[(q * (lanes / 32) + s) * 2] = slot[k].idx, e[(q * (lanes / 32) + s) * 2 + 1] = q * lanes;
		++nwg;
		if (lanes == 32) ++ns_ext_wgs_warp;
	}
	ns_ext_wgs += nwg;
	if (!ns_ext_upload(p->q, p->list, 0, (int64_t)nwg * (NS_EXT_WG / 32) * 2, list, sizeof(int32_t))) return 0;
	{
		cl_uint arg = 0;
		cl_int go = x->opt.go, ge = x->opt.ge, fs = x->opt.fs, xdrop = x->opt.xdrop, end_bonus = x->opt.end_bonus, asize = x->opt.asize, chunk = NS_EXT_CHUNK_ROWS;
		size_t global_size = (size_t)nwg * NS_EXT_WG, local_size = NS_EXT_WG;
#define ARG(value) do { status = clSetKernelArg(ns_ext_kernel, arg++, sizeof(value), &(value)); if (status != CL_SUCCESS) return 0; } while (0)
		ARG(ns_ext_buf[NS_EXT_B_NAS]); ARG(ns_ext_buf[NS_EXT_B_AAS]); ARG(ns_ext_buf[NS_EXT_B_DONOR]); ARG(ns_ext_buf[NS_EXT_B_ACC]);
		ARG(ns_ext_buf[NS_EXT_B_REQ]); ARG(p->list); ARG(ns_ext_buf[NS_EXT_B_STATE]); ARG(ns_ext_buf[NS_EXT_B_MAT]); ARG(ns_ext_buf[NS_EXT_B_PEN]);
		ARG(go); ARG(ge); ARG(fs); ARG(xdrop); ARG(end_bonus); ARG(asize); ARG(chunk);
#undef ARG
		status = clSetKernelArg(ns_ext_kernel, 16, ns_ext_local_shorts(NS_EXT_WG) * sizeof(int16_t), 0); if (status != CL_SUCCESS) return 0;
		status = clSetKernelArg(ns_ext_kernel, 17, ns_ext_local_ints(NS_EXT_WG) * sizeof(int32_t), 0); if (status != CL_SUCCESS) return 0;
		status = clSetKernelArg(ns_ext_kernel, 18, (size_t)x->opt.asize * x->opt.asize, 0); if (status != CL_SUCCESS) return 0;
		status = clEnqueueNDRangeKernel(p->q, ns_ext_kernel, 1, 0, &global_size, &local_size, 0, 0, 0);
		if (status != CL_SUCCESS) return 0;
		__sync_fetch_and_add(&ns_ext_launches, 1);
	}
	// read every live job's records back (contiguous per job); the barrier event marks the whole launch done
	for (jb = x->live_head; jb; jb = jb->next) {
		if (jb->pipe != p->idx || jb->n_live == 0) continue;
		++nj;
		if (clEnqueueReadBuffer(p->q, ns_ext_buf[NS_EXT_B_REQ], CL_FALSE, (size_t)jb->a_req * NS_EXT_REQ_INTS * sizeof(int32_t),
				(size_t)jb->nd * NS_EXT_REQ_INTS * sizeof(int32_t), &x->reqv[jb->a_req * NS_EXT_REQ_INTS], 0, 0, 0) != CL_SUCCESS)
			return 0;
	}
	if (clEnqueueBarrierWithWaitList(p->q, 0, 0, &p->ev) != CL_SUCCESS) return 0;
	++p->seq;
	p->busy = 1, p->fired = 0;
	p->n_jobs = nj, p->n_req = n, p->n_wg = nwg;
	p->t0 = ns_ext_now();
	if (x->n_busy++ == 0) x->t_busy = p->t0;
	clSetEventCallback(p->ev, CL_COMPLETE, ns_ext_pipe_fired, (void*)(uintptr_t)((uintptr_t)p->idx | ((uintptr_t)p->seq << 8)));
	clFlush(p->q);
	return 1;
}

// a launch of pipeline p has completed: fold the records read back into the jobs' results
static void ns_ext_complete(ns_ext_exec_t *x, ns_ext_pipe_t *p)
{
	ns_ext_job_t *jb;
	int32_t k;
	int dbg = ns_ext_debug_on();
	for (jb = x->live_head; jb; jb = jb->next) {
		if (jb->pipe != p->idx || jb->n_live == 0) continue;
		for (k = 0, jb->n_live = 0; k < jb->nd; ++k) {
			const int32_t *rq = &x->reqv[(jb->a_req + k) * NS_EXT_REQ_INTS];
			if (rq[9]) memcpy(&jb->out[k * NS_EXT_OUT_INTS], rq + 10, NS_EXT_OUT_INTS * sizeof(int32_t));
			else ++jb->n_live;
		}
	}
	clReleaseEvent(p->ev);
	p->ev = 0, p->busy = 0, p->fired = 0;
	if (--x->n_busy == 0 && dbg) {
		pthread_mutex_lock(&ns_ext_stat_mutex);
		ns_ext_t_dev += ns_ext_now() - x->t_busy;
		pthread_mutex_unlock(&ns_ext_stat_mutex);
	}
	if (dbg > 1) fprintf(stderr, "[ns_ext_gpu] pipe %d: jobs=%d reqs=%d wgs=%d rows=%d lanes=%d in_flight=%.4f\n", p->idx, p->n_jobs, p->n_req, p->n_wg, p->max_rows, p->max_lanes, ns_ext_now() - p->t0);
}

// release finished jobs (FIFO: the oldest live job keeps the ring heads); returns them via *rel_list
static void ns_ext_retire(ns_ext_exec_t *x, ns_ext_job_t **rel_list, int32_t *n_rel)
{
	*n_rel = 0;
	while (x->live_head && x->live_head->n_live == 0) {
		ns_ext_job_t *jb = x->live_head;
		x->live_head = jb->next;
		if (!x->live_head) x->live_tail = 0;
		--x->n_live;
		x->nas.head = jb->a_nas + jb->l_nas, x->aas.head = jb->a_aas + jb->l_aas, x->sp.head = jb->a_sp + jb->l_sp;
		x->state.head = jb->a_state + jb->state_len, x->req.head = jb->a_req + jb->nd;
		--x->nas.n_live, --x->aas.n_live, --x->sp.n_live, --x->state.n_live, --x->req.n_live;
		if (x->live_head) { // the next job's start is the true head (it may have wrapped)
			x->nas.head = x->live_head->a_nas, x->aas.head = x->live_head->a_aas, x->sp.head = x->live_head->a_sp;
			x->state.head = x->live_head->a_state, x->req.head = x->live_head->a_req;
		}
		rel_list[(*n_rel)++] = jb;
	}
}

// admit pending jobs in order onto pipeline p while they fit and share the active parameters
static ns_ext_job_t *ns_ext_dispatch(ns_ext_exec_t *x, ns_ext_pipe_t *p, ns_ext_job_t **pending_head, ns_ext_job_t **pending_tail)
{
	ns_ext_job_t *jb, *next, *first_admitted = 0;
	for (jb = *pending_head; jb && x->n_live < NS_EXT_MAX_JOBS; jb = next) {
		next = jb->next;
		if (!x->has_opt || x->n_live == 0) {
			if (!x->has_opt || !ns_ext_opt_same(&x->opt, jb->o0)) { // the device is empty: switch parameters (blocking, so every queue sees them)
				x->opt = *jb->o0, x->has_opt = 1;
				memcpy(x->mat, jb->o0->sc, (size_t)x->opt.asize * x->opt.asize);
				memcpy(x->tab, jb->o0->nt4, 256), memcpy(x->tab + 256, jb->o0->aa20, 256), memcpy(x->tab + 512, jb->o0->codon, 64);
				x->opt.sc = x->mat, x->opt.nt4 = x->tab, x->opt.aa20 = x->tab + 256, x->opt.codon = x->tab + 512;
				if (clEnqueueWriteBuffer(p->q, ns_ext_buf[NS_EXT_B_MAT], CL_TRUE, 0, (size_t)x->opt.asize * x->opt.asize, x->opt.sc, 0, 0, 0) != CL_SUCCESS ||
					clEnqueueWriteBuffer(p->q, ns_ext_buf[NS_EXT_B_TAB], CL_TRUE, 0, NS_EXT_TAB_BYTES, x->tab, 0, 0, 0) != CL_SUCCESS) x->failed = 1;
			}
		} else if (!ns_ext_opt_same(&x->opt, jb->o0)) break; // wait until the device is empty, then switch
		if (!ns_ext_admit(x, p, jb)) {
			if (x->n_live == 0) { // can never fit: hand it back to the CPU
				jb->ok = 0;
				pthread_mutex_lock(&ns_ext_q_mutex);
				__atomic_store_n(&jb->done, 1, __ATOMIC_RELEASE);
				__atomic_store_n(&jb->released, 1, __ATOMIC_RELEASE);
				pthread_cond_broadcast(&ns_ext_done_cond);
				pthread_mutex_unlock(&ns_ext_q_mutex);
				*pending_head = next;
				if (!next) *pending_tail = 0;
				continue;
			}
			break;
		}
		if (!first_admitted) first_admitted = jb;
		*pending_head = next;
		if (!next) *pending_tail = 0;
	}
	return first_admitted;
}

// a missed callback must not hang the executor: poll the events of the launches in flight
static void ns_ext_poll_events(ns_ext_exec_t *x)
{
	int g;
	for (g = 0; g < x->n_pipe; ++g) {
		ns_ext_pipe_t *p = &x->pipe[g];
		cl_int st = 1;
		if (!p->busy || p->fired) continue;
		if (clGetEventInfo(p->ev, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(st), &st, 0) == CL_SUCCESS && st <= CL_COMPLETE)
			__atomic_store_n(&p->fired, 1, __ATOMIC_RELEASE);
	}
}

static void *ns_ext_executor_main(void *unused)
{
	ns_ext_exec_t x;
	ns_ext_job_t *pending_head = 0, *pending_tail = 0, **done_list, **rel_list;
	int dbg = ns_ext_debug_on(), stalled = 0, g;
	(void)unused;
	memset(&x, 0, sizeof(x));
	x.nas.cap = NS_EXT_ARENA_SEQ_BYTES, x.aas.cap = NS_EXT_ARENA_AAS_BYTES, x.sp.cap = NS_EXT_ARENA_SEQ_BYTES + 1;
	x.state.cap = NS_EXT_ARENA_STATE_SHORTS, x.req.cap = NS_EXT_ARENA_REQ_MAX;
	x.reqv = (int32_t*)malloc((size_t)NS_EXT_ARENA_REQ_MAX * NS_EXT_REQ_INTS * sizeof(int32_t));
	x.pipe = ns_ext_pipes, x.n_pipe = ns_ext_n_pipes;
	done_list = (ns_ext_job_t**)malloc(sizeof(ns_ext_job_t*) * NS_EXT_MAX_JOBS);
	rel_list = (ns_ext_job_t**)malloc(sizeof(ns_ext_job_t*) * NS_EXT_MAX_JOBS);
	for (;;) {
		ns_ext_job_t *jb, *first_admitted = 0;
		ns_ext_pipe_t *tgt = 0;
		int32_t n_done = 0, n_rel = 0, k, best = INT32_MAX;
		double t0, t1;
		// 1. wait until a launch has completed or new jobs can go to an idle pipeline
		pthread_mutex_lock(&ns_ext_q_mutex);
		for (;;) {
			int any_fired = 0, any_idle = 0;
			if (ns_ext_q_head) {
				if (pending_tail) pending_tail->next = ns_ext_q_head; else pending_head = ns_ext_q_head;
				pending_tail = ns_ext_q_tail;
				ns_ext_q_head = ns_ext_q_tail = 0, ns_ext_q_n = 0;
			}
			for (g = 0; g < x.n_pipe; ++g) {
				if (x.pipe[g].busy) { if (x.pipe[g].fired) any_fired = 1; }
				else any_idle = 1;
			}
			if (x.n_busy == 0) stalled = 0; // nothing in flight: a dispatch always makes progress (switch, admit or CPU hand-back)
			if (any_fired || (any_idle && pending_head && !stalled)) break;
			if (x.n_busy) {
				struct timeval tv;
				struct timespec ts;
				gettimeofday(&tv, 0);
				ts.tv_sec = tv.tv_sec, ts.tv_nsec = (tv.tv_usec + 1000) * 1000L;
				if (ts.tv_nsec >= 1000000000L) ts.tv_nsec -= 1000000000L, ++ts.tv_sec;
				pthread_cond_timedwait(&ns_ext_q_cond, &ns_ext_q_mutex, &ts);
				pthread_mutex_unlock(&ns_ext_q_mutex);
				ns_ext_poll_events(&x);
				pthread_mutex_lock(&ns_ext_q_mutex);
			} else pthread_cond_wait(&ns_ext_q_cond, &ns_ext_q_mutex);
		}
		pthread_mutex_unlock(&ns_ext_q_mutex);
		t0 = dbg? ns_ext_now() : 0;
		// 2. harvest completed launches
		for (g = 0; g < x.n_pipe; ++g)
			if (x.pipe[g].busy && x.pipe[g].fired) ns_ext_complete(&x, &x.pipe[g]);
		// 3. new jobs go to the idle pipeline carrying the least live work; every idle pipeline with live requests is relaunched
		if (!x.failed) {
			for (g = 0; g < x.n_pipe; ++g) {
				int32_t n;
				if (x.pipe[g].busy) continue;
				n = ns_ext_pipe_live(&x, g);
				if (n < best) best = n, tgt = &x.pipe[g];
			}
			if (tgt && pending_head) {
				first_admitted = ns_ext_dispatch(&x, tgt, &pending_head, &pending_tail);
				if (!x.failed && !ns_ext_prepare_inputs(&x, tgt)) x.failed = 1;
				tgt->n_plist = 0;
				if (dbg > 2 && !x.failed && first_admitted)
					for (jb = first_admitted; jb; jb = jb->next) ns_ext_check_prep(&x, tgt, jb);
			}
			for (g = 0; g < x.n_pipe && !x.failed; ++g)
				if (!x.pipe[g].busy && !ns_ext_launch(&x, &x.pipe[g])) x.failed = 1;
		}
		stalled = pending_head != 0; // only a completion can make room or empty the device for a parameter switch
		if (x.failed) { // drain: every live job falls back to the CPU
			for (g = 0; g < x.n_pipe; ++g) {
				ns_ext_pipe_t *p = &x.pipe[g];
				clFinish(p->q);
				if (p->busy) clReleaseEvent(p->ev), p->ev = 0, p->busy = 0, p->fired = 0;
			}
			x.n_busy = 0;
			for (jb = x.live_head; jb; jb = jb->next) jb->n_live = 0;
			stalled = 0;
		}
		// 4. jobs whose requests are all finished are done now; the oldest of them also free their arena slots
		for (jb = x.live_head; jb; jb = jb->next)
			if (jb->n_live == 0 && !jb->done) jb->ok = !x.failed, done_list[n_done++] = jb;
		ns_ext_retire(&x, rel_list, &n_rel);
		if (x.n_live == 0) x.failed = 0;
		if (dbg) {
			t1 = ns_ext_now();
			pthread_mutex_lock(&ns_ext_stat_mutex);
			ns_ext_t_upload += t1 - t0, ++ns_ext_flushes;
			pthread_mutex_unlock(&ns_ext_stat_mutex);
		}
		if (n_done || n_rel) {
			pthread_mutex_lock(&ns_ext_q_mutex);
			for (k = 0; k < n_done; ++k) __atomic_store_n(&done_list[k]->done, 1, __ATOMIC_RELEASE);
			for (k = 0; k < n_rel; ++k) __atomic_store_n(&rel_list[k]->released, 1, __ATOMIC_RELEASE);
			pthread_cond_broadcast(&ns_ext_done_cond);
			pthread_mutex_unlock(&ns_ext_q_mutex);
		}
	}
	return 0;
}

/* Hands the generation's device-eligible requests to the executor and returns at once; the
 * caller polls jb->done. jb->ok == 0 at completion means nothing was computed (all requests
 * then fall back to the CPU); individual requests the kernel couldn't finish carry out[].ok == 0. */
static void ns_ext_gpu_submit(ns_ext_batch_t *b, ns_ext_job_t *jb)
{
	void *km = b->km;
	int32_t k;
	int64_t l_nas = 0, l_aas = 0;
	int dbg = ns_ext_debug_on();
	double t0 = dbg? ns_ext_now() : 0;

	jb->o0 = &jb->dev[0]->opt;
	for (k = 0; k < jb->nd; ++k) {
		const ns_ext_req_t *q = jb->dev[k];
		jb->l_nas += q->nl, jb->l_aas += q->al, jb->l_sp += q->nl + 1;
		if (q->nl > jb->max_nl) jb->max_nl = q->nl;
		if (q->ss) jb->has_ss = 1;
		__sync_fetch_and_add(&ns_ext_cells, (uint64_t)q->nl * q->al);
	}
	jb->rns = Kmalloc(km, uint8_t, jb->l_nas);
	jb->rss = jb->has_ss? Kmalloc(km, uint8_t, jb->l_nas) : 0;
	jb->ras = Kmalloc(km, uint8_t, jb->l_aas);
	for (k = 0; k < jb->nd; ++k) {
		const ns_ext_req_t *q = jb->dev[k];
		memcpy(&jb->rns[l_nas], q->ns, q->nl);
		if (q->ss) memcpy(&jb->rss[l_nas], q->ss, q->nl); // absent tracks leave their range unread
		memcpy(&jb->ras[l_aas], q->as, q->al);
		l_nas += q->nl, l_aas += q->al;
	}
	if (dbg) {
		pthread_mutex_lock(&ns_ext_stat_mutex);
		ns_ext_t_prep += ns_ext_now() - t0;
		pthread_mutex_unlock(&ns_ext_stat_mutex);
	}
	jb->submitted = 1;
	pthread_mutex_lock(&ns_ext_q_mutex);
	if (ns_ext_q_tail) ns_ext_q_tail->next = jb; else ns_ext_q_head = jb;
	ns_ext_q_tail = jb, ++ns_ext_q_n;
	pthread_cond_signal(&ns_ext_q_cond);
	pthread_mutex_unlock(&ns_ext_q_mutex);
}

static int ns_ext_gpu_available(void) { return ns_ext_gpu_prepare(); }

static int ns_ext_gpu_eligible(const ns_ext_req_t *q, const ns_opt_t *o0)
{
	if (q->nl < 3 || q->al < 1 || q->npad > ns_ext_max_npad) return 0;
	if (q->opt.pssm) return 0; // the device kernels only know the 22x22 matrix
	if (o0 && !ns_ext_opt_same(&q->opt, o0)) return 0;
	return 1;
}

#else

static int ns_ext_gpu_available(void) { return 0; }
static int ns_ext_gpu_eligible(const ns_ext_req_t *q, const ns_opt_t *o0) { (void)q; (void)o0; return 0; }
static void ns_ext_gpu_submit(ns_ext_batch_t *b, ns_ext_job_t *jb) { (void)b; jb->done = 1, jb->ok = 0; }

#endif

static int ns_ext_job_done(const ns_ext_job_t *jb) { return __atomic_load_n(&jb->done, __ATOMIC_ACQUIRE); }
// the executor may still reference a submitted job (arena slot, readback target) after it is done
static int ns_ext_job_released(const ns_ext_job_t *jb) { return !jb->submitted || __atomic_load_n(&jb->released, __ATOMIC_ACQUIRE); }

ns_ext_batch_t *ns_ext_batch_init(void *km)
{
	ns_ext_batch_t *b;
	if (!ns_ext_gpu_available()) return 0;
	b = Kcalloc(km, ns_ext_batch_t, 1);
	b->km = km;
	return b;
}

int32_t ns_ext_batch_add(ns_ext_batch_t *b, const char *ns, int32_t nl, const char *as, int32_t al, const ns_opt_t *opt, const uint8_t *ss)
{
	ns_ext_req_t *q;
	int32_t vs;
	if (b->n == b->m) {
		b->m = b->m? b->m << 1 : 16;
		b->req = Krealloc(b->km, ns_ext_req_t, b->req, b->m);
	}
	q = &b->req[b->n];
	memset(q, 0, sizeof(*q));
	q->ns = ns, q->nl = nl, q->as = as, q->al = al, q->ss = ss, q->opt = *opt;
	q->opt.flag &= ~(NS_F_CIGAR | NS_F_GPU);
	vs = ns_ext_stripe_width(&q->opt, al);
	q->npad = (al + vs - 1) / vs * vs;
	q->rst.score = INT32_MIN;
	++b->n;
	return b->next_id++;
}

static void ns_ext_job_free(void *km, ns_ext_job_t *jb)
{
	kfree(km, jb->rns); kfree(km, jb->rss); kfree(km, jb->ras);
	kfree(km, jb->out); kfree(km, jb->dev); kfree(km, jb->req);
	kfree(km, jb);
}

// drop generations whose results have all been read (a submitted job is only freed once the executor is done with it)
static void ns_ext_batch_reap(ns_ext_batch_t *b)
{
	ns_ext_job_t **p = &b->gens;
	while (*p) {
		ns_ext_job_t *jb = *p;
		if (jb->finalized && jb->n_got >= jb->n && ns_ext_job_released(jb)) *p = jb->gnext, ns_ext_job_free(b->km, jb);
		else p = &jb->gnext;
	}
}

int32_t ns_ext_batch_flush(ns_ext_batch_t *b)
{
	int32_t k, n;
	ns_ext_job_t *jb;
	const ns_opt_t *o0 = 0;
	if (!b) return 0;
	ns_ext_batch_reap(b);
	if (b->n == 0) return 0;
	jb = Kcalloc(b->km, ns_ext_job_t, 1);
	jb->base = b->next_id - b->n, jb->n = n = b->n;
	jb->req = b->req, b->req = 0, b->n = b->m = 0;
	jb->dev = Kmalloc(b->km, ns_ext_req_t*, n);
	for (k = 0; k < n; ++k) {
		ns_ext_req_t *q = &jb->req[k];
		q->gpu = ns_ext_gpu_eligible(q, o0);
		if (q->gpu) { if (!o0) o0 = &q->opt; q->dev_idx = jb->nd; jb->dev[jb->nd++] = q; }
	}
	__sync_fetch_and_add(&ns_ext_requests, jb->nd);
	if (b->gens) { ns_ext_job_t *t = b->gens; while (t->gnext) t = t->gnext; t->gnext = jb; }
	else b->gens = jb;
	if (jb->nd > 0) {
		jb->out = Kmalloc(b->km, int32_t, (size_t)jb->nd * NS_EXT_OUT_INTS);
		ns_ext_gpu_submit(b, jb);
	} else jb->done = 1, jb->ok = 1;
	if (jb->nd < n) { // CPU-only requests overlap with the device work
		int dbg = ns_ext_debug_on();
		double t0 = dbg? ns_ext_now() : 0;
		for (k = 0; k < n; ++k)
			if (!jb->req[k].gpu) ns_ext_cpu(b->km, &jb->req[k]);
		if (dbg) {
			pthread_mutex_lock(&ns_ext_stat_mutex);
			ns_ext_t_cpu += ns_ext_now() - t0, ns_ext_cpu_reqs += n - jb->nd;
			pthread_mutex_unlock(&ns_ext_stat_mutex);
		}
	}
	return n;
}

// once the device is done: copy the results, run the CPU fallback for whatever it couldn't compute
static void ns_ext_job_finalize(ns_ext_batch_t *b, ns_ext_job_t *jb)
{
	int32_t k;
	if (jb->nd > 0) {
		for (k = 0; k < jb->nd; ++k) {
			ns_ext_req_t *q = jb->dev[k];
			const int32_t *o = &jb->out[q->dev_idx * NS_EXT_OUT_INTS];
			if (!jb->ok) ns_ext_fallback(b->km, q, "launch failed");
			else if (o[3]) {
				ns_rst_init(&q->rst);
				q->rst.nt_len = o[0], q->rst.aa_len = o[1], q->rst.score = o[2];
			} else ns_ext_fallback(b->km, q, "kernel found no maximum");
		}
	}
	jb->finalized = 1;
}

static ns_ext_job_t *ns_ext_batch_find(const ns_ext_batch_t *b, int32_t id)
{
	ns_ext_job_t *jb;
	for (jb = b->gens; jb; jb = jb->gnext)
		if (id >= jb->base && id < jb->base + jb->n) return jb;
	return 0;
}

int ns_ext_batch_ready(ns_ext_batch_t *b, int32_t id)
{
	ns_ext_job_t *jb = ns_ext_batch_find(b, id);
	if (!jb) return 0;
	if (jb->finalized) return 1;
	if (!ns_ext_job_done(jb)) return 0;
	ns_ext_job_finalize(b, jb);
	return 1;
}

void ns_ext_batch_wait(ns_ext_batch_t *b)
{
	ns_ext_job_t *jb;
	int dbg;
	double t0;
	if (!b) return;
	dbg = ns_ext_debug_on();
	t0 = dbg? ns_ext_now() : 0;
	pthread_mutex_lock(&ns_ext_q_mutex);
	for (;;) {
		int in_flight = 0, fresh = 0;
		for (jb = b->gens; jb; jb = jb->gnext) {
			if (jb->finalized) continue;
			if (ns_ext_job_done(jb)) fresh = 1;
			else in_flight = 1;
		}
		if (fresh || !in_flight) break;
		pthread_cond_wait(&ns_ext_done_cond, &ns_ext_q_mutex);
	}
	pthread_mutex_unlock(&ns_ext_q_mutex);
	if (dbg) {
		pthread_mutex_lock(&ns_ext_stat_mutex);
		ns_ext_t_wait += ns_ext_now() - t0;
		pthread_mutex_unlock(&ns_ext_stat_mutex);
	}
}

const ns_rst_t *ns_ext_batch_get(ns_ext_batch_t *b, int32_t id)
{
	ns_ext_job_t *jb = ns_ext_batch_find(b, id);
	if (!jb) return 0;
	if (!jb->finalized) {
		if (!ns_ext_job_done(jb)) {
			int dbg = ns_ext_debug_on();
			double t0 = dbg? ns_ext_now() : 0;
			pthread_mutex_lock(&ns_ext_q_mutex);
			while (!ns_ext_job_done(jb)) pthread_cond_wait(&ns_ext_done_cond, &ns_ext_q_mutex);
			pthread_mutex_unlock(&ns_ext_q_mutex);
			if (dbg) {
				pthread_mutex_lock(&ns_ext_stat_mutex);
				ns_ext_t_wait += ns_ext_now() - t0;
				pthread_mutex_unlock(&ns_ext_stat_mutex);
			}
		}
		ns_ext_job_finalize(b, jb);
	}
	++jb->n_got;
	return &jb->req[id - jb->base].rst;
}

void ns_ext_batch_destroy(ns_ext_batch_t *b)
{
	ns_ext_job_t *jb, *next;
	if (!b) return;
	for (jb = b->gens; jb; jb = jb->gnext) // the executor may still be uploading from / writing into a job
		if (!ns_ext_job_released(jb)) {
			pthread_mutex_lock(&ns_ext_q_mutex);
			while (!ns_ext_job_released(jb)) pthread_cond_wait(&ns_ext_done_cond, &ns_ext_q_mutex);
			pthread_mutex_unlock(&ns_ext_q_mutex);
		}
	for (jb = b->gens; jb; jb = next) next = jb->gnext, ns_ext_job_free(b->km, jb);
	kfree(b->km, b->req);
	kfree(b->km, b);
}
