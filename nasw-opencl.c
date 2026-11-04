#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "nasw.h"

#ifndef NS_OPENCL
int ns_gpu_init(int32_t device)
{
	(void)device;
	return 0;
}

const char *ns_gpu_error(void)
{
	return "OpenCL backend not compiled; rebuild with 'make gpu=1'";
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

#else

#include <pthread.h>

typedef int32_t cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef cl_ulong cl_bitfield;
typedef cl_bitfield cl_device_type;
typedef cl_bitfield cl_mem_flags;
typedef cl_bitfield cl_command_queue_properties;
typedef struct _cl_platform_id *cl_platform_id;
typedef struct _cl_device_id *cl_device_id;
typedef struct _cl_context *cl_context;
typedef struct _cl_command_queue *cl_command_queue;
typedef struct _cl_mem *cl_mem;
typedef struct _cl_program *cl_program;
typedef struct _cl_kernel *cl_kernel;

#define CL_SUCCESS 0
#define CL_FALSE 0
#define CL_TRUE 1
#define CL_DEVICE_TYPE_GPU (1ULL << 2)
#define CL_MEM_READ_WRITE (1ULL << 0)
#define CL_MEM_WRITE_ONLY (1ULL << 1)
#define CL_MEM_READ_ONLY (1ULL << 2)
#define CL_MEM_COPY_HOST_PTR (1ULL << 5)
#define CL_DEVICE_LOCAL_MEM_SIZE 0x1023
#define CL_DEVICE_NAME 0x102b
#define CL_PROGRAM_BUILD_LOG 0x1183
#define CL_KERNEL_WORK_GROUP_SIZE 0x11b0

extern cl_int clGetPlatformIDs(cl_uint, cl_platform_id*, cl_uint*);
extern cl_int clGetDeviceIDs(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
extern cl_int clGetDeviceInfo(cl_device_id, cl_uint, size_t, void*, size_t*);
extern cl_context clCreateContext(const intptr_t*, cl_uint, const cl_device_id*, void (*)(const char*, const void*, size_t, void*), void*, cl_int*);
extern cl_command_queue clCreateCommandQueue(cl_context, cl_device_id, cl_command_queue_properties, cl_int*);
extern cl_program clCreateProgramWithSource(cl_context, cl_uint, const char**, const size_t*, cl_int*);
extern cl_int clBuildProgram(cl_program, cl_uint, const cl_device_id*, const char*, void (*)(cl_program, void*), void*);
extern cl_int clGetProgramBuildInfo(cl_program, cl_device_id, cl_uint, size_t, void*, size_t*);
extern cl_kernel clCreateKernel(cl_program, const char*, cl_int*);
extern cl_int clGetKernelWorkGroupInfo(cl_kernel, cl_device_id, cl_uint, size_t, void*, size_t*);
extern cl_mem clCreateBuffer(cl_context, cl_mem_flags, size_t, void*, cl_int*);
extern cl_int clSetKernelArg(cl_kernel, cl_uint, size_t, const void*);
extern cl_int clEnqueueNDRangeKernel(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const void*, void*);
extern cl_int clEnqueueWriteBuffer(cl_command_queue, cl_mem, cl_uint, size_t, size_t, const void*, cl_uint, const void*, void*);
extern cl_int clEnqueueReadBuffer(cl_command_queue, cl_mem, cl_uint, size_t, size_t, void*, cl_uint, const void*, void*);
extern cl_int clFinish(cl_command_queue);
extern cl_int clReleaseCommandQueue(cl_command_queue);
extern cl_int clReleaseKernel(cl_kernel);
extern cl_int clReleaseMemObject(cl_mem);

#ifndef NS_GPU_TILE_SIZE
#define NS_GPU_TILE_SIZE 4
#endif
#if NS_GPU_TILE_SIZE < 3
#error "NS_GPU_TILE_SIZE must be at least 3"
#endif
#define NS_GPU_FRONTIER_SHORTS_PER_COLUMN (11 * NS_GPU_TILE_SIZE + 3)
#define NS_GPU_STRINGIFY_INNER(value) #value
#define NS_GPU_STRINGIFY(value) NS_GPU_STRINGIFY_INNER(value)

static const char ns_gpu_source[] =
"#define TILE_SIZE " NS_GPU_STRINGIFY(NS_GPU_TILE_SIZE) "\n"
"#define H_RIGHT 0\n"
"#define H_BOTTOM (TILE_SIZE)\n"
"#define H_CORNER (4 * TILE_SIZE)\n"
"#define I_RIGHT (4 * TILE_SIZE + 3)\n"
"#define D_BOTTOM (5 * TILE_SIZE + 3)\n"
"#define A_BOTTOM (8 * TILE_SIZE + 3)\n"
"#define B_BOTTOM (9 * TILE_SIZE + 3)\n"
"#define C_BOTTOM (10 * TILE_SIZE + 3)\n"
"#define FRONTIER_STRIDE (11 * TILE_SIZE + 3)\n"
"#define NEG (-32768)\n"
"inline int ns_add(int a, int b, int narrow) {\n"
"  int value = a + b;\n"
"  return clamp(value, -32768, 32767);\n"
"}\n"
"inline int ns_sub(int a, int b, int narrow) {\n"
"  int value = a - b;\n"
"  return clamp(value, -32768, 32767);\n"
"}\n"
"inline int ns_edge_h(int i, int j, int fs) {\n"
"  if (j == -1) {\n"
"    if (i == -1) return 0;\n"
"    if (i == 0 || i == 1) return -fs;\n"
"  }\n"
"  return NEG;\n"
"}\n"
"__kernel void ns_splice_dp(\n"
"    __global const uchar *all_nas, __global const uchar *all_aas,\n"
"    __global const char *all_donor, __global const char *all_acceptor,\n"
"    __global const char *all_score, __global const ulong *offsets,\n"
"    __global const int *dimensions, __global const int *parameters,\n"
"    __local short *frontier, __global ushort *all_trace,\n"
"    __global int *result) {\n"
"  const int job = get_group_id(0);\n"
"  const int lane = get_local_id(0);\n"
"  const int lanes = get_local_size(0);\n"
"  __global const ulong *job_offsets = offsets + (long)job * 6;\n"
"  __global const uchar *nas = all_nas + job_offsets[0];\n"
"  __global const uchar *aas = all_aas + job_offsets[1];\n"
"  __global const char *donor = all_donor + job_offsets[2];\n"
"  __global const char *acceptor = all_acceptor + job_offsets[3];\n"
"  __global const char *score = all_score + job_offsets[4];\n"
"  __global ushort *trace = all_trace + job_offsets[5];\n"
"  const int nl = dimensions[job * 3];\n"
"  const int al = dimensions[job * 3 + 1];\n"
"  const int asize = dimensions[job * 3 + 2];\n"
"  const int go = parameters[job * 5];\n"
"  const int ge = parameters[job * 5 + 1];\n"
"  const int io = parameters[job * 5 + 2];\n"
"  const int fs = parameters[job * 5 + 3];\n"
"  const int narrow = parameters[job * 5 + 4];\n"
"  const int matrix_rows = nl - 2;\n"
"  const int tile_rows = (matrix_rows + TILE_SIZE - 1) / TILE_SIZE;\n"
"  const int tile_columns = (al + TILE_SIZE - 1) / TILE_SIZE;\n"
"  const int frontier_plane_size = tile_columns * FRONTIER_STRIDE;\n"
"  short h_rows[4][TILE_SIZE + 1];\n"
"  short d_rows[3][TILE_SIZE];\n"
"  short a_state[TILE_SIZE], b_state[TILE_SIZE], c_state[TILE_SIZE];\n"
"  for (int tile_diagonal = 0; tile_diagonal < tile_rows + tile_columns - 1; ++tile_diagonal) {\n"
"    const int first_tile_row = max(0, tile_diagonal - tile_columns + 1);\n"
"    const int last_tile_row = min(tile_rows - 1, tile_diagonal);\n"
"    for (int tile_row = first_tile_row + lane; tile_row <= last_tile_row; tile_row += lanes) {\n"
"      const int tile_column = tile_diagonal - tile_row;\n"
"      __local short *input_frontier = frontier + ((tile_diagonal + 1) & 1) * frontier_plane_size;\n"
"      __local short *output_frontier = frontier + (tile_diagonal & 1) * frontier_plane_size;\n"
"      __local short *top_boundary = input_frontier + tile_column * FRONTIER_STRIDE;\n"
"      __local short *left_boundary = input_frontier + max(0, tile_column - 1) * FRONTIER_STRIDE;\n"
"      __local short *output_boundary = output_frontier + tile_column * FRONTIER_STRIDE;\n"
"      const int first_i = 2 + tile_row * TILE_SIZE;\n"
"      const int tile_height = min(TILE_SIZE, nl - first_i);\n"
"      const int first_j = tile_column * TILE_SIZE;\n"
"      const int tile_width = min(TILE_SIZE, al - first_j);\n"
"      for (int halo = 0; halo < 3; ++halo) {\n"
"        const int halo_i = first_i - 3 + halo;\n"
"        if (tile_row == 0 || tile_column == 0)\n"
"          h_rows[halo][0] = (short)ns_edge_h(halo_i, first_j - 1, fs);\n"
"        else\n"
"          h_rows[halo][0] = top_boundary[H_CORNER + halo];\n"
"        for (int column = 0; column < tile_width; ++column) {\n"
"          if (tile_row == 0) {\n"
"            h_rows[halo][column + 1] = (short)ns_edge_h(halo_i, first_j + column, fs);\n"
"            d_rows[halo][column] = (short)NEG;\n"
"          } else {\n"
"            h_rows[halo][column + 1] = top_boundary[H_BOTTOM + halo * TILE_SIZE + column];\n"
"            d_rows[halo][column] = top_boundary[D_BOTTOM + halo * TILE_SIZE + column];\n"
"          }\n"
"        }\n"
"      }\n"
"      for (int column = 0; column < tile_width; ++column) {\n"
"        if (tile_row == 0) {\n"
"          a_state[column] = (short)NEG;\n"
"          b_state[column] = (short)NEG;\n"
"          c_state[column] = (short)NEG;\n"
"        } else {\n"
"          a_state[column] = top_boundary[A_BOTTOM + column];\n"
"          b_state[column] = top_boundary[B_BOTTOM + column];\n"
"          c_state[column] = top_boundary[C_BOTTOM + column];\n"
"        }\n"
"      }\n"
"      for (int row = 0; row < tile_height; ++row) {\n"
"        const int i = first_i + row;\n"
"        const int current_h = (row + 3) & 3;\n"
"        const int h_m1 = (row + 2) & 3;\n"
"        const int h_m2 = (row + 1) & 3;\n"
"        const int h_m3 = row & 3;\n"
"        const int current_d = row % 3;\n"
"        int insertion = NEG;\n"
"        if (tile_column == 0)\n"
"          h_rows[current_h][0] = (short)ns_edge_h(i, -1, fs);\n"
"        else {\n"
"          h_rows[current_h][0] = left_boundary[H_RIGHT + row];\n"
"          insertion = left_boundary[I_RIGHT + row];\n"
"        }\n"
"        for (int column = 0; column < tile_width; ++column) {\n"
"          const int j = first_j + column;\n"
"          const long cell = (long)(i - 2) * al + j;\n"
"          int bits = 0, path = 0, candidate, open_score, extension_score;\n"
"          int deletion, intron_a, intron_b, intron_c;\n"
"          int best = ns_add(h_rows[h_m3][column],\n"
"                            score[(int)nas[i] * asize + (int)aas[j]], narrow);\n"
"          open_score = ns_sub(h_rows[current_h][column], go, narrow);\n"
"          extension_score = insertion;\n"
"          if (extension_score > open_score) bits |= 1 << 4;\n"
"          insertion = ns_sub(max(open_score, extension_score), ge, narrow);\n"
"          if (insertion > best) best = insertion, path = 1;\n"
"          open_score = ns_sub(h_rows[h_m3][column + 1], go, narrow);\n"
"          extension_score = d_rows[current_d][column];\n"
"          if (extension_score > open_score) bits |= 1 << 5;\n"
"          deletion = ns_sub(max(open_score, extension_score), nas[i] == 20 ? fs : ge, narrow);\n"
"          if (deletion > best) best = deletion, path = 2;\n"
"          open_score = ns_sub(ns_sub(h_rows[h_m1][column + 1], io, narrow), donor[i - 1], narrow);\n"
"          extension_score = a_state[column];\n"
"          if (extension_score > open_score) bits |= 1 << 6;\n"
"          intron_a = max(open_score, extension_score);\n"
"          candidate = ns_sub(intron_a, acceptor[i], narrow);\n"
"          if (candidate > best) best = candidate, path = 3;\n"
"          open_score = ns_sub(ns_sub(h_rows[h_m1][column], io, narrow), donor[i], narrow);\n"
"          extension_score = b_state[column];\n"
"          if (extension_score > open_score) bits |= 1 << 7;\n"
"          intron_b = max(open_score, extension_score);\n"
"          candidate = ns_sub(intron_b, acceptor[i - 2], narrow);\n"
"          if (candidate > best) best = candidate, path = 4;\n"
"          open_score = ns_sub(ns_sub(h_rows[h_m1][column], io, narrow), donor[i + 1], narrow);\n"
"          extension_score = c_state[column];\n"
"          if (extension_score > open_score) bits |= 1 << 8;\n"
"          intron_c = max(open_score, extension_score);\n"
"          candidate = ns_sub(intron_c, acceptor[i - 1], narrow);\n"
"          if (candidate > best) best = candidate, path = 5;\n"
"          candidate = ns_sub(h_rows[h_m1][column], fs, narrow);\n"
"          if (candidate > best) best = candidate, path = 6;\n"
"          candidate = ns_sub(h_rows[h_m2][column], fs, narrow);\n"
"          if (candidate > best) best = candidate, path = 7;\n"
"          candidate = ns_sub(h_rows[h_m1][column + 1], fs, narrow);\n"
"          if (candidate > best) best = candidate, path = 8;\n"
"          candidate = ns_sub(h_rows[h_m2][column + 1], fs, narrow);\n"
"          if (candidate > best) best = candidate, path = 9;\n"
"          h_rows[current_h][column + 1] = (short)best;\n"
"          d_rows[current_d][column] = (short)deletion;\n"
"          a_state[column] = (short)intron_a;\n"
"          b_state[column] = (short)intron_b;\n"
"          c_state[column] = (short)intron_c;\n"
"          trace[cell] = (ushort)(bits | path);\n"
"        }\n"
"        output_boundary[H_RIGHT + row] = h_rows[current_h][tile_width];\n"
"        output_boundary[I_RIGHT + row] = (short)insertion;\n"
"        if (row + 3 >= tile_height) {\n"
"          const int bottom_row = row + 3 - tile_height;\n"
"          output_boundary[H_CORNER + bottom_row] = h_rows[current_h][0];\n"
"          for (int column = 0; column < tile_width; ++column) {\n"
"            output_boundary[H_BOTTOM + bottom_row * TILE_SIZE + column] = h_rows[current_h][column + 1];\n"
"            output_boundary[D_BOTTOM + bottom_row * TILE_SIZE + column] = d_rows[current_d][column];\n"
"          }\n"
"        }\n"
"      }\n"
"      for (int column = 0; column < tile_width; ++column) {\n"
"        output_boundary[A_BOTTOM + column] = a_state[column];\n"
"        output_boundary[B_BOTTOM + column] = b_state[column];\n"
"        output_boundary[C_BOTTOM + column] = c_state[column];\n"
"      }\n"
"      if (tile_row == tile_rows - 1 && tile_column == tile_columns - 1) {\n"
"        const int last_h = (tile_height + 2) & 3;\n"
"        result[job] = h_rows[last_h][tile_width];\n"
"      }\n"
"    }\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"  }\n"
"}\n";

#if 0
static const char ns_gpu_tiled_source[] =
"inline int ns_h_full(__global short *h, int al, int i, int j, int fs) {\n"
"  if (j == -1) {\n"
"    if (i == -1) return 0;\n"
"    if (i == 0 || i == 1) return -fs;\n"
"  }\n"
"  if (i < 2 || j < 0) return -32768;\n"
"  return h[(long)(i - 2) * al + j];\n"
"}\n"
"__kernel void ns_splice_dp_tiled(\n"
"    __global const uchar *nas, __global const uchar *aas,\n"
"    __global const char *donor, __global const char *acceptor,\n"
"    __global const char *score, __global short *states,\n"
"    __global ushort *trace, __global int *result, int nl, int al, int asize,\n"
"    int go, int ge, int io, int fs, int narrow) {\n"
"  const int tile_size = 8;\n"
"  const int lane = get_local_id(0);\n"
"  const int lanes = get_local_size(0);\n"
"  const long cells = (long)(nl - 2) * al;\n"
"  __global short *h = states;\n"
"  __global short *ins = h + cells;\n"
"  __global short *del = ins + cells;\n"
"  __global short *intron_a_state = del + cells;\n"
"  __global short *intron_b_state = intron_a_state + cells;\n"
"  __global short *intron_c_state = intron_b_state + cells;\n"
"  const int tile_rows = (nl - 2 + tile_size - 1) / tile_size;\n"
"  const int tile_columns = (al + tile_size - 1) / tile_size;\n"
"  for (int tile_diagonal = 0; tile_diagonal < tile_rows + tile_columns - 1; ++tile_diagonal) {\n"
"    const int first_row = max(0, tile_diagonal - tile_columns + 1);\n"
"    const int last_row = min(tile_rows - 1, tile_diagonal);\n"
"    for (int tile_row = first_row + lane; tile_row <= last_row; tile_row += lanes) {\n"
"      const int tile_column = tile_diagonal - tile_row;\n"
"      const int first_i = 2 + tile_row * tile_size;\n"
"      const int last_i = min(nl, first_i + tile_size);\n"
"      const int first_j = tile_column * tile_size;\n"
"      const int last_j = min(al, first_j + tile_size);\n"
"      for (int i = first_i; i < last_i; ++i) {\n"
"        for (int j = first_j; j < last_j; ++j) {\n"
"          const long index = (long)(i - 2) * al + j;\n"
"          int bits = 0, path = 0, candidate, open_score, extension_score;\n"
"          int insertion, deletion, intron_a, intron_b, intron_c;\n"
"          int best = ns_add(ns_h_full(h, al, i - 3, j - 1, fs),\n"
"                            score[(int)nas[i] * asize + (int)aas[j]], narrow);\n"
"          open_score = ns_sub(ns_h_full(h, al, i, j - 1, fs), go, narrow);\n"
"          extension_score = j > 0 ? ins[index - 1] : -32768;\n"
"          if (extension_score > open_score) bits |= 1 << 4;\n"
"          insertion = ns_sub(max(open_score, extension_score), ge, narrow);\n"
"          if (insertion > best) best = insertion, path = 1;\n"
"          open_score = ns_sub(ns_h_full(h, al, i - 3, j, fs), go, narrow);\n"
"          extension_score = i >= 5 ? del[index - (long)3 * al] : -32768;\n"
"          if (extension_score > open_score) bits |= 1 << 5;\n"
"          deletion = ns_sub(max(open_score, extension_score), nas[i] == 20 ? fs : ge, narrow);\n"
"          if (deletion > best) best = deletion, path = 2;\n"
"          open_score = ns_sub(ns_sub(ns_h_full(h, al, i - 1, j, fs), io, narrow), donor[i - 1], narrow);\n"
"          extension_score = i >= 3 ? intron_a_state[index - al] : -32768;\n"
"          if (extension_score > open_score) bits |= 1 << 6;\n"
"          intron_a = max(open_score, extension_score);\n"
"          candidate = ns_sub(intron_a, acceptor[i], narrow);\n"
"          if (candidate > best) best = candidate, path = 3;\n"
"          open_score = ns_sub(ns_sub(ns_h_full(h, al, i - 1, j - 1, fs), io, narrow), donor[i], narrow);\n"
"          extension_score = i >= 3 ? intron_b_state[index - al] : -32768;\n"
"          if (extension_score > open_score) bits |= 1 << 7;\n"
"          intron_b = max(open_score, extension_score);\n"
"          candidate = ns_sub(intron_b, acceptor[i - 2], narrow);\n"
"          if (candidate > best) best = candidate, path = 4;\n"
"          open_score = ns_sub(ns_sub(ns_h_full(h, al, i - 1, j - 1, fs), io, narrow), donor[i + 1], narrow);\n"
"          extension_score = i >= 3 ? intron_c_state[index - al] : -32768;\n"
"          if (extension_score > open_score) bits |= 1 << 8;\n"
"          intron_c = max(open_score, extension_score);\n"
"          candidate = ns_sub(intron_c, acceptor[i - 1], narrow);\n"
"          if (candidate > best) best = candidate, path = 5;\n"
"          candidate = ns_sub(ns_h_full(h, al, i - 1, j - 1, fs), fs, narrow);\n"
"          if (candidate > best) best = candidate, path = 6;\n"
"          candidate = ns_sub(ns_h_full(h, al, i - 2, j - 1, fs), fs, narrow);\n"
"          if (candidate > best) best = candidate, path = 7;\n"
"          candidate = ns_sub(ns_h_full(h, al, i - 1, j, fs), fs, narrow);\n"
"          if (candidate > best) best = candidate, path = 8;\n"
"          candidate = ns_sub(ns_h_full(h, al, i - 2, j, fs), fs, narrow);\n"
"          if (candidate > best) best = candidate, path = 9;\n"
"          h[index] = (short)best;\n"
"          ins[index] = (short)insertion;\n"
"          del[index] = (short)deletion;\n"
"          intron_a_state[index] = (short)intron_a;\n"
"          intron_b_state[index] = (short)intron_b;\n"
"          intron_c_state[index] = (short)intron_c;\n"
"          trace[index] = (ushort)(bits | path);\n"
"        }\n"
"      }\n"
"    }\n"
"    barrier(CLK_GLOBAL_MEM_FENCE);\n"
"  }\n"
"  if (lane == 0) result[0] = h[cells - 1];\n"
"}\n";
#endif

#if 0
static const char ns_gpu_tile_source[] =
"inline int ns_tile_h(__global short *h, int al, int i, int j, int fs) {\n"
"  if (j == -1) {\n"
"    if (i == -1) return 0;\n"
"    if (i == 0 || i == 1) return -fs;\n"
"  }\n"
"  if (i < 2 || j < 0) return -32768;\n"
"  return h[(long)(i - 2) * al + j];\n"
"}\n"
"__kernel void ns_splice_dp_tile(\n"
"    __global const uchar *all_nas, __global const uchar *all_aas,\n"
"    __global const char *all_donor, __global const char *all_acceptor,\n"
"    __global const char *all_score, __global const ulong *offsets,\n"
"    __global const int *dimensions, __global const int *parameters,\n"
"    __global short *all_states, __global ushort *all_trace, __global int *result,\n"
"    int tile_diagonal, int slots_per_job) {\n"
"  const int tile_size = 8;\n"
"  const int global_index = get_global_id(0);\n"
"  const int job = global_index / slots_per_job;\n"
"  const int slot = global_index - job * slots_per_job;\n"
"  __global const ulong *job_offsets = offsets + (long)job * 6;\n"
"  const int nl = dimensions[job * 3];\n"
"  const int al = dimensions[job * 3 + 1];\n"
"  const int asize = dimensions[job * 3 + 2];\n"
"  const int tile_rows = (nl - 2 + tile_size - 1) / tile_size;\n"
"  const int tile_columns = (al + tile_size - 1) / tile_size;\n"
"  const int first_tile_row = max(0, tile_diagonal - tile_columns + 1);\n"
"  const int last_tile_row = min(tile_rows - 1, tile_diagonal);\n"
"  const int tile_row = first_tile_row + slot;\n"
"  if (tile_diagonal >= tile_rows + tile_columns - 1 || tile_row > last_tile_row) return;\n"
"  const int tile_column = tile_diagonal - tile_row;\n"
"  __global const uchar *nas = all_nas + job_offsets[0];\n"
"  __global const uchar *aas = all_aas + job_offsets[1];\n"
"  __global const char *donor = all_donor + job_offsets[2];\n"
"  __global const char *acceptor = all_acceptor + job_offsets[3];\n"
"  __global const char *score = all_score + job_offsets[4];\n"
"  __global ushort *trace = all_trace + job_offsets[5];\n"
"  const long cells = (long)(nl - 2) * al;\n"
"  __global short *h = all_states + job_offsets[5] * 6;\n"
"  __global short *insertion_state = h + cells;\n"
"  __global short *deletion_state = insertion_state + cells;\n"
"  __global short *intron_a_state = deletion_state + cells;\n"
"  __global short *intron_b_state = intron_a_state + cells;\n"
"  __global short *intron_c_state = intron_b_state + cells;\n"
"  const int go = parameters[job * 5];\n"
"  const int ge = parameters[job * 5 + 1];\n"
"  const int io = parameters[job * 5 + 2];\n"
"  const int fs = parameters[job * 5 + 3];\n"
"  const int narrow = parameters[job * 5 + 4];\n"
"  const int first_i = 2 + tile_row * tile_size;\n"
"  const int last_i = min(nl, first_i + tile_size);\n"
"  const int first_j = tile_column * tile_size;\n"
"  const int last_j = min(al, first_j + tile_size);\n"
"  for (int i = first_i; i < last_i; ++i) {\n"
"    for (int j = first_j; j < last_j; ++j) {\n"
"      const long index = (long)(i - 2) * al + j;\n"
"      int bits = 0, path = 0, candidate, open_score, extension_score;\n"
"      int insertion, deletion, intron_a, intron_b, intron_c;\n"
"      int best = ns_add(ns_tile_h(h, al, i - 3, j - 1, fs),\n"
"                        score[(int)nas[i] * asize + (int)aas[j]], narrow);\n"
"      open_score = ns_sub(ns_tile_h(h, al, i, j - 1, fs), go, narrow);\n"
"      extension_score = j > 0 ? insertion_state[index - 1] : -32768;\n"
"      if (extension_score > open_score) bits |= 1 << 4;\n"
"      insertion = ns_sub(max(open_score, extension_score), ge, narrow);\n"
"      if (insertion > best) best = insertion, path = 1;\n"
"      open_score = ns_sub(ns_tile_h(h, al, i - 3, j, fs), go, narrow);\n"
"      extension_score = i >= 5 ? deletion_state[index - (long)3 * al] : -32768;\n"
"      if (extension_score > open_score) bits |= 1 << 5;\n"
"      deletion = ns_sub(max(open_score, extension_score), nas[i] == 20 ? fs : ge, narrow);\n"
"      if (deletion > best) best = deletion, path = 2;\n"
"      open_score = ns_sub(ns_sub(ns_tile_h(h, al, i - 1, j, fs), io, narrow), donor[i - 1], narrow);\n"
"      extension_score = i >= 3 ? intron_a_state[index - al] : -32768;\n"
"      if (extension_score > open_score) bits |= 1 << 6;\n"
"      intron_a = max(open_score, extension_score);\n"
"      candidate = ns_sub(intron_a, acceptor[i], narrow);\n"
"      if (candidate > best) best = candidate, path = 3;\n"
"      open_score = ns_sub(ns_sub(ns_tile_h(h, al, i - 1, j - 1, fs), io, narrow), donor[i], narrow);\n"
"      extension_score = i >= 3 ? intron_b_state[index - al] : -32768;\n"
"      if (extension_score > open_score) bits |= 1 << 7;\n"
"      intron_b = max(open_score, extension_score);\n"
"      candidate = ns_sub(intron_b, acceptor[i - 2], narrow);\n"
"      if (candidate > best) best = candidate, path = 4;\n"
"      open_score = ns_sub(ns_sub(ns_tile_h(h, al, i - 1, j - 1, fs), io, narrow), donor[i + 1], narrow);\n"
"      extension_score = i >= 3 ? intron_c_state[index - al] : -32768;\n"
"      if (extension_score > open_score) bits |= 1 << 8;\n"
"      intron_c = max(open_score, extension_score);\n"
"      candidate = ns_sub(intron_c, acceptor[i - 1], narrow);\n"
"      if (candidate > best) best = candidate, path = 5;\n"
"      candidate = ns_sub(ns_tile_h(h, al, i - 1, j - 1, fs), fs, narrow);\n"
"      if (candidate > best) best = candidate, path = 6;\n"
"      candidate = ns_sub(ns_tile_h(h, al, i - 2, j - 1, fs), fs, narrow);\n"
"      if (candidate > best) best = candidate, path = 7;\n"
"      candidate = ns_sub(ns_tile_h(h, al, i - 1, j, fs), fs, narrow);\n"
"      if (candidate > best) best = candidate, path = 8;\n"
"      candidate = ns_sub(ns_tile_h(h, al, i - 2, j, fs), fs, narrow);\n"
"      if (candidate > best) best = candidate, path = 9;\n"
"      h[index] = (short)best;\n"
"      insertion_state[index] = (short)insertion;\n"
"      deletion_state[index] = (short)deletion;\n"
"      intron_a_state[index] = (short)intron_a;\n"
"      intron_b_state[index] = (short)intron_b;\n"
"      intron_c_state[index] = (short)intron_c;\n"
"      trace[index] = (ushort)(bits | path);\n"
"      if (i == nl - 1 && j == al - 1) result[job] = best;\n"
"    }\n"
"  }\n"
"}\n";
#endif

static pthread_mutex_t ns_gpu_mutex = PTHREAD_MUTEX_INITIALIZER;
static cl_context ns_gpu_context;
static cl_device_id ns_gpu_device;
static cl_program ns_gpu_program;
static int ns_gpu_ready;
static cl_ulong ns_gpu_local_mem;
static size_t ns_gpu_work_group_size;
static uint64_t ns_gpu_dispatches;
static uint64_t ns_gpu_launches;
static char ns_gpu_message[4096] = "GPU backend has not been initialized";
static char ns_gpu_device_name[256] = "none";

#ifndef NS_GPU_MAX_BATCH
#define NS_GPU_MAX_BATCH 128
#endif
#ifndef NS_GPU_EXECUTOR_COUNT
#define NS_GPU_EXECUTOR_COUNT 1
#endif
#ifndef NS_GPU_BATCH_WAIT_NS
#define NS_GPU_BATCH_WAIT_NS 2000000L
#endif
#ifndef NS_GPU_LOCAL_SIZE
#define NS_GPU_LOCAL_SIZE 256
#endif

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

typedef struct ns_gpu_batch_s {
	ns_gpu_request_t *requests[NS_GPU_MAX_BATCH];
	int count;
	struct ns_gpu_batch_s *next;
} ns_gpu_batch_t;

typedef struct {
	pthread_t thread;
	cl_command_queue queue;
	cl_kernel kernel;
	cl_mem buffers[10];
	size_t capacities[10];
} ns_gpu_executor_t;

static pthread_cond_t ns_gpu_condition = PTHREAD_COND_INITIALIZER;
static pthread_t ns_gpu_collector_thread;
static ns_gpu_executor_t ns_gpu_executors[NS_GPU_EXECUTOR_COUNT];
static ns_gpu_request_t *ns_gpu_head, *ns_gpu_tail;
static ns_gpu_batch_t *ns_gpu_batch_head, *ns_gpu_batch_tail;
static int ns_gpu_queue_size, ns_gpu_batches_active, ns_gpu_worker_started;

static int ns_gpu_dispatch_batch(ns_gpu_executor_t *executor, ns_gpu_request_t **requests, int count);
static void *ns_gpu_collect_batches(void *unused);
static void *ns_gpu_execute_batches(void *data);

static int ns_gpu_state_size(int32_t nl, int32_t al, size_t *bytes, int32_t *parallel_tiles)
{
	size_t tile_rows = ((size_t)nl - 2 + NS_GPU_TILE_SIZE - 1) / NS_GPU_TILE_SIZE;
	size_t tile_columns = ((size_t)al + NS_GPU_TILE_SIZE - 1) / NS_GPU_TILE_SIZE;
	if (tile_columns > SIZE_MAX / (2 * NS_GPU_FRONTIER_SHORTS_PER_COLUMN * sizeof(int16_t))) return 0;
	*bytes = 2 * tile_columns * NS_GPU_FRONTIER_SHORTS_PER_COLUMN * sizeof(int16_t);
	*parallel_tiles = (int32_t)(tile_rows < tile_columns ? tile_rows : tile_columns);
	return 1;
}

static void ns_gpu_set_error(const char *operation, cl_int status)
{
	snprintf(ns_gpu_message, sizeof(ns_gpu_message), "%s failed with OpenCL status %d", operation, status);
}

int ns_gpu_init(int32_t requested_device)
{
	cl_platform_id platforms[16];
	cl_device_id devices[64];
	cl_command_queue queue;
	cl_kernel kernel;
	cl_uint n_platforms = 0, n_devices = 0, i;
	cl_int status;
	int threads_started = 0;

	pthread_mutex_lock(&ns_gpu_mutex);
	if (ns_gpu_ready) {
		pthread_mutex_unlock(&ns_gpu_mutex);
		return 1;
	}
	status = clGetPlatformIDs(16, platforms, &n_platforms);
	if (status != CL_SUCCESS) {
		ns_gpu_set_error("clGetPlatformIDs", status);
		goto fail;
	}
	for (i = 0; i < n_platforms && n_devices < 64; ++i) {
		cl_uint count = 0;
		status = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, 0, &count);
		if (status != CL_SUCCESS || count == 0) continue;
		if (count > 64 - n_devices) count = 64 - n_devices;
		status = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, count, &devices[n_devices], 0);
		if (status == CL_SUCCESS) n_devices += count;
	}
	if ((cl_uint)requested_device >= n_devices) {
		snprintf(ns_gpu_message, sizeof(ns_gpu_message), "OpenCL GPU %d does not exist (%u available)", requested_device, n_devices);
		goto fail;
	}
	ns_gpu_device = devices[requested_device];
	ns_gpu_context = clCreateContext(0, 1, &ns_gpu_device, 0, 0, &status);
	if (status != CL_SUCCESS) {
		ns_gpu_set_error("clCreateContext", status);
		goto fail;
	}
	queue = clCreateCommandQueue(ns_gpu_context, ns_gpu_device, 0, &status);
	if (status != CL_SUCCESS) {
		ns_gpu_set_error("clCreateCommandQueue", status);
		goto fail;
	}
	clReleaseCommandQueue(queue);
	{
		const char *source = ns_gpu_source;
		ns_gpu_program = clCreateProgramWithSource(ns_gpu_context, 1, &source, 0, &status);
	}
	if (status != CL_SUCCESS) {
		ns_gpu_set_error("clCreateProgramWithSource", status);
		goto fail;
	}
	status = clBuildProgram(ns_gpu_program, 1, &ns_gpu_device, "-cl-std=CL1.2", 0, 0);
	if (status != CL_SUCCESS) {
		size_t size = 0;
		clGetProgramBuildInfo(ns_gpu_program, ns_gpu_device, CL_PROGRAM_BUILD_LOG, 0, 0, &size);
		if (size > sizeof(ns_gpu_message) - 32) size = sizeof(ns_gpu_message) - 32;
		memcpy(ns_gpu_message, "OpenCL build failed: ", 21);
		clGetProgramBuildInfo(ns_gpu_program, ns_gpu_device, CL_PROGRAM_BUILD_LOG, size, ns_gpu_message + 21, 0);
		ns_gpu_message[21 + size] = 0;
		goto fail;
	}
	kernel = clCreateKernel(ns_gpu_program, "ns_splice_dp", &status);
	if (status != CL_SUCCESS) {
		ns_gpu_set_error("clCreateKernel", status);
		goto fail;
	}
	status = clGetKernelWorkGroupInfo(kernel, ns_gpu_device, CL_KERNEL_WORK_GROUP_SIZE,
			sizeof(ns_gpu_work_group_size), &ns_gpu_work_group_size, 0);
	if (status != CL_SUCCESS || ns_gpu_work_group_size == 0)
		ns_gpu_work_group_size = 256;
	if (ns_gpu_work_group_size > NS_GPU_LOCAL_SIZE)
		ns_gpu_work_group_size = NS_GPU_LOCAL_SIZE;
	while (ns_gpu_work_group_size & (ns_gpu_work_group_size - 1))
		ns_gpu_work_group_size &= ns_gpu_work_group_size - 1;
	clReleaseKernel(kernel);
	status = clGetDeviceInfo(ns_gpu_device, CL_DEVICE_NAME, sizeof(ns_gpu_device_name), ns_gpu_device_name, 0);
	if (status != CL_SUCCESS) strcpy(ns_gpu_device_name, "unnamed OpenCL GPU");
	status = clGetDeviceInfo(ns_gpu_device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(ns_gpu_local_mem), &ns_gpu_local_mem, 0);
	if (status != CL_SUCCESS) ns_gpu_local_mem = 0;
	ns_gpu_device_name[sizeof(ns_gpu_device_name) - 1] = 0;
	for (i = 0; i < NS_GPU_EXECUTOR_COUNT; ++i) {
		ns_gpu_executors[i].queue = clCreateCommandQueue(ns_gpu_context, ns_gpu_device, 0, &status);
		if (status != CL_SUCCESS) {
			ns_gpu_set_error("clCreateCommandQueue", status);
			goto fail;
		}
		ns_gpu_executors[i].kernel = clCreateKernel(ns_gpu_program, "ns_splice_dp", &status);
		if (status != CL_SUCCESS) {
			ns_gpu_set_error("clCreateKernel", status);
			goto fail;
		}
	}
	strcpy(ns_gpu_message, "none");
	ns_gpu_ready = 1;
	for (i = 0; i < NS_GPU_EXECUTOR_COUNT; ++i) {
		status = pthread_create(&ns_gpu_executors[i].thread, 0, ns_gpu_execute_batches, &ns_gpu_executors[i]);
		if (status != 0) {
			snprintf(ns_gpu_message, sizeof(ns_gpu_message), "failed to start GPU executor: %s", strerror(status));
			ns_gpu_ready = 0;
			goto fail;
		}
		pthread_detach(ns_gpu_executors[i].thread);
		++threads_started;
	}
	status = pthread_create(&ns_gpu_collector_thread, 0, ns_gpu_collect_batches, 0);
	if (status != 0) {
		snprintf(ns_gpu_message, sizeof(ns_gpu_message), "failed to start GPU batch collector: %s", strerror(status));
		ns_gpu_ready = 0;
		goto fail;
	}
	pthread_detach(ns_gpu_collector_thread);
	++threads_started;
	ns_gpu_worker_started = 1;
	pthread_mutex_unlock(&ns_gpu_mutex);
	return 1;

fail:
	if (threads_started == 0) {
		for (i = 0; i < NS_GPU_EXECUTOR_COUNT; ++i) {
			if (ns_gpu_executors[i].kernel) clReleaseKernel(ns_gpu_executors[i].kernel);
			if (ns_gpu_executors[i].queue) clReleaseCommandQueue(ns_gpu_executors[i].queue);
			ns_gpu_executors[i].kernel = 0;
			ns_gpu_executors[i].queue = 0;
		}
	}
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

double ns_gpu_h2d_seconds(void) { return 0.0; }
double ns_gpu_kernel_seconds(void) { return 0.0; }
double ns_gpu_d2h_seconds(void) { return 0.0; }

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

static int ns_gpu_global_direct(void *km, const char *ns, int32_t nl, const char *as, int32_t al,
		const ns_opt_t *opt, const uint8_t *ss, ns_rst_t *r)
{
	uint8_t *prepared = 0, *aas = 0;
	int8_t *donor = 0, *acceptor = 0;
	uint16_t *trace = 0;
	cl_mem buffers[10] = {0,0,0,0,0,0,0,0,0,0};
	cl_command_queue queue = 0;
	cl_kernel kernel = 0;
	cl_int status = CL_SUCCESS;
	cl_uint argument = 0;
	int32_t narrow, parallel_tiles, score, max_sub = 0;
	int32_t dimensions[3], parameters[5];
	cl_ulong offsets[6] = {0,0,0,0,0,0};
	int64_t cells;
	size_t global_size, local_size, trace_bytes, state_bytes;
	int success = 0, i;

	if (nl < 3 || al < 1 || opt->asize < 1) return 0;
	cells = (int64_t)(nl - 2) * al;
	if (cells <= 0 || (uint64_t)cells > SIZE_MAX / sizeof(*trace)) return 0;
	if (!ns_gpu_state_size(nl, al, &state_bytes, &parallel_tiles)) return 0;
	trace_bytes = (size_t)cells * sizeof(*trace);
	for (i = 0; i < opt->asize * opt->asize; ++i)
		if (max_sub < opt->sc[i]) max_sub = opt->sc[i];
	narrow = (int64_t)al * max_sub + opt->end_bonus <= 30000;
	if (!narrow || state_bytes > ns_gpu_local_mem) goto cleanup;
	local_size = ns_gpu_work_group_size < 32? ns_gpu_work_group_size : 32;
	while (local_size < (size_t)parallel_tiles && local_size < ns_gpu_work_group_size)
		local_size <<= 1;
	global_size = local_size;
	prepared = ns_gpu_prep_seq(km, ns, nl, as, al, opt, ss, &aas, &donor, &acceptor);
	trace = Kmalloc(km, uint16_t, cells);
	if (prepared == 0 || trace == 0) goto cleanup;
	dimensions[0] = nl, dimensions[1] = al, dimensions[2] = opt->asize;
	parameters[0] = opt->go, parameters[1] = opt->ge, parameters[2] = opt->io;
	parameters[3] = opt->fs, parameters[4] = narrow;

	if (!ns_gpu_ready) goto cleanup;
	queue = clCreateCommandQueue(ns_gpu_context, ns_gpu_device, 0, &status);
	if (status != CL_SUCCESS) goto opencl_error;
	kernel = clCreateKernel(ns_gpu_program, "ns_splice_dp", &status);
	if (status != CL_SUCCESS) goto opencl_error;
	buffers[0] = clCreateBuffer(ns_gpu_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nl, prepared, &status);
	if (status != CL_SUCCESS) goto opencl_error;
	buffers[1] = clCreateBuffer(ns_gpu_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, al, aas, &status);
	if (status != CL_SUCCESS) goto opencl_error;
	buffers[2] = clCreateBuffer(ns_gpu_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nl + 1, donor, &status);
	if (status != CL_SUCCESS) goto opencl_error;
	buffers[3] = clCreateBuffer(ns_gpu_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nl + 1, acceptor, &status);
	if (status != CL_SUCCESS) goto opencl_error;
	buffers[4] = clCreateBuffer(ns_gpu_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, opt->asize * opt->asize, (void*)opt->sc, &status);
	if (status != CL_SUCCESS) goto opencl_error;
	buffers[5] = clCreateBuffer(ns_gpu_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(offsets), offsets, &status);
	if (status != CL_SUCCESS) goto opencl_error;
	buffers[6] = clCreateBuffer(ns_gpu_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(dimensions), dimensions, &status);
	if (status != CL_SUCCESS) goto opencl_error;
	buffers[7] = clCreateBuffer(ns_gpu_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(parameters), parameters, &status);
	if (status != CL_SUCCESS) goto opencl_error;
	buffers[8] = clCreateBuffer(ns_gpu_context, CL_MEM_WRITE_ONLY, trace_bytes, 0, &status);
	if (status != CL_SUCCESS) goto opencl_error;
	buffers[9] = clCreateBuffer(ns_gpu_context, CL_MEM_WRITE_ONLY, sizeof(score), 0, &status);
	if (status != CL_SUCCESS) goto opencl_error;
#define NS_GPU_ARG(value) do { status = clSetKernelArg(kernel, argument++, sizeof(value), &(value)); if (status != CL_SUCCESS) goto opencl_error; } while (0)
	for (i = 0; i < 8; ++i) NS_GPU_ARG(buffers[i]);
	status = clSetKernelArg(kernel, argument++, state_bytes, 0);
	if (status != CL_SUCCESS) goto opencl_error;
	NS_GPU_ARG(buffers[8]); NS_GPU_ARG(buffers[9]);
#undef NS_GPU_ARG
	status = clEnqueueNDRangeKernel(queue, kernel, 1, 0, &global_size, &local_size, 0, 0, 0);
	if (status != CL_SUCCESS) goto opencl_error;
	status = clEnqueueReadBuffer(queue, buffers[8], CL_TRUE, 0, trace_bytes, trace, 0, 0, 0);
	if (status != CL_SUCCESS) goto opencl_error;
	status = clEnqueueReadBuffer(queue, buffers[9], CL_TRUE, 0, sizeof(score), &score, 0, 0, 0);
	if (status != CL_SUCCESS) goto opencl_error;
	__sync_fetch_and_add(&ns_gpu_dispatches, 1);
	__sync_fetch_and_add(&ns_gpu_launches, 1);
	success = 1;
	goto release;

opencl_error:
	ns_gpu_set_error("GPU alignment", status);
release:
	for (i = 0; i < 10; ++i) if (buffers[i]) clReleaseMemObject(buffers[i]);
	if (kernel) clReleaseKernel(kernel);
	if (queue) clReleaseCommandQueue(queue);
	if (success) {
		r->nt_len = nl, r->aa_len = al, r->score = score;
		ns_gpu_backtrack(km, trace, nl, al, r);
	}
cleanup:
	if (trace) kfree(km, trace);
	if (prepared) kfree(km, prepared);
	return success;
}

static int ns_gpu_reserve_buffer(ns_gpu_executor_t *executor, int index, cl_mem_flags flags,
		size_t bytes, cl_int *status)
{
	cl_mem buffer;
	if (executor->buffers[index] && executor->capacities[index] >= bytes) return 1;
	buffer = clCreateBuffer(ns_gpu_context, flags, bytes, 0, status);
	if (*status != CL_SUCCESS) return 0;
	if (executor->buffers[index]) clReleaseMemObject(executor->buffers[index]);
	executor->buffers[index] = buffer;
	executor->capacities[index] = bytes;
	return 1;
}

static int ns_gpu_dispatch_batch(ns_gpu_executor_t *executor, ns_gpu_request_t **requests, int count)
{
	cl_mem *buffers = executor->buffers;
	cl_command_queue queue = executor->queue;
	cl_kernel kernel = executor->kernel;
	cl_int status = CL_SUCCESS;
	cl_mem_flags flags[10] = { CL_MEM_READ_ONLY, CL_MEM_READ_ONLY, CL_MEM_READ_ONLY, CL_MEM_READ_ONLY,
		CL_MEM_READ_ONLY, CL_MEM_READ_ONLY, CL_MEM_READ_ONLY, CL_MEM_READ_ONLY,
		CL_MEM_WRITE_ONLY, CL_MEM_WRITE_ONLY };
	cl_uint argument = 0;
	cl_ulong *offsets = 0;
	int32_t *dimensions = 0, *parameters = 0, *scores = 0;
	uint8_t *nas_data = 0, *aas_data = 0;
	int8_t *donor_data = 0, *acceptor_data = 0, *score_data = 0;
	size_t total_nas = 0, total_aas = 0, total_sites = 0, total_scores = 0, total_trace = 0;
	size_t nas_offset = 0, aas_offset = 0, site_offset = 0, score_offset = 0;
	size_t trace_offset = 0;
	size_t buffer_sizes[10];
	size_t state_bytes = 0, local_size, global_size;
	int32_t max_parallel_tiles = 1;
	int commands_enqueued = 0, success = 0, i;

	for (i = 0; i < count; ++i) {
		ns_gpu_request_t *request = requests[i];
		size_t score_bytes = (size_t)request->asize * request->asize;
		if (total_nas > SIZE_MAX - request->nl || total_aas > SIZE_MAX - request->al ||
			total_sites > SIZE_MAX - (size_t)request->nl - 1 || total_scores > SIZE_MAX - score_bytes ||
			total_trace > SIZE_MAX - request->trace_bytes) {
			snprintf(ns_gpu_message, sizeof(ns_gpu_message), "GPU batch size overflow");
			goto cleanup;
		}
		total_nas += request->nl;
		total_aas += request->al;
		total_sites += (size_t)request->nl + 1;
		total_scores += score_bytes;
		total_trace += request->trace_bytes;
		if (state_bytes < request->state_bytes) state_bytes = request->state_bytes;
		if (max_parallel_tiles < request->parallel_tiles) max_parallel_tiles = request->parallel_tiles;
	}
	local_size = ns_gpu_work_group_size < 32? ns_gpu_work_group_size : 32;
	while (local_size < (size_t)max_parallel_tiles && local_size < ns_gpu_work_group_size)
		local_size <<= 1;
	global_size = local_size * count;
	nas_data = (uint8_t*)malloc(total_nas);
	aas_data = (uint8_t*)malloc(total_aas);
	donor_data = (int8_t*)malloc(total_sites);
	acceptor_data = (int8_t*)malloc(total_sites);
	score_data = (int8_t*)malloc(total_scores);
	offsets = (cl_ulong*)malloc((size_t)count * 6 * sizeof(*offsets));
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
		parameters[i * 5 + 4] = request->narrow;
		memcpy(nas_data + nas_offset, request->prepared, request->nl);
		memcpy(aas_data + aas_offset, request->aas, request->al);
		memcpy(donor_data + site_offset, request->donor, (size_t)request->nl + 1);
		memcpy(acceptor_data + site_offset, request->acceptor, (size_t)request->nl + 1);
		memcpy(score_data + score_offset, request->score_matrix, score_bytes);
		nas_offset += request->nl;
		aas_offset += request->al;
		site_offset += (size_t)request->nl + 1;
		score_offset += score_bytes;
		trace_offset += request->trace_bytes;
	}
	buffer_sizes[0] = total_nas;
	buffer_sizes[1] = total_aas;
	buffer_sizes[2] = buffer_sizes[3] = total_sites;
	buffer_sizes[4] = total_scores;
	buffer_sizes[5] = (size_t)count * 6 * sizeof(*offsets);
	buffer_sizes[6] = (size_t)count * 3 * sizeof(*dimensions);
	buffer_sizes[7] = (size_t)count * 5 * sizeof(*parameters);
	buffer_sizes[8] = total_trace;
	buffer_sizes[9] = (size_t)count * sizeof(*scores);
	for (i = 0; i < 10; ++i)
		if (!ns_gpu_reserve_buffer(executor, i, flags[i], buffer_sizes[i], &status)) goto opencl_error;
#define NS_GPU_WRITE(index, data) do { \
	status = clEnqueueWriteBuffer(queue, buffers[index], CL_FALSE, 0, buffer_sizes[index], data, 0, 0, 0); \
	if (status != CL_SUCCESS) goto opencl_error; \
	commands_enqueued = 1; \
} while (0)
	NS_GPU_WRITE(0, nas_data);
	NS_GPU_WRITE(1, aas_data);
	NS_GPU_WRITE(2, donor_data);
	NS_GPU_WRITE(3, acceptor_data);
	NS_GPU_WRITE(4, score_data);
	NS_GPU_WRITE(5, offsets);
	NS_GPU_WRITE(6, dimensions);
	NS_GPU_WRITE(7, parameters);
#undef NS_GPU_WRITE
#define NS_GPU_BATCH_ARG(value) do { status = clSetKernelArg(kernel, argument++, sizeof(value), &(value)); if (status != CL_SUCCESS) goto opencl_error; } while (0)
	for (i = 0; i < 8; ++i) NS_GPU_BATCH_ARG(buffers[i]);
	status = clSetKernelArg(kernel, argument++, state_bytes, 0);
	if (status != CL_SUCCESS) goto opencl_error;
	NS_GPU_BATCH_ARG(buffers[8]); NS_GPU_BATCH_ARG(buffers[9]);
#undef NS_GPU_BATCH_ARG
	status = clEnqueueNDRangeKernel(queue, kernel, 1, 0, &global_size, &local_size, 0, 0, 0);
	if (status != CL_SUCCESS) goto opencl_error;
	for (i = 0, trace_offset = 0; i < count; ++i) {
		status = clEnqueueReadBuffer(queue, buffers[8], CL_FALSE, trace_offset, requests[i]->trace_bytes,
				requests[i]->trace, 0, 0, 0);
		if (status != CL_SUCCESS) goto opencl_error;
		trace_offset += requests[i]->trace_bytes;
	}
	status = clEnqueueReadBuffer(queue, buffers[9], CL_TRUE, 0, (size_t)count * sizeof(*scores), scores, 0, 0, 0);
	if (status != CL_SUCCESS) goto opencl_error;
	for (i = 0; i < count; ++i) requests[i]->score = scores[i];
	__sync_fetch_and_add(&ns_gpu_dispatches, count);
	__sync_fetch_and_add(&ns_gpu_launches, 1);
	success = 1;
	goto cleanup;

opencl_error:
	ns_gpu_set_error("GPU batch alignment", status);
	if (commands_enqueued) clFinish(queue);
cleanup:
	free(scores); free(parameters); free(dimensions); free(offsets);
	free(score_data); free(acceptor_data); free(donor_data); free(aas_data); free(nas_data);
	return success;
}

static void *ns_gpu_collect_batches(void *unused)
{
	(void)unused;
	for (;;) {
		ns_gpu_batch_t *batch;
		struct timespec deadline;
		int status;
		pthread_mutex_lock(&ns_gpu_mutex);
		while (ns_gpu_head == 0 || ns_gpu_batches_active >= NS_GPU_EXECUTOR_COUNT)
			pthread_cond_wait(&ns_gpu_condition, &ns_gpu_mutex);
		clock_gettime(CLOCK_REALTIME, &deadline);
		deadline.tv_nsec += NS_GPU_BATCH_WAIT_NS;
		if (deadline.tv_nsec >= 1000000000L)
			++deadline.tv_sec, deadline.tv_nsec -= 1000000000L;
		while (ns_gpu_queue_size < NS_GPU_MAX_BATCH) {
			status = pthread_cond_timedwait(&ns_gpu_condition, &ns_gpu_mutex, &deadline);
			if (status == ETIMEDOUT) break;
		}
		batch = (ns_gpu_batch_t*)malloc(sizeof(*batch));
		if (batch == 0) {
			ns_gpu_request_t *request = ns_gpu_head;
			ns_gpu_head = request->next;
			if (--ns_gpu_queue_size == 0) ns_gpu_tail = 0;
			request->success = 0;
			request->done = 1;
			snprintf(ns_gpu_message, sizeof(ns_gpu_message), "failed to allocate GPU batch");
			pthread_cond_broadcast(&ns_gpu_condition);
			pthread_mutex_unlock(&ns_gpu_mutex);
			continue;
		}
		memset(batch, 0, sizeof(*batch));
		while (batch->count < NS_GPU_MAX_BATCH && ns_gpu_head) {
			batch->requests[batch->count++] = ns_gpu_head;
			ns_gpu_head = ns_gpu_head->next;
			--ns_gpu_queue_size;
		}
		if (ns_gpu_head == 0) ns_gpu_tail = 0;
		if (ns_gpu_batch_tail) ns_gpu_batch_tail->next = batch;
		else ns_gpu_batch_head = batch;
		ns_gpu_batch_tail = batch;
		++ns_gpu_batches_active;
		pthread_cond_broadcast(&ns_gpu_condition);
		pthread_mutex_unlock(&ns_gpu_mutex);
	}
	return 0;
}

static void *ns_gpu_execute_batches(void *data)
{
	ns_gpu_executor_t *executor = (ns_gpu_executor_t*)data;
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

		success = ns_gpu_dispatch_batch(executor, batch->requests, batch->count);
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
	if (!request->narrow || !ns_gpu_state_size(nl, al, &request->state_bytes,
			&request->parallel_tiles) || request->state_bytes > ns_gpu_local_mem) goto fail;
	request->prepared = ns_gpu_prep_seq(km, ns, nl, as, al, opt, ss,
			&request->aas, &request->donor, &request->acceptor);
	request->trace = Kmalloc(km, uint16_t, cells);
	request->score_matrix = Kmalloc(km, int8_t, (size_t)opt->asize * opt->asize);
	if (!request->prepared || !request->trace || !request->score_matrix) goto fail;
	memcpy(request->score_matrix, opt->sc, (size_t)opt->asize * opt->asize);
	request->nl = nl, request->al = al, request->asize = opt->asize;
	request->go = opt->go, request->ge = opt->ge, request->io = opt->io, request->fs = opt->fs;
	request->trace_bytes = (size_t)cells * sizeof(uint16_t);

	pthread_mutex_lock(&ns_gpu_mutex);
	if (ns_gpu_tail) ns_gpu_tail->next = request;
	else ns_gpu_head = request;
	ns_gpu_tail = request;
	++ns_gpu_queue_size;
	pthread_cond_broadcast(&ns_gpu_condition);
	pthread_mutex_unlock(&ns_gpu_mutex);
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

#endif

/* --gpu-prefilter requires the CUDA backend (see nasw-cuda.cu); the OpenCL backend never
 * implemented the gapless X-drop kernel, so these stubs apply whether or not NS_OPENCL is defined. */
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