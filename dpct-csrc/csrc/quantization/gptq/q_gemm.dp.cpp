/*
Adapted from https://github.com/turboderp/exllamav2 and https://github.com/qwopqwop200/GPTQ-for-LLaMa
*/

#include <cstdint>
#include <cstdio>

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
/* DPCT_ORIG #include <cuda_runtime.h>*/
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
/* DPCT_ORIG #include <cuda_fp16.h>*/

/* DPCT_ORIG #include "compat.cuh"*/
#include "compat.dp.hpp"
/* DPCT_ORIG #include "matrix_view.cuh"*/
#include "matrix_view.dp.hpp"
/* DPCT_ORIG #include "qdq_4.cuh"*/
#include "qdq_4.dp.hpp"

namespace vllm {
namespace gptq {

#define BLOCK_KN_SIZE 128
#define BLOCK_M_SIZE_MAX 8
#define MAX_GROUPS_IN_BLOCK (BLOCK_KN_SIZE / 32)
#define MAX_Q_GEMM_ROWS 50
#define MAX_ALT_GEMM_ROWS 8
#define THREADS_X 32
#define THREADS_Y 32
#define DIVIDE(x, size) (((x) + (size) - 1) / (size))

#if defined(USE_ROCM)
#include <hipblas/hipblas.h>
__host__ __forceinline__ hipblasStatus_t __compat_hipblasHgemm(hipblasHandle_t    handle,
                                                               hipblasOperation_t transA,
                                                               hipblasOperation_t transB,
                                                               int                m,
                                                               int                n,
                                                               int                k,
                                                               const half*        alpha,
                                                               const half*        AP,
                                                               int                lda,
                                                               const half*        BP,
                                                               int                ldb,
                                                               const half*        beta,
                                                               half*              CP,
                                                               int                ldc) {
    return hipblasHgemm(handle, transA, transB, m, n, k,
                        reinterpret_cast<const hipblasHalf *>(alpha),
                        reinterpret_cast<const hipblasHalf *>(AP), lda,
                        reinterpret_cast<const hipblasHalf *>(BP), ldb,
                        reinterpret_cast<const hipblasHalf *>(beta),
                        reinterpret_cast<hipblasHalf *>(CP), ldc);
}
#define hipblasHgemm __compat_hipblasHgemm

// Previous version of PyTorch were converting to rocBLAS instead of hipBLAS.
#define rocblas_operation_none HIPBLAS_OP_N
#define rocblas_hgemm __compat_hipblasHgemm
#endif

/* DPCT_ORIG __forceinline__ __device__ half2 dot22_8(half2(&dq)[4], const half*
 * a_ptr, const half2 g_result)*/
__dpct_inline__ sycl::half2 dot22_8(sycl::half2 (&dq)[4],
                                    const sycl::half *a_ptr,
                                    const sycl::half2 g_result)
{
/* DPCT_ORIG     half2 result = {};*/
    sycl::half2 result = {};
/* DPCT_ORIG     const half2* a2_ptr = (const half2*)a_ptr;*/
    const sycl::half2 *a2_ptr = (const sycl::half2 *)a_ptr;
#pragma unroll
/* DPCT_ORIG     for (int i = 0; i < 4; i++) result = __hfma2(dq[i], *a2_ptr++,
 * result);*/
    for (int i = 0; i < 4; i++) result = sycl::fma(dq[i], *a2_ptr++, result);
/* DPCT_ORIG     return __hadd2(result, g_result);*/
    return result + g_result;
}

/* DPCT_ORIG __forceinline__ __device__ float dot22_8_f(half2(&dq)[4], const
 * half* a_ptr)*/
__dpct_inline__ float dot22_8_f(sycl::half2 (&dq)[4], const sycl::half *a_ptr)
{
/* DPCT_ORIG     half2 result = {};*/
    sycl::half2 result = {};
/* DPCT_ORIG     const half2* a2_ptr = (const half2*)a_ptr;*/
    const sycl::half2 *a2_ptr = (const sycl::half2 *)a_ptr;
#pragma unroll
/* DPCT_ORIG     for (int i = 0; i < 4; i++) result = __hfma2(dq[i], *a2_ptr++,
 * result);*/
    for (int i = 0; i < 4; i++) result = sycl::fma(dq[i], *a2_ptr++, result);
/* DPCT_ORIG     return __half2float(__low2half(result)) +
 * __half2float(__high2half(result));*/
    return sycl::vec<sycl::half, 1>{result[1]}
               .convert<float, sycl::rounding_mode::automatic>()[0] +
           sycl::vec<sycl::half, 1>{result[0]}
               .convert<float, sycl::rounding_mode::automatic>()[0];
}

typedef void (*fp_gemm_half_q_half_gptq_kernel)(
    /* DPCT_ORIG     const half*,*/
    const sycl::half *, const uint32_t *, const uint32_t *,
    /* DPCT_ORIG     const half*,*/
    const sycl::half *,
    /* DPCT_ORIG     half*,*/
    sycl::half *, const int, const int, const int, const int, const int *);

/* DPCT_ORIG template <bool first_block, int m_count>
__global__ void gemm_half_q_half_gptq_kernel
(
    const half* __restrict__ a,
    const uint32_t* __restrict__ b_q_weight,
    const uint32_t* __restrict__ b_gptq_qzeros,
    const half* __restrict__ b_gptq_scales,
    half* __restrict__ c,
    const int size_m,
    const int size_n,
    const int size_k,
    const int groups,
    const int* __restrict__ b_q_perm*/
template <bool first_block, int m_count>
/*
DPCT1110:21: The total declared local variable size in device function
gemm_half_q_half_gptq_kernel exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
void gemm_half_q_half_gptq_kernel(
    const sycl::half *__restrict__ a, const uint32_t *__restrict__ b_q_weight,
    const uint32_t *__restrict__ b_gptq_qzeros,
    const sycl::half *__restrict__ b_gptq_scales, sycl::half *__restrict__ c,
    const int size_m, const int size_n, const int size_k, const int groups,
    const int *__restrict__ b_q_perm, const sycl::nd_item<3> &item_ct1,
    sycl::local_accessor<sycl::half, 2> block_a)
{
    MatrixView_half a_(a, size_m, size_k);
    MatrixView_half_rw c_(c, size_m, size_n);
    MatrixView_q4_row b_gptq_qzeros_(b_gptq_qzeros, groups, size_n);
    MatrixView_half b_gptq_scales_(b_gptq_scales, groups, size_n);

/* DPCT_ORIG     int t = threadIdx.x;*/
    int t = item_ct1.get_local_id(2);

    // Block
/* DPCT_ORIG     int offset_n = blockIdx.x * BLOCK_KN_SIZE * 4;*/
    int offset_n = item_ct1.get_group(2) * BLOCK_KN_SIZE * 4;
/* DPCT_ORIG     int offset_m = blockIdx.y * m_count;*/
    int offset_m = item_ct1.get_group(1) * m_count;
/* DPCT_ORIG     int offset_k = blockIdx.z * BLOCK_KN_SIZE;*/
    int offset_k = item_ct1.get_group(0) * BLOCK_KN_SIZE;

/* DPCT_ORIG     int end_n = min(offset_n + BLOCK_KN_SIZE * 4, size_n);*/
    int end_n = sycl::min(offset_n + BLOCK_KN_SIZE * 4, size_n);
/* DPCT_ORIG     int end_m = min(offset_m + m_count, size_m);*/
    int end_m = sycl::min(offset_m + m_count, size_m);
/* DPCT_ORIG     int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);*/
    int end_k = sycl::min(offset_k + BLOCK_KN_SIZE, size_k);

    int n = offset_n + t * 4;

    // Preload block_a
/* DPCT_ORIG     __shared__ half block_a[m_count][BLOCK_KN_SIZE];*/

    if (offset_k + t < end_k)
    {
        for (int m = 0; m < m_count; ++m)
        {
/* DPCT_ORIG             const half* a_ptr = a_.item_ptr(offset_m + m, 0);*/
            const sycl::half *a_ptr = a_.item_ptr(offset_m + m, 0);
/* DPCT_ORIG             half* block_a_ptr = block_a[m];*/
            sycl::half *block_a_ptr = block_a[m];

/* DPCT_ORIG             half a0;*/
            sycl::half a0;
            if (b_q_perm) a0 = a_ptr[b_q_perm[offset_k + t]];
            else a0 = a_ptr[offset_k + t];
            block_a_ptr[t] = a0;
        }
    }

    // Zero output
    if (n >= size_n) return;

/* DPCT_ORIG     if (blockIdx.z == 0)*/
    if (item_ct1.get_group(0) == 0)
    {
        for (int m = 0; m < m_count; m++)
            *((uint64_t*)c_.item_ptr(offset_m + m, n)) = 0;
    }

/* DPCT_ORIG     __syncthreads();*/
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Find initial group
    int groupsize = size_k / groups;
    int group = offset_k / groupsize;
    int nextgroup = offset_k + groupsize;

    // a, b offset
    int qk = offset_k / (32 / 4);

    const uint32_t* b_ptr = b_q_weight + qk * size_n + n;
/* DPCT_ORIG     const half* a_ptr = &block_a[0][0];*/
    const sycl::half *a_ptr = &block_a[0][0];
    int a_stride = BLOCK_KN_SIZE;

    // Initial group
    int zeros[4];
    float scales[4];
/* DPCT_ORIG     half2 z1z16[4][2];*/
    sycl::half2 z1z16[4][2];
/* DPCT_ORIG     half2 y1y16[4][2];*/
    sycl::half2 y1y16[4][2];
    b_gptq_qzeros_.item4(zeros, group, n);
    b_gptq_scales_.item4_f(scales, group, n);
    dequant_4bit_8_prep_zero(zeros[0] + 1, z1z16[0], y1y16[0]);
    dequant_4bit_8_prep_zero(zeros[1] + 1, z1z16[1], y1y16[1]);
    dequant_4bit_8_prep_zero(zeros[2] + 1, z1z16[2], y1y16[2]);
    dequant_4bit_8_prep_zero(zeros[3] + 1, z1z16[3], y1y16[3]);

    // Column result
    float block_c[m_count][4] = {};

    // Dequantize and multiply
    int k = offset_k;
    while (k < end_k)
    {
        if (k == nextgroup)
        {
            group++;
            nextgroup += groupsize;
            b_gptq_qzeros_.item4(zeros, group, n);
            b_gptq_scales_.item4_f(scales, group, n);
            dequant_4bit_8_prep_zero(zeros[0] + 1, z1z16[0], y1y16[0]);
            dequant_4bit_8_prep_zero(zeros[1] + 1, z1z16[1], y1y16[1]);
            dequant_4bit_8_prep_zero(zeros[2] + 1, z1z16[2], y1y16[2]);
            dequant_4bit_8_prep_zero(zeros[3] + 1, z1z16[3], y1y16[3]);
        }

        #pragma unroll
        for (int j = 0; j < 4; j++)
        {
/* DPCT_ORIG             const int4* b_ptr4 = (int4*) b_ptr;*/
            const sycl::int4 *b_ptr4 = (sycl::int4 *)b_ptr;
/* DPCT_ORIG             int4 load_int4 = *b_ptr4;*/
            sycl::int4 load_int4 = *b_ptr4;

/* DPCT_ORIG             half2 dq[4][4];*/
            sycl::half2 dq[4][4];
/* DPCT_ORIG             dequant_4bit_8_gptq(load_int4.x, dq[0], z1z16[0],
 * y1y16[0], size_n, false);*/
            dequant_4bit_8_gptq(load_int4.x(), dq[0], z1z16[0], y1y16[0],
                                size_n, false);
/* DPCT_ORIG             dequant_4bit_8_gptq(load_int4.y, dq[1], z1z16[1],
 * y1y16[1], size_n, false);*/
            dequant_4bit_8_gptq(load_int4.y(), dq[1], z1z16[1], y1y16[1],
                                size_n, false);
/* DPCT_ORIG             dequant_4bit_8_gptq(load_int4.z, dq[2], z1z16[2],
 * y1y16[2], size_n, false);*/
            dequant_4bit_8_gptq(load_int4.z(), dq[2], z1z16[2], y1y16[2],
                                size_n, false);
/* DPCT_ORIG             dequant_4bit_8_gptq(load_int4.w, dq[3], z1z16[3],
 * y1y16[3], size_n, false);*/
            dequant_4bit_8_gptq(load_int4.w(), dq[3], z1z16[3], y1y16[3],
                                size_n, false);

#pragma unroll
            for (int m = 0; m < m_count; m++)
            {
                block_c[m][0] = fma(dot22_8_f(dq[0], a_ptr + m * a_stride), scales[0], block_c[m][0]);
                block_c[m][1] = fma(dot22_8_f(dq[1], a_ptr + m * a_stride), scales[1], block_c[m][1]);
                block_c[m][2] = fma(dot22_8_f(dq[2], a_ptr + m * a_stride), scales[2], block_c[m][2]);
                block_c[m][3] = fma(dot22_8_f(dq[3], a_ptr + m * a_stride), scales[3], block_c[m][3]);
            }

            b_ptr += size_n;
            a_ptr += 8;
        }

        k += 32;
    }

    for (int m = 0; m < m_count; m++)
    {
/* DPCT_ORIG         half2 *out = (half2*) c_.item_ptr(offset_m + m, n);*/
        sycl::half2 *out = (sycl::half2 *)c_.item_ptr(offset_m + m, n);
/* DPCT_ORIG         half2 result01 =
 * __halves2half2(__float2half_rn(block_c[m][0]),
 * __float2half_rn(block_c[m][1]));*/
        sycl::half2 result01 = __halves2half2(__float2half_rn(block_c[m][0]),
                                              __float2half_rn(block_c[m][1]));
/* DPCT_ORIG         half2 result23 =
 * __halves2half2(__float2half_rn(block_c[m][2]),
 * __float2half_rn(block_c[m][3]));*/
        sycl::half2 result23 = __halves2half2(__float2half_rn(block_c[m][2]),
                                              __float2half_rn(block_c[m][3]));
        /*
        DPCT1007:62: Migration of half version of atomicAdd is not supported.
        */
        atomicAdd(out, result01);
        /*
        DPCT1007:63: Migration of half version of atomicAdd is not supported.
        */
        atomicAdd(out + 1, result23);
    }
}


fp_gemm_half_q_half_gptq_kernel pick_gemm_half_q_half_gptq_kernel(bool first_block, const int m_count)
{
    #if BLOCK_M_SIZE_MAX >= 1
    if (m_count == 1) return gemm_half_q_half_gptq_kernel<true, 1>;
    #endif
    #if BLOCK_M_SIZE_MAX >= 2
    if (m_count == 2) return gemm_half_q_half_gptq_kernel<true, 2>;
    #endif
    #if BLOCK_M_SIZE_MAX >= 3
    if (m_count == 3) return gemm_half_q_half_gptq_kernel<true, 3>;
    #endif
    #if BLOCK_M_SIZE_MAX >= 4
    if (m_count == 4) return gemm_half_q_half_gptq_kernel<true, 4>;
    #endif
    #if BLOCK_M_SIZE_MAX >= 5
    if (m_count == 5) return gemm_half_q_half_gptq_kernel<true, 5>;
    #endif
    #if BLOCK_M_SIZE_MAX >= 6
    if (m_count == 6) return gemm_half_q_half_gptq_kernel<true, 6>;
    #endif
    #if BLOCK_M_SIZE_MAX >= 7
    if (m_count == 7) return gemm_half_q_half_gptq_kernel<true, 7>;
    #endif
    #if BLOCK_M_SIZE_MAX >= 8
    if (m_count == 8) return gemm_half_q_half_gptq_kernel<true, 8>;
    #endif
    return NULL;
}

void gemm_half_q_half_cuda_part(
    /* DPCT_ORIG     const half* a,*/
    const sycl::half *a, const uint32_t *b_q_weight,
    const uint32_t *b_gptq_qzeros,
    /* DPCT_ORIG     const half* b_gptq_scales,*/
    const sycl::half *b_gptq_scales, const int *b_q_perm,
    /* DPCT_ORIG     half* c,*/
    sycl::half *c, int size_m, int size_n, int size_k, int m_count, int groups)
{
/* DPCT_ORIG     dim3 blockDim, gridDim;*/
    sycl::range<3> blockDim(1, 1, 1), gridDim(1, 1, 1);
/* DPCT_ORIG     blockDim.x = BLOCK_KN_SIZE;*/
    blockDim[2] = BLOCK_KN_SIZE;
/* DPCT_ORIG     blockDim.y = 1;*/
    blockDim[1] = 1;
/* DPCT_ORIG     blockDim.z = 1;*/
    blockDim[0] = 1;
/* DPCT_ORIG     gridDim.x = DIVIDE(size_n, BLOCK_KN_SIZE * 4);*/
    gridDim[2] = DIVIDE(size_n, BLOCK_KN_SIZE * 4);
/* DPCT_ORIG     gridDim.y = DIVIDE(size_m, m_count);*/
    gridDim[1] = DIVIDE(size_m, m_count);
/* DPCT_ORIG     gridDim.z = DIVIDE(size_k, BLOCK_KN_SIZE);*/
    gridDim[0] = DIVIDE(size_k, BLOCK_KN_SIZE);

    fp_gemm_half_q_half_gptq_kernel kernel = pick_gemm_half_q_half_gptq_kernel(true, m_count);

/* DPCT_ORIG     const cudaStream_t stream = at::cuda::getCurrentCUDAStream();*/
    const dpct::queue_ptr stream = at::cuda::getCurrentCUDAStream();
/* DPCT_ORIG     kernel<<<gridDim, blockDim, 0, stream>>>
    (
        a,
        b_q_weight,
        b_gptq_qzeros,
        b_gptq_scales,
        c,
        size_m,
        size_n,
        size_k,
        groups,
        b_q_perm
    );*/
    /*
    DPCT1049:22: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
   stream->parallel_for(sycl::nd_range<3>(gridDim * blockDim, blockDim),
                        [=](sycl::nd_item<3> item_ct1) {
                           (a, b_q_weight, b_gptq_qzeros, b_gptq_scales, c,
                            size_m, size_n, size_k, groups, b_q_perm);
                        });
}

/* DPCT_ORIG __global__ void reconstruct_exllama_kernel
(
    const uint32_t* __restrict__ b_q_weight,
    const int* __restrict__ b_q_perm,
    const uint32_t* __restrict__ b_gptq_qzeros,
    const half* __restrict__ b_gptq_scales,
    const int size_k,
    const int size_n,
    const int groups,
    half* __restrict__ b*/
/*
DPCT1110:23: The total declared local variable size in device function
reconstruct_exllama_kernel exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
void reconstruct_exllama_kernel(const uint32_t *__restrict__ b_q_weight,
                                const int *__restrict__ b_q_perm,
                                const uint32_t *__restrict__ b_gptq_qzeros,
                                const sycl::half *__restrict__ b_gptq_scales,
                                const int size_k, const int size_n,
                                const int groups, sycl::half *__restrict__ b,
                                const sycl::nd_item<3> &item_ct1, int *perm)
{
    MatrixView_half_rw b_(b, size_k, size_n);
    MatrixView_q4_row b_gptq_qzeros_(b_gptq_qzeros, groups, size_n);
    MatrixView_half b_gptq_scales_(b_gptq_scales, groups, size_n);

/* DPCT_ORIG     int offset_k = BLOCK_KN_SIZE * blockIdx.y;*/
    int offset_k = BLOCK_KN_SIZE * item_ct1.get_group(1);
/* DPCT_ORIG     int offset_n = BLOCK_KN_SIZE * blockIdx.x * 4;*/
    int offset_n = BLOCK_KN_SIZE * item_ct1.get_group(2) * 4;

/* DPCT_ORIG     int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);*/
    int end_k = sycl::min(offset_k + BLOCK_KN_SIZE, size_k);

    // Preload remapping table
/* DPCT_ORIG     __shared__ int perm[BLOCK_KN_SIZE];*/

/* DPCT_ORIG     int t = threadIdx.x;*/
    int t = item_ct1.get_local_id(2);

    if (b_q_perm)
    {
        if (offset_k + t < size_k)
            perm[t] = b_q_perm[offset_k + t];
    }

    // Column
    int n = offset_n + t * 4;
    if (n >= size_n) return;

    // Find initial group
    int groupsize = size_k / groups;
    int group = offset_k / groupsize;
    int nextgroup = offset_k + groupsize;

    // b offset
    int qk = offset_k / (32 / 4);

    const uint32_t* b_ptr = b_q_weight + qk * size_n + n;

    // Initial zeros/scale
    int zeros[4];
/* DPCT_ORIG     half2 scales[4];*/
    sycl::half2 scales[4];
/* DPCT_ORIG     half2 z1z16[4][2];*/
    sycl::half2 z1z16[4][2];
/* DPCT_ORIG     half2 y1y16[4][2];*/
    sycl::half2 y1y16[4][2];
    b_gptq_qzeros_.item4(zeros, group, n);
    b_gptq_scales_.item4_h2(scales, group, n);
    dequant_4bit_8_prep_zero(zeros[0] + 1, z1z16[0], y1y16[0]);
    dequant_4bit_8_prep_zero(zeros[1] + 1, z1z16[1], y1y16[1]);
    dequant_4bit_8_prep_zero(zeros[2] + 1, z1z16[2], y1y16[2]);
    dequant_4bit_8_prep_zero(zeros[3] + 1, z1z16[3], y1y16[3]);

/* DPCT_ORIG     __syncthreads();*/
    item_ct1.barrier(sycl::access::fence_space::local_space);

    int k = offset_k;
    int lk = 0;

    while (k < end_k)
    {
        if (k == nextgroup)
        {
            group++;
            nextgroup += groupsize;
            b_gptq_qzeros_.item4(zeros, group, n);
            b_gptq_scales_.item4_h2(scales, group, n);
            dequant_4bit_8_prep_zero(zeros[0] + 1, z1z16[0], y1y16[0]);
            dequant_4bit_8_prep_zero(zeros[1] + 1, z1z16[1], y1y16[1]);
            dequant_4bit_8_prep_zero(zeros[2] + 1, z1z16[2], y1y16[2]);
            dequant_4bit_8_prep_zero(zeros[3] + 1, z1z16[3], y1y16[3]);
        }

        for (int p = 0; p < 4; p++)
        {
/* DPCT_ORIG             half2 dq[4][4];*/
            sycl::half2 dq[4][4];
/* DPCT_ORIG             const int4* b_ptr4 = (int4*) b_ptr;*/
            const sycl::int4 *b_ptr4 = (sycl::int4 *)b_ptr;
/* DPCT_ORIG             int4 load_int4 = *b_ptr4;*/
            sycl::int4 load_int4 = *b_ptr4;

/* DPCT_ORIG             dequant_4bit_8_gptq(load_int4.x, dq[0], z1z16[0],
 * y1y16[0], size_n, false);*/
            dequant_4bit_8_gptq(load_int4.x(), dq[0], z1z16[0], y1y16[0],
                                size_n, false);
/* DPCT_ORIG             dequant_4bit_8_gptq(load_int4.y, dq[1], z1z16[1],
 * y1y16[1], size_n, false);*/
            dequant_4bit_8_gptq(load_int4.y(), dq[1], z1z16[1], y1y16[1],
                                size_n, false);
/* DPCT_ORIG             dequant_4bit_8_gptq(load_int4.z, dq[2], z1z16[2],
 * y1y16[2], size_n, false);*/
            dequant_4bit_8_gptq(load_int4.z(), dq[2], z1z16[2], y1y16[2],
                                size_n, false);
/* DPCT_ORIG             dequant_4bit_8_gptq(load_int4.w, dq[3], z1z16[3],
 * y1y16[3], size_n, false);*/
            dequant_4bit_8_gptq(load_int4.w(), dq[3], z1z16[3], y1y16[3],
                                size_n, false);

            b_ptr += size_n;
            //half* dqh = (half*)dq;
            if (b_q_perm)
            {
                for (int j = 0; j < 4; j++)
                {
/* DPCT_ORIG                     for (int v = 0; v < 4; v++) dq[v][j] =
 * __hmul2(scales[v], dq[v][j]);*/
                    for (int v = 0; v < 4; v++) dq[v][j] = scales[v] * dq[v][j];
/* DPCT_ORIG                     b_.set4(perm[lk++], n, __low2half(dq[0][j]),
 * __low2half(dq[1][j]), __low2half(dq[2][j]), __low2half(dq[3][j]));*/
                    b_.set4(perm[lk++], n, dq[0][j][1], dq[1][j][1],
                            dq[2][j][1], dq[3][j][1]);
/* DPCT_ORIG                     b_.set4(perm[lk++], n, __high2half(dq[0][j]),
 * __high2half(dq[1][j]), __high2half(dq[2][j]), __high2half(dq[3][j]));*/
                    b_.set4(perm[lk++], n, dq[0][j][0], dq[1][j][0],
                            dq[2][j][0], dq[3][j][0]);
                }
            }
            else
            {
                for (int j = 0; j < 4; j++)
                {
/* DPCT_ORIG                     for (int v = 0; v < 4; v++) dq[v][j] =
 * __hmul2(scales[v], dq[v][j]);*/
                    for (int v = 0; v < 4; v++) dq[v][j] = scales[v] * dq[v][j];
/* DPCT_ORIG                     b_.set4(offset_k + lk++, n,
 * __low2half(dq[0][j]), __low2half(dq[1][j]), __low2half(dq[2][j]),
 * __low2half(dq[3][j]));*/
                    b_.set4(offset_k + lk++, n, dq[0][j][1], dq[1][j][1],
                            dq[2][j][1], dq[3][j][1]);
/* DPCT_ORIG                     b_.set4(offset_k + lk++, n,
 * __high2half(dq[0][j]), __high2half(dq[1][j]), __high2half(dq[2][j]),
 * __high2half(dq[3][j]));*/
                    b_.set4(offset_k + lk++, n, dq[0][j][0], dq[1][j][0],
                            dq[2][j][0], dq[3][j][0]);
                }
            }
        }
        k += 32;
    }
}

void reconstruct_exllama(const uint32_t *b_q_weight,
                         const uint32_t *b_gptq_qzeros,
                         /* DPCT_ORIG     const half* b_gptq_scales,*/
                         const sycl::half *b_gptq_scales, const int *b_q_perm,
                         /* DPCT_ORIG     half* out,*/
                         sycl::half *out, int height, int width, int groups)
{
/* DPCT_ORIG     dim3 blockDim, gridDim;*/
    sycl::range<3> blockDim(1, 1, 1), gridDim(1, 1, 1);
/* DPCT_ORIG     blockDim.x = BLOCK_KN_SIZE;*/
    blockDim[2] = BLOCK_KN_SIZE;
/* DPCT_ORIG     blockDim.y = 1;*/
    blockDim[1] = 1;
/* DPCT_ORIG     gridDim.y = DIVIDE(height, BLOCK_KN_SIZE);*/
    gridDim[1] = DIVIDE(height, BLOCK_KN_SIZE);
/* DPCT_ORIG     gridDim.x = DIVIDE(width, BLOCK_KN_SIZE);*/
    gridDim[2] = DIVIDE(width, BLOCK_KN_SIZE);

/* DPCT_ORIG     const cudaStream_t stream = at::cuda::getCurrentCUDAStream();*/
    const dpct::queue_ptr stream = at::cuda::getCurrentCUDAStream();
/* DPCT_ORIG     reconstruct_exllama_kernel<<<gridDim, blockDim, 0, stream>>>
    (
        b_q_weight,
        b_q_perm,
        b_gptq_qzeros,
        b_gptq_scales,
        height,
        width,
        groups,
        out
    );*/
    /*
    DPCT1049:24: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
   {
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});
      stream->submit([&](sycl::handler &cgh) {
         // accessors to device memory
         /*
         DPCT1101:96: 'BLOCK_KN_SIZE' expression was replaced with a value.
         Modify the code to use the original expression, provided in comments,
         if it is correct.
         */
         sycl::local_accessor<int, 1> perm_acc_ct1(
             sycl::range<1>(128 /*BLOCK_KN_SIZE*/), cgh);

         cgh.parallel_for(sycl::nd_range<3>(gridDim * blockDim, blockDim),
                          [=](sycl::nd_item<3> item_ct1) {
                             reconstruct_exllama_kernel(
                                 b_q_weight, b_q_perm, b_gptq_qzeros,
                                 b_gptq_scales, height, width, groups, out,
                                 item_ct1, perm_acc_ct1.get_pointer());
                          });
      });
   }
}

/* DPCT_ORIG __global__ void gemm_half_q_half_alt_kernel(
    const half2* __restrict__ vec,
    const uint32_t* __restrict__ mat,
    half* __restrict__ mul,
    const half* __restrict__ scales,
    const uint32_t* __restrict__ zeros,
    const int* __restrict__ g_idx,
    int batch,
    int height,
    int width*/
/*
DPCT1110:25: The total declared local variable size in device function
gemm_half_q_half_alt_kernel exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
void gemm_half_q_half_alt_kernel(
    const sycl::half2 *__restrict__ vec, const uint32_t *__restrict__ mat,
    sycl::half *__restrict__ mul, const sycl::half *__restrict__ scales,
    const uint32_t *__restrict__ zeros, const int *__restrict__ g_idx,
    int batch, int height, int width, const sycl::nd_item<3> &item_ct1,
    sycl::local_accessor<sycl::half2, 2> blockvec,
    sycl::local_accessor<sycl::half2, 2> deq2)
{
    int zero_width = width / 8;
    int vec_height = height * 4;
    const int blockwidth2 = BLOCK_KN_SIZE / 2;
/* DPCT_ORIG     int b = blockIdx.y * BLOCK_M_SIZE_MAX;*/
    int b = item_ct1.get_group(1) * BLOCK_M_SIZE_MAX;
/* DPCT_ORIG     int b_end = min(BLOCK_M_SIZE_MAX, batch - b);*/
    int b_end = sycl::min(BLOCK_M_SIZE_MAX, batch - b);
/* DPCT_ORIG     int h = BLOCK_KN_SIZE * blockIdx.z / 8;*/
    int h = BLOCK_KN_SIZE * item_ct1.get_group(0) / 8;
/* DPCT_ORIG     int h_end = min(BLOCK_KN_SIZE / 8, height - h) * 4;*/
    int h_end = sycl::min(BLOCK_KN_SIZE / 8, height - h) * 4;
/* DPCT_ORIG     int w = BLOCK_KN_SIZE * blockIdx.x + threadIdx.x;*/
    int w = BLOCK_KN_SIZE * item_ct1.get_group(2) + item_ct1.get_local_id(2);

/* DPCT_ORIG     __shared__ half2 blockvec[BLOCK_M_SIZE_MAX][blockwidth2];*/

/* DPCT_ORIG     if (threadIdx.x < h_end) {*/
    if (item_ct1.get_local_id(2) < h_end) {
        for (int m = 0; m < b_end; ++m) {
/* DPCT_ORIG           blockvec[m][threadIdx.x] =*/
          blockvec[m][item_ct1.get_local_id(2)] =
              /* DPCT_ORIG               vec[(m + b) * vec_height + blockIdx.z *
                 BLOCK_KN_SIZE / 2 +*/
              vec[(m + b) * vec_height +
                  item_ct1.get_group(0) * BLOCK_KN_SIZE / 2 +
                  /* DPCT_ORIG                   threadIdx.x];*/
                  item_ct1.get_local_id(2)];
        }
    }

/* DPCT_ORIG     __shared__ half2 deq2[256][8];*/

/* DPCT_ORIG     int val = threadIdx.x / 8;*/
    int val = item_ct1.get_local_id(2) / 8;
/* DPCT_ORIG     int off = threadIdx.x % 8;*/
    int off = item_ct1.get_local_id(2) % 8;
    for (; val < 256; val += BLOCK_KN_SIZE / 8) {
/* DPCT_ORIG         deq2[val][off] = __halves2half2(
            __int2half_rn(val & 0xF), __int2half_rn(val >> 4)
        );*/
        deq2[val][off] = sycl::half2{
            sycl::vec<int, 1>{(val & 0xF)}
                .convert<sycl::half, sycl::rounding_mode::rte>()[0],
            sycl::vec<int, 1>{(val >> 4)}
                .convert<sycl::half, sycl::rounding_mode::rte>()[0]};
    }

/* DPCT_ORIG     if (blockIdx.z == 0)*/
    if (item_ct1.get_group(0) == 0)
    {
        for (int m = 0; m < b_end; m++)
/* DPCT_ORIG             mul[(b + m) * width + w] = __int2half_rn(0);*/
            mul[(b + m) * width + w] =
                sycl::vec<int, 1>{0}
                    .convert<sycl::half, sycl::rounding_mode::rte>()[0];
    }
/* DPCT_ORIG     __syncthreads();*/
    /*
    DPCT1065:64: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    int i = width * h + w;
    int g_h = h * 8;
    int k = 0;
    int z_w = w / 8;
    int z_mod = (w % 8) * 4;
/* DPCT_ORIG     half2 res2;*/
    sycl::half2 res2;
/* DPCT_ORIG     half res[BLOCK_M_SIZE_MAX] = {};*/
    sycl::half res[BLOCK_M_SIZE_MAX] = {};

    unsigned int tmp;
    while (k < h_end) {
        tmp = mat[i];
/* DPCT_ORIG         half2 scales_tmp[4];*/
        sycl::half2 scales_tmp[4];
/* DPCT_ORIG         half2 zeros_tmp[4];*/
        sycl::half2 zeros_tmp[4];
        for (int tmp_k = 0; tmp_k < 4; tmp_k++) {
            int g = g_idx[g_h + (k + tmp_k) * 2];
            int g2 = g_idx[g_h + (k + tmp_k) * 2 + 1];
/* DPCT_ORIG             half scale_f = scales[g * width + w];*/
            sycl::half scale_f = scales[g * width + w];
/* DPCT_ORIG             half scale_f2 = scales[g2 * width + w];*/
            sycl::half scale_f2 = scales[g2 * width + w];
/* DPCT_ORIG             half2 scale = __halves2half2(scale_f, scale_f2);*/
            sycl::half2 scale = sycl::half2{scale_f, scale_f2};
/* DPCT_ORIG             half2 zero = __halves2half2(
                __hmul(scale_f, __int2half_rn(-((zeros[g * zero_width + z_w] >>
   z_mod) & 0xF) - 1)),
                __hmul(scale_f2, __int2half_rn(-((zeros[g2 * zero_width + z_w]
   >> z_mod) & 0xF) - 1))
            );*/
            sycl::half2 zero = sycl::half2{
                scale_f *
                    sycl::vec<int, 1>{
                        (-((zeros[g * zero_width + z_w] >> z_mod) & 0xF) - 1)}
                        .convert<sycl::half, sycl::rounding_mode::rte>()[0],
                scale_f2 *
                    sycl::vec<int, 1>{
                        (-((zeros[g2 * zero_width + z_w] >> z_mod) & 0xF) - 1)}
                        .convert<sycl::half, sycl::rounding_mode::rte>()[0]};
            scales_tmp[tmp_k] = scale;
            zeros_tmp[tmp_k] = zero;
        }
        for (int m = 0; m < b_end; m++) {
#ifndef USE_ROCM
            res2 = {};
#else
            res2.x = __half_as_ushort(__float2half(0));
            res2.y = __half_as_ushort(__float2half(0));
#endif
/* DPCT_ORIG             res2 = __hfma2(__hfma2(deq2[(tmp >>  0) & 0xff][off],
 * scales_tmp[0], zeros_tmp[0]), blockvec[m][k + 0], res2);*/
            res2 = sycl::fma(sycl::fma(deq2[(tmp >> 0) & 0xff][off],
                                       scales_tmp[0], zeros_tmp[0]),
                             blockvec[m][k + 0], res2);
/* DPCT_ORIG             res2 = __hfma2(__hfma2(deq2[(tmp >>  8) & 0xff][off],
 * scales_tmp[1], zeros_tmp[1]), blockvec[m][k + 1], res2);*/
            res2 = sycl::fma(sycl::fma(deq2[(tmp >> 8) & 0xff][off],
                                       scales_tmp[1], zeros_tmp[1]),
                             blockvec[m][k + 1], res2);
/* DPCT_ORIG             res2 = __hfma2(__hfma2(deq2[(tmp >> 16) & 0xff][off],
 * scales_tmp[2], zeros_tmp[2]), blockvec[m][k + 2], res2);*/
            res2 = sycl::fma(sycl::fma(deq2[(tmp >> 16) & 0xff][off],
                                       scales_tmp[2], zeros_tmp[2]),
                             blockvec[m][k + 2], res2);
/* DPCT_ORIG             res2 = __hfma2(__hfma2(deq2[(tmp >> 24) & 0xff][off],
 * scales_tmp[3], zeros_tmp[3]), blockvec[m][k + 3], res2);*/
            res2 = sycl::fma(sycl::fma(deq2[(tmp >> 24) & 0xff][off],
                                       scales_tmp[3], zeros_tmp[3]),
                             blockvec[m][k + 3], res2);
#ifndef USE_ROCM
/* DPCT_ORIG             res[m] = __hadd(res[m], __hadd(res2.x, res2.y));*/
            res[m] = res[m] + res2.x() + res2.y();
#else
            res[m] = __hadd(res[m], __hadd(__ushort_as_half(res2.x), __ushort_as_half(res2.y)));
#endif
        }
        i += width;
        k += 4;
    }
    for (int m = 0; m < b_end; m++) {
        atomicAdd(&mul[(b + m) * width + w], res[m]);
    }
}

void gemm_half_q_half_alt(
    /* DPCT_ORIG     const half* a,*/
    const sycl::half *a, const uint32_t *b_q_weight,
    const uint32_t *b_gptq_qzeros,
    /* DPCT_ORIG     const half* b_gptq_scales,*/
    const sycl::half *b_gptq_scales, const int *b_g_idx,
    /* DPCT_ORIG     half* c,*/
    sycl::half *c, int size_m, int size_n, int size_k)
{
/* DPCT_ORIG     dim3 blockDim, gridDim;*/
    sycl::range<3> blockDim(1, 1, 1), gridDim(1, 1, 1);
/* DPCT_ORIG     blockDim.x = BLOCK_KN_SIZE;*/
    blockDim[2] = BLOCK_KN_SIZE;
/* DPCT_ORIG     blockDim.y = 1;*/
    blockDim[1] = 1;
/* DPCT_ORIG     blockDim.z = 1;*/
    blockDim[0] = 1;
/* DPCT_ORIG     gridDim.x = DIVIDE(size_n, BLOCK_KN_SIZE);*/
    gridDim[2] = DIVIDE(size_n, BLOCK_KN_SIZE);
/* DPCT_ORIG     gridDim.y = DIVIDE(size_m, BLOCK_M_SIZE_MAX);*/
    gridDim[1] = DIVIDE(size_m, BLOCK_M_SIZE_MAX);
/* DPCT_ORIG     gridDim.z = DIVIDE(size_k, BLOCK_KN_SIZE);*/
    gridDim[0] = DIVIDE(size_k, BLOCK_KN_SIZE);

/* DPCT_ORIG     const cudaStream_t stream = at::cuda::getCurrentCUDAStream();*/
    const dpct::queue_ptr stream = at::cuda::getCurrentCUDAStream();
/* DPCT_ORIG     gemm_half_q_half_alt_kernel<<<gridDim, blockDim, 0, stream>>>
    (
        (const half2*) a,
        b_q_weight,
        c,
        b_gptq_scales,
        b_gptq_qzeros,
        b_g_idx,
        size_m,
        size_k / 8,
        size_n
    );*/
    /*
    DPCT1049:26: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
   {
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});
      stream->submit([&](sycl::handler &cgh) {
         // accessors to device memory
         /*
         DPCT1101:97: 'BLOCK_M_SIZE_MAX' expression was replaced with a value.
         Modify the code to use the original expression, provided in comments,
         if it is correct.
         */
         /*
         DPCT1101:98: 'blockwidth2' expression was replaced with a value.
         Modify the code to use the original expression, provided in comments,
         if it is correct.
         */
         sycl::local_accessor<sycl::half2, 2> blockvec_acc_ct1(
             sycl::range<2>(8 /*BLOCK_M_SIZE_MAX*/, 64 /*blockwidth2*/), cgh);
         sycl::local_accessor<sycl::half2, 2> deq2_acc_ct1(
             sycl::range<2>(256, 8), cgh);

         cgh.parallel_for(sycl::nd_range<3>(gridDim * blockDim, blockDim),
                          [=](sycl::nd_item<3> item_ct1) {
                             gemm_half_q_half_alt_kernel(
                                 (const sycl::half2 *)a, b_q_weight, c,
                                 b_gptq_scales, b_gptq_qzeros, b_g_idx, size_m,
                                 size_k / 8, size_n, item_ct1, blockvec_acc_ct1,
                                 deq2_acc_ct1);
                          });
      });
   }
}

/* DPCT_ORIG __global__ void reconstruct_gptq_kernel
(
    const uint32_t* __restrict__ w,
    const half* __restrict__ w_scales,
    const uint32_t* __restrict__ w_zeros,
    const int* __restrict__ g_idx,
    const int height,
    const int width,
    const int group,
    half* __restrict__ out*/
void reconstruct_gptq_kernel(const uint32_t *__restrict__ w,
                             const sycl::half *__restrict__ w_scales,
                             const uint32_t *__restrict__ w_zeros,
                             const int *__restrict__ g_idx, const int height,
                             const int width, const int group,
                             sycl::half *__restrict__ out,
                             const sycl::nd_item<3> &item_ct1)
{
    // Start of block

/* DPCT_ORIG     int column = BLOCK_KN_SIZE * blockIdx.x + threadIdx.x;*/
    int column =
        BLOCK_KN_SIZE * item_ct1.get_group(2) + item_ct1.get_local_id(2);
/* DPCT_ORIG     int row = blockIdx.y * 8;*/
    int row = item_ct1.get_group(1) * 8;
    if (column >= width) return;

    // Views

    MatrixView_q4_column w_(w, height, width);
    MatrixView_half_rw out_(out, height, width);
    MatrixView_half w_scales_(w_scales, group, width);
    MatrixView_q4_row w_zeros_(w_zeros, group, width);

    uint32_t w_read = w_.item_uint32_t(row, column);
/* DPCT_ORIG     half* out_ptr = out_.item_ptr(row, column);*/
    sycl::half *out_ptr = out_.item_ptr(row, column);

#pragma unroll
    for (int s = 0; s < 32; s += 4)
    {
        int group = g_idx[row + s / 4];
/* DPCT_ORIG         half w_scale = w_scales_.item(group, column);*/
        sycl::half w_scale = w_scales_.item(group, column);
        uint32_t w_zero = w_zeros_.item(group, column) + 1;
/* DPCT_ORIG         half w_item = __hmul(__int2half_rn((int)((w_read >> s) &
 * 0x0f) - w_zero), w_scale);*/
        sycl::half w_item =
            sycl::vec<int, 1>{((int)((w_read >> s) & 0x0f) - w_zero)}
                .convert<sycl::half, sycl::rounding_mode::rte>()[0] *
            w_scale;
        *out_ptr = w_item; out_ptr += out_.width;
    }
}

void reconstruct_gptq(const uint32_t *b_q_weight, const uint32_t *b_gptq_qzeros,
                      /* DPCT_ORIG     const half* b_gptq_scales,*/
                      const sycl::half *b_gptq_scales, const int *b_g_idx,
                      /* DPCT_ORIG     half* out,*/
                      sycl::half *out, int height, int width, int groups)
{
/* DPCT_ORIG     dim3 blockDim, gridDim;*/
    sycl::range<3> blockDim(1, 1, 1), gridDim(1, 1, 1);
/* DPCT_ORIG     blockDim.x = BLOCK_KN_SIZE;*/
    blockDim[2] = BLOCK_KN_SIZE;
/* DPCT_ORIG     blockDim.y = 1;*/
    blockDim[1] = 1;
/* DPCT_ORIG     gridDim.y = DIVIDE(height, 8);*/
    gridDim[1] = DIVIDE(height, 8);
/* DPCT_ORIG     gridDim.x = DIVIDE(width, BLOCK_KN_SIZE);*/
    gridDim[2] = DIVIDE(width, BLOCK_KN_SIZE);
/* DPCT_ORIG     const cudaStream_t stream = at::cuda::getCurrentCUDAStream();*/
    const dpct::queue_ptr stream = at::cuda::getCurrentCUDAStream();
/* DPCT_ORIG     reconstruct_gptq_kernel<<<gridDim, blockDim, 0, stream>>>
    (
        b_q_weight,
        b_gptq_scales,
        b_gptq_qzeros,
        b_g_idx,
        height,
        width,
        groups,
        out
    );*/
    /*
    DPCT1049:27: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
   {
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});
      stream->parallel_for(sycl::nd_range<3>(gridDim * blockDim, blockDim),
                           [=](sycl::nd_item<3> item_ct1) {
                              reconstruct_gptq_kernel(b_q_weight, b_gptq_scales,
                                                      b_gptq_qzeros, b_g_idx,
                                                      height, width, groups,
                                                      out, item_ct1);
                           });
   }
}

void gemm_half_q_half_cuda(cublasHandle_t cublas_handle,
                           /* DPCT_ORIG     const half* a,*/
                           const sycl::half *a, const uint32_t *b_q_weight,
                           const uint32_t *b_gptq_qzeros,
                           /* DPCT_ORIG     const half* b_gptq_scales,*/
                           const sycl::half *b_gptq_scales, const int *b_g_idx,
                           /* DPCT_ORIG     half* c,*/
                           sycl::half *c,
                           /* DPCT_ORIG     half* temp_dq,*/
                           sycl::half *temp_dq, int size_m, int size_n,
                           int size_k, int groups, bool use_exllama)
{
    if ((use_exllama && size_m > MAX_Q_GEMM_ROWS) || (!use_exllama && size_m > MAX_ALT_GEMM_ROWS)) {
        // Reconstruct FP16 matrix, then cuBLAS
        if (use_exllama) {
            reconstruct_exllama(b_q_weight, b_gptq_qzeros, b_gptq_scales, b_g_idx, temp_dq,
                                size_k, size_n, groups);
        }
        else
        {
            reconstruct_gptq(b_q_weight, b_gptq_qzeros, b_gptq_scales, b_g_idx,
                             temp_dq, size_k, size_n, groups);
        }

/* DPCT_ORIG         const half alpha = __float2half(1.0f);*/
        const sycl::half alpha =
            sycl::vec<float, 1>{1.0f}
                .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
/* DPCT_ORIG         const half beta = __float2half(0.0f);*/
        const sycl::half beta =
            sycl::vec<float, 1>{0.0f}
                .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
        cublasHgemm(cublas_handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    size_n, size_m, size_k,
                    &alpha, temp_dq, size_n,
                            a,       size_k,
                    &beta,  c,       size_n);
    }
    else if (use_exllama)
    {
        // Quantized matmul
        int max_chunks = size_m / BLOCK_M_SIZE_MAX;
        int last_chunk = max_chunks * BLOCK_M_SIZE_MAX;
        int last_chunk_size = size_m - last_chunk;

        if (max_chunks)
        {
            gemm_half_q_half_cuda_part(a, b_q_weight, b_gptq_qzeros, b_gptq_scales, b_g_idx,
                                        c, last_chunk, size_n, size_k, BLOCK_M_SIZE_MAX,
                                        groups);
        }

        if (last_chunk_size)
        {
            gemm_half_q_half_cuda_part(a + last_chunk * size_k, b_q_weight, b_gptq_qzeros,
                                        b_gptq_scales, b_g_idx, c + last_chunk * size_n,
                                        last_chunk_size, size_n, size_k, last_chunk_size,
                                        groups);
        }
    }
    else
    {
        gemm_half_q_half_alt(a, b_q_weight, b_gptq_qzeros, b_gptq_scales, b_g_idx,
                             c, size_m, size_n, size_k);
    }
}

/* DPCT_ORIG __global__ void shuffle_kernel
(
    uint32_t* __restrict__ b_q_weight,
    const int size_k,
    const int size_n*/
void shuffle_kernel(uint32_t *__restrict__ b_q_weight, const int size_k,
                    const int size_n, const sycl::nd_item<3> &item_ct1)
{
/* DPCT_ORIG     int n = blockIdx.x * THREADS_X + threadIdx.x;*/
    int n = item_ct1.get_group(2) * THREADS_X + item_ct1.get_local_id(2);
    if (n >= size_n) return;
    int k = 0;
    uint32_t* b_ptr = b_q_weight + n;
    while (k < size_k) { shuffle_4bit_8 (b_ptr, size_n); b_ptr += 1 * size_n; k +=  8; }
}

/* DPCT_ORIG __global__ void make_sequential_kernel
(
    const uint32_t* __restrict__ w,
    uint32_t* __restrict__ w_new,
    const int* __restrict__ q_perm,
    const int w_height,
    const int w_width*/
void make_sequential_kernel(const uint32_t *__restrict__ w,
                            uint32_t *__restrict__ w_new,
                            const int *__restrict__ q_perm, const int w_height,
                            const int w_width, const sycl::nd_item<3> &item_ct1)
{
    const uint64_t* w2 = (uint64_t*) w;
    uint64_t* w_new2 = (uint64_t*) w_new;
    int w2_stride = w_width >> 1;
/* DPCT_ORIG     int w2_column = THREADS_X * blockIdx.x + threadIdx.x;*/
    int w2_column =
        THREADS_X * item_ct1.get_group(2) + item_ct1.get_local_id(2);
    if (w2_column >= w2_stride) return;
/* DPCT_ORIG     int w_new2_row = blockIdx.y;*/
    int w_new2_row = item_ct1.get_group(1);
    int q_perm_idx = w_new2_row << 3;
    uint64_t dst = 0;

    #pragma unroll
    for (int i = 0; i < 8; i++)
    {
        int source_row = q_perm[q_perm_idx++];

        int w2_row = source_row >> 3;
        int w2_subrow = source_row & 0x07;
        int w2_row_shift = w2_subrow << 2;
        int wnew2_row_shift = i << 2;

        uint64_t src = w2[w2_row * w2_stride + w2_column];
        src >>= w2_row_shift;
        src &= 0x0000000f0000000f;
        src <<= wnew2_row_shift;
        dst |= src;
    }
    w_new2[w_new2_row * w2_stride + w2_column] = dst;
}


void shuffle_exllama_weight
(
    uint32_t* q_weight,
    int* q_perm,
    int height,
    int width
)
{
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
    if (q_perm)
    {
        uint32_t* new_qweight = NULL;
        cudaMalloc(&new_qweight, height / 8 * width * sizeof(uint32_t));

/* DPCT_ORIG         dim3 blockDim, gridDim;*/
        sycl::range<3> blockDim(1, 1, 1), gridDim(1, 1, 1);
/* DPCT_ORIG         blockDim.x = THREADS_X;*/
        blockDim[2] = THREADS_X;
/* DPCT_ORIG         blockDim.y = 1;*/
        blockDim[1] = 1;
/* DPCT_ORIG         gridDim.x = DIVIDE(width, THREADS_X);*/
        gridDim[2] = DIVIDE(width, THREADS_X);
/* DPCT_ORIG         gridDim.y = height / 8;*/
        gridDim[1] = height / 8;

/* DPCT_ORIG         const cudaStream_t stream =
 * at::cuda::getCurrentCUDAStream();*/
        const dpct::queue_ptr stream = at::cuda::getCurrentCUDAStream();
/* DPCT_ORIG         make_sequential_kernel<<<gridDim, blockDim, 0, stream>>>
        (
            q_weight,
            new_qweight,
            q_perm,
            height / 8,
            width
        );*/
        /*
        DPCT1049:29: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
      stream->parallel_for(sycl::nd_range<3>(gridDim * blockDim, blockDim),
                           [=](sycl::nd_item<3> item_ct1) {
                              make_sequential_kernel(q_weight, new_qweight,
                                                     q_perm, height / 8, width,
                                                     item_ct1);
                           });
        // Replace qweights
/* DPCT_ORIG         cudaMemcpyAsync(q_weight, new_qweight, height / 8 * width *
 * sizeof(uint32_t), cudaMemcpyDeviceToDevice);*/
        q_ct1.memcpy(q_weight, new_qweight,
                     height / 8 * width * sizeof(uint32_t));
        // Cleanup
/* DPCT_ORIG         cudaDeviceSynchronize();*/
        dev_ct1.queues_wait_and_throw();
/* DPCT_ORIG         cudaFree(new_qweight);*/
        sycl::free(new_qweight, q_ct1);
    }
/* DPCT_ORIG     dim3 blockDim, gridDim;*/
    sycl::range<3> blockDim(1, 1, 1), gridDim(1, 1, 1);
/* DPCT_ORIG     blockDim.x = THREADS_X;*/
    blockDim[2] = THREADS_X;
/* DPCT_ORIG     blockDim.y = 1;*/
    blockDim[1] = 1;
/* DPCT_ORIG     gridDim.x = DIVIDE(width, THREADS_X);*/
    gridDim[2] = DIVIDE(width, THREADS_X);
/* DPCT_ORIG     gridDim.y = 1;*/
    gridDim[1] = 1;
/* DPCT_ORIG     const cudaStream_t stream = at::cuda::getCurrentCUDAStream();*/
    const dpct::queue_ptr stream = at::cuda::getCurrentCUDAStream();
/* DPCT_ORIG     shuffle_kernel<<<gridDim, blockDim, 0, stream>>>(q_weight,
 * height, width);*/
    /*
    DPCT1049:28: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
   stream->parallel_for(sycl::nd_range<3>(gridDim * blockDim, blockDim),
                        [=](sycl::nd_item<3> item_ct1) {
                           shuffle_kernel(q_weight, height, width, item_ct1);
                        });
}

}  // namespace gptq
}  // namespace vllm

torch::Tensor gptq_gemm
(
    torch::Tensor a,
    torch::Tensor b_q_weight,
    torch::Tensor b_gptq_qzeros,
    torch::Tensor b_gptq_scales,
    torch::Tensor b_g_idx,
    bool use_exllama
)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
    auto options = torch::TensorOptions().dtype(a.dtype()).device(a.device());
    at::Tensor c = torch::empty({a.size(0), b_q_weight.size(1)}, options);
    at::Tensor temp_dq = torch::empty({b_q_weight.size(0) * 8, b_q_weight.size(1)}, options);

    vllm::gptq::gemm_half_q_half_cuda
    (
        at::cuda::getCurrentCUDABlasHandle(),
        (const half*) a.data_ptr(),
        (const uint32_t*) b_q_weight.data_ptr(),
        (const uint32_t*)b_gptq_qzeros.data_ptr(),
        (const half*) b_gptq_scales.data_ptr(),
        b_g_idx.device().is_meta() ? NULL : (const int*) b_g_idx.data_ptr(),
        (half*) c.data_ptr(),
        (half*) temp_dq.data_ptr(),
        c.size(0),  // m
        c.size(1),  // n
        a.size(1),  // k
        b_gptq_qzeros.size(0),  // group number
        use_exllama
    );
    return c;
}

void gptq_shuffle
(
    torch::Tensor q_weight,
    torch::Tensor q_perm
)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(q_weight));
    vllm::gptq::shuffle_exllama_weight(
        (uint32_t*) q_weight.data_ptr(),
        q_perm.device().is_meta() ? NULL : (int*) q_perm.data_ptr(),
        q_weight.size(0) * 8,
        q_weight.size(1)
    );
}
