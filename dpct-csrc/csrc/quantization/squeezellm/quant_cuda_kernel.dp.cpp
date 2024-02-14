#include <torch/all.h>
#include <torch/python.h>
/* DPCT_ORIG #include <cuda.h>*/
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
/* DPCT_ORIG #include <cuda_runtime.h>*/
/* DPCT_ORIG #include <cuda_fp16.h>*/

// half-tensor
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDATensorMethods.cuh>
#include <c10/cuda/CUDAGuard.h>

#define BLOCKWIDTH 128
#define BLOCKHEIGHT4 16

namespace vllm {
namespace squeezellm {

/* DPCT_ORIG __device__ inline unsigned int as_unsigned(int i) {*/
inline unsigned int as_unsigned(int i) {
  return *reinterpret_cast<unsigned int*>(&i);
}

// 4-bit matvec kernel (LUT-based)
/* DPCT_ORIG __global__ void NUQ4MatMulKernel(
#ifndef USE_ROCM
    const  half2* __restrict__ vec,
#else
    const  __half2* __restrict__ vec,
#endif
    const    int* __restrict__ mat,
#ifndef USE_ROCM
           half2* __restrict__ mul,
#else
          float2* __restrict__ mul,
#endif
    const  __half* __restrict__ lookup_table,
    int height,
    int width,
    int batch,
    int vec_height*/
void NUQ4MatMulKernel(
#ifndef USE_ROCM
    const sycl::half2 *__restrict__ vec,
#else
    const __half2 *__restrict__ vec,
#endif
    const int *__restrict__ mat,
#ifndef USE_ROCM
    sycl::half2 *__restrict__ mul,
#else
    float2 *__restrict__ mul,
#endif
    const sycl::half *__restrict__ lookup_table, int height, int width,
    int batch, int vec_height, const sycl::nd_item<3> &item_ct1,
    sycl::half2 *blockvec, sycl::local_accessor<sycl::half, 2> deq2) {

  const int blockwidth2 = BLOCKWIDTH / 2;

/* DPCT_ORIG   int row = BLOCKHEIGHT4 * blockIdx.x;*/
  int row = BLOCKHEIGHT4 * item_ct1.get_group(2);
/* DPCT_ORIG   int col =  BLOCKWIDTH * blockIdx.y + threadIdx.x;*/
  int col = BLOCKWIDTH * item_ct1.get_group(1) + item_ct1.get_local_id(2);

#ifndef USE_ROCM
/* DPCT_ORIG   __shared__ half2 blockvec[blockwidth2];*/

#else
  __shared__ __half2 blockvec[blockwidth2];
#endif

/* DPCT_ORIG   __shared__ __half deq2[16][BLOCKWIDTH];*/

/* DPCT_ORIG   int off = threadIdx.x;*/
  int off = item_ct1.get_local_id(2);
  int column_offset = col * 16;
  for (int val = 0; val < 16; val += 1) {
    int lut_index = column_offset + val;
    deq2[val][off] = lookup_table[lut_index];
  }

/* DPCT_ORIG   __half res;*/
  sycl::half res;
#ifndef USE_ROCM
/* DPCT_ORIG   half2 res2;*/
  sycl::half2 res2;
/* DPCT_ORIG   half2 tmp2;*/
  sycl::half2 tmp2;
#else
  __half2 res2;
  __half2 tmp2;
#endif

  int i;
  int k;

  unsigned int tmp1;
  unsigned int lut_index1, lut_index2;

  for (int b = 0; b < batch; ++b){
    i = width * row + col;
/* DPCT_ORIG     res = __int2half_rd(0);*/
    res =
        sycl::vec<int, 1>{0}.convert<sycl::half, sycl::rounding_mode::rtn>()[0];
    k = 0;

/* DPCT_ORIG     __syncthreads();*/
    /*
    DPCT1118:30: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:68: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
/* DPCT_ORIG     if (threadIdx.x < blockwidth2)*/
    if (item_ct1.get_local_id(2) < blockwidth2)
/* DPCT_ORIG       blockvec[threadIdx.x] = vec[b * vec_height / 2 + (row /
 * BLOCKHEIGHT4) * blockwidth2 + threadIdx.x];*/
      blockvec[item_ct1.get_local_id(2)] =
          vec[b * vec_height / 2 + (row / BLOCKHEIGHT4) * blockwidth2 +
              item_ct1.get_local_id(2)];
/* DPCT_ORIG     __syncthreads();*/
    /*
    DPCT1118:31: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:69: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    while (k < blockwidth2) {
      tmp1 = as_unsigned(mat[i]);

#ifndef USE_ROCM
      res2 = {};
      tmp2 = {};
#else
      res2.x = __half_as_ushort(__float2half(0));
      res2.y = __half_as_ushort(__float2half(0));
      tmp2.x = __half_as_ushort(__float2half(0));
      tmp2.y = __half_as_ushort(__float2half(0));
#endif

      lut_index1 = tmp1 & 0xF;
      lut_index2 = (tmp1 >> 4) & 0xF;
#ifndef USE_ROCM
/* DPCT_ORIG       tmp2.x = deq2[lut_index1][off];*/
      tmp2.x() = deq2[lut_index1][off];
/* DPCT_ORIG       tmp2.y = deq2[lut_index2][off];*/
      tmp2.y() = deq2[lut_index2][off];
#else
      tmp2.x = __half_as_ushort(deq2[lut_index1][off]);
      tmp2.y = __half_as_ushort(deq2[lut_index2][off]);
#endif
/* DPCT_ORIG       res2 = __hfma2(tmp2, blockvec[k + 0], res2);*/
      res2 = sycl::fma(tmp2, blockvec[k + 0], res2);

      lut_index1 = (tmp1 >> 8) & 0xF;
      lut_index2 = (tmp1 >> 12) & 0xF;
#ifndef USE_ROCM
/* DPCT_ORIG       tmp2.x = deq2[lut_index1][off];*/
      tmp2.x() = deq2[lut_index1][off];
/* DPCT_ORIG       tmp2.y = deq2[lut_index2][off];*/
      tmp2.y() = deq2[lut_index2][off];
#else
      tmp2.x = __half_as_ushort(deq2[lut_index1][off]);
      tmp2.y = __half_as_ushort(deq2[lut_index2][off]);
#endif
/* DPCT_ORIG       res2 = __hfma2(tmp2, blockvec[k + 1], res2);*/
      res2 = sycl::fma(tmp2, blockvec[k + 1], res2);

      lut_index1 = (tmp1 >> 16) & 0xF;
      lut_index2 = (tmp1 >> 20) & 0xF;
#ifndef USE_ROCM
/* DPCT_ORIG       tmp2.x = deq2[lut_index1][off];*/
      tmp2.x() = deq2[lut_index1][off];
/* DPCT_ORIG       tmp2.y = deq2[lut_index2][off];*/
      tmp2.y() = deq2[lut_index2][off];
#else
      tmp2.x = __half_as_ushort(deq2[lut_index1][off]);
      tmp2.y = __half_as_ushort(deq2[lut_index2][off]);
#endif
/* DPCT_ORIG       res2 = __hfma2(tmp2, blockvec[k + 2], res2);*/
      res2 = sycl::fma(tmp2, blockvec[k + 2], res2);

      lut_index1 = (tmp1 >> 24) & 0xF;
      lut_index2 = (tmp1 >> 28) & 0xF;
#ifndef USE_ROCM
/* DPCT_ORIG       tmp2.x = deq2[lut_index1][off];*/
      tmp2.x() = deq2[lut_index1][off];
/* DPCT_ORIG       tmp2.y = deq2[lut_index2][off];*/
      tmp2.y() = deq2[lut_index2][off];
#else
      tmp2.x = __half_as_ushort(deq2[lut_index1][off]);
      tmp2.y = __half_as_ushort(deq2[lut_index2][off]);
#endif
/* DPCT_ORIG       res2 = __hfma2(tmp2, blockvec[k + 3], res2);*/
      res2 = sycl::fma(tmp2, blockvec[k + 3], res2);

#ifndef USE_ROCM
/* DPCT_ORIG       res = __hadd(__hadd(res2.x, res2.y), res);*/
      res = res2.x() + res2.y() + res;
#else
      res = __hadd(__hadd(__ushort_as_half(res2.x), __ushort_as_half(res2.y)), res);
#endif

      i += width;
      k += 4;
    }

    // col%2 -> only set one of the two values
#ifndef USE_ROCM
/* DPCT_ORIG     half2 res3 = {};*/
    sycl::half2 res3 = {};
    if (col % 2 == 0) {
/* DPCT_ORIG       res3.x = res;*/
      res3.x() = res;
    } else {
/* DPCT_ORIG       res3.y = res;*/
      res3.y() = res;
    }
#else
    __half2 res3;
    res3.x = __half_as_ushort(__float2half(0));
    res3.y = __half_as_ushort(__float2half(0));
    if (col % 2 == 0) {
      res3.x = __half_as_ushort(res);
    } else {
      res3.y = __half_as_ushort(res);
    }
#endif

#ifndef USE_ROCM
    /*
    DPCT1007:70: Migration of half version of atomicAdd is not supported.
    */
    atomicAdd(&mul[b * width / 2 + col / 2], res3);
#else
    int tmp_addr = b * width / 2 + col / 2;
    atomicAdd(&(mul[tmp_addr].x), __half2float(__ushort_as_half(res3.x)));
    atomicAdd(&(mul[tmp_addr].y), __half2float(__ushort_as_half(res3.y)));
#endif
  }
}

} // namespace squeezellm
} // namespace vllm

// 4-bit matvec kernel (LUT-based)
void squeezellm_gemm(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table
) {
  int height = mat.size(0);
  int width = mat.size(1);

  int batch = vec.size(0);
  int vec_height = vec.size(1);

/* DPCT_ORIG   dim3 blocks(
    (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );*/
  sycl::range<3> blocks(1, (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
                        (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4);
/* DPCT_ORIG   dim3 threads(BLOCKWIDTH);*/
  sycl::range<3> threads(1, 1, BLOCKWIDTH);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
/* DPCT_ORIG   const cudaStream_t stream = at::cuda::getCurrentCUDAStream();*/
  const dpct::queue_ptr stream = at::cuda::getCurrentCUDAStream();
  vllm::squeezellm::NUQ4MatMulKernel<<<blocks, threads, 0, stream>>>(
#ifndef USE_ROCM
    (half2*) vec.data<at::Half>(),
#else
    (__half2*) vec.data_ptr<at::Half>(),
#endif
    mat.data_ptr<int>(),
#ifndef USE_ROCM
    (half2*) mul.data<at::Half>(),
    (__half*) lookup_table.data<at::Half>(),
#else
    (float2*) mul.data_ptr<float>(),
    (__half*) lookup_table.data_ptr<at::Half>(),
#endif
    height, width, batch, vec_height
  );
}

#undef BLOCKWIDTH
#undef BLOCKHEIGHT4
