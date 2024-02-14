#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "dispatch_utils.h"
/* DPCT_ORIG #include "reduction_utils.cuh"*/
#include "reduction_utils.dp.hpp"

namespace vllm {

// TODO(woosuk): Further optimize this kernel.
/* DPCT_ORIG template<typename scalar_t>
__global__ void rms_norm_kernel(
  scalar_t* __restrict__ out,
  const scalar_t* __restrict__ input,
  const scalar_t* __restrict__ weight,
  const float epsilon,
  const int num_tokens,
  const int hidden_size) {*/
template <typename scalar_t>
void rms_norm_kernel(scalar_t *__restrict__ out,          // [..., hidden_size]
                     const scalar_t *__restrict__ input,  // [..., hidden_size]
                     const scalar_t *__restrict__ weight, // [hidden_size]
                     const float epsilon, const int num_tokens,
                     const int hidden_size, const sycl::nd_item<3> &item_ct1,
                     float &s_variance) {
/* DPCT_ORIG   __shared__ float s_variance;*/

  float variance = 0.0f;

/* DPCT_ORIG   for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x)
 * {*/
  for (int idx = item_ct1.get_local_id(2); idx < hidden_size;
       idx += item_ct1.get_local_range(2)) {
/* DPCT_ORIG     const float x = (float) input[blockIdx.x * hidden_size +
 * idx];*/
    const float x = (float)input[item_ct1.get_group(2) * hidden_size + idx];
    variance += x * x;
  }
  variance = blockReduceSum<float>(variance);
/* DPCT_ORIG   if (threadIdx.x == 0) {*/
  if (item_ct1.get_local_id(2) == 0) {
/* DPCT_ORIG     s_variance = rsqrtf(variance / hidden_size + epsilon);*/
    s_variance = sycl::rsqrt(variance / hidden_size + epsilon);
  }
/* DPCT_ORIG   __syncthreads();*/
  /*
  DPCT1065:94: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

/* DPCT_ORIG   for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x)
 * {*/
  for (int idx = item_ct1.get_local_id(2); idx < hidden_size;
       idx += item_ct1.get_local_range(2)) {
/* DPCT_ORIG     float x = (float) input[blockIdx.x * hidden_size + idx];*/
    float x = (float)input[item_ct1.get_group(2) * hidden_size + idx];
/* DPCT_ORIG     out[blockIdx.x * hidden_size + idx] = ((scalar_t) (x *
 * s_variance)) * weight[idx];*/
    out[item_ct1.get_group(2) * hidden_size + idx] =
        ((scalar_t)(x * s_variance)) * weight[idx];
  }
}

// TODO: Further optimize this kernel.
/* DPCT_ORIG template<typename scalar_t>
__global__ void fused_add_rms_norm_kernel(
  scalar_t* __restrict__ input,
  scalar_t* __restrict__ residual,
  const scalar_t* __restrict__ weight,
  const float epsilon,
  const int num_tokens,
  const int hidden_size) {*/
template <typename scalar_t>
void fused_add_rms_norm_kernel(
    scalar_t *__restrict__ input,        // [..., hidden_size]
    scalar_t *__restrict__ residual,     // [..., hidden_size]
    const scalar_t *__restrict__ weight, // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size,
    const sycl::nd_item<3> &item_ct1, float &s_variance) {
/* DPCT_ORIG   __shared__ float s_variance;*/

  float variance = 0.0f;

/* DPCT_ORIG   for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x)
 * {*/
  for (int idx = item_ct1.get_local_id(2); idx < hidden_size;
       idx += item_ct1.get_local_range(2)) {
/* DPCT_ORIG     float x = (float) input[blockIdx.x * hidden_size + idx];*/
    float x = (float)input[item_ct1.get_group(2) * hidden_size + idx];
/* DPCT_ORIG     x += (float) residual[blockIdx.x * hidden_size + idx];*/
    x += (float)residual[item_ct1.get_group(2) * hidden_size + idx];
    variance += x * x;
/* DPCT_ORIG     residual[blockIdx.x * hidden_size + idx] = (scalar_t) x;*/
    residual[item_ct1.get_group(2) * hidden_size + idx] = (scalar_t)x;
  }
  variance = blockReduceSum<float>(variance);
/* DPCT_ORIG   if (threadIdx.x == 0) {*/
  if (item_ct1.get_local_id(2) == 0) {
/* DPCT_ORIG     s_variance = rsqrtf(variance / hidden_size + epsilon);*/
    s_variance = sycl::rsqrt(variance / hidden_size + epsilon);
  }
/* DPCT_ORIG   __syncthreads();*/
  /*
  DPCT1065:95: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

/* DPCT_ORIG   for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x)
 * {*/
  for (int idx = item_ct1.get_local_id(2); idx < hidden_size;
       idx += item_ct1.get_local_range(2)) {
/* DPCT_ORIG     float x = (float) residual[blockIdx.x * hidden_size + idx];*/
    float x = (float)residual[item_ct1.get_group(2) * hidden_size + idx];
/* DPCT_ORIG     input[blockIdx.x * hidden_size + idx] = ((scalar_t) (x *
 * s_variance)) * weight[idx];*/
    input[item_ct1.get_group(2) * hidden_size + idx] =
        ((scalar_t)(x * s_variance)) * weight[idx];
  }
}

} // namespace vllm

void rms_norm(
  torch::Tensor& out,      // [..., hidden_size]
  torch::Tensor& input,    // [..., hidden_size]
  torch::Tensor& weight,   // [hidden_size]
  float epsilon) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

/* DPCT_ORIG   dim3 grid(num_tokens);*/
  sycl::range<3> grid(1, 1, num_tokens);
/* DPCT_ORIG   dim3 block(std::min(hidden_size, 1024));*/
  sycl::range<3> block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
/* DPCT_ORIG   const cudaStream_t stream = at::cuda::getCurrentCUDAStream();*/
  const dpct::queue_ptr stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
    input.scalar_type(),
    "rms_norm_kernel",
    [&] {
      vllm::rms_norm_kernel<scalar_t><<<grid, block, 0, stream>>>(
        out.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        epsilon,
        num_tokens,
        hidden_size);
    });
}

void fused_add_rms_norm(
  torch::Tensor& input,    // [..., hidden_size]
  torch::Tensor& residual, // [..., hidden_size]
  torch::Tensor& weight,   // [hidden_size]
  float epsilon) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

/* DPCT_ORIG   dim3 grid(num_tokens);*/
  sycl::range<3> grid(1, 1, num_tokens);
/* DPCT_ORIG   dim3 block(std::min(hidden_size, 1024));*/
  sycl::range<3> block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
/* DPCT_ORIG   const cudaStream_t stream = at::cuda::getCurrentCUDAStream();*/
  const dpct::queue_ptr stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
    input.scalar_type(),
    "fused_add_rms_norm_kernel",
    [&] {
      vllm::fused_add_rms_norm_kernel<scalar_t><<<grid, block, 0, stream>>>(
        input.data_ptr<scalar_t>(),
        residual.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        epsilon,
        num_tokens,
        hidden_size);
    });
}
