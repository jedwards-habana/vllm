/*
 * Adapted from https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/reduce_kernel_utils.cuh
 * Copyright (c) 2023, The vLLM team.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "cuda_compat.h"

namespace vllm {

/* DPCT_ORIG template<typename T>
__inline__ __device__ T warpReduceSum(T val) {*/
template <typename T> __inline__ T warpReduceSum(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val += VLLM_SHFL_XOR_SYNC(val, mask);
  return val;
}

/* Calculate the sum of all elements in a block */
/* DPCT_ORIG template<typename T>
__inline__ __device__ T blockReduceSum(T val) {*/
template <typename T>
__inline__ T blockReduceSum(T val, const sycl::nd_item<3> &item_ct1,
                            T *shared) {
/* DPCT_ORIG   static __shared__ T shared[32];*/

/* DPCT_ORIG   int lane = threadIdx.x & 0x1f;*/
  int lane = item_ct1.get_local_id(2) & 0x1f;
/* DPCT_ORIG   int wid = threadIdx.x >> 5;*/
  int wid = item_ct1.get_local_id(2) >> 5;

  val = warpReduceSum<T>(val);

  if (lane == 0)
    shared[wid] = val;

/* DPCT_ORIG   __syncthreads();*/
  /*
  DPCT1065:93: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
  // blockDim.x is not divided by 32
/* DPCT_ORIG   val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] :
 * (T)(0.0f);*/
  val = (item_ct1.get_local_id(2) < (item_ct1.get_local_range(2) / 32.f))
            ? shared[lane]
            : (T)(0.0f);
  val = warpReduceSum<T>(val);
  return val;
}

} // namespace vllm
