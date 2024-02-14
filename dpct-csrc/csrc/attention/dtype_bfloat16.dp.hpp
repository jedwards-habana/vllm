/*
 * Adapted from https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp
 * and https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention_utils.h
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

/* DPCT_ORIG #include "attention_generic.cuh"*/
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "attention_generic.dp.hpp"
/* DPCT_ORIG #include "dtype_float32.cuh"*/
#include "dtype_float32.dp.hpp"

#ifndef USE_ROCM
/* DPCT_ORIG   #include <cuda_bf16.h>*/
  /* DPCT_ORIG   #include <cuda_fp16.h>*/
#else
#include <hip/hip_bf16.h>
  #include <hip/hip_fp16.h>

  typedef __hip_bfloat162 __nv_bfloat162;
  typedef __hip_bfloat16 __nv_bfloat16;
#endif

#include <stdint.h>

namespace vllm {

// Define custom BF16 vector data types.
struct bf16_4_t {
/* DPCT_ORIG   __nv_bfloat162 x;*/
  sycl::marray<sycl::ext::oneapi::bfloat16, 2> x;
/* DPCT_ORIG   __nv_bfloat162 y;*/
  sycl::marray<sycl::ext::oneapi::bfloat16, 2> y;
};

struct bf16_8_t {
/* DPCT_ORIG   __nv_bfloat162 x;*/
  sycl::marray<sycl::ext::oneapi::bfloat16, 2> x;
/* DPCT_ORIG   __nv_bfloat162 y;*/
  sycl::marray<sycl::ext::oneapi::bfloat16, 2> y;
/* DPCT_ORIG   __nv_bfloat162 z;*/
  sycl::marray<sycl::ext::oneapi::bfloat16, 2> z;
/* DPCT_ORIG   __nv_bfloat162 w;*/
  sycl::marray<sycl::ext::oneapi::bfloat16, 2> w;
};

// BF16 vector types for Q, K, V.
template <>
/* DPCT_ORIG struct Vec<__nv_bfloat16, 1> {*/
struct Vec<sycl::ext::oneapi::bfloat16, 1> {
/* DPCT_ORIG   using Type = __nv_bfloat16;*/
  using Type = sycl::ext::oneapi::bfloat16;
};
template <>
/* DPCT_ORIG struct Vec<__nv_bfloat16, 2> {*/
struct Vec<sycl::ext::oneapi::bfloat16, 2> {
/* DPCT_ORIG   using Type = __nv_bfloat162;*/
  using Type = sycl::marray<sycl::ext::oneapi::bfloat16, 2>;
};
template <>
/* DPCT_ORIG struct Vec<__nv_bfloat16, 4> {*/
struct Vec<sycl::ext::oneapi::bfloat16, 4> {
  using Type = bf16_4_t;
};
template <>
/* DPCT_ORIG struct Vec<__nv_bfloat16, 8> {*/
struct Vec<sycl::ext::oneapi::bfloat16, 8> {
  using Type = bf16_8_t;
};

// FP32 accumulator vector types corresponding to Vec.
template <>
/* DPCT_ORIG struct FloatVec<__nv_bfloat16> {*/
struct FloatVec<sycl::ext::oneapi::bfloat16> {
  using Type = float;
};
template <>
/* DPCT_ORIG struct FloatVec<__nv_bfloat162> {*/
struct FloatVec<sycl::marray<sycl::ext::oneapi::bfloat16, 2>> {
/* DPCT_ORIG   using Type = float2;*/
  using Type = sycl::float2;
};
template<>
struct FloatVec<bf16_4_t> {
  using Type = Float4_;
};
template<>
struct FloatVec<bf16_8_t> {
  using Type = Float8_;
};

// Utility functions for type conversions.
/* DPCT_ORIG inline __device__ float2 bf1622float2(const __nv_bfloat162 val) {*/
inline sycl::float2
bf1622float2(const sycl::marray<sycl::ext::oneapi::bfloat16, 2> val) {
/* DPCT_ORIG #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800*/
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP < 800
  /*
  DPCT1007:43: Migration of __assert_fail is not supported.
  */
  assert(false);
#else
  return __bfloat1622float2(val);
#endif
}

/* DPCT_ORIG inline __device__ __nv_bfloat162 bf162bf162(const __nv_bfloat16
 * val) {*/
inline sycl::marray<sycl::ext::oneapi::bfloat16, 2>
bf162bf162(const sycl::ext::oneapi::bfloat16 val) {
/* DPCT_ORIG #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800*/
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP < 800
  /*
  DPCT1007:44: Migration of __assert_fail is not supported.
  */
  assert(false);
#else
  return __bfloat162bfloat162(val);
#endif
}

// Vector addition.
/* DPCT_ORIG inline __device__ __nv_bfloat16 add(__nv_bfloat16 a, __nv_bfloat16
 * b) {*/
inline sycl::ext::oneapi::bfloat16 add(sycl::ext::oneapi::bfloat16 a,
                                       sycl::ext::oneapi::bfloat16 b) {
/* DPCT_ORIG #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800*/
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP < 800
  /*
  DPCT1007:45: Migration of __assert_fail is not supported.
  */
  assert(false);
#else
  #ifndef USE_ROCM
    return a + b;
  #else
    return __hadd(a, b);
  #endif
#endif
}

/* DPCT_ORIG inline __device__ __nv_bfloat162 add(__nv_bfloat162 a,
 * __nv_bfloat162 b) {*/
inline sycl::marray<sycl::ext::oneapi::bfloat16, 2>
add(sycl::marray<sycl::ext::oneapi::bfloat16, 2> a,
    sycl::marray<sycl::ext::oneapi::bfloat16, 2> b) {
/* DPCT_ORIG #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800*/
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP < 800
  /*
  DPCT1007:46: Migration of __assert_fail is not supported.
  */
  assert(false);
#else
  return __hadd2(a, b);
#endif
}

/* DPCT_ORIG inline __device__ bf16_4_t add(bf16_4_t a, bf16_4_t b) {*/
inline bf16_4_t add(bf16_4_t a, bf16_4_t b) {
  bf16_4_t c;
  c.x = add(a.x, b.x);
  c.y = add(a.y, b.y);
  return c;
}

/* DPCT_ORIG inline __device__ bf16_8_t add(bf16_8_t a, bf16_8_t b) {*/
inline bf16_8_t add(bf16_8_t a, bf16_8_t b) {
  bf16_8_t c;
  c.x = add(a.x, b.x);
  c.y = add(a.y, b.y);
  c.z = add(a.z, b.z);
  c.w = add(a.w, b.w);
  return c;
}

/* DPCT_ORIG inline __device__ float2 add(__nv_bfloat162 a, float2 fb) {*/
inline sycl::float2 add(sycl::marray<sycl::ext::oneapi::bfloat16, 2> a,
                        sycl::float2 fb) {
/* DPCT_ORIG   float2 fa = bf1622float2(a);*/
  sycl::float2 fa = bf1622float2(a);
  return add(fa, fb);
}

/* DPCT_ORIG inline __device__ Float4_ add(bf16_4_t a, Float4_ fb) {*/
inline Float4_ add(bf16_4_t a, Float4_ fb) {
  Float4_ fc;
  fc.x = add(a.x, fb.x);
  fc.y = add(a.y, fb.y);
  return fc;
}

/* DPCT_ORIG inline __device__ Float8_ add(bf16_8_t a, Float8_ fb) {*/
inline Float8_ add(bf16_8_t a, Float8_ fb) {
  Float8_ fc;
  fc.x = add(a.x, fb.x);
  fc.y = add(a.y, fb.y);
  fc.z = add(a.z, fb.z);
  fc.w = add(a.w, fb.w);
  return fc;
}

// Vector multiplication.
/* DPCT_ORIG template<>
inline __device__ __nv_bfloat16 mul(__nv_bfloat16 a, __nv_bfloat16 b) {*/
template <>
inline sycl::ext::oneapi::bfloat16 mul(sycl::ext::oneapi::bfloat16 a,
                                       sycl::ext::oneapi::bfloat16 b) {
/* DPCT_ORIG #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800*/
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP < 800
  /*
  DPCT1007:47: Migration of __assert_fail is not supported.
  */
  assert(false);
#else
  return __hmul(a, b);
#endif
}

/* DPCT_ORIG template<>
inline __device__ __nv_bfloat162 mul(__nv_bfloat162 a, __nv_bfloat162 b) {*/
template <>
inline sycl::marray<sycl::ext::oneapi::bfloat16, 2>
mul(sycl::marray<sycl::ext::oneapi::bfloat16, 2> a,
    sycl::marray<sycl::ext::oneapi::bfloat16, 2> b) {
/* DPCT_ORIG #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800*/
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP < 800
  /*
  DPCT1007:48: Migration of __assert_fail is not supported.
  */
  assert(false);
#else
  return __hmul2(a, b);
#endif
}

/* DPCT_ORIG template<>
inline __device__ __nv_bfloat162 mul(__nv_bfloat16 a, __nv_bfloat162 b) {*/
template <>
inline sycl::marray<sycl::ext::oneapi::bfloat16, 2>
mul(sycl::ext::oneapi::bfloat16 a,
    sycl::marray<sycl::ext::oneapi::bfloat16, 2> b) {
/* DPCT_ORIG   return mul<__nv_bfloat162, __nv_bfloat162,
 * __nv_bfloat162>(bf162bf162(a), b);*/
  return mul<sycl::marray<sycl::ext::oneapi::bfloat16, 2>,
             sycl::marray<sycl::ext::oneapi::bfloat16, 2>,
             sycl::marray<sycl::ext::oneapi::bfloat16, 2>>(bf162bf162(a), b);
}

/* DPCT_ORIG template<>
inline __device__ bf16_4_t mul(bf16_4_t a, bf16_4_t b) {*/
template <> inline bf16_4_t mul(bf16_4_t a, bf16_4_t b) {
  bf16_4_t c;
/* DPCT_ORIG   c.x = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(a.x,
 * b.x);*/
  c.x = mul<sycl::marray<sycl::ext::oneapi::bfloat16, 2>,
            sycl::marray<sycl::ext::oneapi::bfloat16, 2>,
            sycl::marray<sycl::ext::oneapi::bfloat16, 2>>(a.x, b.x);
/* DPCT_ORIG   c.y = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(a.y,
 * b.y);*/
  c.y = mul<sycl::marray<sycl::ext::oneapi::bfloat16, 2>,
            sycl::marray<sycl::ext::oneapi::bfloat16, 2>,
            sycl::marray<sycl::ext::oneapi::bfloat16, 2>>(a.y, b.y);
  return c;
}

/* DPCT_ORIG template<>
inline __device__ bf16_4_t mul(__nv_bfloat16 a, bf16_4_t b) {*/
template <> inline bf16_4_t mul(sycl::ext::oneapi::bfloat16 a, bf16_4_t b) {
/* DPCT_ORIG   __nv_bfloat162 s = bf162bf162(a);*/
  sycl::marray<sycl::ext::oneapi::bfloat16, 2> s = bf162bf162(a);
  bf16_4_t c;
/* DPCT_ORIG   c.x = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(s,
 * b.x);*/
  c.x = mul<sycl::marray<sycl::ext::oneapi::bfloat16, 2>,
            sycl::marray<sycl::ext::oneapi::bfloat16, 2>,
            sycl::marray<sycl::ext::oneapi::bfloat16, 2>>(s, b.x);
/* DPCT_ORIG   c.y = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(s,
 * b.y);*/
  c.y = mul<sycl::marray<sycl::ext::oneapi::bfloat16, 2>,
            sycl::marray<sycl::ext::oneapi::bfloat16, 2>,
            sycl::marray<sycl::ext::oneapi::bfloat16, 2>>(s, b.y);
  return c;
}

/* DPCT_ORIG template<>
inline __device__ bf16_8_t mul(bf16_8_t a, bf16_8_t b) {*/
template <> inline bf16_8_t mul(bf16_8_t a, bf16_8_t b) {
  bf16_8_t c;
/* DPCT_ORIG   c.x = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(a.x,
 * b.x);*/
  c.x = mul<sycl::marray<sycl::ext::oneapi::bfloat16, 2>,
            sycl::marray<sycl::ext::oneapi::bfloat16, 2>,
            sycl::marray<sycl::ext::oneapi::bfloat16, 2>>(a.x, b.x);
/* DPCT_ORIG   c.y = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(a.y,
 * b.y);*/
  c.y = mul<sycl::marray<sycl::ext::oneapi::bfloat16, 2>,
            sycl::marray<sycl::ext::oneapi::bfloat16, 2>,
            sycl::marray<sycl::ext::oneapi::bfloat16, 2>>(a.y, b.y);
/* DPCT_ORIG   c.z = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(a.z,
 * b.z);*/
  c.z = mul<sycl::marray<sycl::ext::oneapi::bfloat16, 2>,
            sycl::marray<sycl::ext::oneapi::bfloat16, 2>,
            sycl::marray<sycl::ext::oneapi::bfloat16, 2>>(a.z, b.z);
/* DPCT_ORIG   c.w = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(a.w,
 * b.w);*/
  c.w = mul<sycl::marray<sycl::ext::oneapi::bfloat16, 2>,
            sycl::marray<sycl::ext::oneapi::bfloat16, 2>,
            sycl::marray<sycl::ext::oneapi::bfloat16, 2>>(a.w, b.w);
  return c;
}

/* DPCT_ORIG template<>
inline __device__ bf16_8_t mul(__nv_bfloat16 a, bf16_8_t b) {*/
template <> inline bf16_8_t mul(sycl::ext::oneapi::bfloat16 a, bf16_8_t b) {
/* DPCT_ORIG   __nv_bfloat162 s = bf162bf162(a);*/
  sycl::marray<sycl::ext::oneapi::bfloat16, 2> s = bf162bf162(a);
  bf16_8_t c;
/* DPCT_ORIG   c.x = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(s,
 * b.x);*/
  c.x = mul<sycl::marray<sycl::ext::oneapi::bfloat16, 2>,
            sycl::marray<sycl::ext::oneapi::bfloat16, 2>,
            sycl::marray<sycl::ext::oneapi::bfloat16, 2>>(s, b.x);
/* DPCT_ORIG   c.y = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(s,
 * b.y);*/
  c.y = mul<sycl::marray<sycl::ext::oneapi::bfloat16, 2>,
            sycl::marray<sycl::ext::oneapi::bfloat16, 2>,
            sycl::marray<sycl::ext::oneapi::bfloat16, 2>>(s, b.y);
/* DPCT_ORIG   c.z = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(s,
 * b.z);*/
  c.z = mul<sycl::marray<sycl::ext::oneapi::bfloat16, 2>,
            sycl::marray<sycl::ext::oneapi::bfloat16, 2>,
            sycl::marray<sycl::ext::oneapi::bfloat16, 2>>(s, b.z);
/* DPCT_ORIG   c.w = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(s,
 * b.w);*/
  c.w = mul<sycl::marray<sycl::ext::oneapi::bfloat16, 2>,
            sycl::marray<sycl::ext::oneapi::bfloat16, 2>,
            sycl::marray<sycl::ext::oneapi::bfloat16, 2>>(s, b.w);
  return c;
}

/* DPCT_ORIG template<>
inline __device__ float mul(__nv_bfloat16 a, __nv_bfloat16 b) {*/
template <>
inline float mul(sycl::ext::oneapi::bfloat16 a, sycl::ext::oneapi::bfloat16 b) {
/* DPCT_ORIG   float fa = __bfloat162float(a);*/
  float fa = static_cast<float>(a);
/* DPCT_ORIG   float fb = __bfloat162float(b);*/
  float fb = static_cast<float>(b);
  return fa * fb;
}

/* DPCT_ORIG template<>
inline __device__ float2 mul(__nv_bfloat162 a, __nv_bfloat162 b) {*/
template <>
inline sycl::float2 mul(sycl::marray<sycl::ext::oneapi::bfloat16, 2> a,
                        sycl::marray<sycl::ext::oneapi::bfloat16, 2> b) {
/* DPCT_ORIG   float2 fa = bf1622float2(a);*/
  sycl::float2 fa = bf1622float2(a);
/* DPCT_ORIG   float2 fb = bf1622float2(b);*/
  sycl::float2 fb = bf1622float2(b);
/* DPCT_ORIG   return mul<float2, float2, float2>(fa, fb);*/
  return mul<sycl::float2, sycl::float2, sycl::float2>(fa, fb);
}

/* DPCT_ORIG template<>
inline __device__ float2 mul(__nv_bfloat16 a, __nv_bfloat162 b) {*/
template <>
inline sycl::float2 mul(sycl::ext::oneapi::bfloat16 a,
                        sycl::marray<sycl::ext::oneapi::bfloat16, 2> b) {
/* DPCT_ORIG   return mul<float2, __nv_bfloat162, __nv_bfloat162>(bf162bf162(a),
 * b);*/
  return mul<sycl::float2, sycl::marray<sycl::ext::oneapi::bfloat16, 2>,
             sycl::marray<sycl::ext::oneapi::bfloat16, 2>>(bf162bf162(a), b);
}

/* DPCT_ORIG template<>
inline __device__ Float4_ mul(bf16_4_t a, bf16_4_t b) {*/
template <> inline Float4_ mul(bf16_4_t a, bf16_4_t b) {
  Float4_ fc;
/* DPCT_ORIG   fc.x = mul<float2, __nv_bfloat162, __nv_bfloat162>(a.x, b.x);*/
  fc.x = mul<sycl::float2, sycl::marray<sycl::ext::oneapi::bfloat16, 2>,
             sycl::marray<sycl::ext::oneapi::bfloat16, 2>>(a.x, b.x);
/* DPCT_ORIG   fc.y = mul<float2, __nv_bfloat162, __nv_bfloat162>(a.y, b.y);*/
  fc.y = mul<sycl::float2, sycl::marray<sycl::ext::oneapi::bfloat16, 2>,
             sycl::marray<sycl::ext::oneapi::bfloat16, 2>>(a.y, b.y);
  return fc;
}

/* DPCT_ORIG template<>
inline __device__ Float4_ mul(__nv_bfloat16 a, bf16_4_t b) {*/
template <> inline Float4_ mul(sycl::ext::oneapi::bfloat16 a, bf16_4_t b) {
/* DPCT_ORIG   __nv_bfloat162 s = bf162bf162(a);*/
  sycl::marray<sycl::ext::oneapi::bfloat16, 2> s = bf162bf162(a);
  Float4_ fc;
/* DPCT_ORIG   fc.x = mul<float2, __nv_bfloat162, __nv_bfloat162>(s, b.x);*/
  fc.x = mul<sycl::float2, sycl::marray<sycl::ext::oneapi::bfloat16, 2>,
             sycl::marray<sycl::ext::oneapi::bfloat16, 2>>(s, b.x);
/* DPCT_ORIG   fc.y = mul<float2, __nv_bfloat162, __nv_bfloat162>(s, b.y);*/
  fc.y = mul<sycl::float2, sycl::marray<sycl::ext::oneapi::bfloat16, 2>,
             sycl::marray<sycl::ext::oneapi::bfloat16, 2>>(s, b.y);
  return fc;
}

/* DPCT_ORIG template<>
inline __device__ Float8_ mul(bf16_8_t a, bf16_8_t b) {*/
template <> inline Float8_ mul(bf16_8_t a, bf16_8_t b) {
  Float8_ fc;
/* DPCT_ORIG   fc.x = mul<float2, __nv_bfloat162, __nv_bfloat162>(a.x, b.x);*/
  fc.x = mul<sycl::float2, sycl::marray<sycl::ext::oneapi::bfloat16, 2>,
             sycl::marray<sycl::ext::oneapi::bfloat16, 2>>(a.x, b.x);
/* DPCT_ORIG   fc.y = mul<float2, __nv_bfloat162, __nv_bfloat162>(a.y, b.y);*/
  fc.y = mul<sycl::float2, sycl::marray<sycl::ext::oneapi::bfloat16, 2>,
             sycl::marray<sycl::ext::oneapi::bfloat16, 2>>(a.y, b.y);
/* DPCT_ORIG   fc.z = mul<float2, __nv_bfloat162, __nv_bfloat162>(a.z, b.z);*/
  fc.z = mul<sycl::float2, sycl::marray<sycl::ext::oneapi::bfloat16, 2>,
             sycl::marray<sycl::ext::oneapi::bfloat16, 2>>(a.z, b.z);
/* DPCT_ORIG   fc.w = mul<float2, __nv_bfloat162, __nv_bfloat162>(a.w, b.w);*/
  fc.w = mul<sycl::float2, sycl::marray<sycl::ext::oneapi::bfloat16, 2>,
             sycl::marray<sycl::ext::oneapi::bfloat16, 2>>(a.w, b.w);
  return fc;
}

/* DPCT_ORIG template<>
inline __device__ Float8_ mul(__nv_bfloat16 a, bf16_8_t b) {*/
template <> inline Float8_ mul(sycl::ext::oneapi::bfloat16 a, bf16_8_t b) {
/* DPCT_ORIG   __nv_bfloat162 s = bf162bf162(a);*/
  sycl::marray<sycl::ext::oneapi::bfloat16, 2> s = bf162bf162(a);
  Float8_ fc;
/* DPCT_ORIG   fc.x = mul<float2, __nv_bfloat162, __nv_bfloat162>(s, b.x);*/
  fc.x = mul<sycl::float2, sycl::marray<sycl::ext::oneapi::bfloat16, 2>,
             sycl::marray<sycl::ext::oneapi::bfloat16, 2>>(s, b.x);
/* DPCT_ORIG   fc.y = mul<float2, __nv_bfloat162, __nv_bfloat162>(s, b.y);*/
  fc.y = mul<sycl::float2, sycl::marray<sycl::ext::oneapi::bfloat16, 2>,
             sycl::marray<sycl::ext::oneapi::bfloat16, 2>>(s, b.y);
/* DPCT_ORIG   fc.z = mul<float2, __nv_bfloat162, __nv_bfloat162>(s, b.z);*/
  fc.z = mul<sycl::float2, sycl::marray<sycl::ext::oneapi::bfloat16, 2>,
             sycl::marray<sycl::ext::oneapi::bfloat16, 2>>(s, b.z);
/* DPCT_ORIG   fc.w = mul<float2, __nv_bfloat162, __nv_bfloat162>(s, b.w);*/
  fc.w = mul<sycl::float2, sycl::marray<sycl::ext::oneapi::bfloat16, 2>,
             sycl::marray<sycl::ext::oneapi::bfloat16, 2>>(s, b.w);
  return fc;
}

// Vector fused multiply-add.
/* DPCT_ORIG inline __device__ __nv_bfloat162 fma(__nv_bfloat162 a,
 * __nv_bfloat162 b, __nv_bfloat162 c) {*/
inline sycl::marray<sycl::ext::oneapi::bfloat16, 2>
fma(sycl::marray<sycl::ext::oneapi::bfloat16, 2> a,
    sycl::marray<sycl::ext::oneapi::bfloat16, 2> b,
    sycl::marray<sycl::ext::oneapi::bfloat16, 2> c) {
/* DPCT_ORIG #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800*/
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP < 800
  /*
  DPCT1007:49: Migration of __assert_fail is not supported.
  */
  assert(false);
#else
  return __hfma2(a, b, c);
#endif
}

/* DPCT_ORIG inline __device__ __nv_bfloat162 fma(__nv_bfloat16 a,
 * __nv_bfloat162 b, __nv_bfloat162 c) {*/
inline sycl::marray<sycl::ext::oneapi::bfloat16, 2>
fma(sycl::ext::oneapi::bfloat16 a,
    sycl::marray<sycl::ext::oneapi::bfloat16, 2> b,
    sycl::marray<sycl::ext::oneapi::bfloat16, 2> c) {
/* DPCT_ORIG #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800*/
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP < 800
  /*
  DPCT1007:50: Migration of __assert_fail is not supported.
  */
  assert(false);
#else
  return __hfma2(bf162bf162(a), b, c);
#endif
}

/* DPCT_ORIG inline __device__ bf16_4_t fma(bf16_4_t a, bf16_4_t b, bf16_4_t c)
 * {*/
inline bf16_4_t fma(bf16_4_t a, bf16_4_t b, bf16_4_t c) {
  bf16_4_t d;
  d.x = fma(a.x, b.x, c.x);
  d.y = fma(a.y, b.y, c.y);
  return d;
}

/* DPCT_ORIG inline __device__ bf16_4_t fma(__nv_bfloat16 a, bf16_4_t b,
 * bf16_4_t c) {*/
inline bf16_4_t fma(sycl::ext::oneapi::bfloat16 a, bf16_4_t b, bf16_4_t c) {
/* DPCT_ORIG   __nv_bfloat162 s = bf162bf162(a);*/
  sycl::marray<sycl::ext::oneapi::bfloat16, 2> s = bf162bf162(a);
  bf16_4_t d;
  d.x = fma(s, b.x, c.x);
  d.y = fma(s, b.y, c.y);
  return d;
}

/* DPCT_ORIG inline __device__ bf16_8_t fma(bf16_8_t a, bf16_8_t b, bf16_8_t c)
 * {*/
inline bf16_8_t fma(bf16_8_t a, bf16_8_t b, bf16_8_t c) {
  bf16_8_t d;
  d.x = fma(a.x, b.x, c.x);
  d.y = fma(a.y, b.y, c.y);
  d.z = fma(a.z, b.z, c.z);
  d.w = fma(a.w, b.w, c.w);
  return d;
}

/* DPCT_ORIG inline __device__ bf16_8_t fma(__nv_bfloat16 a, bf16_8_t b,
 * bf16_8_t c) {*/
inline bf16_8_t fma(sycl::ext::oneapi::bfloat16 a, bf16_8_t b, bf16_8_t c) {
/* DPCT_ORIG   __nv_bfloat162 s = bf162bf162(a);*/
  sycl::marray<sycl::ext::oneapi::bfloat16, 2> s = bf162bf162(a);
  bf16_8_t d;
  d.x = fma(s, b.x, c.x);
  d.y = fma(s, b.y, c.y);
  d.z = fma(s, b.z, c.z);
  d.w = fma(s, b.w, c.w);
  return d;
}

/* DPCT_ORIG inline __device__ float fma(__nv_bfloat16 a, __nv_bfloat16 b, float
 * fc) {*/
inline float fma(sycl::ext::oneapi::bfloat16 a, sycl::ext::oneapi::bfloat16 b,
                 float fc) {
/* DPCT_ORIG   return __bfloat162float(a) * __bfloat162float(b) + fc;*/
  return static_cast<float>(a) * static_cast<float>(b) + fc;
}

/* DPCT_ORIG inline __device__ float2 fma(__nv_bfloat162 a, __nv_bfloat162 b,
 * float2 fc) {*/
inline sycl::float2 fma(sycl::marray<sycl::ext::oneapi::bfloat16, 2> a,
                        sycl::marray<sycl::ext::oneapi::bfloat16, 2> b,
                        sycl::float2 fc) {
/* DPCT_ORIG   float2 fa = bf1622float2(a);*/
  sycl::float2 fa = bf1622float2(a);
/* DPCT_ORIG   float2 fb = bf1622float2(b);*/
  sycl::float2 fb = bf1622float2(b);
  return fma(fa, fb, fc);
}

/* DPCT_ORIG inline __device__ float2 fma(__nv_bfloat16 a, __nv_bfloat162 b,
 * float2 fc) {*/
inline sycl::float2 fma(sycl::ext::oneapi::bfloat16 a,
                        sycl::marray<sycl::ext::oneapi::bfloat16, 2> b,
                        sycl::float2 fc) {
  return fma(bf162bf162(a), b, fc);
}

/* DPCT_ORIG inline __device__ Float4_ fma(bf16_4_t a, bf16_4_t b, Float4_ fc)
 * {*/
inline Float4_ fma(bf16_4_t a, bf16_4_t b, Float4_ fc) {
  Float4_ fd;
  fd.x = fma(a.x, b.x, fc.x);
  fd.y = fma(a.y, b.y, fc.y);
  return fd;
}

/* DPCT_ORIG inline __device__ Float4_ fma(__nv_bfloat16 a, bf16_4_t b, Float4_
 * fc) {*/
inline Float4_ fma(sycl::ext::oneapi::bfloat16 a, bf16_4_t b, Float4_ fc) {
/* DPCT_ORIG   __nv_bfloat162 s = bf162bf162(a);*/
  sycl::marray<sycl::ext::oneapi::bfloat16, 2> s = bf162bf162(a);
  Float4_ fd;
  fd.x = fma(s, b.x, fc.x);
  fd.y = fma(s, b.y, fc.y);
  return fd;
}

/* DPCT_ORIG inline __device__ Float8_ fma(bf16_8_t a, bf16_8_t b, Float8_ fc)
 * {*/
inline Float8_ fma(bf16_8_t a, bf16_8_t b, Float8_ fc) {
  Float8_ fd;
  fd.x = fma(a.x, b.x, fc.x);
  fd.y = fma(a.y, b.y, fc.y);
  fd.z = fma(a.z, b.z, fc.z);
  fd.w = fma(a.w, b.w, fc.w);
  return fd;
}

/* DPCT_ORIG inline __device__ Float8_ fma(__nv_bfloat16 a, bf16_8_t b, Float8_
 * fc) {*/
inline Float8_ fma(sycl::ext::oneapi::bfloat16 a, bf16_8_t b, Float8_ fc) {
/* DPCT_ORIG   __nv_bfloat162 s = bf162bf162(a);*/
  sycl::marray<sycl::ext::oneapi::bfloat16, 2> s = bf162bf162(a);
  Float8_ fd;
  fd.x = fma(s, b.x, fc.x);
  fd.y = fma(s, b.y, fc.y);
  fd.z = fma(s, b.z, fc.z);
  fd.w = fma(s, b.w, fc.w);
  return fd;
}

// Vector sum.
/* DPCT_ORIG template<>
inline __device__ float sum(__nv_bfloat16 v) {*/
template <> inline float sum(sycl::ext::oneapi::bfloat16 v) {
/* DPCT_ORIG   return __bfloat162float(v);*/
  return static_cast<float>(v);
}

/* DPCT_ORIG template<>
inline __device__ float sum(__nv_bfloat162 v) {*/
template <> inline float sum(sycl::marray<sycl::ext::oneapi::bfloat16, 2> v) {
/* DPCT_ORIG   float2 vf = bf1622float2(v);*/
  sycl::float2 vf = bf1622float2(v);
/* DPCT_ORIG   return vf.x + vf.y;*/
  return vf.x() + vf.y();
}

/* DPCT_ORIG template<>
inline __device__ float sum(bf16_4_t v) {*/
template <> inline float sum(bf16_4_t v) {
  return sum(v.x) + sum(v.y);
}

/* DPCT_ORIG template<>
inline __device__ float sum(bf16_8_t v) {*/
template <> inline float sum(bf16_8_t v) {
  return sum(v.x) + sum(v.y) + sum(v.z) + sum(v.w);
}

// From float32 to bfloat16.
/* DPCT_ORIG inline __device__ void from_float(__nv_bfloat16& dst, float src)
 * {*/
inline void from_float(sycl::ext::oneapi::bfloat16 &dst, float src) {
/* DPCT_ORIG   dst = __float2bfloat16(src);*/
  dst = sycl::ext::oneapi::bfloat16(src);
}

/* DPCT_ORIG inline __device__ void from_float(__nv_bfloat162& dst, float2 src)
 * {*/
inline void from_float(sycl::marray<sycl::ext::oneapi::bfloat16, 2> &dst,
                       sycl::float2 src) {
/* DPCT_ORIG #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800*/
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP < 800
  /*
  DPCT1007:51: Migration of __assert_fail is not supported.
  */
  assert(false);
#else
  dst = __float22bfloat162_rn(src);
#endif
}

/* DPCT_ORIG inline __device__ void from_float(bf16_4_t& dst, Float4_ src) {*/
inline void from_float(bf16_4_t &dst, Float4_ src) {
/* DPCT_ORIG #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800*/
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP < 800
  /*
  DPCT1007:52: Migration of __assert_fail is not supported.
  */
  assert(false);
#else
  dst.x = __float22bfloat162_rn(src.x);
  dst.y = __float22bfloat162_rn(src.y);
#endif
}

/* DPCT_ORIG inline __device__ void from_float(bf16_8_t& dst, Float8_ src) {*/
inline void from_float(bf16_8_t &dst, Float8_ src) {
/* DPCT_ORIG #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800*/
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP < 800
  /*
  DPCT1007:53: Migration of __assert_fail is not supported.
  */
  assert(false);
#else
  dst.x = __float22bfloat162_rn(src.x);
  dst.y = __float22bfloat162_rn(src.y);
  dst.z = __float22bfloat162_rn(src.z);
  dst.w = __float22bfloat162_rn(src.w);
#endif
}

// From bfloat16 to float32.
/* DPCT_ORIG inline __device__ float to_float(__nv_bfloat16 u) {*/
inline float to_float(sycl::ext::oneapi::bfloat16 u) {
/* DPCT_ORIG   return __bfloat162float(u);*/
  return static_cast<float>(u);
}

// Zero-out a variable.
/* DPCT_ORIG inline __device__ void zero(__nv_bfloat16& dst) {*/
inline void zero(sycl::ext::oneapi::bfloat16 &dst) {
/* DPCT_ORIG #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800*/
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP < 800
  /*
  DPCT1007:54: Migration of __assert_fail is not supported.
  */
  assert(false);
#else
  // Same as CUDART_ZERO_BF16 introduced in CUDA 12.2.
  dst = __ushort_as_bfloat16((unsigned short)0x0000U);
#endif
}

} // namespace vllm
