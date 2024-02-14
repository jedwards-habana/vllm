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

#ifdef USE_ROCM
  #include <hip/hip_fp16.h>
#endif

#include <stdint.h>

namespace vllm {

// FP16 vector types for Q, K, V.
template<>
struct Vec<uint16_t, 1> {
  using Type = uint16_t;
};
template<>
struct Vec<uint16_t, 2> {
  using Type = uint32_t;
};
template<>
struct Vec<uint16_t, 4> {
/* DPCT_ORIG   using Type = uint2;*/
  using Type = sycl::uint2;
};
template<>
struct Vec<uint16_t, 8> {
/* DPCT_ORIG   using Type = uint4;*/
  using Type = sycl::uint4;
};

// FP32 accumulator vector types corresponding to Vec.
template<>
struct FloatVec<uint16_t> {
  using Type = float;
};
template<>
struct FloatVec<uint32_t> {
/* DPCT_ORIG   using Type = float2;*/
  using Type = sycl::float2;
};
template <>
/* DPCT_ORIG struct FloatVec<uint2> {*/
struct FloatVec<sycl::uint2> {
  using Type = Float4_;
};
template <>
/* DPCT_ORIG struct FloatVec<uint4> {*/
struct FloatVec<sycl::uint4> {
  using Type = Float8_;
};

// Utility functions for type conversions.
/* DPCT_ORIG inline __device__ uint32_t h0_h0(uint16_t a) {*/
inline uint32_t h0_h0(uint16_t a) {
#ifndef USE_ROCM
  uint32_t b;
/* DPCT_ORIG   asm volatile("mov.b32 %0, {%1, %1};" : "=r"(b) : "h"(a));*/
   b = {a, a};
  return b;
#else
  union {
   uint32_t u32;
   uint16_t u16[2];
  } tmp;
  tmp.u16[0] = a;
  tmp.u16[1] = a;
  return tmp.u32;
#endif
}

/* DPCT_ORIG inline __device__ float half_to_float(uint16_t h) {*/
inline float half_to_float(uint16_t h) {
  float f;
#ifndef USE_ROCM
  /*
  DPCT1053:10: Migration of device assembly code is not supported.
  */
  asm volatile("cvt.f32.f16 %0, %1;\n" : "=f"(f) : "h"(h));
#else
  asm volatile("v_cvt_f32_f16 %0, %1;" : "=v"(f) : "v"(h));
#endif
  return f;
}

/* DPCT_ORIG inline __device__ float2 half2_to_float2(uint32_t v) {*/
inline sycl::float2 half2_to_float2(uint32_t v) {
#ifndef USE_ROCM
  uint16_t lo, hi;
/* DPCT_ORIG   asm volatile("mov.b32 {%0, %1}, %2;\n" : "=h"(lo), "=h"(hi) :
 * "r"(v));*/
   {lo, hi} = v;
/* DPCT_ORIG   return make_float2(half_to_float(lo), half_to_float(hi));*/
  return sycl::float2(half_to_float(lo), half_to_float(hi));
#else
  union {
    uint32_t u32;
    uint16_t u16[2];
  } tmp;
  tmp.u32 = v;
  float2 ret;
  ret.x = half_to_float(tmp.u16[0]);
  ret.y = half_to_float(tmp.u16[1]);
  return ret;
#endif
}

/* DPCT_ORIG inline __device__ uint16_t float_to_half(float f) {*/
inline uint16_t float_to_half(float f) {
  union {
    uint32_t u32;
    uint16_t u16[2];
  } tmp;
#ifndef USE_ROCM
  /*
  DPCT1053:11: Migration of device assembly code is not supported.
  */
  asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(tmp.u16[0]) : "f"(f));
#else
  asm volatile("v_cvt_f16_f32 %0, %1;\n" : "=v"(tmp.u32) : "v"(f));
#endif
  return tmp.u16[0];
}

/* DPCT_ORIG inline __device__ uint32_t float2_to_half2(float2 f) {*/
inline uint32_t float2_to_half2(sycl::float2 f) {
  union {
    uint32_t u32;
    uint16_t u16[2];
  } tmp;
#ifndef USE_ROCM
/* DPCT_ORIG   #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800*/
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP >= 800
    asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(tmp.u32) : "f"(f.y), "f"(f.x));
#else
/* DPCT_ORIG     asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(tmp.u16[0]) :
 * "f"(f.x));*/
    /*
    DPCT1053:12: Migration of device assembly code is not supported.
    */
    asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(tmp.u16[0]) : "f"(f.x()));
/* DPCT_ORIG     asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(tmp.u16[1]) :
 * "f"(f.y));*/
    /*
    DPCT1053:13: Migration of device assembly code is not supported.
    */
    asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(tmp.u16[1]) : "f"(f.y()));
#endif
#else
  tmp.u16[0] = float_to_half(f.x);
  tmp.u16[1] = float_to_half(f.y);
#endif
  return tmp.u32;
}

// Vector addition.
/* DPCT_ORIG inline __device__ uint16_t add(uint16_t a, uint16_t b) {*/
inline uint16_t add(uint16_t a, uint16_t b) {
  uint16_t c;
#ifndef USE_ROCM
  /*
  DPCT1053:14: Migration of device assembly code is not supported.
  */
  asm volatile("add.f16 %0, %1, %2;\n" : "=h"(c) : "h"(a), "h"(b));
#else
  asm volatile("v_add_f16 %0, %1, %2;\n" : "=v"(c) : "v"(a), "v"(b));
#endif
  return c;
}

/* DPCT_ORIG inline __device__ uint32_t add(uint32_t a, uint32_t b) {*/
inline uint32_t add(uint32_t a, uint32_t b) {
  uint32_t c;
#ifndef USE_ROCM
  /*
  DPCT1053:15: Migration of device assembly code is not supported.
  */
  asm volatile("add.f16x2 %0, %1, %2;\n" : "=r"(c) : "r"(a), "r"(b));
#else
  asm volatile("v_pk_add_f16 %0, %1, %2;\n" : "=v"(c) : "v"(a), "v"(b));
#endif
  return c;
}

/* DPCT_ORIG inline __device__ uint2 add(uint2 a, uint2 b) {*/
inline sycl::uint2 add(sycl::uint2 a, sycl::uint2 b) {
/* DPCT_ORIG   uint2 c;*/
  sycl::uint2 c;
/* DPCT_ORIG   c.x = add(a.x, b.x);*/
  c.x() = add(a.x(), b.x());
/* DPCT_ORIG   c.y = add(a.y, b.y);*/
  c.y() = add(a.y(), b.y());
  return c;
}

/* DPCT_ORIG inline __device__ uint4 add(uint4 a, uint4 b) {*/
inline sycl::uint4 add(sycl::uint4 a, sycl::uint4 b) {
/* DPCT_ORIG   uint4 c;*/
  sycl::uint4 c;
/* DPCT_ORIG   c.x = add(a.x, b.x);*/
  c.x() = add(a.x(), b.x());
/* DPCT_ORIG   c.y = add(a.y, b.y);*/
  c.y() = add(a.y(), b.y());
/* DPCT_ORIG   c.z = add(a.z, b.z);*/
  c.z() = add(a.z(), b.z());
/* DPCT_ORIG   c.w = add(a.w, b.w);*/
  c.w() = add(a.w(), b.w());
  return c;
}

/* DPCT_ORIG inline __device__ float2 add(uint32_t a, float2 fb) {*/
inline sycl::float2 add(uint32_t a, sycl::float2 fb) {
/* DPCT_ORIG   float2 fa = half2_to_float2(a);*/
  sycl::float2 fa = half2_to_float2(a);
  return add(fa, fb);
}

/* DPCT_ORIG inline __device__ Float4_ add(uint2 a, Float4_ fb) {*/
inline Float4_ add(sycl::uint2 a, Float4_ fb) {
  Float4_ fc;
/* DPCT_ORIG   fc.x = add(a.x, fb.x);*/
  fc.x = add(a.x(), fb.x);
/* DPCT_ORIG   fc.y = add(a.y, fb.y);*/
  fc.y = add(a.y(), fb.y);
  return fc;
}

/* DPCT_ORIG inline __device__ Float8_ add(uint4 a, Float8_ fb) {*/
inline Float8_ add(sycl::uint4 a, Float8_ fb) {
  Float8_ fc;
/* DPCT_ORIG   fc.x = add(a.x, fb.x);*/
  fc.x = add(a.x(), fb.x);
/* DPCT_ORIG   fc.y = add(a.y, fb.y);*/
  fc.y = add(a.y(), fb.y);
/* DPCT_ORIG   fc.z = add(a.z, fb.z);*/
  fc.z = add(a.z(), fb.z);
/* DPCT_ORIG   fc.w = add(a.w, fb.w);*/
  fc.w = add(a.w(), fb.w);
  return fc;
}

// Vector multiplication.
/* DPCT_ORIG template<>
inline __device__ uint16_t mul(uint16_t a, uint16_t b) {*/
template <> inline uint16_t mul(uint16_t a, uint16_t b) {
  uint16_t c;
#ifndef USE_ROCM
  /*
  DPCT1053:16: Migration of device assembly code is not supported.
  */
  asm volatile("mul.f16 %0, %1, %2;\n" : "=h"(c) : "h"(a), "h"(b));
#else
  asm volatile("v_mul_f16 %0, %1, %2;\n" : "=v"(c) : "v"(a), "v"(b));
#endif
  return c;
}

/* DPCT_ORIG template<>
inline __device__ uint32_t mul(uint32_t a, uint32_t b) {*/
template <> inline uint32_t mul(uint32_t a, uint32_t b) {
  uint32_t c;
#ifndef USE_ROCM
  /*
  DPCT1053:17: Migration of device assembly code is not supported.
  */
  asm volatile("mul.f16x2 %0, %1, %2;\n" : "=r"(c) : "r"(a), "r"(b));
#else
  asm volatile("v_pk_mul_f16 %0, %1, %2;\n" : "=v"(c) : "v"(a), "v"(b));
#endif
  return c;
}

/* DPCT_ORIG template<>
inline __device__ uint32_t mul(uint16_t a, uint32_t b) {*/
template <> inline uint32_t mul(uint16_t a, uint32_t b) {
  return mul<uint32_t, uint32_t, uint32_t>(h0_h0(a), b);
}

/* DPCT_ORIG template<>
inline __device__ uint2 mul(uint2 a, uint2 b) {*/
template <> inline sycl::uint2 mul(sycl::uint2 a, sycl::uint2 b) {
/* DPCT_ORIG   uint2 c;*/
  sycl::uint2 c;
/* DPCT_ORIG   c.x = mul<uint32_t, uint32_t, uint32_t>(a.x, b.x);*/
  c.x() = mul<uint32_t, uint32_t, uint32_t>(a.x(), b.x());
/* DPCT_ORIG   c.y = mul<uint32_t, uint32_t, uint32_t>(a.y, b.y);*/
  c.y() = mul<uint32_t, uint32_t, uint32_t>(a.y(), b.y());
  return c;
}

/* DPCT_ORIG template<>
inline __device__ uint2 mul(uint16_t a, uint2 b) {*/
template <> inline sycl::uint2 mul(uint16_t a, sycl::uint2 b) {
  uint32_t s = h0_h0(a);
/* DPCT_ORIG   uint2 c;*/
  sycl::uint2 c;
/* DPCT_ORIG   c.x = mul<uint32_t, uint32_t, uint32_t>(s, b.x);*/
  c.x() = mul<uint32_t, uint32_t, uint32_t>(s, b.x());
/* DPCT_ORIG   c.y = mul<uint32_t, uint32_t, uint32_t>(s, b.y);*/
  c.y() = mul<uint32_t, uint32_t, uint32_t>(s, b.y());
  return c;
}

/* DPCT_ORIG template<>
inline __device__ uint4 mul(uint4 a, uint4 b) {*/
template <> inline sycl::uint4 mul(sycl::uint4 a, sycl::uint4 b) {
/* DPCT_ORIG   uint4 c;*/
  sycl::uint4 c;
/* DPCT_ORIG   c.x = mul<uint32_t, uint32_t, uint32_t>(a.x, b.x);*/
  c.x() = mul<uint32_t, uint32_t, uint32_t>(a.x(), b.x());
/* DPCT_ORIG   c.y = mul<uint32_t, uint32_t, uint32_t>(a.y, b.y);*/
  c.y() = mul<uint32_t, uint32_t, uint32_t>(a.y(), b.y());
/* DPCT_ORIG   c.z = mul<uint32_t, uint32_t, uint32_t>(a.z, b.z);*/
  c.z() = mul<uint32_t, uint32_t, uint32_t>(a.z(), b.z());
/* DPCT_ORIG   c.w = mul<uint32_t, uint32_t, uint32_t>(a.w, b.w);*/
  c.w() = mul<uint32_t, uint32_t, uint32_t>(a.w(), b.w());
  return c;
}

/* DPCT_ORIG template<>
inline __device__ uint4 mul(uint16_t a, uint4 b) {*/
template <> inline sycl::uint4 mul(uint16_t a, sycl::uint4 b) {
  uint32_t s = h0_h0(a);
/* DPCT_ORIG   uint4 c;*/
  sycl::uint4 c;
/* DPCT_ORIG   c.x = mul<uint32_t, uint32_t, uint32_t>(s, b.x);*/
  c.x() = mul<uint32_t, uint32_t, uint32_t>(s, b.x());
/* DPCT_ORIG   c.y = mul<uint32_t, uint32_t, uint32_t>(s, b.y);*/
  c.y() = mul<uint32_t, uint32_t, uint32_t>(s, b.y());
/* DPCT_ORIG   c.z = mul<uint32_t, uint32_t, uint32_t>(s, b.z);*/
  c.z() = mul<uint32_t, uint32_t, uint32_t>(s, b.z());
/* DPCT_ORIG   c.w = mul<uint32_t, uint32_t, uint32_t>(s, b.w);*/
  c.w() = mul<uint32_t, uint32_t, uint32_t>(s, b.w());
  return c;
}

/* DPCT_ORIG template<>
inline __device__ float mul(uint16_t a, uint16_t b) {*/
template <> inline float mul(uint16_t a, uint16_t b) {
  float fa = half_to_float(a);
  float fb = half_to_float(b);
  return fa * fb;
}

/* DPCT_ORIG template<>
inline __device__ float2 mul(uint32_t a, uint32_t b) {*/
template <> inline sycl::float2 mul(uint32_t a, uint32_t b) {
/* DPCT_ORIG   float2 fa = half2_to_float2(a);*/
  sycl::float2 fa = half2_to_float2(a);
/* DPCT_ORIG   float2 fb = half2_to_float2(b);*/
  sycl::float2 fb = half2_to_float2(b);
/* DPCT_ORIG   return mul<float2, float2, float2>(fa, fb);*/
  return mul<sycl::float2, sycl::float2, sycl::float2>(fa, fb);
}

/* DPCT_ORIG template<>
inline __device__ float2 mul(uint16_t a, uint32_t b) {*/
template <> inline sycl::float2 mul(uint16_t a, uint32_t b) {
/* DPCT_ORIG   return mul<float2, uint32_t, uint32_t>(h0_h0(a), b);*/
  return mul<sycl::float2, uint32_t, uint32_t>(h0_h0(a), b);
}

/* DPCT_ORIG template<>
inline __device__ Float4_ mul(uint2 a, uint2 b) {*/
template <> inline Float4_ mul(sycl::uint2 a, sycl::uint2 b) {
  Float4_ fc;
/* DPCT_ORIG   fc.x = mul<float2, uint32_t, uint32_t>(a.x, b.x);*/
  fc.x = mul<sycl::float2, uint32_t, uint32_t>(a.x(), b.x());
/* DPCT_ORIG   fc.y = mul<float2, uint32_t, uint32_t>(a.y, b.y);*/
  fc.y = mul<sycl::float2, uint32_t, uint32_t>(a.y(), b.y());
  return fc;
}

/* DPCT_ORIG template<>
inline __device__ Float4_ mul(uint16_t a, uint2 b) {*/
template <> inline Float4_ mul(uint16_t a, sycl::uint2 b) {
  uint32_t s = h0_h0(a);
  Float4_ fc;
/* DPCT_ORIG   fc.x = mul<float2, uint32_t, uint32_t>(s, b.x);*/
  fc.x = mul<sycl::float2, uint32_t, uint32_t>(s, b.x());
/* DPCT_ORIG   fc.y = mul<float2, uint32_t, uint32_t>(s, b.y);*/
  fc.y = mul<sycl::float2, uint32_t, uint32_t>(s, b.y());
  return fc;
}

/* DPCT_ORIG template<>
inline __device__ Float8_ mul(uint4 a, uint4 b) {*/
template <> inline Float8_ mul(sycl::uint4 a, sycl::uint4 b) {
  Float8_ fc;
/* DPCT_ORIG   fc.x = mul<float2, uint32_t, uint32_t>(a.x, b.x);*/
  fc.x = mul<sycl::float2, uint32_t, uint32_t>(a.x(), b.x());
/* DPCT_ORIG   fc.y = mul<float2, uint32_t, uint32_t>(a.y, b.y);*/
  fc.y = mul<sycl::float2, uint32_t, uint32_t>(a.y(), b.y());
/* DPCT_ORIG   fc.z = mul<float2, uint32_t, uint32_t>(a.z, b.z);*/
  fc.z = mul<sycl::float2, uint32_t, uint32_t>(a.z(), b.z());
/* DPCT_ORIG   fc.w = mul<float2, uint32_t, uint32_t>(a.w, b.w);*/
  fc.w = mul<sycl::float2, uint32_t, uint32_t>(a.w(), b.w());
  return fc;
}

/* DPCT_ORIG template<>
inline __device__ Float8_ mul(uint16_t a, uint4 b) {*/
template <> inline Float8_ mul(uint16_t a, sycl::uint4 b) {
  uint32_t s = h0_h0(a);
  Float8_ fc;
/* DPCT_ORIG   fc.x = mul<float2, uint32_t, uint32_t>(s, b.x);*/
  fc.x = mul<sycl::float2, uint32_t, uint32_t>(s, b.x());
/* DPCT_ORIG   fc.y = mul<float2, uint32_t, uint32_t>(s, b.y);*/
  fc.y = mul<sycl::float2, uint32_t, uint32_t>(s, b.y());
/* DPCT_ORIG   fc.z = mul<float2, uint32_t, uint32_t>(s, b.z);*/
  fc.z = mul<sycl::float2, uint32_t, uint32_t>(s, b.z());
/* DPCT_ORIG   fc.w = mul<float2, uint32_t, uint32_t>(s, b.w);*/
  fc.w = mul<sycl::float2, uint32_t, uint32_t>(s, b.w());
  return fc;
}

// Vector fused multiply-add.
/* DPCT_ORIG inline __device__ uint32_t fma(uint32_t a, uint32_t b, uint32_t c)
 * {*/
inline uint32_t fma(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
#ifndef USE_ROCM
  /*
  DPCT1053:18: Migration of device assembly code is not supported.
  */
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
               : "=r"(d)
               : "r"(a), "r"(b), "r"(c));
#else
  asm volatile("v_pk_fma_f16 %0, %1, %2, %3;\n" : "=v"(d) : "v"(a), "v"(b), "v"(c));
#endif
  return d;
}

/* DPCT_ORIG inline __device__ uint32_t fma(uint16_t a, uint32_t b, uint32_t c)
 * {*/
inline uint32_t fma(uint16_t a, uint32_t b, uint32_t c) {
  return fma(h0_h0(a), b, c);
}

/* DPCT_ORIG inline __device__ uint2 fma(uint2 a, uint2 b, uint2 c) {*/
inline sycl::uint2 fma(sycl::uint2 a, sycl::uint2 b, sycl::uint2 c) {
/* DPCT_ORIG   uint2 d;*/
  sycl::uint2 d;
/* DPCT_ORIG   d.x = fma(a.x, b.x, c.x);*/
  d.x() = fma(a.x(), b.x(), c.x());
/* DPCT_ORIG   d.y = fma(a.y, b.y, c.y);*/
  d.y() = fma(a.y(), b.y(), c.y());
  return d;
}

/* DPCT_ORIG inline __device__ uint2 fma(uint16_t a, uint2 b, uint2 c) {*/
inline sycl::uint2 fma(uint16_t a, sycl::uint2 b, sycl::uint2 c) {
  uint32_t s = h0_h0(a);
/* DPCT_ORIG   uint2 d;*/
  sycl::uint2 d;
/* DPCT_ORIG   d.x = fma(s, b.x, c.x);*/
  d.x() = fma(s, b.x(), c.x());
/* DPCT_ORIG   d.y = fma(s, b.y, c.y);*/
  d.y() = fma(s, b.y(), c.y());
  return d;
}

/* DPCT_ORIG inline __device__ uint4 fma(uint4 a, uint4 b, uint4 c) {*/
inline sycl::uint4 fma(sycl::uint4 a, sycl::uint4 b, sycl::uint4 c) {
/* DPCT_ORIG   uint4 d;*/
  sycl::uint4 d;
/* DPCT_ORIG   d.x = fma(a.x, b.x, c.x);*/
  d.x() = fma(a.x(), b.x(), c.x());
/* DPCT_ORIG   d.y = fma(a.y, b.y, c.y);*/
  d.y() = fma(a.y(), b.y(), c.y());
/* DPCT_ORIG   d.z = fma(a.z, b.z, c.z);*/
  d.z() = fma(a.z(), b.z(), c.z());
/* DPCT_ORIG   d.w = fma(a.w, b.w, c.w);*/
  d.w() = fma(a.w(), b.w(), c.w());
  return d;
}

/* DPCT_ORIG inline __device__ uint4 fma(uint16_t a, uint4 b, uint4 c) {*/
inline sycl::uint4 fma(uint16_t a, sycl::uint4 b, sycl::uint4 c) {
  uint32_t s = h0_h0(a);
/* DPCT_ORIG   uint4 d;*/
  sycl::uint4 d;
/* DPCT_ORIG   d.x = fma(s, b.x, c.x);*/
  d.x() = fma(s, b.x(), c.x());
/* DPCT_ORIG   d.y = fma(s, b.y, c.y);*/
  d.y() = fma(s, b.y(), c.y());
/* DPCT_ORIG   d.z = fma(s, b.z, c.z);*/
  d.z() = fma(s, b.z(), c.z());
/* DPCT_ORIG   d.w = fma(s, b.w, c.w);*/
  d.w() = fma(s, b.w(), c.w());
  return d;
}

/* DPCT_ORIG inline __device__ float fma(uint16_t a, uint16_t b, float fc) {*/
inline float fma(uint16_t a, uint16_t b, float fc) {
  float fa = half_to_float(a);
  float fb = half_to_float(b);
  return fa * fb + fc;
}

/* DPCT_ORIG inline __device__ float2 fma(uint32_t a, uint32_t b, float2 fc) {*/
inline sycl::float2 fma(uint32_t a, uint32_t b, sycl::float2 fc) {
/* DPCT_ORIG   float2 fa = half2_to_float2(a);*/
  sycl::float2 fa = half2_to_float2(a);
/* DPCT_ORIG   float2 fb = half2_to_float2(b);*/
  sycl::float2 fb = half2_to_float2(b);
  return fma(fa, fb, fc);
}

/* DPCT_ORIG inline __device__ float2 fma(uint16_t a, uint32_t b, float2 fc) {*/
inline sycl::float2 fma(uint16_t a, uint32_t b, sycl::float2 fc) {
  return fma(h0_h0(a), b, fc);
}

/* DPCT_ORIG inline __device__ Float4_ fma(uint2 a, uint2 b, Float4_ fc) {*/
inline Float4_ fma(sycl::uint2 a, sycl::uint2 b, Float4_ fc) {
  Float4_ fd;
/* DPCT_ORIG   fd.x = fma(a.x, b.x, fc.x);*/
  fd.x = fma(a.x(), b.x(), fc.x);
/* DPCT_ORIG   fd.y = fma(a.y, b.y, fc.y);*/
  fd.y = fma(a.y(), b.y(), fc.y);
  return fd;
}

/* DPCT_ORIG inline __device__ Float4_ fma(uint16_t a, uint2 b, Float4_ fc) {*/
inline Float4_ fma(uint16_t a, sycl::uint2 b, Float4_ fc) {
  uint32_t s = h0_h0(a);
  Float4_ fd;
/* DPCT_ORIG   fd.x = fma(s, b.x, fc.x);*/
  fd.x = fma(s, b.x(), fc.x);
/* DPCT_ORIG   fd.y = fma(s, b.y, fc.y);*/
  fd.y = fma(s, b.y(), fc.y);
  return fd;
}

/* DPCT_ORIG inline __device__ Float8_ fma(uint4 a, uint4 b, Float8_ fc) {*/
inline Float8_ fma(sycl::uint4 a, sycl::uint4 b, Float8_ fc) {
  Float8_ fd;
/* DPCT_ORIG   fd.x = fma(a.x, b.x, fc.x);*/
  fd.x = fma(a.x(), b.x(), fc.x);
/* DPCT_ORIG   fd.y = fma(a.y, b.y, fc.y);*/
  fd.y = fma(a.y(), b.y(), fc.y);
/* DPCT_ORIG   fd.z = fma(a.z, b.z, fc.z);*/
  fd.z = fma(a.z(), b.z(), fc.z);
/* DPCT_ORIG   fd.w = fma(a.w, b.w, fc.w);*/
  fd.w = fma(a.w(), b.w(), fc.w);
  return fd;
}

/* DPCT_ORIG inline __device__ Float8_ fma(uint16_t a, uint4 b, Float8_ fc) {*/
inline Float8_ fma(uint16_t a, sycl::uint4 b, Float8_ fc) {
  uint32_t s = h0_h0(a);
  Float8_ fd;
/* DPCT_ORIG   fd.x = fma(s, b.x, fc.x);*/
  fd.x = fma(s, b.x(), fc.x);
/* DPCT_ORIG   fd.y = fma(s, b.y, fc.y);*/
  fd.y = fma(s, b.y(), fc.y);
/* DPCT_ORIG   fd.z = fma(s, b.z, fc.z);*/
  fd.z = fma(s, b.z(), fc.z);
/* DPCT_ORIG   fd.w = fma(s, b.w, fc.w);*/
  fd.w = fma(s, b.w(), fc.w);
  return fd;
}

// Vector sum.
/* DPCT_ORIG template<>
inline __device__ float sum(uint16_t v) {*/
template <> inline float sum(uint16_t v) {
  return half_to_float(v);
}

/* DPCT_ORIG template<>
inline __device__ float sum(uint32_t v) {*/
template <> inline float sum(uint32_t v) {
/* DPCT_ORIG   float2 tmp = half2_to_float2(v);*/
  sycl::float2 tmp = half2_to_float2(v);
/* DPCT_ORIG   return tmp.x + tmp.y;*/
  return tmp.x() + tmp.y();
}

/* DPCT_ORIG template<>
inline __device__ float sum(uint2 v) {*/
template <> inline float sum(sycl::uint2 v) {
/* DPCT_ORIG   uint32_t c = add(v.x, v.y);*/
  uint32_t c = add(v.x(), v.y());
  return sum(c);
}

/* DPCT_ORIG template<>
inline __device__ float sum(uint4 v) {*/
template <> inline float sum(sycl::uint4 v) {
/* DPCT_ORIG   uint32_t c = add(v.x, v.y);*/
  uint32_t c = add(v.x(), v.y());
/* DPCT_ORIG   c = add(c, v.z);*/
  c = add(c, v.z());
/* DPCT_ORIG   c = add(c, v.w);*/
  c = add(c, v.w());
  return sum(c);
}

// From float32 to float16.
/* DPCT_ORIG inline __device__ void from_float(uint16_t& dst, float src) {*/
inline void from_float(uint16_t &dst, float src) {
  dst = float_to_half(src);
}

/* DPCT_ORIG inline __device__ void from_float(uint32_t& dst, float2 src) {*/
inline void from_float(uint32_t &dst, sycl::float2 src) {
  dst = float2_to_half2(src);
}

/* DPCT_ORIG inline __device__ void from_float(uint2& dst, Float4_ src) {*/
inline void from_float(sycl::uint2 &dst, Float4_ src) {
/* DPCT_ORIG   dst.x = float2_to_half2(src.x);*/
  dst.x() = float2_to_half2(src.x);
/* DPCT_ORIG   dst.y = float2_to_half2(src.y);*/
  dst.y() = float2_to_half2(src.y);
}

/* DPCT_ORIG inline __device__ void from_float(uint4& dst, Float8_ src) {*/
inline void from_float(sycl::uint4 &dst, Float8_ src) {
/* DPCT_ORIG   dst.x = float2_to_half2(src.x);*/
  dst.x() = float2_to_half2(src.x);
/* DPCT_ORIG   dst.y = float2_to_half2(src.y);*/
  dst.y() = float2_to_half2(src.y);
/* DPCT_ORIG   dst.z = float2_to_half2(src.z);*/
  dst.z() = float2_to_half2(src.z);
/* DPCT_ORIG   dst.w = float2_to_half2(src.w);*/
  dst.w() = float2_to_half2(src.w);
}

// From float16 to float32.
/* DPCT_ORIG inline __device__ float to_float(uint16_t u) {*/
inline float to_float(uint16_t u) {
  return half_to_float(u);
}

/* DPCT_ORIG inline __device__ float2 to_float(uint32_t u) {*/
inline sycl::float2 to_float(uint32_t u) {
  return half2_to_float2(u);
}

/* DPCT_ORIG inline __device__ Float4_ to_float(uint2 u) {*/
inline Float4_ to_float(sycl::uint2 u) {
  Float4_ tmp;
/* DPCT_ORIG   tmp.x = half2_to_float2(u.x);*/
  tmp.x = half2_to_float2(u.x());
/* DPCT_ORIG   tmp.y = half2_to_float2(u.y);*/
  tmp.y = half2_to_float2(u.y());
  return tmp;
}

/* DPCT_ORIG inline __device__ Float8_ to_float(uint4 u) {*/
inline Float8_ to_float(sycl::uint4 u) {
  Float8_ tmp;
/* DPCT_ORIG   tmp.x = half2_to_float2(u.x);*/
  tmp.x = half2_to_float2(u.x());
/* DPCT_ORIG   tmp.y = half2_to_float2(u.y);*/
  tmp.y = half2_to_float2(u.y());
/* DPCT_ORIG   tmp.z = half2_to_float2(u.z);*/
  tmp.z = half2_to_float2(u.z());
/* DPCT_ORIG   tmp.w = half2_to_float2(u.w);*/
  tmp.w = half2_to_float2(u.w());
  return tmp;
}

// Zero-out a variable.
/* DPCT_ORIG inline __device__ void zero(uint16_t& dst) {*/
inline void zero(uint16_t &dst) {
  dst = uint16_t(0);
}

} // namespace vllm
