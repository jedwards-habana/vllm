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

#include <stdint.h>

namespace vllm {

// Define custom FP32 vector data types.
struct Float4_ {
/* DPCT_ORIG   float2 x;*/
  sycl::float2 x;
/* DPCT_ORIG   float2 y;*/
  sycl::float2 y;
};

struct Float8_ {
/* DPCT_ORIG   float2 x;*/
  sycl::float2 x;
/* DPCT_ORIG   float2 y;*/
  sycl::float2 y;
/* DPCT_ORIG   float2 z;*/
  sycl::float2 z;
/* DPCT_ORIG   float2 w;*/
  sycl::float2 w;
};

// FP32 vector types for Q, K, V.
template<>
struct Vec<float, 1> {
  using Type = float;
};
template<>
struct Vec<float, 2> {
/* DPCT_ORIG   using Type = float2;*/
  using Type = sycl::float2;
};
template<>
struct Vec<float, 4> {
/* DPCT_ORIG   using Type = float4;*/
  using Type = sycl::float4;
};

// FP32 accumulator vector types corresponding to Vec.
template<>
struct FloatVec<float> {
  using Type = float;
};
template <>
/* DPCT_ORIG struct FloatVec<float2> {*/
struct FloatVec<sycl::float2> {
/* DPCT_ORIG   using Type = float2;*/
  using Type = sycl::float2;
};
template <>
/* DPCT_ORIG struct FloatVec<float4> {*/
struct FloatVec<sycl::float4> {
/* DPCT_ORIG   using Type = float4;*/
  using Type = sycl::float4;
};

// Vector addition.
/* DPCT_ORIG inline __device__ float add(float a, float b) {*/
inline float add(float a, float b) {
  return a + b;
}

/* DPCT_ORIG inline __device__ float2 add(float2 a, float2 b) {*/
inline sycl::float2 add(sycl::float2 a, sycl::float2 b) {
/* DPCT_ORIG   float2 c;*/
  sycl::float2 c;
/* DPCT_ORIG   c.x = add(a.x, b.x);*/
  c.x() = add(a.x(), b.x());
/* DPCT_ORIG   c.y = add(a.y, b.y);*/
  c.y() = add(a.y(), b.y());
  return c;
}

/* DPCT_ORIG inline __device__ float4 add(float4 a, float4 b) {*/
inline sycl::float4 add(sycl::float4 a, sycl::float4 b) {
/* DPCT_ORIG   float4 c;*/
  sycl::float4 c;
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

// Vector multiplication.
/* DPCT_ORIG template<>
inline __device__ float mul<float, float>(float a, float b) {*/
template <> inline float mul<float, float>(float a, float b) {
  return a * b;
}

/* DPCT_ORIG template<>
inline __device__ float2 mul(float2 a, float2 b) {*/
template <> inline sycl::float2 mul(sycl::float2 a, sycl::float2 b) {
/* DPCT_ORIG   float2 c;*/
  sycl::float2 c;
/* DPCT_ORIG   c.x = a.x * b.x;*/
  c.x() = a.x() * b.x();
/* DPCT_ORIG   c.y = a.y * b.y;*/
  c.y() = a.y() * b.y();
  return c;
}

/* DPCT_ORIG template<>
inline __device__ float2 mul(float a, float2 b) {*/
template <> inline sycl::float2 mul(float a, sycl::float2 b) {
/* DPCT_ORIG   float2 c;*/
  sycl::float2 c;
/* DPCT_ORIG   c.x = a * b.x;*/
  c.x() = a * b.x();
/* DPCT_ORIG   c.y = a * b.y;*/
  c.y() = a * b.y();
  return c;
}

/* DPCT_ORIG template<>
inline __device__ float4 mul(float4 a, float4 b) {*/
template <> inline sycl::float4 mul(sycl::float4 a, sycl::float4 b) {
/* DPCT_ORIG   float4 c;*/
  sycl::float4 c;
/* DPCT_ORIG   c.x = a.x * b.x;*/
  c.x() = a.x() * b.x();
/* DPCT_ORIG   c.y = a.y * b.y;*/
  c.y() = a.y() * b.y();
/* DPCT_ORIG   c.z = a.z * b.z;*/
  c.z() = a.z() * b.z();
/* DPCT_ORIG   c.w = a.w * b.w;*/
  c.w() = a.w() * b.w();
  return c;
}

/* DPCT_ORIG template<>
inline __device__ float4 mul(float a, float4 b) {*/
template <> inline sycl::float4 mul(float a, sycl::float4 b) {
/* DPCT_ORIG   float4 c;*/
  sycl::float4 c;
/* DPCT_ORIG   c.x = a * b.x;*/
  c.x() = a * b.x();
/* DPCT_ORIG   c.y = a * b.y;*/
  c.y() = a * b.y();
/* DPCT_ORIG   c.z = a * b.z;*/
  c.z() = a * b.z();
/* DPCT_ORIG   c.w = a * b.w;*/
  c.w() = a * b.w();
  return c;
}

// Vector fused multiply-add.
/* DPCT_ORIG inline __device__ float fma(float a, float b, float c) {*/
inline float fma(float a, float b, float c) {
  return a * b + c;
}

/* DPCT_ORIG inline __device__ float2 fma(float2 a, float2 b, float2 c) {*/
inline sycl::float2 fma(sycl::float2 a, sycl::float2 b, sycl::float2 c) {
/* DPCT_ORIG   float2 d;*/
  sycl::float2 d;
/* DPCT_ORIG   d.x = fma(a.x, b.x, c.x);*/
  d.x() = fma(a.x(), b.x(), c.x());
/* DPCT_ORIG   d.y = fma(a.y, b.y, c.y);*/
  d.y() = fma(a.y(), b.y(), c.y());
  return d;
}

/* DPCT_ORIG inline __device__ float2 fma(float a, float2 b, float2 c) {*/
inline sycl::float2 fma(float a, sycl::float2 b, sycl::float2 c) {
/* DPCT_ORIG   float2 d;*/
  sycl::float2 d;
/* DPCT_ORIG   d.x = fma(a, b.x, c.x);*/
  d.x() = fma(a, b.x(), c.x());
/* DPCT_ORIG   d.y = fma(a, b.y, c.y);*/
  d.y() = fma(a, b.y(), c.y());
  return d;
}

/* DPCT_ORIG inline __device__ float4 fma(float4 a, float4 b, float4 c) {*/
inline sycl::float4 fma(sycl::float4 a, sycl::float4 b, sycl::float4 c) {
/* DPCT_ORIG   float4 d;*/
  sycl::float4 d;
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

/* DPCT_ORIG inline __device__ float4 fma(float a, float4 b, float4 c) {*/
inline sycl::float4 fma(float a, sycl::float4 b, sycl::float4 c) {
/* DPCT_ORIG   float4 d;*/
  sycl::float4 d;
/* DPCT_ORIG   d.x = fma(a, b.x, c.x);*/
  d.x() = fma(a, b.x(), c.x());
/* DPCT_ORIG   d.y = fma(a, b.y, c.y);*/
  d.y() = fma(a, b.y(), c.y());
/* DPCT_ORIG   d.z = fma(a, b.z, c.z);*/
  d.z() = fma(a, b.z(), c.z());
/* DPCT_ORIG   d.w = fma(a, b.w, c.w);*/
  d.w() = fma(a, b.w(), c.w());
  return d;
}

/* DPCT_ORIG inline __device__ Float4_ fma(float a, Float4_ b, Float4_ c) {*/
inline Float4_ fma(float a, Float4_ b, Float4_ c) {
  Float4_ d;
  d.x = fma(a, b.x, c.x);
  d.y = fma(a, b.y, c.y);
  return d;
}

/* DPCT_ORIG inline __device__ Float8_ fma(float a, Float8_ b, Float8_ c) {*/
inline Float8_ fma(float a, Float8_ b, Float8_ c) {
  Float8_ d;
  d.x = fma(a, b.x, c.x);
  d.y = fma(a, b.y, c.y);
  d.z = fma(a, b.z, c.z);
  d.w = fma(a, b.w, c.w);
  return d;
}

// Vector sum.
/* DPCT_ORIG template<>
inline __device__ float sum(float v) {*/
template <> inline float sum(float v) {
  return v;
}

/* DPCT_ORIG template<>
inline __device__ float sum(float2 v) {*/
template <> inline float sum(sycl::float2 v) {
/* DPCT_ORIG   return v.x + v.y;*/
  return v.x() + v.y();
}

/* DPCT_ORIG template<>
inline __device__ float sum(float4 v) {*/
template <> inline float sum(sycl::float4 v) {
/* DPCT_ORIG   return v.x + v.y + v.z + v.w;*/
  return v.x() + v.y() + v.z() + v.w();
}

/* DPCT_ORIG template<>
inline __device__ float sum(Float4_ v) {*/
template <> inline float sum(Float4_ v) {
/* DPCT_ORIG   return v.x.x + v.x.y + v.y.x + v.y.y;*/
  return v.x.x() + v.x.y() + v.y.x() + v.y.y();
}

/* DPCT_ORIG template<>
inline __device__ float sum(Float8_ v) {*/
template <> inline float sum(Float8_ v) {
/* DPCT_ORIG   return v.x.x + v.x.y + v.y.x + v.y.y + v.z.x + v.z.y + v.w.x +
 * v.w.y;*/
  return v.x.x() + v.x.y() + v.y.x() + v.y.y() + v.z.x() + v.z.y() + v.w.x() +
         v.w.y();
}

// Vector dot product.
/* DPCT_ORIG inline __device__ float dot(float a, float b) {*/
inline float dot(float a, float b) {
  return a * b;
}

/* DPCT_ORIG inline __device__ float dot(float2 a, float2 b) {*/
inline float dot(sycl::float2 a, sycl::float2 b) {
/* DPCT_ORIG   float2 c = mul<float2, float2, float2>(a, b);*/
  sycl::float2 c = mul<sycl::float2, sycl::float2, sycl::float2>(a, b);
/* DPCT_ORIG   return c.x + c.y;*/
  return c.x() + c.y();
}

/* DPCT_ORIG inline __device__ float dot(Float4_ a, Float4_ b) {*/
inline float dot(Float4_ a, Float4_ b) {
/* DPCT_ORIG   float2 acc = mul<float2, float2, float2>(a.x, b.x);*/
  sycl::float2 acc = mul<sycl::float2, sycl::float2, sycl::float2>(a.x, b.x);
  acc = fma(a.y, b.y, acc);
/* DPCT_ORIG   return acc.x + acc.y;*/
  return acc.x() + acc.y();
}

/* DPCT_ORIG inline __device__ float dot(Float8_ a, Float8_ b) {*/
inline float dot(Float8_ a, Float8_ b) {
/* DPCT_ORIG   float2 acc = mul<float2, float2, float2>(a.x, b.x);*/
  sycl::float2 acc = mul<sycl::float2, sycl::float2, sycl::float2>(a.x, b.x);
  acc = fma(a.y, b.y, acc);
  acc = fma(a.z, b.z, acc);
  acc = fma(a.w, b.w, acc);
/* DPCT_ORIG   return acc.x + acc.y;*/
  return acc.x() + acc.y();
}

// From float to float.
/* DPCT_ORIG inline __device__ void from_float(float& dst, float src) {*/
inline void from_float(float &dst, float src) {
  dst = src;
}

/* DPCT_ORIG inline __device__ void from_float(float2& dst, float2 src) {*/
inline void from_float(sycl::float2 &dst, sycl::float2 src) {
  dst = src;
}

/* DPCT_ORIG inline __device__ void from_float(float4& dst, float4 src) {*/
inline void from_float(sycl::float4 &dst, sycl::float4 src) {
  dst = src;
}

// From float to float.
/* DPCT_ORIG inline __device__ float to_float(float u) {*/
inline float to_float(float u) {
  return u;
}

/* DPCT_ORIG inline __device__ float2 to_float(float2 u) {*/
inline sycl::float2 to_float(sycl::float2 u) {
  return u;
}

/* DPCT_ORIG inline __device__ float4 to_float(float4 u) {*/
inline sycl::float4 to_float(sycl::float4 u) {
  return u;
}

/* DPCT_ORIG inline __device__ Float4_ to_float(Float4_ u) {*/
inline Float4_ to_float(Float4_ u) {
  return u;
}

/* DPCT_ORIG inline __device__ Float8_ to_float(Float8_ u) {*/
inline Float8_ to_float(Float8_ u) {
  return u;
}

// Zero-out a variable.
/* DPCT_ORIG inline __device__ void zero(float& dst) {*/
inline void zero(float &dst) {
  dst = 0.f;
}

} // namespace vllm
