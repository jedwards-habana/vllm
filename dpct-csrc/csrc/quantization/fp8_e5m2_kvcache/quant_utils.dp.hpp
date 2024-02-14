#pragma once

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <assert.h>
#include <stdint.h>
#include <float.h>
#include <type_traits>
#include "../../attention/attention_dtypes.h"
/* DPCT_ORIG #include "../../attention/dtype_float32.cuh"*/
#include "../../attention/dtype_float32.dp.hpp"
/* DPCT_ORIG #include "../../attention/dtype_float16.cuh"*/
#include "../../attention/dtype_float16.dp.hpp"
/* DPCT_ORIG #include "../../attention/dtype_bfloat16.cuh"*/
#include "../../attention/dtype_bfloat16.dp.hpp"

#pragma once

namespace vllm {
#ifdef ENABLE_FP8_E5M2
namespace fp8_e5m2_unscaled {

/* DPCT_ORIG template<typename Tout, typename Tin>
__inline__ __device__ Tout vec_conversion(const Tin& x)*/
template <typename Tout, typename Tin>
__inline__ Tout vec_conversion(const Tin &x)
{
    return x;
}

// fp8 -> half
/* DPCT_ORIG template<>
__inline__ __device__ uint16_t vec_conversion<uint16_t, uint8_t>(const uint8_t&
a)*/
template <>
__inline__ uint16_t vec_conversion<uint16_t, uint8_t>(const uint8_t &a)
{
/* DPCT_ORIG     __half_raw res = __nv_cvt_fp8_to_halfraw(a, __NV_E5M2);*/
    /*
    DPCT1007:55: Migration of __nv_cvt_fp8_to_halfraw is not supported.
    */
    uint16_t res = __nv_cvt_fp8_to_halfraw(a, __NV_E5M2);
/* DPCT_ORIG     return res.x;*/
    return res;
}

// fp8x2 -> half2
/* DPCT_ORIG template<>
__inline__ __device__ uint32_t vec_conversion<uint32_t, uint16_t>(const
uint16_t& a)*/
template <>
__inline__ uint32_t vec_conversion<uint32_t, uint16_t>(const uint16_t &a)
{
    union {
        uint16_t u16[2];
        uint32_t u32;
    } tmp;
    /*
    DPCT1007:56: Migration of __nv_cvt_fp8x2_to_halfraw2 is not supported.
    */
    __half2_raw res = __nv_cvt_fp8x2_to_halfraw2(a, __NV_E5M2);
    tmp.u16[0] = res.x;
    tmp.u16[1] = res.y;
    return tmp.u32;
}

// fp8x4 -> half2x2
/* DPCT_ORIG template<>
__inline__ __device__ uint2 vec_conversion<uint2, uint32_t>(const uint32_t& a)*/
template <>
__inline__ sycl::uint2 vec_conversion<uint2, uint32_t>(const uint32_t &a)
{
    union {
/* DPCT_ORIG         uint2    u32x2;*/
        sycl::uint2 u32x2{};
        uint32_t u32[2];
    } tmp;
    tmp.u32[0] = vec_conversion<uint32_t, uint16_t>((uint16_t)a);
    tmp.u32[1] = vec_conversion<uint32_t, uint16_t>((uint16_t)(a >> 16U));
    return tmp.u32x2;
}

// fp8x8 -> half2x4
/* DPCT_ORIG template<>
__inline__ __device__ uint4 vec_conversion<uint4, uint2>(const uint2& a)*/
template <>
__inline__ sycl::uint4 vec_conversion<uint4, uint2>(const sycl::uint2 &a)
{
    union {
/* DPCT_ORIG         uint4 u64x2;*/
        sycl::uint4 u64x2{};
/* DPCT_ORIG         uint2 u64[2];*/
        sycl::uint2 u64[2];
    } tmp;
/* DPCT_ORIG     tmp.u64[0] = vec_conversion<uint2, uint32_t>(a.x);*/
    tmp.u64[0] = vec_conversion<sycl::uint2, uint32_t>(a.x());
/* DPCT_ORIG     tmp.u64[1] = vec_conversion<uint2, uint32_t>(a.y);*/
    tmp.u64[1] = vec_conversion<sycl::uint2, uint32_t>(a.y());
    return tmp.u64x2;
}

// fp8 -> __nv_bfloat16
/* DPCT_ORIG template<>
__inline__ __device__ __nv_bfloat16 vec_conversion<__nv_bfloat16, uint8_t>(const
uint8_t& a)*/
template <>
__inline__ sycl::ext::oneapi::bfloat16
vec_conversion<__nv_bfloat16, uint8_t>(const uint8_t &a)
{
    // Note there is no direct convert function from fp8 to bf16.
    // fp8 -> half
/* DPCT_ORIG     __half_raw res = __nv_cvt_fp8_to_halfraw(a, __NV_E5M2);*/
    /*
    DPCT1007:57: Migration of __nv_cvt_fp8_to_halfraw is not supported.
    */
    uint16_t res = __nv_cvt_fp8_to_halfraw(a, __NV_E5M2);
    // half -> float -> bf16
/* DPCT_ORIG     float tmp = half_to_float(res.x);*/
    float tmp = half_to_float(res);
/* DPCT_ORIG     return __float2bfloat16(tmp);*/
    return sycl::ext::oneapi::bfloat16(tmp);
}

// fp8x2 -> __nv_bfloat162
/* DPCT_ORIG template<>
__inline__ __device__ __nv_bfloat162 vec_conversion<__nv_bfloat162,
uint16_t>(const uint16_t& a)*/
template <>
__inline__ sycl::marray<sycl::ext::oneapi::bfloat16, 2>
vec_conversion<__nv_bfloat162, uint16_t>(const uint16_t &a)
{
/* DPCT_ORIG     __nv_bfloat162 res;*/
    sycl::marray<sycl::ext::oneapi::bfloat16, 2> res;
/* DPCT_ORIG     res.x = vec_conversion<__nv_bfloat16, uint8_t>((uint8_t)a);*/
    res[0] = vec_conversion<sycl::ext::oneapi::bfloat16, uint8_t>((uint8_t)a);
/* DPCT_ORIG     res.y = vec_conversion<__nv_bfloat16, uint8_t>((uint8_t)(a >>
 * 8U));*/
    res[1] = vec_conversion<sycl::ext::oneapi::bfloat16, uint8_t>(
        (uint8_t)(a >> 8U));
    return res;
}

// fp8x4 -> bf16_4_t
/* DPCT_ORIG template<>
__inline__ __device__ bf16_4_t vec_conversion<bf16_4_t, uint32_t>(const
uint32_t& a)*/
template <>
__inline__ bf16_4_t vec_conversion<bf16_4_t, uint32_t>(const uint32_t &a)
{
    bf16_4_t res;
/* DPCT_ORIG     res.x = vec_conversion<__nv_bfloat162,
 * uint16_t>((uint16_t)a);*/
    res.x =
        vec_conversion<sycl::marray<sycl::ext::oneapi::bfloat16, 2>, uint16_t>(
            (uint16_t)a);
/* DPCT_ORIG     res.y = vec_conversion<__nv_bfloat162, uint16_t>((uint16_t)(a
 * >> 16U));*/
    res.y =
        vec_conversion<sycl::marray<sycl::ext::oneapi::bfloat16, 2>, uint16_t>(
            (uint16_t)(a >> 16U));
    return res;
}

// fp8x8 -> bf16_8_t
/* DPCT_ORIG template<>
__inline__ __device__ bf16_8_t vec_conversion<bf16_8_t, uint2>(const uint2& a)*/
template <>
__inline__ bf16_8_t vec_conversion<bf16_8_t, uint2>(const sycl::uint2 &a)
{
    bf16_4_t tmp1, tmp2;
/* DPCT_ORIG     tmp1 = vec_conversion<bf16_4_t, uint32_t>(a.x);*/
    tmp1 = vec_conversion<bf16_4_t, uint32_t>(a.x());
/* DPCT_ORIG     tmp2 = vec_conversion<bf16_4_t, uint32_t>(a.y);*/
    tmp2 = vec_conversion<bf16_4_t, uint32_t>(a.y());
    bf16_8_t res;
    res.x = tmp1.x;
    res.y = tmp1.y;
    res.z = tmp2.x;
    res.w = tmp2.y;
    return res;
}

// fp8 -> float
/* DPCT_ORIG template<>
__inline__ __device__ float vec_conversion<float, uint8_t>(const uint8_t& a)*/
template <> __inline__ float vec_conversion<float, uint8_t>(const uint8_t &a)
{
    // fp8 -> half
    uint16_t tmp = vec_conversion<uint16_t, uint8_t>(a);
    // half -> float
    return half_to_float(tmp);
}

// fp8x2 -> float2
/* DPCT_ORIG template<>
__inline__ __device__ float2 vec_conversion<float2, uint16_t>(const uint16_t&
a)*/
template <>
__inline__ sycl::float2 vec_conversion<float2, uint16_t>(const uint16_t &a)
{
    // fp8x2 -> half2
    uint32_t tmp = vec_conversion<uint32_t, uint16_t>(a);
    // half2 -> float2
    return half2_to_float2(tmp);
}

// fp8x4 -> float4
/* DPCT_ORIG template<>
__inline__ __device__ Float4_ vec_conversion<Float4_, uint32_t>(const uint32_t&
a)*/
template <>
__inline__ Float4_ vec_conversion<Float4_, uint32_t>(const uint32_t &a)
{
    Float4_ res;
/* DPCT_ORIG     res.x = vec_conversion<float2, uint16_t>((uint16_t)a);*/
    res.x = vec_conversion<sycl::float2, uint16_t>((uint16_t)a);
/* DPCT_ORIG     res.y = vec_conversion<float2, uint16_t>((uint16_t)(a >>
 * 16U));*/
    res.y = vec_conversion<sycl::float2, uint16_t>((uint16_t)(a >> 16U));
    return res;
}

// fp8x8 -> float8
/* DPCT_ORIG template<>
__inline__ __device__ Float8_ vec_conversion<Float8_, uint2>(const uint2& a)*/
template <>
__inline__ Float8_ vec_conversion<Float8_, uint2>(const sycl::uint2 &a)
{
    Float4_ tmp1, tmp2;
/* DPCT_ORIG     tmp1 = vec_conversion<Float4_, uint32_t>(a.x);*/
    tmp1 = vec_conversion<Float4_, uint32_t>(a.x());
/* DPCT_ORIG     tmp2 = vec_conversion<Float4_, uint32_t>(a.y);*/
    tmp2 = vec_conversion<Float4_, uint32_t>(a.y());
    Float8_ res;
    res.x = tmp1.x;
    res.y = tmp1.y;
    res.z = tmp2.x;
    res.w = tmp2.y;
    return res;
}


// half -> fp8
/* DPCT_ORIG template<>
__inline__ __device__ uint8_t vec_conversion<uint8_t, uint16_t>(const uint16_t&
a)*/
template <>
__inline__ uint8_t vec_conversion<uint8_t, uint16_t>(const uint16_t &a)
{
/* DPCT_ORIG     __half_raw tmp;*/
    uint16_t tmp;
/* DPCT_ORIG     tmp.x = a;*/
    tmp = a;
/* DPCT_ORIG     __nv_fp8_storage_t res = __nv_cvt_halfraw_to_fp8(tmp,
 * __NV_SATFINITE, __NV_E5M2);*/
    /*
    DPCT1007:58: Migration of __nv_cvt_halfraw_to_fp8 is not supported.
    */
    __nv_fp8_storage_t res = __nv_cvt_halfraw_to_fp8(
        sycl::bit_cast<sycl::half>(tmp), __NV_SATFINITE, __NV_E5M2);
    return (uint8_t)res;
}

// bf16 -> fp8
/* DPCT_ORIG template<>
__inline__ __device__ uint8_t vec_conversion<uint8_t, __nv_bfloat16>(const
__nv_bfloat16& a)*/
template <>
__inline__ uint8_t
vec_conversion<uint8_t, __nv_bfloat16>(const sycl::ext::oneapi::bfloat16 &a)
{
/* DPCT_ORIG #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800*/
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP < 800
    /*
    DPCT1007:59: Migration of __assert_fail is not supported.
    */
    assert(false);
#else
    __nv_fp8_storage_t res = __nv_cvt_bfloat16raw_to_fp8(__nv_bfloat16_raw(a), __NV_SATFINITE, __NV_E5M2);
    return (uint8_t)res;
#endif
}

// float -> fp8
/* DPCT_ORIG template<>
__inline__ __device__ uint8_t vec_conversion<uint8_t, float>(const float& a)*/
template <> __inline__ uint8_t vec_conversion<uint8_t, float>(const float &a)
{
    /*
    DPCT1007:60: Migration of __nv_cvt_float_to_fp8 is not supported.
    */
    __nv_fp8_storage_t res =
        __nv_cvt_float_to_fp8(a, __NV_SATFINITE, __NV_E5M2);
    return (uint8_t)res;
}

// fp8x4 -> float4
/* DPCT_ORIG template<>
__inline__ __device__ float4 vec_conversion<float4, uint32_t>(const uint32_t&
a)*/
template <>
__inline__ sycl::float4 vec_conversion<float4, uint32_t>(const uint32_t &a)
{
    Float4_ tmp = vec_conversion<Float4_, uint32_t>(a);
/* DPCT_ORIG     float4 res = make_float4(tmp.x.x, tmp.x.y, tmp.y.x, tmp.y.y);*/
    sycl::float4 res = sycl::float4(tmp.x.x(), tmp.x.y(), tmp.y.x(), tmp.y.y());
    return res;
}

/* DPCT_ORIG template<>
__inline__ __device__ uint32_t vec_conversion<uint32_t, float2>(const float2&
a)*/
template <>
__inline__ uint32_t vec_conversion<uint32_t, float2>(const sycl::float2 &a)
{
    union {
/* DPCT_ORIG         half2    float16;*/
        sycl::half2 float16;
        uint32_t uint32;
    };

/* DPCT_ORIG     float16 = __float22half2_rn(a);*/
    float16 = a.convert<sycl::half, sycl::rounding_mode::rte>();
    return uint32;
}

/* DPCT_ORIG template<>
__inline__ __device__ uint2 vec_conversion<uint2, Float4_>(const Float4_& a)*/
template <>
__inline__ sycl::uint2 vec_conversion<uint2, Float4_>(const Float4_ &a)
{
/* DPCT_ORIG     uint2  b;*/
    sycl::uint2 b;
/* DPCT_ORIG     float2 val;*/
    sycl::float2 val;
/* DPCT_ORIG     val.x = a.x.x;*/
    val.x() = a.x.x();
/* DPCT_ORIG     val.y = a.x.y;*/
    val.y() = a.x.y();
/* DPCT_ORIG     b.x   = vec_conversion<uint32_t, float2>(val);*/
    b.x() = vec_conversion<uint32_t, sycl::float2>(val);

/* DPCT_ORIG     val.x = a.y.x;*/
    val.x() = a.y.x();
/* DPCT_ORIG     val.y = a.y.y;*/
    val.y() = a.y.y();
/* DPCT_ORIG     b.y   = vec_conversion<uint32_t, float2>(val);*/
    b.y() = vec_conversion<uint32_t, sycl::float2>(val);

    return b;
}

/* DPCT_ORIG template<>
__inline__ __device__ float4 vec_conversion<float4, Float4_>(const Float4_& a)*/
template <>
__inline__ sycl::float4 vec_conversion<float4, Float4_>(const Float4_ &a)
{
/* DPCT_ORIG     float4 b;*/
    sycl::float4 b;
/* DPCT_ORIG     b.x = a.x.x;*/
    b.x() = a.x.x();
/* DPCT_ORIG     b.y = a.x.y;*/
    b.y() = a.x.y();
/* DPCT_ORIG     b.z = a.y.x;*/
    b.z() = a.y.x();
/* DPCT_ORIG     b.w = a.y.y;*/
    b.w() = a.y.y();
    return b;
}

/* DPCT_ORIG template<>
__inline__ __device__ uint4 vec_conversion<uint4, Float8_>(const Float8_& a)*/
template <>
__inline__ sycl::uint4 vec_conversion<uint4, Float8_>(const Float8_ &a)
{
/* DPCT_ORIG     uint4 b;*/
    sycl::uint4 b;
/* DPCT_ORIG     b.x = vec_conversion<uint32_t, float2>(a.x);*/
    b.x() = vec_conversion<uint32_t, sycl::float2>(a.x);
/* DPCT_ORIG     b.y = vec_conversion<uint32_t, float2>(a.y);*/
    b.y() = vec_conversion<uint32_t, sycl::float2>(a.y);
/* DPCT_ORIG     b.z = vec_conversion<uint32_t, float2>(a.z);*/
    b.z() = vec_conversion<uint32_t, sycl::float2>(a.z);
/* DPCT_ORIG     b.w = vec_conversion<uint32_t, float2>(a.w);*/
    b.w() = vec_conversion<uint32_t, sycl::float2>(a.w);
    return b;
}

/* DPCT_ORIG template<>
__inline__ __device__ __nv_bfloat162 vec_conversion<__nv_bfloat162,
float2>(const float2 &a) {*/
template <>
__inline__ sycl::marray<sycl::ext::oneapi::bfloat16, 2>
vec_conversion<__nv_bfloat162, float2>(const sycl::float2 &a) {
/* DPCT_ORIG     __nv_bfloat162 b;*/
    sycl::marray<sycl::ext::oneapi::bfloat16, 2> b;
    from_float(b, a);
    return b;
}

/* DPCT_ORIG template<>
__inline__ __device__ bf16_4_t vec_conversion<bf16_4_t, Float4_>(const Float4_
&a) {*/
template <>
__inline__ bf16_4_t vec_conversion<bf16_4_t, Float4_>(const Float4_ &a) {
    bf16_4_t b;
    from_float(b, a);
    return b;
}

/* DPCT_ORIG template<>
__inline__ __device__ bf16_8_t vec_conversion<bf16_8_t, Float8_>(const Float8_
&a) {*/
template <>
__inline__ bf16_8_t vec_conversion<bf16_8_t, Float8_>(const Float8_ &a) {
    bf16_8_t b;
    from_float(b, a);
    return b;
}

} // namespace fp8_e5m2_unscaled
#endif // ENABLE_FP8_E5M2
} // namespace vllm
