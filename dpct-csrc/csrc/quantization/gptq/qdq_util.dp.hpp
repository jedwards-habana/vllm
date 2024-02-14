#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
/*
Copied from https://github.com/turboderp/exllamav2
*/

#ifndef _qdq_util_cuh
#define _qdq_util_cuh

namespace vllm {
namespace gptq {

union half2_uint32
{
    uint32_t as_uint32;
/* DPCT_ORIG     half2 as_half2;*/
    sycl::half2 as_half2;
/* DPCT_ORIG     __device__ half2_uint32(uint32_t val) : as_uint32(val) {}*/
    half2_uint32(uint32_t val) : as_uint32(val) {}
/* DPCT_ORIG     __device__ half2_uint32(half2 val) : as_half2(val) {}*/
    half2_uint32(sycl::half2 val) : as_half2(val) {}
};

union half_uint16
{
    uint16_t as_uint16;
/* DPCT_ORIG     half as_half;*/
    sycl::half as_half;
/* DPCT_ORIG     __device__ half_uint16(uint16_t val) : as_uint16(val) {}*/
    half_uint16(uint16_t val) : as_uint16(val) {}
/* DPCT_ORIG     __device__ half_uint16(half val) : as_half(val) {}*/
    half_uint16(sycl::half val) : as_half(val) {}
};

// Max_scale premultiplied by 1/256

/* DPCT_ORIG __forceinline__ __device__ half dq_scale(const int qs, const half
 * max_scale)*/
__dpct_inline__ sycl::half dq_scale(const int qs, const sycl::half max_scale)
{
    int qs_i = qs + 1;
/* DPCT_ORIG     half qs_h = __int2half_rn(qs_i * qs_i);*/
    sycl::half qs_h = sycl::vec<int, 1>{(qs_i * qs_i)}
                          .convert<sycl::half, sycl::rounding_mode::rte>()[0];
/* DPCT_ORIG     qs_h = __hmul(qs_h, max_scale);*/
    qs_h = qs_h * max_scale;
    return qs_h;
}

/* DPCT_ORIG __forceinline__ __device__ half dq(const int q, const int qzero,
 * const half scale)*/
__dpct_inline__ sycl::half dq(const int q, const int qzero,
                              const sycl::half scale)
{
/* DPCT_ORIG     return __hmul(__int2half_rn(q - qzero), scale);*/
    return sycl::vec<int, 1>{(q - qzero)}
               .convert<sycl::half, sycl::rounding_mode::rte>()[0] *
           scale;
}

/* DPCT_ORIG __forceinline__ __device__ half dq_ns(const int q, const int
 * qzero)*/
__dpct_inline__ sycl::half dq_ns(const int q, const int qzero)
{
    //return __hsub(__int2half_rn(q), __int2half_rn(qzero));
/* DPCT_ORIG     return __int2half_rn(q - qzero);*/
    return sycl::vec<int, 1>{(q - qzero)}
        .convert<sycl::half, sycl::rounding_mode::rte>()[0];
}

/* DPCT_ORIG __forceinline__ __device__ int exb(const uint32_t q, const int
 * shift, const int mask)*/
__dpct_inline__ int exb(const uint32_t q, const int shift, const int mask)
{
    return (int)((q >> shift) & mask);
}

/* DPCT_ORIG __forceinline__ __device__ int exb(const uint32_t q1, const
 * uint32_t q0, const int shift, const int mask)*/
__dpct_inline__ int exb(const uint32_t q1, const uint32_t q0, const int shift,
                        const int mask)
{
/* DPCT_ORIG     return (int)(__funnelshift_rc(q0, q1, shift) & mask);*/
    /*
    DPCT1098:61: The ((upsample(hi, lo) >> min(shift, 32)) & 0xFFFFFFFF)
    expression is used instead of the __funnelshift_rc call. These two
    expressions do not provide the exact same functionality. Check the generated
    code for potential precision and/or performance issues.
    */
    return (int)(((sycl::upsample<unsigned>(q1, q0) >> sycl::min(shift, 32)) &
                  0xFFFFFFFF) &
                 mask);
}

}  // namespace gptq
}  // namespace vllm
#endif
