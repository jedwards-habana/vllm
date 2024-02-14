#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
/*
Copied from https://github.com/turboderp/exllamav2
*/

#ifndef _compat_cuh
#define _compat_cuh

namespace vllm {
namespace gptq {
// atomicAdd for half types, to support CC < 7.x

/* DPCT_ORIG __device__ __forceinline__ void atomicAdd_half(half* address, half
 * val)*/
__dpct_inline__ void atomicAdd_half(sycl::half *address, sycl::half val)
{
    unsigned int * address_as_ui = (unsigned int *) ((char *)address - ((size_t)address & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;

    do
    {
        assumed = old;
/* DPCT_ORIG         __half_raw hsum;*/
        uint16_t hsum;
/* DPCT_ORIG         hsum.x = (size_t)address & 2 ? (old >> 16) : (old &
 * 0xffff);*/
        hsum = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
/* DPCT_ORIG         half tmpres = __hadd(hsum, val);*/
        sycl::half tmpres = hsum + val;
/* DPCT_ORIG         hsum = __half_raw(tmpres);*/
        hsum = uint16_t(tmpres);
/* DPCT_ORIG         old = (size_t)address & 2 ? (old & 0xffff) | (hsum.x << 16)
 * : (old & 0xffff0000) | hsum.x;*/
        old = (size_t)address & 2 ? (old & 0xffff) | (hsum << 16)
                                  : (old & 0xffff0000) | hsum;
/* DPCT_ORIG         old = atomicCAS(address_as_ui, assumed, old);*/
        old = dpct::atomic_compare_exchange_strong<
            sycl::access::address_space::generic_space>(address_as_ui, assumed,
                                                        old);
    }
    while (assumed != old);
}

// atomicAdd for half2 types

/* DPCT_ORIG __device__ __forceinline__ void atomicAdd_half2(half2* address,
 * half2 val)*/
__dpct_inline__ void atomicAdd_half2(sycl::half2 *address, sycl::half2 val)
{
    unsigned int* address_as_ui = (unsigned int*)address;
    unsigned int old = *address_as_ui;
    unsigned int assumed;
    do
    {
        assumed = old;
/* DPCT_ORIG         half2 old_val = *((half2*)&old);*/
        sycl::half2 old_val = *((sycl::half2 *)&old);
/* DPCT_ORIG         half2 new_val = __hadd2(old_val, val);*/
        sycl::half2 new_val = old_val + val;
/* DPCT_ORIG         old = atomicCAS(address_as_ui, assumed, *((unsigned
 * int*)&new_val));*/
        old = dpct::atomic_compare_exchange_strong<
            sycl::access::address_space::generic_space>(
            address_as_ui, assumed, *((unsigned int *)&new_val));
    }
    while (assumed != old);
}

//

/* DPCT_ORIG #if defined(__CUDA_ARCH__) || defined(USE_ROCM)*/
#if defined(DPCT_COMPATIBILITY_TEMP) || defined(USE_ROCM)
/* DPCT_ORIG #if __CUDA_ARCH__ < 700 || defined(USE_ROCM)*/
#if DPCT_COMPATIBILITY_TEMP < 700 || defined(USE_ROCM)

/* DPCT_ORIG __device__ __forceinline__ void atomicAdd(half* address, half val)
 * { atomicAdd_half(address, val); }*/
__dpct_inline__ void atomicAdd(sycl::half *address, sycl::half val) {
    atomicAdd_half(address, val);
}

/* DPCT_ORIG #if __CUDA_ARCH__ < 600 || defined(USE_ROCM)*/
#if DPCT_COMPATIBILITY_TEMP < 600 || defined(USE_ROCM)
__device__ __forceinline__ void atomicAdd(half2* address, half2 val) { atomicAdd_half2(address, val); }
#endif

#endif
#endif

}  // namespace gptq
}  // namespace vllm
#endif
