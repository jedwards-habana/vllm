#pragma once

/* DPCT_ORIG #include "attention_generic.cuh"*/
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "attention_generic.dp.hpp"

#include <stdint.h>
#ifdef ENABLE_FP8_E5M2
/* DPCT_ORIG #include <cuda_fp8.h>*/
#endif

namespace vllm {
#ifdef ENABLE_FP8_E5M2
// fp8 vector types for quantization of kv cache

template<>
struct Vec<uint8_t, 1> {
    using Type = uint8_t;
};

template<>
struct Vec<uint8_t, 2> {
    using Type = uint16_t;
};

template<>
struct Vec<uint8_t, 4> {
    using Type = uint32_t;
};

template<>
struct Vec<uint8_t, 8> {
/* DPCT_ORIG     using Type = uint2;*/
    using Type = sycl::uint2;
};
#endif // ENABLE_FP8_E5M2

} // namespace vllm
