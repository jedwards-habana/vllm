#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#pragma once

#ifndef USE_ROCM
/* DPCT_ORIG   #define VLLM_LDG(arg) __ldg(arg)*/
  /*
DPCT1098:82: The '*' expression is used instead of the __ldg call. These two
expressions do not provide the exact same functionality. Check the generated
code for potential precision and/or performance issues.
*/
  /*
DPCT1064:83: Migrated __ldg call is used in a macro/template definition and may
not be valid for all macro/template uses. Adjust the code.
*/
#define VLLM_LDG(arg) *(sin_ptr + x_index / 2)
#else
  #define VLLM_LDG(arg) *(arg)
#endif

#ifndef USE_ROCM
/* DPCT_ORIG   #define VLLM_SHFL_XOR_SYNC(var, lane_mask)
 * __shfl_xor_sync(uint32_t(-1), var, lane_mask)*/
  /*
DPCT1023:32: The SYCL sub-group does not support mask options for
dpct::permute_sub_group_by_xor. You can specify
"--use-experimental-features=masked-sub-group-operation" to use the experimental
helper function to migrate __shfl_xor_sync.
*/
#define VLLM_SHFL_XOR_SYNC(var, lane_mask)                                     \
    dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), var, lane_mask)
#else
  #define VLLM_SHFL_XOR_SYNC(var, lane_mask) __shfl_xor(var, lane_mask)
#endif

#ifndef USE_ROCM
/* DPCT_ORIG   #define VLLM_SHFL_SYNC(var, src_lane) __shfl_sync(uint32_t(-1),
 * var, src_lane)*/
  /*
DPCT1023:33: The SYCL sub-group does not support mask options for
dpct::select_from_sub_group. You can specify
"--use-experimental-features=masked-sub-group-operation" to use the experimental
helper function to migrate __shfl_sync.
*/
#define VLLM_SHFL_SYNC(var, src_lane)                                          \
    dpct::select_from_sub_group(item_ct1.get_sub_group(), var, src_lane)
#else
  #define VLLM_SHFL_SYNC(var, src_lane) __shfl(var, src_lane)
#endif

#ifndef USE_ROCM
  /*
DPCT1027:81: The call to cudaFuncSetAttribute was replaced with 0 because SYCL
currently does not support corresponding setting.
*/
#define VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(FUNC, VAL)        \
    /* DPCT_ORIG     cudaFuncSetAttribute(FUNC,                                \
     * cudaFuncAttributeMaxDynamicSharedMemorySize, VAL)*/                     \
    0
#else
  #define VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(FUNC, VAL) \
    hipFuncSetAttribute(FUNC, hipFuncAttributeMaxDynamicSharedMemorySize, VAL)
#endif

