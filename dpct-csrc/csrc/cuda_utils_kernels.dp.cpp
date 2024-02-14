#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#ifdef USE_ROCM
#include <hip/hip_runtime.h>
  #include <hip/hip_runtime_api.h>
#endif
int get_device_attribute(
    int attribute,
    int device_id)
{
    int device, value;
    if (device_id < 0) {
/* DPCT_ORIG         cudaGetDevice(&device);*/
        device = dpct::dev_mgr::instance().current_device_id();
    }
    else {
        device = device_id;
    }
/* DPCT_ORIG     cudaDeviceGetAttribute(&value,
 * static_cast<cudaDeviceAttr>(attribute), device);*/
    /*
    DPCT1076:84: The device attribute was not recognized. You may need to adjust
    the code.
    */
    cudaDeviceGetAttribute(&value, static_cast<int>(attribute), device);
    return value;
}


int get_max_shared_memory_per_block_device_attribute(
    int device_id)
{
int attribute;    
// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html
// cudaDevAttrMaxSharedMemoryPerBlockOptin = 97 if not is_hip() else 74

#ifdef USE_ROCM
    attribute = hipDeviceAttributeMaxSharedMemoryPerBlock;
#else
/* DPCT_ORIG     attribute = cudaDevAttrMaxSharedMemoryPerBlockOptin;*/
    attribute = 97;
#endif

    return get_device_attribute(attribute, device_id);
}
