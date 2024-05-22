#ifndef _CUDA_UTILS
#define _CUDA_UTILS

#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <math_constants.h>
#include <cuda_profiler_api.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#define H2D cudaMemcpyHostToDevice
#define D2H cudaMemcpyDeviceToHost
#define D2D cudaMemcpyDeviceToDevice
#define DEF cudaMemcpyDefault
#define FILELINE(mess) std::cout << __FILE__ << ":" << __LINE__ << " " << mess << std::endl;

#define CUDA_TRY(func)                                                                                                 \
    {                                                                                                                  \
        cudaError_t e = func;                                                                                          \
        if (e != cudaSuccess) {                                                                                        \
            fprintf(stderr, "CUDA_TRY() failed at %s:%i : %s\n", __FILE__, __LINE__, cudaGetErrorString(e));           \
            std::abort();                                                                                              \
        }                                                                                                              \
    }

inline void
CUDA_CHECK_ERROR(const char* file, const int line)
{
    cudaError err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
        std::abort();
    }

    // More careful checking. However, this will affect performance.
    // Uncomment if needed for debugging, but do not commit uncommented.
    // err = cudaDeviceSynchronize();
    // if (cudaSuccess != err) {
    //     fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
    //     exit(-1);
    // }

    return;
}

// Check if the pointer is a true device pointer (from cudaMalloc)
inline bool
is_device_pointer(const void* ptr)
{
    cudaPointerAttributes attr;
    if (cudaPointerGetAttributes(&attr, ptr) == cudaErrorInvalidValue || attr.devicePointer == NULL)
        return false;
    if (attr.type == cudaMemoryTypeDevice)
        return true;
    return false;
}

#endif
