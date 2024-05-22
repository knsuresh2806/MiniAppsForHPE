#ifndef _CART_VOLUME_GPU_KERNELS
#define _CART_VOLUME_GPU_KERNELS

#include "cuda_utils.h"
#include "cart_volume.h"
#include "axis.h"
#include "std_const.h"
#include "emsl_error.h"
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <new>
#include <stdint.h>
#include <float.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

// set kernel, working on 4 elements per thread, using 32 x 32 threads
// Assuming that data is float4 aligned.
__global__ __launch_bounds__(1024) void set(float* __restrict__ data, int start0, int n0, int n1, int stride1,
                                            int stride2, float value);

// set kernel, using 128 x 1 threads.  No alignment requirement
__global__ __launch_bounds__(128) void set_unaligned(float* __restrict__ data, int n0, int n1, int stride1, int stride2,
                                                     float value);

// Kernel to unpack cudaMalloced plan2d to destination
// The thread is fixed in calling code with (128,1,1)
// The pointers are pointing to the beginning of data to be copied
// The nx is number of floats need copy in x axis,
// The ldimx_gpu and ldimx_plan2d are the length of x axis for each data
__global__ void unpack_2d(float* gpu_plan, float* plan2d, int nx, int ldimx_gpu, int ldimx_plan2d);

// Kernel to pack source data to cudaMalloced plan2d (dim x and y)
// The thread is fixed in calling code with (128,1,1)
// The pointers are pointing to the beginning of data to be copied
// The nx is number of floats need copy in x dimension,
// The ldimx_gpu and ldimx_plan2d are the length of x dimension for each data
__global__ void pack_2d(float* plan2d, float* gpu_plan, int nx, int ldimx_plan2d, int ldimx_gpu);

// Kernel to copy data from another cartVolume, changing the stride if necessary
// This is used in halo copy, as well as for the copyFrom function.
// The thread block shape is tuned based on the type of copy
// The pointers are pointing to the beginning of the exchange zones
__global__ __launch_bounds__(1024) void copy_and_restride_data(float* __restrict__ dest, float* __restrict__ src,
                                                               int n0, int n1, int n2, int dest_ldim0, int dest_ldim1,
                                                               int src_ldim0, int src_ldim1);

#endif //_CART_VOLUME_GPU_KERNELS
