
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

#include "cart_volume_gpu_kernels.h"

namespace cg = cooperative_groups;

// GTC
// set kernel, working on 4 elements per thread, using 32 x 32 threads
// Assuming that data is float4 aligned.
__global__ __launch_bounds__(1024) void set(float* __restrict__ data, int start0, int n0, int n1, int stride1,
                                            int stride2, float value)
{
    int i0 = 4 * (blockIdx.x * 32 + threadIdx.x); // 4 elements per thread
    int i1 = blockIdx.y * 32 + threadIdx.y;
    int i2 = blockIdx.z;
    int index = i2 * stride2 + i1 * stride1 + i0;
    float4* data_vec4 = reinterpret_cast<float4*>(data + index);

    // Get rid of the threads which are out of bounds
    if (i0 + 3 < start0 || i0 >= start0 + n0 || i1 >= n1)
        return;

    // Which components of the float4 are active
    bool active0 = i0 >= start0 && i0 < start0 + n0;
    bool active1 = i0 + 1 >= start0 && i0 + 1 < start0 + n0;
    bool active2 = i0 + 2 >= start0 && i0 + 2 < start0 + n0;
    bool active3 = i0 + 3 >= start0 && i0 + 3 < start0 + n0;

    // If all elements are active, store as one float4
    // Otherwise, store the active values individually
    if (active0 && active1 && active2 && active3)
        *data_vec4 = make_float4(value, value, value, value);
    else {
        if (active0)
            data[index + 0] = value;
        if (active1)
            data[index + 1] = value;
        if (active2)
            data[index + 2] = value;
        if (active3)
            data[index + 3] = value;
    }
}

// set kernel, using 128 x 1 threads.  No alignment requirement
__global__ __launch_bounds__(128) void set_unaligned(float* __restrict__ data, int n0, int n1, int stride1, int stride2,
                                                     float value)
{
    int i0 = blockIdx.x * 128 + threadIdx.x;
    int i1 = blockIdx.y;
    int i2 = blockIdx.z;

    // Get rid of the threads which are out of bounds
    if (i0 >= n0) {
        return;
    }

    int index = i2 * stride2 + i1 * stride1 + i0;
    data[index] = value;
}

// Kernel to unpack cudaMalloced plan2d to destination
// The thread is fixed in calling code with (128,1,1)
// The pointers are pointing to the beginning of data to be copied
// The nx is number of floats need copy in x axis,
// The ldimx_gpu and ldimx_plan2d are the length of x axis for each data
__global__ void
unpack_2d(float* gpu_plan, float* plan2d, int nx, int ldimx_gpu, int ldimx_plan2d)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y;
    if (ix >= nx)
        return;
    gpu_plan[iy * ldimx_gpu + ix] = plan2d[iy * ldimx_plan2d + ix];
}

// Kernel to pack source data to cudaMalloced plan2d (dim x and y)
// The thread is fixed in calling code with (128,1,1)
// The pointers are pointing to the beginning of data to be copied
// The nx is number of floats need copy in x dimension,
// The ldimx_gpu and ldimx_plan2d are the length of x dimension for each data
__global__ void
pack_2d(float* plan2d, float* gpu_plan, int nx, int ldimx_plan2d, int ldimx_gpu)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y;
    if (ix >= nx)
        return;
    plan2d[iy * ldimx_plan2d + ix] = gpu_plan[iy * ldimx_gpu + ix];
}

// Kernel to copy data from another cartVolume, changing the stride if necessary
// This is used in halo copy, as well as for the copyFrom function.
// The thread block shape is tuned based on the type of copy
// The pointers are pointing to the beginning of the exchange zones
__global__ __launch_bounds__(1024) void copy_and_restride_data(float* __restrict__ dest, float* __restrict__ src,
                                                               int n0, int n1, int n2, int dest_ldim0, int dest_ldim1,
                                                               int src_ldim0, int src_ldim1)
{
    int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    int i1 = blockIdx.y * blockDim.y + threadIdx.y;
    int i2 = blockIdx.z * blockDim.z + threadIdx.z;
    if (i0 >= n0 || i1 >= n1 || i2 >= n2)
        return;
    int dest_index = (i2 * dest_ldim1 + i1) * dest_ldim0 + i0;
    int src_index = (i2 * src_ldim1 + i1) * src_ldim0 + i0;
    dest[dest_index] = src[src_index];
}
