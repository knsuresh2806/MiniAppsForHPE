#include "kernel_type2_small.h"

namespace kernel3_gpu_kernels {

__global__ void
kernel_loop4_derivatives(float* dfield1dx, float* dfield1dy, float* dfield1dz, float* dfield2dx, float* dfield2dy, float* dfield2dz,
                          const size_t xtot, const size_t ytot, const int izbeg)
{
    size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    size_t iy = blockIdx.y * blockDim.y + threadIdx.y;
    size_t iz = blockIdx.z;
    if (ix >= xtot || iy >= ytot || iz > izbeg) {
        return;
    }

    size_t global_index = (iz * ytot * xtot) + (iy * xtot) + ix;
    if (iz == izbeg) {
        dfield1dx[global_index] = 0.0f;
        dfield1dy[global_index] = 0.0f;
        dfield2dx[global_index] = 0.0f;
        dfield2dy[global_index] = 0.0f;

    } else {
        // Mirror x and y derivative
        // Odd symmetry
        // iz + izmirror = 2 * izbeg
        size_t mirror_index_xy_derivative = (2 * izbeg - iz) * (ytot * xtot) + (iy * xtot) + ix;
        dfield1dx[global_index] = -dfield1dx[mirror_index_xy_derivative];
        dfield1dy[global_index] = -dfield1dy[mirror_index_xy_derivative];
        dfield2dx[global_index] = -dfield2dx[mirror_index_xy_derivative];
        dfield2dy[global_index] = -dfield2dy[mirror_index_xy_derivative];
    }

    // Mirror z derivative
    // Even symmetry
    // iz + izmirror = 2 * izbeg + 1
    size_t mirror_index_z_derivative = (2 * izbeg + 1 - iz) * (ytot * xtot) + (iy * xtot) + ix;
    dfield1dz[global_index] = dfield1dz[mirror_index_z_derivative];
    dfield2dz[global_index] = dfield2dz[mirror_index_z_derivative];
}

__global__ void
update_Vp_gradient(volume_index idx, volume_index idx_adj, float* __restrict__ snap_field3, float* __restrict__ snap_field4,
                   float* __restrict__ grad_Vp, float* __restrict__ adj_field3, float* __restrict__ adj_field4, int ixbeg,
                   int ixend, int iybeg, int iyend, int izbeg, int izend)
{
    int ix = ixbeg + blockIdx.x * blockDim.x + threadIdx.x;
    int iy = iybeg + blockIdx.y * blockDim.y + threadIdx.y;
    int iz = izbeg + blockIdx.z * blockDim.z + threadIdx.z;

    if (ix > ixend || iy > iyend || iz > izend)
        return;

    float field30 = idx(snap_field3, ix - ixbeg, iy - iybeg, iz - izbeg);
    float field40 = idx(snap_field4, ix - ixbeg, iy - iybeg, iz - izbeg);

    idx(grad_Vp, ix - ixbeg, iy - iybeg, iz - izbeg) +=
        idx_adj(adj_field3, ix, iy, iz) * field30 + idx_adj(adj_field4, ix, iy, iz) * field40;
}

__global__ void
finalize_Vp_gradient(volume_index idx, volume_index model3_idx, float* __restrict__ model3_, float* __restrict__ grad_Vp_,
                     int ixbeg, int ixend, int iybeg, int iyend, int izbeg, int izend, const float dt_xcorr)
{
    int ix = ixbeg + blockIdx.x * blockDim.x + threadIdx.x;
    int iy = iybeg + blockIdx.y * blockDim.y + threadIdx.y;
    int iz = izbeg + blockIdx.z * blockDim.z + threadIdx.z;

    if (ix > ixend || iy > iyend || iz > izend)
        return;

    float vp = sqrt(model3_idx(model3_, ix, iy, iz));
    idx(grad_Vp_, ix - ixbeg, iy - iybeg, iz - izbeg) *= -dt_xcorr * 2.0 / vp;
}

}
