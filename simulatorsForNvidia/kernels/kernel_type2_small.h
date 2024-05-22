#ifndef KERNEL_TYPE2_SMALL
#define KERNEL_TYPE2_SMALL

#include "volume_index.h"

namespace kernel3_gpu_kernels {

__global__ void kernel_loop4_derivatives(float* dfield1dx, float* dfield1dy, float* dfield1dz, float* dfield2dx, float* dfield2dy,
                                          float* dfield2dz, const size_t xtot, const size_t ytot, const int izbeg);

__global__ void update_Vp_gradient(volume_index idx, volume_index idx_adj, float* __restrict__ snap_field3,
                                   float* __restrict__ snap_field4, float* __restrict__ grad_Vp, float* __restrict__ adj_field3,
                                   float* __restrict__ adj_field4, int ixbeg, int ixend, int iybeg, int iyend, int izbeg,
                                   int izend);

__global__ void finalize_Vp_gradient(volume_index idx, volume_index model3_idx, float* __restrict__ model3_,
                                     float* __restrict__ grad_Vp_, int ixbeg, int ixend, int iybeg, int iyend,
                                     int izbeg, int izend, const float dt_xcorr);

}

#endif