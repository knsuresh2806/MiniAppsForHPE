#pragma once

#include <cuda_runtime.h>
#include <vector_types.h>

namespace kernel3_gpu_kernels {

void launch_update_adj_kernel_main_3(float* field3, float* field4, float const* field1_dx, float const* field2_dx,
                                          float const* field1_dy, float const* field2_dy, float const* field1_dz,
                                          float const* field2_dz, float const* field3_rhs, float const* field4_rhs,
                                          float2 const* ab_xx, float2 const* ab_yy, float2 const* ab_zz, int ixbeg,
                                          int ixend, int iybeg, int iyend, int izbeg, int izend, int ldimx, int ldimy,
                                          float dt, double dx, double dy, double dz, bool pmlx, bool pmly, bool pmlz,
                                          int order, bool sponge_active, cudaStream_t stream = 0);

}
