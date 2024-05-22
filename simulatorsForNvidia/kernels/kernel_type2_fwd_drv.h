#ifndef KERNEL_TYPE2_FWD_DRV_H
#define KERNEL_TYPE2_FWD_DRV_H

#include <cuda_runtime.h>

namespace kernel3_gpu_kernels {
void launch_fwd_kernel2_drv(const float* field1, const float* field2, const float* Model11, float* dfield1dx, float* dfield1dy,
                           float* dfield1dz, float* dfield2dx, float* dfield2dy, float* dfield2dz, const int ixbeg, const int ixend,
                           const int iybeg, const int iyend, const int izbeg, const int izend, const float invdx,
                           const float invdy, const float invdz, const int ldimx, const int ldimy, const int order,
                           cudaStream_t stream);
}

#endif // KERNEL_TYPE2_FWD_DRV_H
