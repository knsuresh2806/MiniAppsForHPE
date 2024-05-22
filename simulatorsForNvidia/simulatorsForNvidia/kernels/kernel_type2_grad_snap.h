#ifndef KERNEL_TYPE2_GRAD_SNAP_H
#define KERNEL_TYPE2_GRAD_SNAP_H

#include <cuda_runtime.h>

namespace adj_kernel3_derivatives {
void launch_adj_kernel2_drv(float* dfield1dx, float* dfield2dx, float* dfield1dy, float* dfield2dy, float* dfield1dz, float* dfield2dz,
                           float* field3_rhs, float* field4_rhs, float* field1, float* field2, float* Model11, float* model1, float* model2,
                           float* model3, float* model4, float* model5, float* model6, float* model7, int ixbeg, int ixend, int iybeg,
                           int iyend, int izbeg, int izend, float invdxx, float invdyy, float invdzz, float invdxy,
                           float invdyz, float invdzx, int ldimx, int ldimy, int order, float dt, float invdx,
                           float invdy, float invdz, cudaStream_t stream);
}

#endif // KERNEL_TYPE2_GRAD_SNAP_H
