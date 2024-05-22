#pragma once

#include <cuda_runtime.h>
#include <vector_types.h>

void launch_update_rho_fwd_main_loop_2_1(
    float* field3, float* field4, float const* field1_dx, float const* field2_dx, float const* field1_dy, float const* field2_dy,
    float const* field1_dz, float const* field2_dz, float const* field1, float const* field2, float const* irho, float const* model1,
    float const* model2, float const* model3, float const* model4, float const* rx, float const* ry, float const* rz,
    float* pml_field1_xx, float* pml_field2_xx, float* pml_field1_yy, float* pml_field2_yy, float* pml_field1_zz, float* pml_field2_zz,
    float2 const* pml_ab_x, float2 const* pml_ab_y, float2 const* pml_ab_z, float2 const* sponge_ab_xx,
    float2 const* sponge_ab_yy, float2 const* sponge_ab_zz, int ixbeg, int ixend, int iybeg, int iyend, int izbeg, int izend,
    int ldimx, int ldimy, float dt, double dx, double dy, double dz, bool sponge_active, int order, bool pml_x,
    bool pml_y, bool pml_z, cudaStream_t stream);
