#pragma once

#include <cuda_runtime.h>
#include <vector_types.h>
//------------------------------------------------------------------------------
// clang-format off
//------------------------------------------------------------------------------
//
// Author: Igor.
//
// Launches variable density rho-gradient kernel.
// Can be configured to launch either the simple or the optimized gradient kernel.
// Updates grad in [ixbeg, ixend] x [iybeg, iyend] x [izbeg, izend] range.
//
// Array arguments point to:
//   - grid point = -1/2 along staggered axis A.
//   - grid point = 0 along regular axis A.
// E.g., field1 points to {0, 0, 0}, fwd_field1_dy points to {0, -1/2, 0}.
//
// Currently, order must be a multiple of 8 (i.e., 8 or 16).
// With minor code adjustments, could use any order (2, 4, ..., 14, 16).
void launch_update_rho_gradient_rho (
    float* grad,
    float const* fwd_field1_dx, float const* fwd_field2_dx,
    float const* fwd_field1_dy, float const* fwd_field2_dy,
    float const* fwd_field1_dz, float const* fwd_field2_dz,
    float const* field1, float const*  field2, float const* irho,
    float const* model1, float const* model2, float const* model3, float const* model4,
    float const* rx, float const* ry, float const* rz,
    int ixbeg, int ixend, int iybeg, int iyend, int izbeg, int izend,
    int ldimx, int ldimy, int grad_ldimx, int grad_ldimy,
    double dx, double dy, double dz, int order, bool simple_gradient, cudaStream_t stream = 0);
//------------------------------------------------------------------------------
// clang-format on
