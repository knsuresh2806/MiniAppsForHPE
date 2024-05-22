#include <cuda_pipeline.h>
#include "kernel_type2_fwd.h"
#include "helper_kernels_gpu.h"
#include "cuda_utils.h"
#include "kernel_utils.h"
#include <cassert>

namespace {

template <int order, int bx, int by>
struct smstruct_base
{
    float dfield1dx[by][bx + order];
    float dfield2dx[by][bx + order];
    float dfield1dy[by + order][bx];
    float dfield2dy[by + order][bx];
    float dfield1dz[by][bx];
    float dfield2dz[by][bx];
    float model1[by][bx];
    float model2[by][bx];
    float model3[by][bx];
    float model4[by][bx];
    float irho[by][bx];
};

template <int order, int bx, int by, bool pml_any>
struct smstruct_interior
{
    float field1[by + order][bx + order];
    float field2[by + order][bx + order];
    float rx[by][bx];
    float ry[by][bx];
    float rz[by][bx];
};

template <int order, int bx, int by>
struct smstruct_interior<order, bx, by, true>
{};

template <int order, int bx, int by, bool pml_any>
struct smstruct : public smstruct_base<order, bx, by>, public smstruct_interior<order, bx, by, pml_any>
{};

// Load the shared memory struct.
// load_main=false  loads only the values needed for the priming loop (field1/field2 for mixed derivatives)
// load_main=true   loads for "main loop" rather than priming loop
template <bool load_main, int order, int bx, int by, bool pml_any>
__device__ void
load_xy_planes(smstruct<order, bx, by, pml_any>& sm, float const* field1_dx, float const* field2_dx, float const* field1_dy,
               float const* field2_dy, float const* field1_dz, float const* field2_dz, float const* field1, float const* field2,
               float const* irho, float const* model1, float const* model2, float const* model3, float const* model4,
               float const* rx, float const* ry, float const* rz, int index_load_x, int index_load_y, int index_load_xy,
               int index_load_nohalo, int index_load_nohalo_z_offset, bool loader_x, bool loader_y, bool loader_xy,
               bool loader_nohalo, int txload, int tyload)
{
    if constexpr (load_main) {
        if (loader_x) {
            __pipeline_memcpy_async(&sm.dfield1dx[tyload][txload], &field1_dx[index_load_x], sizeof(float4));
            __pipeline_memcpy_async(&sm.dfield2dx[tyload][txload], &field2_dx[index_load_x], sizeof(float4));
        }

        if (loader_y) {
            __pipeline_memcpy_async(&sm.dfield1dy[tyload][txload], &field1_dy[index_load_y], sizeof(float4));
            __pipeline_memcpy_async(&sm.dfield2dy[tyload][txload], &field2_dy[index_load_y], sizeof(float4));
        }

        if (loader_nohalo) {
            __pipeline_memcpy_async(&sm.model1[tyload][txload], &model1[index_load_nohalo], sizeof(float4));
            __pipeline_memcpy_async(&sm.model2[tyload][txload], &model2[index_load_nohalo], sizeof(float4));
            __pipeline_memcpy_async(&sm.model3[tyload][txload], &model3[index_load_nohalo], sizeof(float4));
            __pipeline_memcpy_async(&sm.model4[tyload][txload], &model4[index_load_nohalo], sizeof(float4));
            __pipeline_memcpy_async(&sm.irho[tyload][txload], &irho[index_load_nohalo], sizeof(float4));
            __pipeline_memcpy_async(&sm.dfield1dz[tyload][txload], &field1_dz[index_load_nohalo + index_load_nohalo_z_offset],
                                    sizeof(float4));
            __pipeline_memcpy_async(&sm.dfield2dz[tyload][txload], &field2_dz[index_load_nohalo + index_load_nohalo_z_offset],
                                    sizeof(float4));
        }

        if constexpr (!pml_any) {
            if (loader_nohalo) {
                __pipeline_memcpy_async(&sm.rx[tyload][txload], &rx[index_load_nohalo], sizeof(float4));
                __pipeline_memcpy_async(&sm.ry[tyload][txload], &ry[index_load_nohalo], sizeof(float4));
                __pipeline_memcpy_async(&sm.rz[tyload][txload], &rz[index_load_nohalo], sizeof(float4));
            }
        }
    }

    if constexpr (!pml_any) {
        if (loader_xy) {
            __pipeline_memcpy_async(&sm.field1[tyload][txload], &field1[index_load_xy], sizeof(float4));
            __pipeline_memcpy_async(&sm.field2[tyload][txload], &field2[index_load_xy], sizeof(float4));
        }
    }

    __pipeline_commit();
}

// Updates the 2nd Z derivatives.  These are calculated piecewise using one XY plane of the 1st Z derivatives
// at a time.  The intermediate values are stored in a register queue.  This function is also be used to
// rotate the queue, by using different registers for the field1_zz and field1_zz_prev variables.
// Parameter i is the offset, must be 0 <= i < order.  i - (radius-1) is the location used to update the derivatives
template <FD_COEF_TYPE ctype, int order>
__device__ void
update_d2z(float field1_z, float field2_z, int i, float& field1_zz, float& field2_zz, float field1_zz_prev, float field2_zz_prev)
{
    int radius = order / 2;
    float c;
    if (i < radius) {
        c = -1.0f * helper_kernels_gpu::df_coef<order, ctype>(radius - i);
    } else {
        c = helper_kernels_gpu::df_coef<order, ctype>(i - radius + 1);
    }
    field1_zz = field1_zz_prev + c * field1_z;
    field2_zz = field2_zz_prev + c * field2_z;
}

// Updates the mixed derivatives involving Z.  These are calculated piecewise using one XY plane of field1/field2
// at a time.  The intermediate values are stored in a register queue.  This function is also be used to
// rotate the queue, by using different registers for the field1_zx and field1_zx_prev variables.
// Parameter i is the offset, must be -radius <= i <= radius
template <FD_COEF_TYPE ctype, int order, int bx, int by>
__device__ void
update_z_mixed(smstruct<order, bx, by, false>& sm, int i, float& field1_zx, float& field2_zx, float& field1_yz, float& field2_yz,
               float field1_zx_prev, float field2_zx_prev, float field1_yz_prev, float field2_yz_prev)
{
    int radius = order / 2;
    float c;
    c = helper_kernels_gpu::df2_coef<order, ctype>(abs(i));
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    field1_zx = field1_zx_prev;
    field1_yz = field1_yz_prev;
    field2_zx = field2_zx_prev;
    field2_yz = field2_yz_prev;
    if (i != 0) {
        field1_zx += c * (sm.field1[ty + radius][tx + radius + i] - sm.field1[ty + radius][tx + radius - i]);
        field1_yz += c * (sm.field1[ty + radius + i][tx + radius] - sm.field1[ty + radius - i][tx + radius]);
        field2_zx += c * (sm.field2[ty + radius][tx + radius + i] - sm.field2[ty + radius][tx + radius - i]);
        field2_yz += c * (sm.field2[ty + radius + i][tx + radius] - sm.field2[ty + radius - i][tx + radius]);
    }
}

// Calculates dxy using the field1/field2 values in shared memory.
template <FD_COEF_TYPE ctype, int order, int bx, int by>
__device__ void
compute_dxy(smstruct<order, bx, by, false>& sm, float invdxdy, int ldimx, float& field1_xy, float& field2_xy)
{
    int radius = order / 2;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    field1_xy = 0;
    field2_xy = 0;
#pragma unroll
    for (int i = -radius; i <= radius; ++i) {
        if (i != 0) {
            float c;
            c = helper_kernels_gpu::df2_coef<order, ctype>(abs(i));
            field1_xy += c * (sm.field1[ty + i + radius][tx + i + radius] - sm.field1[ty - i + radius][tx + i + radius]);
            field2_xy += c * (sm.field2[ty + i + radius][tx + i + radius] - sm.field2[ty - i + radius][tx + i + radius]);
        }
    }
    field1_xy *= invdxdy;
    field2_xy *= invdxdy;
}

// Calculates d2x and d2y using the derivative values in shared memory.
template <FD_COEF_TYPE ctype, int order, int bx, int by, bool pml_any>
__device__ void
compute_d2x_d2y(smstruct<order, bx, by, pml_any>& sm, float invdx, float invdy, float& field1_xx, float& field2_xx,
                float& field1_yy, float& field2_yy)
{
    int radius = order / 2;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    field1_xx = field2_xx = field1_yy = field2_yy = 0.0f;

#pragma unroll
    for (int i = 0; i < order; ++i) {
        float c;
        if (i < radius) {
            c = -1.0f * helper_kernels_gpu::df_coef<order, ctype>(radius - i);
        } else {
            c = helper_kernels_gpu::df_coef<order, ctype>(i - radius + 1);
        }

        field1_xx += c * sm.dfield1dx[ty][tx + i + 1];
        field2_xx += c * sm.dfield2dx[ty][tx + i + 1];
        field1_yy += c * sm.dfield1dy[ty + i + 1][tx];
        field2_yy += c * sm.dfield2dy[ty + i + 1][tx];
    }
    field1_xx *= invdx;
    field2_xx *= invdx;
    field1_yy *= invdy;
    field2_yy *= invdy;
}

// Calculate the result for interior domain.
template <int order, int bx, int by>
__device__ void
calculate_result_interior(smstruct<order, bx, by, false>& sm, float& field3_rhs, float& field4_rhs, float field1_xx, float field2_xx,
                          float field1_yy, float field2_yy, float field1_zz, float field2_zz, float field1_xy, float field2_xy,
                          float field1_yz, float field2_yz, float field1_zx, float field2_zx)
{

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float rho = 1.0f / sm.irho[ty][tx];

    float rx2 = sm.rx[ty][tx] * sm.rx[ty][tx];
    float ry2 = sm.ry[ty][tx] * sm.ry[ty][tx];
    float rz2 = sm.rz[ty][tx] * sm.rz[ty][tx];
    float rxy = 2.0f * sm.rx[ty][tx] * sm.ry[ty][tx];
    float ryz = 2.0f * sm.ry[ty][tx] * sm.rz[ty][tx];
    float rzx = 2.0f * sm.rz[ty][tx] * sm.rx[ty][tx];

    // Make normal direvative nondimensional
    field1_xx *= rho;
    field1_yy *= rho;
    field1_zz *= rho;

    field2_xx *= rho;
    field2_yy *= rho;
    field2_zz *= rho;

    float field1_V = rx2 * field1_xx + ry2 * field1_yy + rz2 * field1_zz + rxy * field1_xy + rzx * field1_zx + ryz * field1_yz;

    float field1_H = field1_xx + field1_yy + field1_zz - field1_V;

    float field2_V = rx2 * field2_xx + ry2 * field2_yy + rz2 * field2_zz + rxy * field2_xy + rzx * field2_zx + ryz * field2_yz;
    float field2_H = field2_xx + field2_yy + field2_zz - field2_V;

    field3_rhs = sm.model1[ty][tx] * field1_H + sm.model4[ty][tx] * field1_V + sm.model2[ty][tx] * field2_V;
    field4_rhs = sm.model2[ty][tx] * field1_H + sm.model3[ty][tx] * field2_V + sm.model4[ty][tx] * field2_H;
}

// ****************************************************************************
// Optimized fwd_main_loop_2 kernel.
// Supports 8th and 16th order (template parameter)
// Alignment: Aligned for non-halo reads and writes (most arrays are not accessed with halos)
// To simplify pointer arithmetic, we expect the volumes to point after the Y and Z halos, but not the X halos,
// ie. an offset of (0, iybeg, izbeg).
//
// Launch with a thread block of size (bx, by)
// For 8th order, a block size of (32,16) or (32,8) should work well.
// For 16th order, a block size of (16,16) is needed to use <= 256 threads (to get > 180 registers)
// This kernel uses lots of registers (especially 16th order) due to the numerous register queues.
// The Z dimension is partitioned into "bz" sub-blocks to increase thread block parallelism.

// Strategy:
//  Load the shared memory at two z's - current z plane (i.e. index_load_z)
//       and then offset by radius * zstride (for z derivatives)
//  Compute Z 2nd derivatives and store in order - 1 size register queues
//  Compute the mixed X & Y derivatives and store in half (i.e radius ) size register queues
//  Compute the mixed Z derivatives and store in full (i.e order) size register queues
//     These store partial contributions and are rotated with element 0 being complete
//  (the above 3 steps require a priming loop and a final iteration outside of the main loop)
//  Compute the x and y 2nd within the main loop
//  (If in PML) update the PML variables and derivatives
//  Compute the final result at iz using the oldest values in the register queues
//  Update field3/field4 with the new values, using sponge coefficients if sponge is active
// Uses float4 to load all arrays asynchronously in shared memory to hide the latencies.

template <FD_COEF_TYPE ctype, int order, int bx, int by, int bz, bool pml_x, bool pml_y, bool pml_z, bool sponge_active>
__global__
__launch_bounds__(bx* by) void fwd_main_loop_2(
    float* __restrict__ field3, float* __restrict__ field4, float const* __restrict__ field1_x, float const* __restrict__ field2_x,
    float const* __restrict__ field1_y, float const* __restrict__ field2_y, float const* __restrict__ field1_z,
    float const* __restrict__ field2_z, float const* __restrict__ field1, float const* __restrict__ field2,
    float const* __restrict__ irho, float const* __restrict__ model1, float const* __restrict__ model2,
    float const* __restrict__ model3, float const* __restrict__ model4, float const* __restrict__ rx,
    float const* __restrict__ ry, float const* __restrict__ rz, float* __restrict__ pml_field1_xx,
    float* __restrict__ pml_field2_xx, float* __restrict__ pml_field1_yy, float* __restrict__ pml_field2_yy,
    float* __restrict__ pml_field1_zz, float* __restrict__ pml_field2_zz, float2 const* __restrict__ pml_ab_x,
    float2 const* __restrict__ pml_ab_y, float2 const* __restrict__ pml_ab_z, float2 const* __restrict__ sponge_ab_xx,
    float2 const* __restrict__ sponge_ab_yy, float2 const* __restrict__ sponge_ab_zz, int ixbeg, int nx, int ny, int nz,
    int ldimx, int ldimy, float invdx, float invdy, float invdz, float invdxdy, float invdydz, float invdzdx, float dt)
{
    constexpr bool pml_any = pml_x || pml_y || pml_z;
    static_assert(!(pml_any && sponge_active), "Can't have both PML and sponge");

    constexpr int radius = order / 2;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int ixblock = blockIdx.x * bx;
    int iyblock = blockIdx.y * by;
    int izblock = blockIdx.z * bz;
    int zstride = ldimx * ldimy;

    // Load the shared memory as float4, and remap the threads from (bx, by) into (bx/2, by*2)
    const int tid = threadIdx.y * bx + threadIdx.x;
    static_assert(bx >= order && by >= order, "Incompatible block size for this order");
    const int txload = 4 * (tid % (bx / 2));
    const int tyload = tid / (bx / 2);

    // For arrays with X halos: start at (ixblock-radius,iyblock,iz) and use the float4 thread mapping.
    // Since the input arrays are offset for y and z halos, but not x, when ixload_x is negative, it will
    // be reading garbage from the previous line. This isn't a "problem" necessarily, because nothing uses
    // that data from shared memory (see active variable), but it wastes some bandwidth, so the loader_x
    // variable should be false if ixload_x is negative.  Same applies for loader_xy
    const int ixload_x = ixblock + txload - radius;
    const int iyload_x = iyblock + tyload;
    const bool loader_x = (txload < bx + order && tyload < by) && ixload_x >= 0 && (ixload_x < ldimx && iyload_x < ny);
    int index_load_x = (izblock * ldimy + iyload_x) * ldimx + ixload_x;

    // For arrays with Y halos: start at (ixblock,iyblock-radius,iz) and use the float4 thread mapping.
    const int ixload_y = ixblock + txload;
    const int iyload_y = iyblock + tyload - radius;
    const bool loader_y = (txload < bx && tyload < by + order) && (ixload_y < ldimx && iyload_y < ny + radius);
    int index_load_y = (izblock * ldimy + iyload_y) * ldimx + ixload_y;

    // For arrays with XY halos: start at (ixblock-radius,iyblock-radius,iz) and use the float4 thread mapping.
    const int ixload_xy = ixblock + txload - radius;
    const int iyload_xy = iyblock + tyload - radius;
    const bool loader_xy = (txload < bx + order && tyload < by + order) && ixload_xy >= 0 &&
                           (ixload_xy < ldimx && iyload_xy < ny + radius);
    int index_load_xy = ((izblock - radius) * ldimy + iyload_xy) * ldimx + ixload_xy;

    // For arrays without halos: start at (ixblock,iyblock,iz) and use the float4 thread mapping.
    // This is also used for loading the z derivatives, since those only use a halo in the Z direction, not X or Y.
    const int ixload_nohalo = ixblock + txload;
    const int iyload_nohalo = iyblock + tyload;
    const int loader_nohalo = (txload < bx && tyload < by) && (ixload_nohalo < ldimx && iyload_nohalo < ny);
    int index_load_nohalo = (izblock * ldimy + iyload_nohalo) * ldimx + ixload_nohalo;

    static_assert(order == 8 || order == 16,
                  "Radius must be a multiple of 4 due to using float4 for loading shared memory");

    extern __shared__ char sm_[];
    auto sm = reinterpret_cast<smstruct<order, bx, by, pml_any>*>(sm_);

    int ix = ixblock + threadIdx.x;
    int iy = iyblock + threadIdx.y;
    int ixend = ixbeg + nx;
    int active = ix >= ixbeg && ix < ixend && iy < ny;

    // The index of where to write the output
    int index_out = (izblock * ldimy + iy) * ldimx + ix;
    // The index for loading the z derivatives.  This is only used in the loops to prime the z derivative
    // register queues.  The main loop uses shared memory async prefetching to load the data for the z
    // derivatives.  We attempted to use async prefetching for the priming loops as well, but it made the
    // code significantly more complex and provided less than 1% overall speedup, so it was scrapped.
    int index_load_z = index_out - (radius - 1) * zstride;

    // Register queues for 2nd derivatives only involving Z.  The stencil for these is the 1D staggered
    // stencil, which is why they only use order - 1 registers.
    float field1_zz[order - 1] = { 0 };
    float field2_zz[order - 1] = { 0 };
    // Register queues for mixed derivatives involving Z.  These are unused in PML.
    float field1_yz[order] = { 0 };
    float field1_zx[order] = { 0 };
    float field2_yz[order] = { 0 };
    float field2_zx[order] = { 0 };
    // Register queues for XY mixed derivatives.  These are unused in PML.
    // These are calculated at iz+radius because the needed data has already been loaded for the mixed
    // derivatives involving Z.  They are put in the register queue until they are needed at iz.
    float field1_xy[radius] = { 0 };
    float field2_xy[radius] = { 0 };

    // Prime the register queues for 2nd Z derivative.
    // Both loops need to be unrolled to allow proper inlining and resolution of FD coefficients.
    if (active) {
#pragma unroll
        for (int i = 0; i < order - 1; ++i) {
#pragma unroll
            for (int j = 0; j <= i; ++j) {
                // prev and new are the same because we don't rotate the queue in the priming loop
                update_d2z<ctype, order>(field1_z[index_load_z], field2_z[index_load_z], i - j, field1_zz[j], field2_zz[j],
                                         field1_zz[j], field2_zz[j]);
            }
            index_load_z += zstride;
        }
    }

    int ism = 0;

    // Prime the register queues for the mixed derivatives.  These values are not used for PML.
    // Both loops need to be unrolled to allow proper inlining and resolution of FD coefficients.
    // Kick off the first shared memory pipeline load
    load_xy_planes<false>(sm[ism], nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, field1, field2, nullptr, nullptr,
                            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, index_load_x, index_load_y,
                            index_load_xy, index_load_nohalo, zstride * radius, loader_x, loader_y, loader_xy,
                            loader_nohalo, txload, tyload);
    // In the priming loop, we only use field1 and field2 which are loaded with XY halos.  So we only update index_load_xy
    // rather than all the index_* variables.
    index_load_xy += zstride;
    ism ^= 1;

#pragma unroll
    for (int i = 0; i < order; ++i) {
        int radius = order / 2;
        __syncthreads(); // Protect shared memory between iterations
        if (i == order - 1) {
            // This is the last iteration through the priming loop, so we need to kick off the "full" shared memory
            // preload for the main loop.
            load_xy_planes<true>(sm[ism], field1_x, field2_x, field1_y, field2_y, field1_z, field2_z, field1, field2, irho, model1, model2, model3,
                                    model4, rx, ry, rz, index_load_x, index_load_y, index_load_xy, index_load_nohalo,
                                    zstride * radius, loader_x, loader_y, loader_xy, loader_nohalo, txload, tyload);
        } else {
            load_xy_planes<false>(sm[ism], nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, field1, field2, nullptr,
                                    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, index_load_x,
                                    index_load_y, index_load_xy, index_load_nohalo, zstride * radius, loader_x,
                                    loader_y, loader_xy, loader_nohalo, txload, tyload);
        }
        ism ^= 1;
        __pipeline_wait_prior(1);
        __syncthreads(); // Require pipeline load to be complete

        if (active) {
#pragma unroll
            for (int j = 0; j <= i; ++j) {
                // prev and new are the same because we don't rotate the queue in the priming loop
                update_z_mixed<ctype, order>(sm[ism], i - j - radius, field1_zx[j], field2_zx[j], field1_yz[j], field2_yz[j],
                                                field1_zx[j], field2_zx[j], field1_yz[j], field2_yz[j]);
            }

            // We only need to calculate the mixed XY derivatives once we get out of the halo region of the priming loop
            if (i >= radius) {
                compute_dxy<ctype, order>(sm[ism], invdxdy, ldimx, field1_xy[i - radius], field2_xy[i - radius]);
            }
        }

        index_load_xy += zstride;
    }

    // These were used in the last load_xy_planes call of the loop, so increment them now.
    index_load_x += zstride;
    index_load_y += zstride;
    index_load_nohalo += zstride;

    int nzloop = min(bz, nz - izblock);
    for (int iz = 0; iz < nzloop - 1; ++iz) {
        __syncthreads(); // Protect shared memory between iterations
        load_xy_planes<true>(sm[ism], field1_x, field2_x, field1_y, field2_y, field1_z, field2_z, field1, field2, irho, model1, model2, model3, model4, rx,
                             ry, rz, index_load_x, index_load_y, index_load_xy, index_load_nohalo, zstride * radius,
                             loader_x, loader_y, loader_xy, loader_nohalo, txload, tyload);
        ism ^= 1;
        __pipeline_wait_prior(1);
        __syncthreads(); // Wait for pipeline load to complete

        if (active) {
            // Update the top ZZ derivative values in place
            update_d2z<ctype, order>(sm[ism].dfield1dz[ty][tx], sm[ism].dfield2dz[ty][tx], order - 1, field1_zz[0], field2_zz[0],
                                     field1_zz[0], field2_zz[0]);
            field1_zz[0] *= invdz;
            field2_zz[0] *= invdz;

            if constexpr (!pml_any) {
                // Update the top derivative values involving Z in place
                update_z_mixed<ctype, order, bx, by>(sm[ism], radius, field1_zx[0], field2_zx[0], field1_yz[0], field2_yz[0],
                                                     field1_zx[0], field2_zx[0], field1_yz[0], field2_yz[0]);
                field1_yz[0] *= invdydz;
                field1_zx[0] *= invdzdx;
                field2_yz[0] *= invdydz;
                field2_zx[0] *= invdzdx;
            }

            float field1_xx = 0, field1_yy = 0, field2_xx = 0, field2_yy = 0;
            compute_d2x_d2y<ctype>(sm[ism], invdx, invdy, field1_xx, field2_xx, field1_yy, field2_yy);

            float field3_rhs = 0, field4_rhs = 0;
            calculate_result_interior(sm[ism], field3_rhs, field4_rhs, field1_xx, field2_xx, field1_yy, field2_yy, field1_zz[0], field2_zz[0],
                                      field1_xy[0], field2_xy[0], field1_yz[0], field2_yz[0], field1_zx[0], field2_zx[0]);

            atomicAdd(&field3[index_out], field3_rhs * dt);
            atomicAdd(&field4[index_out], field4_rhs * dt);

            // Update the ZZ derivative queues
#pragma unroll
            for (int i = 1; i < order - 1; ++i) {
                update_d2z<ctype, order>(sm[ism].dfield1dz[ty][tx], sm[ism].dfield2dz[ty][tx], order - 1 - i, field1_zz[i - 1],
                                         field2_zz[i - 1], field1_zz[i], field2_zz[i]);
            }
            update_d2z<ctype, order>(sm[ism].dfield1dz[ty][tx], sm[ism].dfield2dz[ty][tx], 0, field1_zz[order - 2],
                                     field2_zz[order - 2], 0.0f, 0.0f);

            if constexpr (!pml_any) {
                // Update the mixed derivatives involving Z
#pragma unroll
                for (int i = 1; i < order; ++i) {
                    update_z_mixed<ctype, order, bx, by>(sm[ism], radius - i, field1_zx[i - 1], field2_zx[i - 1],
                                                         field1_yz[i - 1], field2_yz[i - 1], field1_zx[i], field2_zx[i], field1_yz[i],
                                                         field2_yz[i]);
                }

                update_z_mixed<ctype, order, bx, by>(sm[ism], -radius, field1_zx[order - 1], field2_zx[order - 1],
                                                     field1_yz[order - 1], field2_yz[order - 1], 0.0f, 0.0f, 0.0f, 0.0f);

                //Rotate XY queues
#pragma unroll
                for (int i = 0; i < radius - 1; i++) {
                    field1_xy[i] = field1_xy[i + 1];
                    field2_xy[i] = field2_xy[i + 1];
                }
                // Calculate new value for XY queue and put it at the end of the queue
                compute_dxy<ctype, order>(sm[ism], invdxdy, ldimx, field1_xy[radius - 1], field2_xy[radius - 1]);
            }
        }

        index_out += zstride;
        index_load_xy += zstride;
        index_load_x += zstride;
        index_load_y += zstride;
        index_load_nohalo += zstride;
    }

    // Complete final iteration with the shared memory that has been prefetched.

    int iz = nzloop - 1;
    __syncthreads(); // Protect shared memory between iterations
    ism ^= 1;
    __pipeline_wait_prior(0);
    __syncthreads(); // Ensure previous pipeline load is finished

    if (active) {
        update_d2z<ctype, order>(sm[ism].dfield1dz[ty][tx], sm[ism].dfield2dz[ty][tx], order - 1, field1_zz[0], field2_zz[0],
                                 field1_zz[0], field2_zz[0]);
        field1_zz[0] *= invdz;
        field2_zz[0] *= invdz;

        if constexpr (!pml_any) {
            update_z_mixed<ctype, order, bx, by>(sm[ism], radius, field1_zx[0], field2_zx[0], field1_yz[0], field2_yz[0], field1_zx[0],
                                                 field2_zx[0], field1_yz[0], field2_yz[0]);
            field1_yz[0] *= invdydz;
            field1_zx[0] *= invdzdx;
            field2_yz[0] *= invdydz;
            field2_zx[0] *= invdzdx;
        }

        float field1_xx = 0, field1_yy = 0, field2_xx = 0, field2_yy = 0;
        compute_d2x_d2y<ctype>(sm[ism], invdx, invdy, field1_xx, field2_xx, field1_yy, field2_yy);

        float field3_rhs = 0, field4_rhs = 0;
        calculate_result_interior(sm[ism], field3_rhs, field4_rhs, field1_xx, field2_xx, field1_yy, field2_yy, field1_zz[0], field2_zz[0],
                                      field1_xy[0], field2_xy[0], field1_yz[0], field2_yz[0], field1_zx[0], field2_zx[0]);

        atomicAdd(&field3[index_out], field3_rhs * dt);
        atomicAdd(&field4[index_out], field4_rhs * dt);
    }
}

// Returns the optimal block size, given the order and pml directions
template <int order, bool pml_x, bool pml_y, bool pml_z>
constexpr int3
get_optimal_block_size()
{
    static_assert(order == 8 || order == 16, "Only 8th or 16th order supported.");

    // TODO these can probably be tuned, but it's probably not worth doing right now
    if constexpr (order == 8) {
        return { 32, 16, 64 };
    } else {
        return { 16, 16, 64 };
    }
}

// Returns the amount of shared memory required for the kernel metaparameters (blocksize/order/pml/etc)
// As a side effect, sets the function attribute to allow it to use more shared memory.
template <int order, int bx, int by, int bz, bool pml_x, bool pml_y, bool pml_z, bool sponge_active>
int
get_shm()
{
    int shmsize = sizeof(smstruct < order, bx, by, pml_x || pml_y || pml_z > [2]);

    static bool set_shmsize = false;
    if (!set_shmsize) {
        set_shmsize = true;
        CUDA_TRY(cudaFuncSetAttribute(
            fwd_main_loop_2<_FD_COEF_LEAST_SQUARE, order, bx, by, bz, pml_x, pml_y, pml_z, sponge_active>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shmsize));
    }

    return shmsize;
}

} // end anonymous namespace

void
launch_update_rho_fwd_main_loop_2_1(float* field3, float* field4, float const* field1_dx, float const* field2_dx, float const* field1_dy,
                                 float const* field2_dy, float const* field1_dz, float const* field2_dz, float const* field1,
                                 float const* field2, float const* irho, float const* model1, float const* model2,
                                 float const* model3, float const* model4, float const* rx, float const* ry, float const* rz,
                                 float* pml_field1_xx, float* pml_field2_xx, float* pml_field1_yy, float* pml_field2_yy,
                                 float* pml_field1_zz, float* pml_field2_zz, float2 const* pml_ab_x, float2 const* pml_ab_y,
                                 float2 const* pml_ab_z, float2 const* sponge_ab_xx, float2 const* sponge_ab_yy,
                                 float2 const* sponge_ab_zz, int ixbeg, int ixend, int iybeg, int iyend, int izbeg,
                                 int izend, int ldimx, int ldimy, float dt, double dx, double dy, double dz,
                                 bool sponge_active, int order, bool pml_x, bool pml_y, bool pml_z, cudaStream_t stream)
{
    int nx = ixend - ixbeg + 1;
    int ny = iyend - iybeg + 1;
    int nz = izend - izbeg + 1;

    int subblock_z = 64;

    // Offset all input and output volumes to skip y and z halos (not x halo)
    int offset = (izbeg * ldimy + iybeg) * ldimx;
    field3 += offset;
    field4 += offset;
    field1_dx += offset;
    field2_dx += offset;
    field1_dy += offset;
    field2_dy += offset;
    field1_dz += offset;
    field2_dz += offset;
    field1 += offset;
    field2 += offset;
    irho += offset;
    model1 += offset;
    model2 += offset;
    model3 += offset;
    model4 += offset;
    rx += offset;
    ry += offset;
    rz += offset;
    sponge_ab_yy += iybeg;
    sponge_ab_zz += izbeg;
    pml_ab_y += iybeg;
    pml_ab_z += izbeg;
    float invdx;
    float invdy;
    float invdz;
    float invdzdx;
    float invdxdy;
    float invdydz;

    float unused;
    kernel_utils::compute_fd_const(dx, dy, dz, invdx, invdy, invdz, unused, unused, unused, invdxdy, invdydz, invdzdx);


    if (order == 8) {
        constexpr int3 optimal_block = get_optimal_block_size<8, false, false, false>();
        constexpr int bx = optimal_block.x;
        constexpr int by = optimal_block.y;
        constexpr int bz = optimal_block.z;

        dim3 threads(bx, by, 1);
        dim3 blocks((nx + ixbeg - 1) / threads.x + 1, (ny - 1) / threads.y + 1, (nz - 1) / bz + 1);

        int shmsize = get_shm<8, bx, by, bz, false, false, false, false>();
        fwd_main_loop_2<_FD_COEF_LEAST_SQUARE, 8, bx, by, bz, false, false, false, false>
            <<<blocks, threads, shmsize, stream>>>(
                field3, field4, field1_dx, field2_dx, field1_dy, field2_dy, field1_dz, field2_dz, field1, field2, irho, model1, model2, model3, model4, rx,
                ry, rz, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                sponge_ab_xx, sponge_ab_yy, sponge_ab_zz, ixbeg, nx, ny, nz, ldimx, ldimy, invdx, invdy, invdz,
                invdxdy, invdydz, invdzdx, dt);
    } else {
        constexpr int3 optimal_block = get_optimal_block_size<16, false, false, false>();
        constexpr int bx = optimal_block.x;
        constexpr int by = optimal_block.y;
        constexpr int bz = optimal_block.z;

        dim3 threads(bx, by, 1);
        dim3 blocks((nx + ixbeg - 1) / threads.x + 1, (ny - 1) / threads.y + 1, (nz - 1) / bz + 1);

        int shmsize = get_shm<16, bx, by, bz, false, false, false, false>();
        fwd_main_loop_2<_FD_COEF_LEAST_SQUARE, 16, bx, by, bz, false, false, false, false>
            <<<blocks, threads, shmsize, stream>>>(
                field3, field4, field1_dx, field2_dx, field1_dy, field2_dy, field1_dz, field2_dz, field1, field2, irho, model1, model2, model3, model4, rx,
                ry, rz, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                sponge_ab_xx, sponge_ab_yy, sponge_ab_zz, ixbeg, nx, ny, nz, ldimx, ldimy, invdx, invdy, invdz,
                invdxdy, invdydz, invdzdx, dt);
    }
    CUDA_TRY(cudaPeekAtLastError());
}
