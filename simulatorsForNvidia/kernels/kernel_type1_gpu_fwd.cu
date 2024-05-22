#include <cuda_pipeline.h>
#include "helper_kernels_gpu.h"
#include "kernel_utils.h"
#include "kernel_type1_gpu_fwd.h"


namespace fwd_main_loop_2_1_inner_gpu {

template <int order, FD_COEF_TYPE ctype, bool calc_snap>
__global__ void __launch_bounds__(32 * 4 * 4)
    fwd_main_loop_2_1_simple(volume_index vol3d_in, volume_index vol3d_out, float* __restrict__ field3, float* __restrict__ field4,
                          const float* __restrict__ field1, const float* __restrict__ field2, const float* __restrict__ model1,
                          const float* __restrict__ model2, const float* __restrict__ model3, const float* __restrict__ model4,
                          const float* __restrict__ model5, const float* __restrict__ model6, const float* __restrict__ model7,
                          const float2* __restrict__ ABx, const float2* __restrict__ ABy,
                          const float2* __restrict__ ABz, int ixbeg, int ixend, int iybeg, int iyend, int izbeg,
                          int izend, float dt, float invdxx, float invdyy, float invdzz, float invdxy, float invdyz,
                          float invdzx)
{
    // Global position of the thread, using 3D thread blocks for improved locality
    int ix = ixbeg + blockIdx.x * blockDim.x + threadIdx.x;
    int iy = iybeg + blockIdx.y * blockDim.y + threadIdx.y;
    int iz = izbeg + blockIdx.z * blockDim.z + threadIdx.z;

    // All threads outside the boundaries can exit
    if (ix > ixend || iy > iyend || iz > izend)
        return;

    const float field1_xx = helper_kernels_gpu::drv11<order, ctype>(field1, vol3d_in, ix, iy, iz, invdxx);
    const float field1_yy = helper_kernels_gpu::drv22<order, ctype>(field1, vol3d_in, ix, iy, iz, invdyy);
    const float field1_zz = helper_kernels_gpu::drv33<order, ctype>(field1, vol3d_in, ix, iy, iz, invdzz);
    const float field1_xy = helper_kernels_gpu::drv12<order, ctype>(field1, vol3d_in, ix, iy, iz, invdxy);
    const float field1_yz = helper_kernels_gpu::drv23<order, ctype>(field1, vol3d_in, ix, iy, iz, invdyz);
    const float field1_zx = helper_kernels_gpu::drv31<order, ctype>(field1, vol3d_in, ix, iy, iz, invdzx);

    const float field2_xx = helper_kernels_gpu::drv11<order, ctype>(field2, vol3d_in, ix, iy, iz, invdxx);
    const float field2_yy = helper_kernels_gpu::drv22<order, ctype>(field2, vol3d_in, ix, iy, iz, invdyy);
    const float field2_zz = helper_kernels_gpu::drv33<order, ctype>(field2, vol3d_in, ix, iy, iz, invdzz);
    const float field2_xy = helper_kernels_gpu::drv12<order, ctype>(field2, vol3d_in, ix, iy, iz, invdxy);
    const float field2_yz = helper_kernels_gpu::drv23<order, ctype>(field2, vol3d_in, ix, iy, iz, invdyz);
    const float field2_zx = helper_kernels_gpu::drv31<order, ctype>(field2, vol3d_in, ix, iy, iz, invdzx);

    const float rxx = vol3d_in(model5, ix, iy, iz) * vol3d_in(model5, ix, iy, iz);
    const float ryy = vol3d_in(model6, ix, iy, iz) * vol3d_in(model6, ix, iy, iz);
    const float rzz = vol3d_in(model7, ix, iy, iz) * vol3d_in(model7, ix, iy, iz);
    const float rxy = 2.0f * vol3d_in(model5, ix, iy, iz) * vol3d_in(model6, ix, iy, iz);
    const float ryz = 2.0f * vol3d_in(model6, ix, iy, iz) * vol3d_in(model7, ix, iy, iz);
    const float rzx = 2.0f * vol3d_in(model7, ix, iy, iz) * vol3d_in(model5, ix, iy, iz);

    const float field1_V = rxx * field1_xx + ryy * field1_yy + rzz * field1_zz + rxy * field1_xy + rzx * field1_zx + ryz * field1_yz;
    const float field1_H = field1_xx + field1_yy + field1_zz - field1_V;

    const float field2_V = rxx * field2_xx + ryy * field2_yy + rzz * field2_zz + rxy * field2_xy + rzx * field2_zx + ryz * field2_yz;
    const float field2_H = field2_xx + field2_yy + field2_zz - field2_V;

    const float field3_rhs =
        vol3d_in(model1, ix, iy, iz) * field1_H + vol3d_in(model4, ix, iy, iz) * field1_V + vol3d_in(model2, ix, iy, iz) * field2_V;
    const float field4_rhs =
        vol3d_in(model2, ix, iy, iz) * field1_H + vol3d_in(model3, ix, iy, iz) * field2_V + vol3d_in(model4, ix, iy, iz) * field2_H;

    if constexpr (calc_snap) {
        // Calc snap has no halo
        vol3d_out(field3, ix - ixbeg, iy - iybeg, iz - izbeg) = field3_rhs;
        vol3d_out(field4, ix - ixbeg, iy - iybeg, iz - izbeg) = field4_rhs;
    } else {
        float A_xyz = min(ABx[ix].x, min(ABy[iy].x, ABz[iz].x));
        float B_xyz = min(ABx[ix].y, min(ABy[iy].y, ABz[iz].y));
        vol3d_out(field3, ix, iy, iz) = A_xyz * vol3d_out(field3, ix, iy, iz) + B_xyz * field3_rhs * dt;
        vol3d_out(field4, ix, iy, iz) = A_xyz * vol3d_out(field4, ix, iy, iz) + B_xyz * field4_rhs * dt;
    }
}

// ****************************************************************************
// ****************************************************************************
// ****************************************************************************
// Optimized kernel, based on partial contributions, walking through the Z dimension.
// All the inputs are loaded asynchronously in shared memory to hide latencies.

// Structure for the shared memory arrays
template <int order, int bx, int by>
struct smstruct
{
    float field1[by + order][bx + order];
    float field2[by + order][bx + order];
    float model1[by][bx];
    float model2[by][bx];
    float model3[by][bx];
    float model4[by][bx];
    float model5[by][bx];
    float model6[by][bx];
    float model7[by][bx];
};

// ****************************************************************************
// Device function to asynchronously load the shared memory as float4
// Loading only the T arrays while in the Z halos
template <bool t_only = false, int order, int bx, int by>
__device__ inline void
async_load(smstruct<order, bx, by>* sm, int index_load, int index_load_halos, bool loader, bool loader_halos,
           int txload, int tyload, const float* field1, const float* field2, const float* model1, const float* model2,
           const float* model3, const float* model4, const float* model5, const float* model6, const float* model7)
{
    if (loader_halos) {
        __pipeline_memcpy_async(&sm->field1[tyload][txload], field1 + index_load_halos, sizeof(float4));
        __pipeline_memcpy_async(&sm->field2[tyload][txload], field2 + index_load_halos, sizeof(float4));
    }
    if (!t_only && loader) {
        __pipeline_memcpy_async(&sm->model1[tyload][txload], model1 + index_load, sizeof(float4));
        __pipeline_memcpy_async(&sm->model2[tyload][txload], model2 + index_load, sizeof(float4));
        __pipeline_memcpy_async(&sm->model3[tyload][txload], model3 + index_load, sizeof(float4));
        __pipeline_memcpy_async(&sm->model4[tyload][txload], model4 + index_load, sizeof(float4));
        __pipeline_memcpy_async(&sm->model5[tyload][txload], model5 + index_load, sizeof(float4));
        __pipeline_memcpy_async(&sm->model6[tyload][txload], model6 + index_load, sizeof(float4));
        __pipeline_memcpy_async(&sm->model7[tyload][txload], model7 + index_load, sizeof(float4));
    }
    __pipeline_commit();
}

// ****************************************************************************
template <int order, int bx, int by>
__device__ inline void
compute_result(const smstruct<order, bx, by>* sm, float& field3_rhs, float& field4_rhs, float field1_xx, float field1_yy,
               float field1_zz, float field1_xy, float field1_yz, float field1_zx, float field2_xx, float field2_yy, float field2_zz,
               float field2_xy, float field2_yz, float field2_zx)
{
    const float rxx = sm->model5[threadIdx.y][threadIdx.x] * sm->model5[threadIdx.y][threadIdx.x];
    const float ryy = sm->model6[threadIdx.y][threadIdx.x] * sm->model6[threadIdx.y][threadIdx.x];
    const float rzz = sm->model7[threadIdx.y][threadIdx.x] * sm->model7[threadIdx.y][threadIdx.x];
    const float rxy = 2.0f * sm->model5[threadIdx.y][threadIdx.x] * sm->model6[threadIdx.y][threadIdx.x];
    const float ryz = 2.0f * sm->model6[threadIdx.y][threadIdx.x] * sm->model7[threadIdx.y][threadIdx.x];
    const float rzx = 2.0f * sm->model7[threadIdx.y][threadIdx.x] * sm->model5[threadIdx.y][threadIdx.x];

    const float field1_V = rxx * field1_xx + ryy * field1_yy + rzz * field1_zz + rxy * field1_xy + rzx * field1_zx + ryz * field1_yz;
    const float field1_H = field1_xx + field1_yy + field1_zz - field1_V;

    const float field2_V = rxx * field2_xx + ryy * field2_yy + rzz * field2_zz + rxy * field2_xy + rzx * field2_zx + ryz * field2_yz;
    const float field2_H = field2_xx + field2_yy + field2_zz - field2_V;

    field3_rhs = sm->model1[threadIdx.y][threadIdx.x] * field1_H + sm->model4[threadIdx.y][threadIdx.x] * field1_V +
             sm->model2[threadIdx.y][threadIdx.x] * field2_V;
    field4_rhs = sm->model2[threadIdx.y][threadIdx.x] * field1_H + sm->model3[threadIdx.y][threadIdx.x] * field2_V +
             sm->model4[threadIdx.y][threadIdx.x] * field2_H;
}

// ****************************************************************************
// Contributions of the shared memory at iz+ir, with ir between [-order/2 : order/2].
// The computation is done out-of-place to allow rotation of the register queues on the fly.
// When everything is unrolled and inlined properly, the FD coefficient is known at compile time
// and the conditionals should disappear
template <FD_COEF_TYPE ctype, int order, int bx, int by>
__device__ inline void
compute_z_contributions(const smstruct<order, bx, by>* sm, int ir, float invdzz, float invdyz, float invdzx,
                        // Results
                        float& field1_zz, float& field1_zx, float& field1_yz, float& field2_zz, float& field2_zx, float& field2_yz,
                        // Previous values to accumulate
                        float pxx_zz, float pxx_zx, float pxx_yz, float pzz_zz, float pzz_zx, float pzz_yz)
{
    const int radius = order / 2;
    const int tx = threadIdx.x + radius;
    const int ty = threadIdx.y + radius;
    float coef = helper_kernels_gpu::df2_coef<order, ctype>(abs(ir));
    field1_zz = pxx_zz + coef * sm->field1[ty][tx];
    field2_zz = pzz_zz + coef * sm->field2[ty][tx];
    if (ir == 0) {
        // No mixed contributions for zero offset
        field1_zx = pxx_zx;
        field2_zx = pzz_zx;
        field1_yz = pxx_yz;
        field2_yz = pzz_yz;
    } else {
        // Offset in Z -> use 2 diagonals
        field1_zx = pxx_zx + coef * (sm->field1[ty][tx + ir] - sm->field1[ty][tx - ir]);
        field2_zx = pzz_zx + coef * (sm->field2[ty][tx + ir] - sm->field2[ty][tx - ir]);
        field1_yz = pxx_yz + coef * (sm->field1[ty + ir][tx] - sm->field1[ty - ir][tx]);
        field2_yz = pzz_yz + coef * (sm->field2[ty + ir][tx] - sm->field2[ty - ir][tx]);
    }
    if (ir == order / 2) // Final contribution to this value : apply the grid coefs
    {
        field1_zz *= invdzz;
        field2_zz *= invdzz;
        field1_zx *= invdzx;
        field2_zx *= invdzx;
        field1_yz *= invdyz;
        field2_yz *= invdyz;
    }
}

// ****************************************************************************
// Contributions of the shared memory to the X and Y derivatives
template <FD_COEF_TYPE ctype, int order, int bx, int by>
__device__ inline void
compute_xy_contributions(const smstruct<order, bx, by>* sm, float invdxx, float invdyy, float invdxy, float& field1_xx,
                         float& field1_yy, float& field1_xy, float& field2_xx, float& field2_yy, float& field2_xy)
{
    const int radius = order / 2;
    const int tx = threadIdx.x + radius;
    const int ty = threadIdx.y + radius;
    float coef = helper_kernels_gpu::df2_coef<order, ctype>(0);
    field1_xx = coef * sm->field1[ty][tx];
    field2_xx = coef * sm->field2[ty][tx];
    field1_yy = coef * sm->field1[ty][tx];
    field2_yy = coef * sm->field2[ty][tx];
    field1_xy = 0.0f;
    field2_xy = 0.0f;
#pragma unroll
    for (int i = 1; i <= order / 2; i++) {
        float coef = helper_kernels_gpu::df2_coef<order, ctype>(i);
        field1_xx += coef * (sm->field1[ty][tx + i] + sm->field1[ty][tx - i]);
        field2_xx += coef * (sm->field2[ty][tx + i] + sm->field2[ty][tx - i]);
        field1_yy += coef * (sm->field1[ty + i][tx] + sm->field1[ty - i][tx]);
        field2_yy += coef * (sm->field2[ty + i][tx] + sm->field2[ty - i][tx]);
        field1_xy += coef * (sm->field1[ty + i][tx + i] - sm->field1[ty + i][tx - i] + sm->field1[ty - i][tx - i] -
                          sm->field1[ty - i][tx + i]);
        field2_xy += coef * (sm->field2[ty + i][tx + i] - sm->field2[ty + i][tx - i] + sm->field2[ty - i][tx - i] -
                          sm->field2[ty - i][tx + i]);
    }
    field1_xx *= invdxx;
    field2_xx *= invdxx;
    field1_yy *= invdyy;
    field2_yy *= invdyy;
    field1_xy *= invdxy;
    field2_xy *= invdxy;
}

template <int order, FD_COEF_TYPE ctype, int bx, int by, int bz, bool calc_snap, bool use_sponge>
__global__
__launch_bounds__(bx* by) void fwd_main_loop_2_1(
    float* __restrict__ field3, float* __restrict__ field4, const float* __restrict__ field1, const float* __restrict__ field2,
    const float* __restrict__ model1, const float* __restrict__ model2, const float* __restrict__ model3,
    const float* __restrict__ model4, const float* __restrict__ model5, const float* __restrict__ model6,
    const float* __restrict__ model7, const float2* __restrict__ ABx, const float2* __restrict__ ABy,
    const float2* __restrict__ ABz, int ixbeg, int nx, int ny, int nz, int ldimx_in, int ldimy_in, int ldimx_out,
    int ldimy_out, float dt, float invdxx, float invdyy, float invdzz, float invdxy, float invdyz, float invdzx)
{
    static_assert(!(calc_snap && use_sponge), "calc_snap and use_sponge flags cannot both be true");

    constexpr int radius = order / 2;
    // Global position of the thread block, using 2D thread blocks
    const int ixblock = blockIdx.x * bx;
    const int iyblock = blockIdx.y * by;
    const int izblock = blockIdx.z * bz;
    const int stride_in = ldimx_in * ldimy_in;

    static_assert(order == 8 || order == 16); // Radius must be a multiple of 4 for now

    // Shared memory (~44KB for 8th order (32,16), and ~30KB for 16th order (16,16))
    __shared__ smstruct<order, bx, by> sm[2];

    // Which threads are active (writing values), and their global index
    // calc_snap mode has no halo on the output
    const int ix = ixblock + threadIdx.x;
    const int iy = iyblock + threadIdx.y;
    const bool active = ix >= ixbeg && ix < ixbeg + nx && iy < ny;
    int index_out = (izblock * ldimy_out + iy) * ldimx_out + ix;
    if (calc_snap) {
        index_out -= ixbeg;
    }
    const int stride_out = ldimx_out * ldimy_out;

    float A_xy, B_xy;
    if (use_sponge && active) {
        A_xy = min(ABx[ix].x, ABy[iy].x);
        B_xy = min(ABx[ix].y, ABy[iy].y);
    }

    // Load the shared memory as float4, and remap the threads from (bx, by) into (bx/2, by*2)
    const int tid = threadIdx.y * bx + threadIdx.x;
    static_assert(bx >= order && by >= order, "Incompatible block size for this order");
    const int txload = 4 * (tid % (bx / 2));
    const int tyload = tid / (bx / 2);
    // For arrays without halos: start at (ixblock,iyblock,iz) and use the float4 thread mapping.
    const int ixload = ixblock + txload;
    const int iyload = iyblock + tyload;
    const bool loader = (txload < bx && tyload < by) && (ixload < ldimx_in && iyload < ny);
    int index_load = (izblock * ldimy_in + iyload) * ldimx_in + ixload;
    // For arrays with halos: remove "radius" points in the 3 dimensions, and use the float4 thread mapping.
    const int ixload_halos = ixload - radius; // Can be negative and OOB since the X dimension starts inside the halo.
    const int iyload_halos = iyload - radius;
    const bool loader_halos = ((bx == order || txload < bx + order) && (by == order || tyload < by + order)) &&
                              (ixload_halos >= 0 && ixload_halos < ldimx_in && iyload_halos < ny + radius);
    int index_load_halos = ((izblock - radius) * ldimy_in + iyload_halos) * ldimx_in + ixload_halos;

    // Register queues for the derivatives which involve Z
    float field1_zz[order];
    float field1_zx[order];
    float field1_yz[order];
    float field2_zz[order];
    float field2_zx[order];
    float field2_yz[order];
    // Half register queues for the derivatives without Z
    float field1_xx[radius];
    float field1_xy[radius];
    float field1_yy[radius];
    float field2_xx[radius];
    float field2_xy[radius];
    float field2_yy[radius];

    float field3_rhs, field4_rhs;

    int ism = 0; // Flip-flop between the 2 shared memory buffers
    // Async load the first plans for field1 and field2 only
    async_load<true>(sm + ism, index_load, index_load_halos, loader, loader_halos, txload, tyload, field1, field2, model1, model2,
                     model3, model4, model5, model6, model7);
    index_load_halos += stride_in;
    ism ^= 1;

    // Prime the register queues
    // Both loops must be unrolled to allow proper inlining and resolution of FD coefficients.
#pragma unroll
    for (int i = 0; i < order; i++) {
        // Protect the shared memory between iterations
        __syncthreads();
        // Prefetch the next arrays in shared memory
        if (i < order - 1)
            async_load<true>(sm + ism, index_load, index_load_halos, loader, loader_halos, txload, tyload, field1, field2,
                             model1, model2, model3, model4, model5, model6, model7);
        else // Last iteration of priming: prefetch all arrays
        {
            async_load(sm + ism, index_load, index_load_halos, loader, loader_halos, txload, tyload, field1, field2, model1, model2,
                       model3, model4, model5, model6, model7);
            index_load += stride_in;
        }
        index_load_halos += stride_in;
        // Wait until the previous shared memory loads are complete
        __pipeline_wait_prior(1);
        __syncthreads();
        ism ^= 1;

        // Compute the derivatives with Z
#pragma unroll
        for (int j = 1; j <= i; j++)
            compute_z_contributions<ctype>(sm + ism, j - radius, invdzz, invdyz, invdzx, field1_zz[i - j], field1_zx[i - j],
                                           field1_yz[i - j], field2_zz[i - j], field2_zx[i - j], field2_yz[i - j], field1_zz[i - j],
                                           field1_zx[i - j], field1_yz[i - j], field2_zz[i - j], field2_zx[i - j], field2_yz[i - j]);
        compute_z_contributions<ctype>(sm + ism, -radius, invdzz, invdyz, invdzx, field1_zz[i], field1_zx[i], field1_yz[i],
                                       field2_zz[i], field2_zx[i], field2_yz[i], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
        // Compute the X Y derivatives once we're out of the Z halo
        if (i >= radius)
            compute_xy_contributions<ctype>(sm + ism, invdxx, invdyy, invdxy, field1_xx[i - radius], field1_yy[i - radius],
                                            field1_xy[i - radius], field2_xx[i - radius], field2_yy[i - radius],
                                            field2_xy[i - radius]);
    }

    int nzloop = min(bz, nz - izblock);
    // Loop on the Z dimension, except the last one
    for (int izloop = 0; izloop < nzloop - 1; izloop++) {
        // Protect the shared memory between iterations
        __syncthreads();
        // Prefetch the next arrays in shared memory
        async_load(sm + ism, index_load, index_load_halos, loader, loader_halos, txload, tyload, field1, field2, model1, model2,
                   model3, model4, model5, model6, model7);
        index_load += stride_in;
        index_load_halos += stride_in;
        // Wait until the previous shared memory loads are complete
        __pipeline_wait_prior(1);
        __syncthreads();
        ism ^= 1;

        // Finalize the oldest derivatives with Z
        compute_z_contributions<ctype>(sm + ism, radius, invdzz, invdyz, invdzx, field1_zz[0], field1_zx[0], field1_yz[0],
                                       field2_zz[0], field2_zx[0], field2_yz[0], field1_zz[0], field1_zx[0], field1_yz[0], field2_zz[0],
                                       field2_zx[0], field2_yz[0]);
        compute_result(sm + ism, field3_rhs, field4_rhs, field1_xx[0], field1_yy[0], field1_zz[0], field1_xy[0], field1_yz[0], field1_zx[0],
                       field2_xx[0], field2_yy[0], field2_zz[0], field2_xy[0], field2_yz[0], field2_zx[0]);
        if (active) {
            if constexpr (calc_snap) {
                field3[index_out] = field3_rhs;
                field4[index_out] = field4_rhs;
            } else {
                if constexpr (use_sponge) {
                    float A_xyz = min(A_xy, ABz[izblock + izloop].x);
                    float B_xyz = min(B_xy, ABz[izblock + izloop].y);
                    field3[index_out] = A_xyz * field3[index_out] + B_xyz * field3_rhs * dt;
                    field4[index_out] = A_xyz * field4[index_out] + B_xyz * field4_rhs * dt;
                } else {
                    // Use atomics as an optimization, we just let the L2 cache take care of the operation
                    // instead of bringing the original value all the way to the SM
                    atomicAdd(&field3[index_out], field3_rhs * dt);
                    atomicAdd(&field4[index_out], field4_rhs * dt);
                }
            }
        }
        index_out += stride_out;

        // Update the partial contributions, rotate them (read from i write to i-1)
#pragma unroll
        for (int i = 1; i < order; i++)
            compute_z_contributions<ctype>(sm + ism, radius - i, invdzz, invdyz, invdzx, field1_zz[i - 1], field1_zx[i - 1],
                                           field1_yz[i - 1], field2_zz[i - 1], field2_zx[i - 1], field2_yz[i - 1], field1_zz[i],
                                           field1_zx[i], field1_yz[i], field2_zz[i], field2_zx[i], field2_yz[i]);
        compute_z_contributions<ctype>(sm + ism, -radius, invdzz, invdyz, invdzx, field1_zz[order - 1], field1_zx[order - 1],
                                       field1_yz[order - 1], field2_zz[order - 1], field2_zx[order - 1], field2_yz[order - 1], 0.0f,
                                       0.0f, 0.0f, 0.0f, 0.0f, 0.0f);

        // Rotate the register queues without Z
        for (int i = 0; i < radius; i++) {
            field1_xx[i] = field1_xx[i + 1];
            field1_xy[i] = field1_xy[i + 1];
            field1_yy[i] = field1_yy[i + 1];
            field2_xx[i] = field2_xx[i + 1];
            field2_xy[i] = field2_xy[i + 1];
            field2_yy[i] = field2_yy[i + 1];
        }

        // Compute new derivatives in X and Y at the end of the register queue
        compute_xy_contributions<ctype>(sm + ism, invdxx, invdyy, invdxy, field1_xx[radius - 1], field1_yy[radius - 1],
                                        field1_xy[radius - 1], field2_xx[radius - 1], field2_yy[radius - 1], field2_xy[radius - 1]);
    }

    // Last iteration (no prefetch)
    // Wait until the last shared memory loads are complete
    __pipeline_wait_prior(0);
    __syncthreads();
    ism ^= 1;
    // Finalize the oldest derivatives with Z
    compute_z_contributions<ctype>(sm + ism, radius, invdzz, invdyz, invdzx, field1_zz[0], field1_zx[0], field1_yz[0], field2_zz[0],
                                   field2_zx[0], field2_yz[0], field1_zz[0], field1_zx[0], field1_yz[0], field2_zz[0], field2_zx[0],
                                   field2_yz[0]);
    compute_result(sm + ism, field3_rhs, field4_rhs, field1_xx[0], field1_yy[0], field1_zz[0], field1_xy[0], field1_yz[0], field1_zx[0],
                   field2_xx[0], field2_yy[0], field2_zz[0], field2_xy[0], field2_yz[0], field2_zx[0]);
    if (active) {
        if constexpr (calc_snap) {
            field3[index_out] = field3_rhs;
            field4[index_out] = field4_rhs;
        } else {
            if constexpr (use_sponge) {
                float A_xyz = min(A_xy, ABz[izblock + nzloop - 1].x);
                float B_xyz = min(B_xy, ABz[izblock + nzloop - 1].y);
                field3[index_out] = A_xyz * field3[index_out] + B_xyz * field3_rhs * dt;
                field4[index_out] = A_xyz * field4[index_out] + B_xyz * field4_rhs * dt;
            } else {
                atomicAdd(&field3[index_out], field3_rhs * dt);
                atomicAdd(&field4[index_out], field4_rhs * dt);
            }
        }
    }
}

// ****************************************************************************
// Kernel launcher

void
launch(float* field3, float* field4, const float* field1, const float* field2, const float* model1, const float* model2, const float* model3,
       const float* model4, const float* model5, const float* model6, const float* model7, const float2* ABx, const float2* ABy,
       const float2* ABz, int ixbeg, int ixend, int iybeg, int iyend, int izbeg, int izend, int ldimx_in, int ldimy_in,
       int ldimz_in, int ldimx_out, int ldimy_out, int ldimz_out, float dt, double dx, double dy, double dz, int order,
       cudaStream_t stream, bool calc_snap, bool use_sponge, bool simple)
{
    constexpr FD_COEF_TYPE ctype = _FD_COEF_LEAST_SQUARE;
    const int nx = ixend - ixbeg + 1;
    const int ny = iyend - iybeg + 1;
    const int nz = izend - izbeg + 1;

    float unused, invdxx, invdyy, invdzz, invdxy, invdyz, invdzx;
    kernel_utils::compute_fd_const(dx, dy, dz, unused, unused, unused, invdxx, invdyy, invdzz, invdxy, invdyz, invdzx);

    // Simple kernel
    if (simple) {
        volume_index vol3d_in(ldimx_in, ldimy_in, ldimz_in);
        volume_index vol3d_out(ldimx_out, ldimy_out, ldimz_out);
        dim3 threads(32, 4, 4); // This should match the launch_bounds attribute
        dim3 blocks((nx - 1) / threads.x + 1, (ny - 1) / threads.y + 1, (nz - 1) / threads.z + 1);
        if (!calc_snap) {
            if (order == 8)
                fwd_main_loop_2_1_simple<8, ctype, false><<<blocks, threads, 0, stream>>>(
                    vol3d_in, vol3d_out, field3, field4, field1, field2, model1, model2, model3, model4, model5, model6, model7, ABx, ABy, ABz, ixbeg, ixend,
                    iybeg, iyend, izbeg, izend, dt, invdxx, invdyy, invdzz, invdxy, invdyz, invdzx);
            if (order == 16)
                fwd_main_loop_2_1_simple<16, ctype, false><<<blocks, threads, 0, stream>>>(
                    vol3d_in, vol3d_out, field3, field4, field1, field2, model1, model2, model3, model4, model5, model6, model7, ABx, ABy, ABz, ixbeg, ixend,
                    iybeg, iyend, izbeg, izend, dt, invdxx, invdyy, invdzz, invdxy, invdyz, invdzx);
        } else {
            if (order == 8) {
                fwd_main_loop_2_1_simple<8, ctype, true><<<blocks, threads, 0, stream>>>(
                    vol3d_in, vol3d_out, field3, field4, field1, field2, model1, model2, model3, model4, model5, model6, model7, ABx, ABy, ABz, ixbeg, ixend,
                    iybeg, iyend, izbeg, izend, dt, invdxx, invdyy, invdzz, invdxy, invdyz, invdzx);
            } else if (order == 16) {
                fwd_main_loop_2_1_simple<16, ctype, true><<<blocks, threads, 0, stream>>>(
                    vol3d_in, vol3d_out, field3, field4, field1, field2, model1, model2, model3, model4, model5, model6, model7, ABx, ABy, ABz, ixbeg, ixend,
                    iybeg, iyend, izbeg, izend, dt, invdxx, invdyy, invdzz, invdxy, invdyz, invdzx);
            }
        }
    }
    // Optimized kernel
    else {
        // The base pointers skip the halos in Y and Z, but not in X.
        off_t off = (izbeg * ldimy_in + iybeg) * ldimx_in;

        // A sub-block of 64 for Z seems to work fine, might adjust it to 32 if running small volumes?
        constexpr int bz = 64;

        // We must launch enough threads to cover (nx + order / 2) points
        if (order == 8) {
            constexpr int bx = 32;
            constexpr int by = 16;
            dim3 b(bx, by);
            dim3 blocks((nx + 3) / bx + 1, (ny - 1) / by + 1, (nz - 1) / bz + 1);
            if (!calc_snap) {
                if (use_sponge) {
                    fwd_main_loop_2_1<8, ctype, bx, by, bz, false, true><<<blocks, b, 0, stream>>>(
                        field3 + off, field4 + off, field1 + off, field2 + off, model1 + off, model2 + off, model3 + off, model4 + off, model5 + off,
                        model6 + off, model7 + off, ABx, ABy + iybeg, ABz + izbeg, ixbeg, nx, ny, nz, ldimx_in, ldimy_in,
                        ldimx_out, ldimy_out, dt, invdxx, invdyy, invdzz, invdxy, invdyz, invdzx);
                } else {
                    fwd_main_loop_2_1<8, ctype, bx, by, bz, false, false><<<blocks, b, 0, stream>>>(
                        field3 + off, field4 + off, field1 + off, field2 + off, model1 + off, model2 + off, model3 + off, model4 + off, model5 + off,
                        model6 + off, model7 + off, nullptr, nullptr, nullptr, ixbeg, nx, ny, nz, ldimx_in, ldimy_in, ldimx_out,
                        ldimy_out, dt, invdxx, invdyy, invdzz, invdxy, invdyz, invdzx);
                }
            } else {
                // field3 and field4 are actually snap_field1 and snap_field2 here, and they have no halo, so don't add off
                fwd_main_loop_2_1<8, ctype, bx, by, bz, true, false><<<blocks, b, 0, stream>>>(
                    field3, field4, field1 + off, field2 + off, model1 + off, model2 + off, model3 + off, model4 + off, model5 + off, model6 + off,
                    model7 + off, nullptr, nullptr, nullptr, ixbeg, nx, ny, nz, ldimx_in, ldimy_in, ldimx_out, ldimy_out,
                    dt, invdxx, invdyy, invdzz, invdxy, invdyz, invdzx);
            }
        }
        if (order == 16) {
            // 16th order should use a block size of 16x16. More than 256 threads will spill registers.
            constexpr int bx = 16;
            constexpr int by = 16;
            constexpr dim3 b(bx, by);
            dim3 blocks((nx + 7) / bx + 1, (ny - 1) / by + 1, (nz - 1) / bz + 1);
            if (!calc_snap) {
                if (use_sponge) {
                    fwd_main_loop_2_1<16, ctype, bx, by, bz, false, true><<<blocks, b, 0, stream>>>(
                        field3 + off, field4 + off, field1 + off, field2 + off, model1 + off, model2 + off, model3 + off, model4 + off, model5 + off,
                        model6 + off, model7 + off, ABx, ABy + iybeg, ABz + izbeg, ixbeg, nx, ny, nz, ldimx_in, ldimy_in,
                        ldimx_out, ldimy_out, dt, invdxx, invdyy, invdzz, invdxy, invdyz, invdzx);
                } else {
                    fwd_main_loop_2_1<16, ctype, bx, by, bz, false, false><<<blocks, b, 0, stream>>>(
                        field3 + off, field4 + off, field1 + off, field2 + off, model1 + off, model2 + off, model3 + off, model4 + off, model5 + off,
                        model6 + off, model7 + off, nullptr, nullptr, nullptr, ixbeg, nx, ny, nz, ldimx_in, ldimy_in, ldimx_out,
                        ldimy_out, dt, invdxx, invdyy, invdzz, invdxy, invdyz, invdzx);
                }
            } else {
                // field3 and field4 are actually snap_field1 and snap_field2 here, and they have no halo, so don't add off
                fwd_main_loop_2_1<16, ctype, bx, by, bz, true, false><<<blocks, b, 0, stream>>>(
                    field3, field4, field1 + off, field2 + off, model1 + off, model2 + off, model3 + off, model4 + off, model5 + off, model6 + off,
                    model7 + off, nullptr, nullptr, nullptr, ixbeg, nx, ny, nz, ldimx_in, ldimy_in, ldimx_out, ldimy_out,
                    dt, invdxx, invdyy, invdzz, invdxy, invdyz, invdzx);
            }
        }
    }
}

} // namespace fwd_main_loop_2_1_inner_gpu
