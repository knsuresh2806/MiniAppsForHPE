#include <cuda_pipeline.h>
#include "helper_kernels_gpu.h"
#include "kernel_utils.h"
#include "kernel_type1_gpu_adj.h"
#include <stdio.h>

namespace adj_main_loop_2_inner_gpu {

template <int order, FD_COEF_TYPE ctype, int bx, int by, int bz>
__global__
__launch_bounds__(bx* by* bz) void adj_main_loop_2_simple(
    volume_index vol3d, float* __restrict__ field3, float* __restrict__ field4, const float* __restrict__ field1,
    const float* __restrict__ field2, const float* __restrict__ model1, const float* __restrict__ model2,
    const float* __restrict__ model3, const float* __restrict__ model4, const float* __restrict__ model5,
    const float* __restrict__ model6, const float* __restrict__ model7, const float2* __restrict__ ABx,
    const float2* __restrict__ ABy, const float2* __restrict__ ABz, int ixbeg, int ixend, int iybeg, int iyend,
    int izbeg, int izend, float dt, float invdxx, float invdyy, float invdzz, float invdxy, float invdyz, float invdzx)
{
    int ix = ixbeg + blockIdx.x * bx + threadIdx.x;
    int iy = iybeg + blockIdx.y * by + threadIdx.y;
    int iz = izbeg + blockIdx.z * bz + threadIdx.z;

    // All threads outside the boundaries can exit
    if (ix > ixend || iy > iyend || iz > izend)
        return;

    // Compute all the derivatives using the helper functions
    float field3_xx, field3_yy, field3_zz, field3_xy, field3_yz, field3_xz;
    float field4_xx, field4_yy, field4_zz, field4_xy, field4_yz, field4_xz;
    helper_kernels_gpu::drfield3X_cg_simulator2_adj<order, ctype>(field1, field2, model5, model1, model2, model3, model4, vol3d, ix, iy, iz, invdxx,
                                                       field3_xx, field4_xx);
    helper_kernels_gpu::drvYY_cg_simulator2_adj<order, ctype>(field1, field2, model6, model1, model2, model3, model4, vol3d, ix, iy, iz, invdyy,
                                                       field3_yy, field4_yy);
    helper_kernels_gpu::drfield3Y_cg_simulator2_adj<order, ctype>(field1, field2, model5, model6, model1, model2, model3, model4, vol3d, ix, iy, iz, invdxy,
                                                       field3_xy, field4_xy);
    helper_kernels_gpu::drfield4Z_cg_simulator2_adj<order, ctype>(field1, field2, model7, model1, model2, model3, model4, vol3d, ix, iy, iz, invdzz,
                                                       field3_zz, field4_zz);
    helper_kernels_gpu::drfield3Z_cg_simulator2_adj<order, ctype>(field1, field2, model5, model7, model1, model2, model3, model4, vol3d, ix, iy, iz, invdzx,
                                                       field3_xz, field4_xz);
    helper_kernels_gpu::drvYZ_cg_simulator2_adj<order, ctype>(field1, field2, model6, model7, model1, model2, model3, model4, vol3d, ix, iy, iz, invdyz,
                                                       field3_yz, field4_yz);

    float field3_rhs = dt * (field3_xx + field3_yy + field3_zz + field3_xy + field3_yz + field3_xz);
    float field4_rhs = dt * (field4_xx + field4_yy + field4_zz + field4_xy + field4_yz + field4_xz);

    // Write the results
    float A_xyz = min(ABx[ix].x, min(ABy[iy].x, ABz[iz].x));
    float B_xyz = min(ABx[ix].y, min(ABy[iy].y, ABz[iz].y));
    vol3d(field3, ix, iy, iz) = A_xyz * vol3d(field3, ix, iy, iz) + B_xyz * field3_rhs;
    vol3d(field4, ix, iy, iz) = A_xyz * vol3d(field4, ix, iy, iz) + B_xyz * field4_rhs;
}

template <int order, int bx, int by>
struct smstruct
{
    // Inputs
    float field1[by + order][bx + order];
    float field2[by + order][bx + order];
    float model1[by + order][bx + order];
    float model2[by + order][bx + order];
    float model3[by + order][bx + order];
    float model4[by + order][bx + order];
    float model5[by + order][bx + order];
    float model6[by + order][bx + order];
    float model7[by + order][bx + order];
    // Temp arrays, generated from the inputs above, then used for derivatives.
    float field3_xx[by][bx + order];
    float field4_xx[by][bx + order];
    float field3_yy[by + order][bx];
    float field4_yy[by + order][bx];
    float field3_zz[by][bx];
    float field4_zz[by][bx];
    float field3_xy[by + order][bx + order];
    float field4_xy[by + order][bx + order];
    float field3_yz[by + order][bx];
    float field4_yz[by + order][bx];
    float field3_zx[by][bx + order];
    float field4_zx[by][bx + order];
};

// ****************************************************************************
// Device function to asynchronously load the shared memory inputs as float4
template <int order, int bx, int by>
__device__ inline void
async_load(smstruct<order, bx, by>* sm, int index_load, bool loader, int txload, int tyload, const float* field1,
           const float* field2, const float* model1, const float* model2, const float* model3, const float* model4, const float* model5,
           const float* model6, const float* model7)
{
    if (loader) {
        __pipeline_memcpy_async(&sm->field1[tyload][txload], field1 + index_load, sizeof(float4));
        __pipeline_memcpy_async(&sm->field2[tyload][txload], field2 + index_load, sizeof(float4));
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
// Compute the shared memory temporary arrays, from the shared memory inputs.
// The threads are responsible for the float4 values they loaded (same mapping)
// z_only: In Z halo, we only need to compute Z-components (ZZ, YZ, ZX)

template <int order, int bx, int by>
__device__ inline void
compute_temp_arrays(smstruct<order, bx, by>* sm, bool loader, int txload, int tyload, bool center_x, bool center_y,
                    bool z_only = false)
{
    const int r = order / 2;
    if (loader) {
        float4 rx = *reinterpret_cast<float4*>(&sm->model5[tyload][txload]);
        float4 ry = *reinterpret_cast<float4*>(&sm->model6[tyload][txload]);
        float4 rz = *reinterpret_cast<float4*>(&sm->model7[tyload][txload]);
        float4 model1 = *reinterpret_cast<float4*>(&sm->model1[tyload][txload]);
        float4 model2 = *reinterpret_cast<float4*>(&sm->model2[tyload][txload]);
        float4 model3 = *reinterpret_cast<float4*>(&sm->model3[tyload][txload]);
        float4 model4 = *reinterpret_cast<float4*>(&sm->model4[tyload][txload]);
        float4 field1 = *reinterpret_cast<float4*>(&sm->field1[tyload][txload]);
        float4 field2 = *reinterpret_cast<float4*>(&sm->field2[tyload][txload]);

        // Using float4 math
        using namespace helper_kernels_gpu;
        float4 rxx = rx * rx;
        float4 ryy = ry * ry;
        float4 rzz = rz * rz;
        float4 rxy = 2.0f * rx * ry;
        float4 ryz = 2.0f * ry * rz;
        float4 rzx = 2.0f * rz * rx;

        float4 lap_x = model1 * field1 + model2 * field2;
        float4 hz_x = (-model1 + model4) * field1 - model2 * field2;

        float4 lap_z = model4 * field2;
        float4 hz_z = model2 * field1 + (-model4 + model3) * field2;

        // Write XY values (X and Y halos)
        if (!z_only) {
            *reinterpret_cast<float4*>(&sm->field3_xy[tyload][txload]) = rxy * hz_x;
            *reinterpret_cast<float4*>(&sm->field4_xy[tyload][txload]) = rxy * hz_z;
        }

        // write XX and XZ values (X halo but no Y halo)
        if (center_y) {
            if (!z_only) {
                *reinterpret_cast<float4*>(&sm->field3_xx[tyload - r][txload]) = lap_x + rxx * hz_x;
                *reinterpret_cast<float4*>(&sm->field4_xx[tyload - r][txload]) = lap_z + rxx * hz_z;
            }
            *reinterpret_cast<float4*>(&sm->field3_zx[tyload - r][txload]) = rzx * hz_x;
            *reinterpret_cast<float4*>(&sm->field4_zx[tyload - r][txload]) = rzx * hz_z;
        }
        // Write YY and YZ values (Y halo but no X halo)
        if (center_x) {
            if (!z_only) {
                *reinterpret_cast<float4*>(&sm->field3_yy[tyload][txload - r]) = lap_x + ryy * hz_x;
                *reinterpret_cast<float4*>(&sm->field4_yy[tyload][txload - r]) = lap_z + ryy * hz_z;
            }
            *reinterpret_cast<float4*>(&sm->field3_yz[tyload][txload - r]) = ryz * hz_x;
            *reinterpret_cast<float4*>(&sm->field4_yz[tyload][txload - r]) = ryz * hz_z;
        }
        // Write ZZ values (no X nor Y halos)
        if (center_x && center_y) {
            *reinterpret_cast<float4*>(&sm->field3_zz[tyload - r][txload - r]) = lap_x + rzz * hz_x;
            *reinterpret_cast<float4*>(&sm->field4_zz[tyload - r][txload - r]) = lap_z + rzz * hz_z;
        }
    }
}

// ****************************************************************************
// Contributions of the temp shared memory at iz+ir, with ir between [-order/2 : order/2].
// The computation is done out-of-place to allow rotation of the register queues on the fly.
// When everything is unrolled and inlined properly, the FD coefficient is known at compile time
// and the conditionals should disappear

template <FD_COEF_TYPE ctype, int order, int bx, int by>
__device__ inline void
compute_z_contributions(const smstruct<order, bx, by>* sm, int ir, float invdzz, float invdyz, float invdzx,
                        float& field3_rhs, float& field4_rhs, float prefield3_rhs, float prefield4_rhs)
{
    const int radius = order / 2;
    const int tx = threadIdx.x + radius;
    const int ty = threadIdx.y + radius;
    float coef = helper_kernels_gpu::df2_coef<order, ctype>(abs(ir));
    if (ir == 0) {
        // No offset in Z -> no mixed derivative contribution, only ZZ
        field3_rhs = prefield3_rhs + invdzz * coef * sm->field3_zz[threadIdx.y][threadIdx.x];
        field4_rhs = prefield4_rhs + invdzz * coef * sm->field4_zz[threadIdx.y][threadIdx.x];
    } else {
        field3_rhs = prefield3_rhs + invdzz * coef * sm->field3_zz[threadIdx.y][threadIdx.x] +
                 invdyz * coef * (sm->field3_yz[ty + ir][threadIdx.x] - sm->field3_yz[ty - ir][threadIdx.x]) +
                 invdzx * coef * (sm->field3_zx[threadIdx.y][tx + ir] - sm->field3_zx[threadIdx.y][tx - ir]);
        field4_rhs = prefield4_rhs + invdzz * coef * sm->field4_zz[threadIdx.y][threadIdx.x] +
                 invdyz * coef * (sm->field4_yz[ty + ir][threadIdx.x] - sm->field4_yz[ty - ir][threadIdx.x]) +
                 invdzx * coef * (sm->field4_zx[threadIdx.y][tx + ir] - sm->field4_zx[threadIdx.y][tx - ir]);
    }
}

// ****************************************************************************
// Compute the X and Y contributions of the temporary shared memory arrays

template <FD_COEF_TYPE ctype, int order, int bx, int by>
__device__ inline void
compute_xy_contributions(const smstruct<order, bx, by>* sm, float invdxx, float invdyy, float invdxy, float& field3_rhs,
                         float& field4_rhs)
{
    const int radius = order / 2;
    const int tx = threadIdx.x + radius;
    const int ty = threadIdx.y + radius;
    float coef = helper_kernels_gpu::df2_coef<order, ctype>(0);
    float tmpx_xx = coef * sm->field3_xx[threadIdx.y][tx];
    float tmpz_xx = coef * sm->field4_xx[threadIdx.y][tx];
    float tmpx_yy = coef * sm->field3_yy[ty][threadIdx.x];
    float tmpz_yy = coef * sm->field4_yy[ty][threadIdx.x];
    float tmpx_xy = 0.0f;
    float tmpz_xy = 0.0f;
#pragma unroll
    for (int i = 1; i <= radius; i++) {
        coef = helper_kernels_gpu::df2_coef<order, ctype>(i);
        tmpx_xx += coef * (sm->field3_xx[threadIdx.y][tx + i] + sm->field3_xx[threadIdx.y][tx - i]);
        tmpz_xx += coef * (sm->field4_xx[threadIdx.y][tx + i] + sm->field4_xx[threadIdx.y][tx - i]);
        tmpx_yy += coef * (sm->field3_yy[ty + i][threadIdx.x] + sm->field3_yy[ty - i][threadIdx.x]);
        tmpz_yy += coef * (sm->field4_yy[ty + i][threadIdx.x] + sm->field4_yy[ty - i][threadIdx.x]);
        tmpx_xy += coef * (sm->field3_xy[ty + i][tx + i] - sm->field3_xy[ty + i][tx - i] + sm->field3_xy[ty - i][tx - i] -
                           sm->field3_xy[ty - i][tx + i]);
        tmpz_xy += coef * (sm->field4_xy[ty + i][tx + i] - sm->field4_xy[ty + i][tx - i] + sm->field4_xy[ty - i][tx - i] -
                           sm->field4_xy[ty - i][tx + i]);
    }
    field3_rhs += invdxx * tmpx_xx + invdyy * tmpx_yy + invdxy * tmpx_xy;
    field4_rhs += invdxx * tmpz_xx + invdyy * tmpz_yy + invdxy * tmpz_xy;
}


template <int order, FD_COEF_TYPE ctype, int bx, int by, int bz, bool use_sponge>
__global__
__launch_bounds__(bx* by) void adj_main_loop_2(
    float* __restrict__ field3, float* __restrict__ field4, const float* __restrict__ field1, const float* __restrict__ field2,
    const float* __restrict__ model1, const float* __restrict__ model2, const float* __restrict__ model3,
    const float* __restrict__ model4, const float* __restrict__ model5, const float* __restrict__ model6,
    const float* __restrict__ model7, const float2* __restrict__ ABx, const float2* __restrict__ ABy,
    const float2* __restrict__ ABz, int nx, int ny, int nz, int ldimx, int ldimy, float dt, float invdxx, float invdyy,
    float invdzz, float invdxy, float invdyz, float invdzx)
{
    static_assert(order == 8 || order == 16); // Radius must be a multiple of 4
    constexpr int radius = order / 2;

    // Global position of the thread block, using 2D thread blocks
    const int ixblock = blockIdx.x * bx;
    const int iyblock = blockIdx.y * by;
    const int izblock = blockIdx.z * bz;
    const int stride = ldimx * ldimy;

    // Dynamic shared memory. Using a non-templated pointer to get the base address,
    // because we can't have a templated extern declaration with different template args,
    // and we need dynamic shared memory to use > 48KB per thread block.
    extern __shared__ char dsmem[];
    // Now we can cast the base pointer to the proper templated struct
    smstruct<order, bx, by>* sm = reinterpret_cast<smstruct<order, bx, by>*>(dsmem);

    // Which threads are active (writing values), and their global index
    const int ix = ixblock + threadIdx.x;
    const int iy = iyblock + threadIdx.y;
    const bool active = ix < nx && iy < ny;
    int index = (izblock * ldimy + iy) * ldimx + ix;

    float A_xy, B_xy;
    if (use_sponge && active) {
        A_xy = min(ABx[ix].x, ABy[iy].x);
        B_xy = min(ABx[ix].y, ABy[iy].y);
    }

    // Load the shared memory as float4, and remap the threads from (bx, by) into (bx/2, by*2)
    // These reads are offset by -radius from the block offset, to read the halos.
    const int tid = threadIdx.y * bx + threadIdx.x;
    static_assert(bx >= order && by >= order, "Incompatible block size for this order");
    const int txload = 4 * (tid % (bx / 2));
    const int tyload = tid / (bx / 2);
    const int ixload = ixblock + txload - radius;
    const int iyload = iyblock + tyload - radius;
    const bool loader = ((bx == order || txload < bx + order) && (by == order || tyload < by + order)) &&
                        (ixload < nx + radius && iyload < ny + radius);
    int index_load = ((izblock - radius) * ldimy + iyload) * ldimx + ixload;
    // The threads will compute the temp shared arrays with the same float4 thread mapping.
    // We need to identify the threads which are not in the X or Y halos
    const bool center_x = txload >= radius && txload < bx + radius;
    const bool center_y = tyload >= radius && tyload < by + radius;

    // Register queues for the results
    float field3_rhs[order];
    float field4_rhs[order];

    // Async load the first input values in shared memory
    async_load(sm, index_load, loader, txload, tyload, field1, field2, model1, model2, model3, model4, model5, model6, model7);
    index_load += stride;

    // Prime the register queues
    // Both loops must be unrolled to allow proper inlining and resolution of FD coefficients.
#pragma unroll
    for (int i = 0; i < order; i++) {
        // Wait for the async load, compute the temporary arrays in shared memory
        __pipeline_wait_prior(0);
        __syncthreads();

        // In the first "halo" iterations, we only compute the Z components
        compute_temp_arrays(sm, loader, txload, tyload, center_x, center_y, i < radius);

        __syncthreads();

        // Async load the next inputs
        async_load(sm, index_load, loader, txload, tyload, field1, field2, model1, model2, model3, model4, model5, model6, model7);
        index_load += stride;

        // Compute the Z contributions of the temp arrays to the register queues
#pragma unroll
        for (int j = 1; j <= i; j++)
            compute_z_contributions<ctype>(sm, j - radius, invdzz, invdyz, invdzx, field3_rhs[i - j], field4_rhs[i - j],
                                           field3_rhs[i - j], field4_rhs[i - j]);
        compute_z_contributions<ctype>(sm, -radius, invdzz, invdyz, invdzx, field3_rhs[i], field4_rhs[i], 0.0f, 0.0f);

        // Compute the X Y contributions of the temp arrays once we're out of the Z halo
        if (i >= radius)
            compute_xy_contributions<ctype>(sm, invdxx, invdyy, invdxy, field3_rhs[i - radius], field4_rhs[i - radius]);
    }

    int nzloop = min(bz, nz - izblock);
    // Loop on the Z dimension, except the last one
    for (int izloop = 0; izloop < nzloop - 1; izloop++) {
        // Wait for the async load, compute the temporary arrays in shared memory
        __pipeline_wait_prior(0);
        __syncthreads();
        compute_temp_arrays(sm, loader, txload, tyload, center_x, center_y);

        __syncthreads();

        // Async load the next inputs
        async_load(sm, index_load, loader, txload, tyload, field1, field2, model1, model2, model3, model4, model5, model6, model7);
        index_load += stride;

        // Compute the final contribution to the oldest values in the register queue
        compute_z_contributions<ctype>(sm, radius, invdzz, invdyz, invdzx, field3_rhs[0], field4_rhs[0], field3_rhs[0], field4_rhs[0]);

        // Write the results
        if (active) {
            if constexpr (use_sponge) {
                float A_xyz = min(A_xy, ABz[izblock + izloop].x);
                float B_xyz = min(B_xy, ABz[izblock + izloop].y);
                field3[index] = A_xyz * field3[index] + B_xyz * field3_rhs[0] * dt;
                field4[index] = A_xyz * field4[index] + B_xyz * field4_rhs[0] * dt;
            } else {
                atomicAdd(&field3[index], field3_rhs[0] * dt);
                atomicAdd(&field4[index], field4_rhs[0] * dt);
            }
        }
        index += stride;

        // Compute the Z contributions and rotate the register queues on the fly (i -> i-1)
#pragma unroll
        for (int i = 1; i < order; i++)
            compute_z_contributions<ctype>(sm, radius - i, invdzz, invdyz, invdzx, field3_rhs[i - 1], field4_rhs[i - 1],
                                           field3_rhs[i], field4_rhs[i]);
        compute_z_contributions<ctype>(sm, -radius, invdzz, invdyz, invdzx, field3_rhs[order - 1], field4_rhs[order - 1], 0.0f,
                                       0.0f);

        // Compute the X and Y contributions
        compute_xy_contributions<ctype>(sm, invdxx, invdyy, invdxy, field3_rhs[radius - 1], field4_rhs[radius - 1]);
    }
    // Last Z (no prefetch)
    // Wait for the async load, compute the temporary arrays in shared memory
    __pipeline_wait_prior(0);
    __syncthreads();
    compute_temp_arrays(sm, loader, txload, tyload, center_x, center_y);

    __syncthreads();

    // Compute the final contribution to the oldest values in the register queue
    compute_z_contributions<ctype>(sm, radius, invdzz, invdyz, invdzx, field3_rhs[0], field4_rhs[0], field3_rhs[0], field4_rhs[0]);

    // Write the results
    if (active) {
        if constexpr (use_sponge) {
            float A_xyz = min(A_xy, ABz[izblock + nzloop - 1].x);
            float B_xyz = min(B_xy, ABz[izblock + nzloop - 1].y);
            field3[index] = A_xyz * field3[index] + B_xyz * field3_rhs[0] * dt;
            field4[index] = A_xyz * field4[index] + B_xyz * field4_rhs[0] * dt;
        } else {
            atomicAdd(&field3[index], field3_rhs[0] * dt);
            atomicAdd(&field4[index], field4_rhs[0] * dt);
        }
    }
}

// ****************************************************************************
// Kernel launcher

void
launch(float* field3, float* field4, const float* field1, const float* field2, const float* model1, const float* model2, const float* model3,
       const float* model4, const float* model5, const float* model6, const float* model7, const float2* ABx, const float2* ABy,
       const float2* ABz, int ixbeg, int ixend, int iybeg, int iyend, int izbeg, int izend, int ldimx, int ldimy,
       int ldimz, float dt, double dx, double dy, double dz, int order, cudaStream_t stream, bool use_sponge,
       bool simple)
{
    constexpr FD_COEF_TYPE ctype = _FD_COEF_LEAST_SQUARE;
    const int nx = ixend - ixbeg + 1;
    const int ny = iyend - iybeg + 1;
    const int nz = izend - izbeg + 1;

    float invdxx, invdyy, invdzz, invdxy, invdyz, invdzx, unused;
    kernel_utils::compute_fd_const(dx, dy, dz, unused, unused, unused, invdxx, invdyy, invdzz, invdxy, invdyz, invdzx);

    if (simple) {
        volume_index vol3d(ldimx, ldimy, ldimz);
        // Using a fixed block size to cover the 3D volume
        constexpr int bx = 32;
        constexpr int by = 4;
        constexpr int bz = 4;
        constexpr dim3 b(bx, by, bz);
        const dim3 blocks((nx - 1) / bx + 1, (ny - 1) / by + 1, (nz - 1) / bz + 1);
        if (order == 8) {
            adj_main_loop_2_simple<8, ctype, bx, by, bz><<<blocks, b, 0, stream>>>(
                vol3d, field3, field4, field1, field2, model1, model2, model3, model4, model5, model6, model7, ABx, ABy, ABz, ixbeg, ixend, iybeg, iyend,
                izbeg, izend, dt, invdxx, invdyy, invdzz, invdxy, invdyz, invdzx);
        }
        if (order == 16) {
            adj_main_loop_2_simple<16, ctype, bx, by, bz><<<blocks, b, 0, stream>>>(
                vol3d, field3, field4, field1, field2, model1, model2, model3, model4, model5, model6, model7, ABx, ABy, ABz, ixbeg, ixend, iybeg, iyend,
                izbeg, izend, dt, invdxx, invdyy, invdzz, invdxy, invdyz, invdzx);
        }

    } else {
        // A sub-block of 64 for Z seems to work fine, might adjust it to 32 if running small volumes?
        constexpr int bz = 64;

        // The base pointers skip the halos in X, Y and Z.
        off_t off = (izbeg * ldimy + iybeg) * ldimx + ixbeg;

        // A block size of 32 x 16 should work well for both 8th order and 16th order.
        constexpr int bx = 32;
        constexpr int by = 16;
        constexpr dim3 b(bx, by);
        dim3 blocks((nx - 1) / bx + 1, (ny - 1) / by + 1, (nz - 1) / bz + 1);

        if (order == 8) {
            // Dynamic shared memory, and allow > 48KB
            size_t shm = sizeof(smstruct<8, bx, by>);
            if (use_sponge) {
                cudaFuncSetAttribute(adj_main_loop_2<8, ctype, bx, by, bz, true>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, shm);
                adj_main_loop_2<8, ctype, bx, by, bz, true><<<blocks, b, shm, stream>>>(
                    field3 + off, field4 + off, field1 + off, field2 + off, model1 + off, model2 + off, model3 + off, model4 + off, model5 + off,
                    model6 + off, model7 + off, ABx + ixbeg, ABy + iybeg, ABz + izbeg, nx, ny, nz, ldimx, ldimy, dt, invdxx,
                    invdyy, invdzz, invdxy, invdyz, invdzx);
            } else {
                cudaFuncSetAttribute(adj_main_loop_2<8, ctype, bx, by, bz, false>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, shm);
                adj_main_loop_2<8, ctype, bx, by, bz, false><<<blocks, b, shm, stream>>>(
                    field3 + off, field4 + off, field1 + off, field2 + off, model1 + off, model2 + off, model3 + off, model4 + off, model5 + off,
                    model6 + off, model7 + off, nullptr, nullptr, nullptr, nx, ny, nz, ldimx, ldimy, dt, invdxx, invdyy, invdzz,
                    invdxy, invdyz, invdzx);
            }
        }
        if (order == 16) {
            // Dynamic shared memory to allow > 48KB
            size_t shm = sizeof(smstruct<16, bx, by>);
            if (use_sponge) {
                cudaFuncSetAttribute(adj_main_loop_2<16, ctype, bx, by, bz, true>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, shm);
                adj_main_loop_2<16, ctype, bx, by, bz, true><<<blocks, b, shm, stream>>>(
                    field3 + off, field4 + off, field1 + off, field2 + off, model1 + off, model2 + off, model3 + off, model4 + off, model5 + off,
                    model6 + off, model7 + off, ABx + ixbeg, ABy + iybeg, ABz + izbeg, nx, ny, nz, ldimx, ldimy, dt, invdxx,
                    invdyy, invdzz, invdxy, invdyz, invdzx);
            } else {
                cudaFuncSetAttribute(adj_main_loop_2<16, ctype, bx, by, bz, false>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, shm);
                adj_main_loop_2<16, ctype, bx, by, bz, false><<<blocks, b, shm, stream>>>(
                    field3 + off, field4 + off, field1 + off, field2 + off, model1 + off, model2 + off, model3 + off, model4 + off, model5 + off,
                    model6 + off, model7 + off, nullptr, nullptr, nullptr, nx, ny, nz, ldimx, ldimy, dt, invdxx, invdyy, invdzz,
                    invdxy, invdyz, invdzx);
            }
        }
    }
}

} 

namespace adj_main_loop_2_pml_gpu {} // namespace adj_main_loop_2_pml_gpu
