#include <cuda_pipeline.h>
#include "cuda_utils.h"
#include "helper_kernels_gpu.h"
#include "kernel_type2_grad_snap.h"

namespace adj_kernel3_derivatives {

// Structure for the shared memory arrays
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
    float Model11[by + order][bx + order];

    // Temp arrays, generated from the inputs above, then used for derivatives.
    float field1_xx[by][bx + order];
    float field2_xx[by][bx + order];
    float field1_yy[by + order][bx];
    float field2_yy[by + order][bx];
    float field1_zz[by][bx];
    float field2_zz[by][bx];
    float field1_xy[by + order][bx + order];
    float field2_xy[by + order][bx + order];
    float field1_yz[by + order][bx];
    float field2_yz[by + order][bx];
    float field1_zx[by][bx + order];
    float field2_zx[by][bx + order];
    // TODO: VRKG
    // we issue an async load immediately after computing temp arrays
    // when we hit compute_xy_contributions shared memory (non mixed derivatives buffers) is modified with next zplane's values
    float Model11Dup[by + order][bx + order];
};

// ****************************************************************************
// Device function to asynchronously load the shared memory inputs as float4
template <int order, int bx, int by>
__device__ inline void
async_load(smstruct<order, bx, by>* sm, int index_load, bool loader, int txload, int tyload, const float* field1,
           const float* field2, const float* model1, const float* model2, const float* model3, const float* model4, const float* model5,
           const float* model6, const float* model7, const float* Model11)
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
        __pipeline_memcpy_async(&sm->Model11[tyload][txload], Model11 + index_load, sizeof(float4));
    }
    __pipeline_commit();
}

template <int order, int bx, int by>
__device__ inline void
compute_temp_arrays(smstruct<order, bx, by>* sm, bool loader, int txload, int tyload, bool center_x, bool center_y,
                    bool z_only = false)
{
    constexpr int r = order / 2;
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
        float4 invrho = *reinterpret_cast<float4*>(&sm->Model11[tyload][txload]);

        // VRKG
        // copy the inv_rho for future calculation
        *reinterpret_cast<float4*>(&sm->Model11Dup[tyload][txload]) = invrho;

        // Using float4 math
        using namespace helper_kernels_gpu;
        float4 rho = 1.0f / invrho;
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
            *reinterpret_cast<float4*>(&sm->field1_xy[tyload][txload]) = rxy * hz_x;
            *reinterpret_cast<float4*>(&sm->field2_xy[tyload][txload]) = rxy * hz_z;
        }
        if (center_y) {
            if (!z_only) {
                *reinterpret_cast<float4*>(&sm->field1_xx[tyload - r][txload]) = rho * (lap_x + rxx * hz_x);
                *reinterpret_cast<float4*>(&sm->field2_xx[tyload - r][txload]) = rho * (lap_z + rxx * hz_z);
            }
            *reinterpret_cast<float4*>(&sm->field1_zx[tyload - r][txload]) = rzx * hz_x;
            *reinterpret_cast<float4*>(&sm->field2_zx[tyload - r][txload]) = rzx * hz_z;
        }
        // Write YY and YZ values (Y halo but no X halo)
        if (center_x) {
            if (!z_only) {
                *reinterpret_cast<float4*>(&sm->field1_yy[tyload][txload - r]) = rho * (lap_x + ryy * hz_x);
                *reinterpret_cast<float4*>(&sm->field2_yy[tyload][txload - r]) = rho * (lap_z + ryy * hz_z);
            }
            *reinterpret_cast<float4*>(&sm->field1_yz[tyload][txload - r]) = ryz * hz_x;
            *reinterpret_cast<float4*>(&sm->field2_yz[tyload][txload - r]) = ryz * hz_z;
        }
        // Write ZZ values (no X nor Y halos)
        if (center_x && center_y) {
            *reinterpret_cast<float4*>(&sm->field1_zz[tyload - r][txload - r]) = rho * (lap_x + rzz * hz_x); // * Model11
            *reinterpret_cast<float4*>(&sm->field2_zz[tyload - r][txload - r]) = rho * (lap_z + rzz * hz_z); // * Model11
        }
    }
}

// Compute z contributions for dfield1 contributions
template <FD_COEF_TYPE ctype, int order, int bx, int by>
__device__ inline void
compute_z_contributionsdtd(const smstruct<order, bx, by>* sm, int ir, float invdz, float& dfield1dz, float& dfield2dz,
                           float prevdfield1dz, float prevdfield2dz)
{
    constexpr int radius = order / 2;
    float alpha = 0.0f;
    dfield1dz = 0.0f;
    dfield2dz = 0.0f;
    const int ty = threadIdx.y + radius;
    const int tx = threadIdx.x + radius;

    if (ir < 0) {
        alpha = -invdz * helper_kernels_gpu::df_coef<order, ctype>(abs(ir));
    } else {
        alpha = invdz * helper_kernels_gpu::df_coef<order, ctype>(ir + 1);
    }
    // VRKG
    // Model11 is accounted before writing the results
    dfield1dz = prevdfield1dz + alpha * sm->field1_zz[threadIdx.y][threadIdx.x];
    dfield2dz = prevdfield2dz + alpha * sm->field2_zz[threadIdx.y][threadIdx.x];
}

template <FD_COEF_TYPE ctype, int order, int bx, int by>
__device__ inline void
compute_z_contributionsvrhs(const smstruct<order, bx, by>* sm, int ir, float invdyz, float invdzx, float& field3_rhs,
                            float& field4_rhs, float prefield3_rhs, float prefield4_rhs)
{
    constexpr int radius = order / 2;
    const int tx = threadIdx.x + radius;
    const int ty = threadIdx.y + radius;
    float coef = helper_kernels_gpu::df2_coef<order, ctype>(abs(ir));

    if (ir != 0) {
        // scatterDyz, scatterDzx
        field3_rhs = prefield3_rhs + (invdyz * coef * (sm->field1_yz[ty + ir][threadIdx.x] - sm->field1_yz[ty - ir][threadIdx.x])) +
                 (invdzx * coef * (sm->field1_zx[threadIdx.y][tx + ir] - sm->field1_zx[threadIdx.y][tx - ir]));

        field4_rhs = prefield4_rhs + (invdyz * coef * (sm->field2_yz[ty + ir][threadIdx.x] - sm->field2_yz[ty - ir][threadIdx.x])) +
                 (invdzx * coef * (sm->field2_zx[threadIdx.y][tx + ir] - sm->field2_zx[threadIdx.y][tx - ir]));
    } else {
        field3_rhs = prefield3_rhs;
        field4_rhs = prefield4_rhs;
    }
}

// ****************************************************************************
// Compute the X and Y contributions of the temporary shared memory arrays

template <FD_COEF_TYPE ctype, int order, int bx, int by>
__device__ inline void
compute_xy_contributions(const smstruct<order, bx, by>* sm, float invdx, float invdy, float invdxy, float& dfield1dx,
                         float& dfield2dx, float& dfield1dy, float& dfield2dy, float& field3_rhs, float& field4_rhs, float& inv_rho_rq)
{
    const int radius = order / 2;
    const int tx = threadIdx.x + radius;
    const int ty = threadIdx.y + radius;
    float alpha_x = 0.0f;
    float alpha_y = 0.0f;
    float alpha_xy = 0.0f;
    dfield1dx = 0.0f;
    dfield2dx = 0.0f;
    dfield1dy = 0.0f;
    dfield2dy = 0.0f;

    inv_rho_rq = sm->Model11Dup[ty][tx];

#pragma unroll
    for (int i = 1; i <= radius; i++) {
        // scatterDx
        alpha_x = invdx * helper_kernels_gpu::df_coef<order, ctype>(i);
        dfield1dx +=
            alpha_x * (sm->field1_xx[threadIdx.y][tx + i - 1] - sm->field1_xx[threadIdx.y][tx - i]) * sm->Model11Dup[ty][tx];
        dfield2dx +=
            alpha_x * (sm->field2_xx[threadIdx.y][tx + i - 1] - sm->field2_xx[threadIdx.y][tx - i]) * sm->Model11Dup[ty][tx];

        // scatterDy
        alpha_y = invdy * helper_kernels_gpu::df_coef<order, ctype>(i);
        dfield1dy +=
            alpha_y * (sm->field1_yy[ty + i - 1][threadIdx.x] - sm->field1_yy[ty - i][threadIdx.x]) * sm->Model11Dup[ty][tx];
        dfield2dy +=
            alpha_y * (sm->field2_yy[ty + i - 1][threadIdx.x] - sm->field2_yy[ty - i][threadIdx.x]) * sm->Model11Dup[ty][tx];

        // scatterDxy
        /*
                  ------> ix
                0 1 2 3 4 5 6 7 8
            0   a               A
       |    1     b           B  
    iy |    2       c       C    
       |    3         d   D      
       \/   4   . . . . X . . . . 
            5         f  F       
            6       g       G     
            7     h           H  
            8   i               I


        */
        alpha_xy = invdxy * helper_kernels_gpu::df2_coef<order, ctype>(i);
        field3_rhs += alpha_xy * (sm->field1_xy[ty + i][tx + i] - sm->field1_xy[ty + i][tx - i] + sm->field1_xy[ty - i][tx - i] -
                              sm->field1_xy[ty - i][tx + i]);
        field4_rhs += alpha_xy * (sm->field2_xy[ty + i][tx + i] - sm->field2_xy[ty + i][tx - i] + sm->field2_xy[ty - i][tx - i] -
                              sm->field2_xy[ty - i][tx + i]);
    }
}

template <int order, FD_COEF_TYPE ctype, int bx, int by, int bz>
__global__
__launch_bounds__(bx* by) void adj_main_loop_kernel(
    float* __restrict__ dfield1dx, float* __restrict__ dfield2dx, float* __restrict__ dfield1dy, float* __restrict__ dfield2dy,
    float* __restrict__ dfield1dz, float* __restrict__ dfield2dz, float* __restrict__ field3_rhs1, float* __restrict__ field4_rhs1,
    float* __restrict__ field1, float* __restrict__ field2, float* __restrict__ Model11, float* __restrict__ model1,
    float* __restrict__ model2, float* __restrict__ model3, float* __restrict__ model4, float* __restrict__ model5,
    float* __restrict__ model6, float* __restrict__ model7, const int ixbeg, float invdxx, float invdyy, float invdzz,
    float invdxy, float invdyz, float invdzx, const int ldimx, const int ldimy, const int nx, const int ny,
    const int nz, float dt, float invdx, float invdy, float invdz)
{
    static_assert(order == 8 || order == 16, "Only 8th and 16th orders are supported.");
    static_assert(bx >= order && by >= order, "Incompatible block size for this order.");

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

    // Load the shared memory as float4, and remap the threads from (bx, by) into (bx/2, by*2)
    // These reads are offset by -radius from the block offset, to read the halos.
    const int tid = threadIdx.y * bx + threadIdx.x;
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
    float field3_rhs[order] = { 0.0f };
    float field4_rhs[order] = { 0.0f };
    float inv_rho_rq[radius] = { 0.0f };
    // TODO: VRKG
    // For Guillaume's comment on No register queues for derivatives involving non z values,
    // I think, it is still needed since we start computing results in the priming loop when i >= radius
    // when we write the first result, the z plane is at iz = 8... if we don't have register queues how do we store the x & y derivatives values for those z planes from iz = 4 to 7???
    // may be we have to discuss more on this?
    float dt2xdx[radius] = { 0.0f };
    float dt2xdy[radius] = { 0.0f };
    float dt2xdz[order] = { 0.0f }; // ok
    float dt2zdx[radius] = { 0.0f };
    float dt2zdy[radius] = { 0.0f };
    float dt2zdz[order] = { 0.0f }; // ok

    // Async load the first input values in shared memory
    async_load(sm, index_load, loader, txload, tyload, field1, field2, model1, model2, model3, model4, model5, model6, model7, Model11);
    index_load += stride;

#pragma unroll
    for (int i = 0; i < order; ++i) {
        // Wait for the async load, compute the temporary arrays in shared memory
        __pipeline_wait_prior(0);
        __syncthreads();
        // In the first "halo" iterations, we only compute the Z components
        compute_temp_arrays(sm, loader, txload, tyload, center_x, center_y, i < radius);

        __syncthreads();

        // Async load the next inputs
        async_load(sm, index_load, loader, txload, tyload, field1, field2, model1, model2, model3, model4, model5, model6, model7, Model11);
        index_load += stride;

        // Compute the Z contributions of the temp arrays to the register queues
#pragma unroll
        for (int j = 1; j <= i; j++) {
            compute_z_contributionsdtd<ctype>(sm, j - radius, invdz, dt2xdz[i - j], dt2zdz[i - j], dt2xdz[i - j],
                                              dt2zdz[i - j]);

            compute_z_contributionsvrhs<ctype>(sm, j - radius, invdyz, invdzx, field3_rhs[i - j], field4_rhs[i - j],
                                               field3_rhs[i - j], field4_rhs[i - j]);
        }

        compute_z_contributionsdtd<ctype>(sm, -radius, invdz, dt2xdz[i], dt2zdz[i], 0.0f, 0.0f);
        compute_z_contributionsvrhs<ctype>(sm, -radius, invdyz, invdzx, field3_rhs[i], field4_rhs[i], 0.0f, 0.0f);

        // Compute the X Y contributions of the temp arrays once we're out of the Z halo
        if (i >= radius) {
            compute_xy_contributions<ctype>(sm, invdx, invdy, invdxy, dt2xdx[i - radius], dt2zdx[i - radius],
                                            dt2xdy[i - radius], dt2zdy[i - radius], field3_rhs[i - radius],
                                            field4_rhs[i - radius], inv_rho_rq[i - radius]);
        }
    }

    int nzloop = min(bz, nz - izblock);
    // Loop on the Z dimension, except the last one
    for (int izloop = 0; izloop < nzloop - 1; ++izloop) {
        // Wait for the async load, compute the temporary arrays in shared memory
        __pipeline_wait_prior(0);
        __syncthreads();
        compute_temp_arrays(sm, loader, txload, tyload, center_x, center_y);
        __syncthreads();

        // Async load the next inputs
        async_load(sm, index_load, loader, txload, tyload, field1, field2, model1, model2, model3, model4, model5, model6, model7, Model11);
        index_load += stride;

        // Compute the final contribution to the oldest values in the register queue
        // VRKG we do not need to compute_z_contributionsdtd as we have increased the register queue by 1 and we include the final contributions in the last iteration of priming loop
        // the stencil is from -4 to +3... ie from 0 to 7 indices which is covered in priming loop
        compute_z_contributionsvrhs<ctype>(sm, radius, invdyz, invdzx, field3_rhs[0], field4_rhs[0], field3_rhs[0], field4_rhs[0]);
        // Write the results
        if (active) {
            field3_rhs1[index] = field3_rhs[0];
            field4_rhs1[index] = field4_rhs[0];
            dfield1dx[index] = dt2xdx[0];
            dfield1dy[index] = dt2xdy[0];
            dfield1dz[index] = dt2xdz[0] * inv_rho_rq[0];
            dfield2dx[index] = dt2zdx[0];
            dfield2dy[index] = dt2zdy[0];
            dfield2dz[index] = dt2zdz[0] * inv_rho_rq[0];
        }
        index += stride;

        // Compute the Z contributions and rotate the register queues on the fly (i -> i-1)
#pragma unroll
        for (int i = 1; i < order; i++) {
            compute_z_contributionsdtd<ctype>(sm, radius - i, invdz, dt2xdz[i - 1], dt2zdz[i - 1], dt2xdz[i],
                                              dt2zdz[i]);

            compute_z_contributionsvrhs<ctype>(sm, radius - i, invdyz, invdzx, field3_rhs[i - 1], field4_rhs[i - 1], field3_rhs[i],
                                               field4_rhs[i]);
        }
        compute_z_contributionsdtd<ctype>(sm, -radius, invdz, dt2xdz[order - 1], dt2zdz[order - 1], 0.0f, 0.0f);
        compute_z_contributionsvrhs<ctype>(sm, -radius, invdyz, invdzx, field3_rhs[order - 1], field4_rhs[order - 1], 0.0f,
                                           0.0f);

#pragma unroll
        for (int i = 1; i < radius; ++i) {
            dt2xdx[i - 1] = dt2xdx[i];
            dt2zdx[i - 1] = dt2zdx[i];
            dt2xdy[i - 1] = dt2xdy[i];
            dt2zdy[i - 1] = dt2zdy[i];
            inv_rho_rq[i - 1] = inv_rho_rq[i];
        }
        // Compute the X and Y contributions
        compute_xy_contributions<ctype>(sm, invdx, invdy, invdxy, dt2xdx[radius - 1], dt2zdx[radius - 1],
                                        dt2xdy[radius - 1], dt2zdy[radius - 1], field3_rhs[radius - 1], field4_rhs[radius - 1],
                                        inv_rho_rq[radius - 1]);
    }
    // Last Z (no prefetch)
    // Wait for the async load, compute the temporary arrays in shared memory
    __pipeline_wait_prior(0);
    __syncthreads();
    compute_temp_arrays(sm, loader, txload, tyload, center_x, center_y);
    __syncthreads();

    // Compute the final contribution to the oldest values in the register queue
    // VRKG we do not need to compute_z_contributionsdtd as we have increased the register queue by 1 and we include the final contributions in the izloop
    compute_z_contributionsvrhs<ctype>(sm, radius, invdyz, invdzx, field3_rhs[0], field4_rhs[0], field3_rhs[0], field4_rhs[0]);

    // Write the results
    if (active) {
        field3_rhs1[index] = field3_rhs[0];
        field4_rhs1[index] = field4_rhs[0];
        dfield1dx[index] = dt2xdx[0];
        dfield1dy[index] = dt2xdy[0];
        dfield1dz[index] = dt2xdz[0] * inv_rho_rq[0];
        dfield2dx[index] = dt2zdx[0];
        dfield2dy[index] = dt2zdy[0];
        dfield2dz[index] = dt2zdz[0] * inv_rho_rq[0];
    }
}

void
launch_adj_kernel2_drv(float* dfield1dx, float* dfield2dx, float* dfield1dy, float* dfield2dy, float* dfield1dz, float* dfield2dz,
                      float* field3_rhs, float* field4_rhs, float* field1, float* field2, float* Model11, float* model1, float* model2,
                      float* model3, float* model4, float* model5, float* model6, float* model7, int ixbeg, int ixend, int iybeg,
                      int iyend, int izbeg, int izend, float invdxx, float invdyy, float invdzz, float invdxy,
                      float invdyz, float invdzx, int ldimx, int ldimy, int order, float dt, float invdx, float invdy,
                      float invdz, cudaStream_t stream)
{
    constexpr FD_COEF_TYPE ctype = _FD_COEF_LEAST_SQUARE;
    const int nx = ixend - ixbeg + 1;
    const int ny = iyend - iybeg + 1;
    const int nz = izend - izbeg + 1;
    const int radius = order / 2;
    off_t offset = (izbeg * ldimx * ldimy) + (iybeg * ldimx) + ixbeg;
    constexpr int z_threads = 64;

    if (order == 8) {
        constexpr int bx = 32;
        constexpr int by = 16;

        dim3 threads(bx, by);
        dim3 blocks((nx - 1) / bx + 1, (ny - 1) / by + 1, (nz - 1) / z_threads + 1);
        size_t shm = sizeof(smstruct<8, bx, by>);
        cudaFuncSetAttribute(adj_main_loop_kernel<8, ctype, bx, by, z_threads>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, shm);
        adj_main_loop_kernel<8, ctype, bx, by, z_threads><<<blocks, threads, shm, stream>>>(
            dfield1dx + offset, dfield2dx + offset, dfield1dy + offset, dfield2dy + offset, dfield1dz + offset, dfield2dz + offset,
            field3_rhs + offset, field4_rhs + offset, field1 + offset, field2 + offset, Model11 + offset, model1 + offset, model2 + offset,
            model3 + offset, model4 + offset, model5 + offset, model6 + offset, model7 + offset, ixbeg, invdxx, invdyy, invdzz, invdxy,
            invdyz, invdzx, ldimx, ldimy, nx, ny, nz, dt, invdx, invdy, invdz);
    }
    if (order == 16) {
        constexpr int bx = 16;
        constexpr int by = 16;

        dim3 threads(bx, by);
        dim3 blocks((nx - 1) / bx + 1, (ny - 1) / by + 1, (nz - 1) / z_threads + 1);
        size_t shm = sizeof(smstruct<16, bx, by>);
        cudaFuncSetAttribute(adj_main_loop_kernel<16, ctype, bx, by, z_threads>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, shm);
        adj_main_loop_kernel<16, ctype, bx, by, z_threads><<<blocks, threads, shm, stream>>>(
            dfield1dx + offset, dfield2dx + offset, dfield1dy + offset, dfield2dy + offset, dfield1dz + offset, dfield2dz + offset,
            field3_rhs + offset, field4_rhs + offset, field1 + offset, field2 + offset, Model11 + offset, model1 + offset, model2 + offset,
            model3 + offset, model4 + offset, model5 + offset, model6 + offset, model7 + offset, ixbeg, invdxx, invdyy, invdzz, invdxy,
            invdyz, invdzx, ldimx, ldimy, nx, ny, nz, dt, invdx, invdy, invdz);
    }

    CUDA_TRY(cudaPeekAtLastError());
}
} //namespace adj_kernel3_derivatives