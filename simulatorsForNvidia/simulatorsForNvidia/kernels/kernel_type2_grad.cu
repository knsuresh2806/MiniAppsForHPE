#include <cuda_runtime.h>
#include <vector_types.h>
#include <cuda_pipeline.h>
#include <fstream>
#include "emsl_error.h"
#include "fd_coefficients.h"
#include "kernel_utils.h"
#include "helper_kernels_gpu.h"
#include "kernel_type2_grad.h"

//Structure for shared memory arrays.  2D arrays for data in XY plane
template <int ORD, int BX, int BY>
struct smStruct
{
    //input cart vols that require full X and Y halos in shared mem.
    float field1[BY + ORD][BX + ORD];
    float field2[BY + ORD][BX + ORD];
    float model1[BY + ORD][BX + ORD];
    float model2[BY + ORD][BX + ORD];
    float model3[BY + ORD][BX + ORD];
    float model4[BY + ORD][BX + ORD];
    float irho[BY + ORD][BX + ORD];
    float rx[BY + ORD][BX + ORD];
    float ry[BY + ORD][BX + ORD];
    float rz[BY + ORD][BX + ORD];

    //Fwd derivatives, only need halos in direction derivative will be computed.
    //Except Z derivatives, those halos will be taken care of with register queues.
    float fwd_field1_dx[BY][BX + ORD];
    float fwd_field1_dy[BY + ORD][BX];
    float fwd_field1_dz[BY][BX];
    float fwd_field2_dx[BY][BX + ORD];
    float fwd_field2_dy[BY + ORD][BX];
    float fwd_field2_dz[BY][BX];

    //adj input values, only need halos in direction derivative will be computed.
    float adj_field1_x[BY][BX + ORD];
    float adj_field2_x[BY][BX + ORD];
    float adj_field1_y[BY + ORD][BX];
    float adj_field2_y[BY + ORD][BX];
    float adj_field1_z[BY][BX];
    float adj_field2_z[BY][BX];
};

//
// Kernel registers.
//
template <int ORD>
struct RegBase
{
    float d_fwd_field1_dx;
    float d_fwd_field2_dx;
    float d_fwd_field1_dy;
    float d_fwd_field2_dy;
    float d_fwd_field1_dz;
    float d_fwd_field2_dz;

    float d_adj_field1_x;
    float d_adj_field2_x;
    float d_adj_field1_y;
    float d_adj_field2_y;
    float d_adj_field1_z;
    float d_adj_field2_z;

    float grad_rhs;

    float fwd_field1_dz[ORD + 1];
    float fwd_field2_dz[ORD + 1];

    //Half register queues
    float grad_rhs_xy[ORD / 2 + 1];
    float irho[ORD / 2 + 1];

    //Full register queues for Z adjoint inputs, fwd Z derivatives, and adj Z derivatives
    float adj_field1_z[ORD + 1];
    float adj_field2_z[ORD + 1];
};

// Device function to asynchronously load the shared memory inputs as float4
template <int ORD, int BX, int BY>
__device__ inline void
async_load(smStruct<ORD, BX, BY>* sm, int index_load, bool loader, bool center_x, bool center_y, int txload, int tyload,
           const float* field1, const float* field2, const float* model1, const float* model2, const float* model3, const float* model4,
           const float* model5, const float* model6, const float* model7, const float* irho, const float* fwd_field1_dx,
           const float* fwd_field1_dy, const float* fwd_field1_dz, const float* fwd_field2_dx, const float* fwd_field2_dy,
           const float* fwd_field2_dz)
{
    constexpr int radius = ORD / 2;

    if (loader) {
        __pipeline_memcpy_async(&sm->field1[tyload][txload], field1 + index_load, sizeof(float4));
        __pipeline_memcpy_async(&sm->field2[tyload][txload], field2 + index_load, sizeof(float4));
        __pipeline_memcpy_async(&sm->model1[tyload][txload], model1 + index_load, sizeof(float4));
        __pipeline_memcpy_async(&sm->model2[tyload][txload], model2 + index_load, sizeof(float4));
        __pipeline_memcpy_async(&sm->model3[tyload][txload], model3 + index_load, sizeof(float4));
        __pipeline_memcpy_async(&sm->model4[tyload][txload], model4 + index_load, sizeof(float4));
        __pipeline_memcpy_async(&sm->rx[tyload][txload], model5 + index_load, sizeof(float4));
        __pipeline_memcpy_async(&sm->ry[tyload][txload], model6 + index_load, sizeof(float4));
        __pipeline_memcpy_async(&sm->rz[tyload][txload], model7 + index_load, sizeof(float4));
        __pipeline_memcpy_async(&sm->irho[tyload][txload], irho + index_load, sizeof(float4));
        if (center_y) {
            __pipeline_memcpy_async(&sm->fwd_field1_dx[tyload - radius][txload], fwd_field1_dx + index_load, sizeof(float4));
            __pipeline_memcpy_async(&sm->fwd_field2_dx[tyload - radius][txload], fwd_field2_dx + index_load, sizeof(float4));
        }
        if (center_x) {
            __pipeline_memcpy_async(&sm->fwd_field1_dy[tyload][txload - radius], fwd_field1_dy + index_load, sizeof(float4));
            __pipeline_memcpy_async(&sm->fwd_field2_dy[tyload][txload - radius], fwd_field2_dy + index_load, sizeof(float4));
        }
        if (center_x && center_y) {
            __pipeline_memcpy_async(&sm->fwd_field1_dz[tyload - radius][txload - radius], fwd_field1_dz + index_load,
                                    sizeof(float4));
            __pipeline_memcpy_async(&sm->fwd_field2_dz[tyload - radius][txload - radius], fwd_field2_dz + index_load,
                                    sizeof(float4));
        }
    }
    __pipeline_commit();
}

//
// Adjoint values.
// @param i Global index.
//
template <int ORD, int BX, int BY>
__device__ __forceinline__ void
compute_temp_adj_arrays(smStruct<ORD, BX, BY>* sm, bool loader, int txload, int tyload, bool center_x, bool center_y,
                        bool z_only = false)
{

    const int r = ORD / 2;
    if (loader) {
        float4 rx = *reinterpret_cast<float4*>(&sm->rx[tyload][txload]);
        float4 ry = *reinterpret_cast<float4*>(&sm->ry[tyload][txload]);
        float4 rz = *reinterpret_cast<float4*>(&sm->rz[tyload][txload]);
        float4 irho = *reinterpret_cast<float4*>(&sm->irho[tyload][txload]);
        float4 model1 = *reinterpret_cast<float4*>(&sm->model1[tyload][txload]);
        float4 model2 = *reinterpret_cast<float4*>(&sm->model2[tyload][txload]);
        float4 model3 = *reinterpret_cast<float4*>(&sm->model3[tyload][txload]);
        float4 model4 = *reinterpret_cast<float4*>(&sm->model4[tyload][txload]);
        float4 field1 = *reinterpret_cast<float4*>(&sm->field1[tyload][txload]);
        float4 field2 = *reinterpret_cast<float4*>(&sm->field2[tyload][txload]);

        // Using float4 math
        using namespace helper_kernels_gpu;

        float4 field1_ = field1;
        float4 field2_ = field2;
        float4 rho_ = 1.0f / irho;
        float4 model1_ = model1;
        float4 model2_ = model2;
        float4 model3_ = model3;
        float4 model4_ = model4;

        float4 rx_rx = rx * rx;
        float4 ry_ry = ry * ry;
        float4 rz_rz = rz * rz;
        float4 lx_zz = (model4_ - model1_) * field1_ - model2_ * field2_;
        float4 lz_zz = model2_ * field1_ + (model3_ - model4_) * field2_;

        // write adj_field1_x, adj_field2_x values (X halo but no Y halo)
        if (center_y) {
            if (!z_only) {
                *reinterpret_cast<float4*>(&sm->adj_field1_x[tyload - r][txload]) =
                    rho_ * (model1_ * field1_ + model2_ * field2_ + rx_rx * lx_zz);
                *reinterpret_cast<float4*>(&sm->adj_field2_x[tyload - r][txload]) = rho_ * (model4_ * field2_ + rx_rx * lz_zz);
            }
        }
        // Write adj_field1_y, adj_field2_y values (Y halo but no X halo)
        if (center_x) {
            if (!z_only) {
                *reinterpret_cast<float4*>(&sm->adj_field1_y[tyload][txload - r]) =
                    rho_ * (model1_ * field1_ + model2_ * field2_ + ry_ry * lx_zz);
                *reinterpret_cast<float4*>(&sm->adj_field2_y[tyload][txload - r]) = rho_ * (model4_ * field2_ + ry_ry * lz_zz);
            }
        }
        // Write adj_field1_z, adj_field2_z values (no Y halo and no X halo)
        if (center_x && center_y) {
            *reinterpret_cast<float4*>(&sm->adj_field1_z[tyload - r][txload - r]) =
                rho_ * (model1_ * field1_ + model2_ * field2_ + rz_rz * lx_zz);
            *reinterpret_cast<float4*>(&sm->adj_field2_z[tyload - r][txload - r]) = rho_ * (model4_ * field2_ + rz_rz * lz_zz);
        }
    }
}

//
// Adjoint values.
// @param i Global index.
//
__device__ __forceinline__ float2
adj(float const* __restrict__ field1, float const* __restrict__ field2, float const* __restrict__ irho,
    float const* __restrict__ model1, float const* __restrict__ model2, float const* __restrict__ model3,
    float const* __restrict__ model4, float const* __restrict__ ra, int i)
{
    float field1_ = field1[i];
    float field2_ = field2[i];
    float rho_ = 1.0f / irho[i];
    float model1_ = model1[i];
    float model2_ = model2[i];
    float model3_ = model3[i];
    float model4_ = model4[i];

    float ra_ra = ra[i] * ra[i];
    float lx_zz = (model4_ - model1_) * field1_ - model2_ * field2_;
    float lz_zz = model2_ * field1_ + (model3_ - model4_) * field2_;

    return { rho_ * (model1_ * field1_ + model2_ * field2_ + ra_ra * lx_zz), rho_ * (model4_ * field2_ + ra_ra * lz_zz) };
}

//
// Staggered 1D derivative.
// @tparam RIGHT Staggered grid shift with respect to regular grid.
// @param h Grid step.
// @param i Global index.
// @param s Stride.
//
template <int ORD, bool RIGHT>
__device__ __forceinline__ float
stg(float const* __restrict__ t, float h, int i, int s)
{
    constexpr int P = RIGHT ? 1 : 0;
    constexpr int M = RIGHT ? 0 : 1;

    float d = 0.0f;

#pragma unroll
    for (int r = 0; r < ORD / 2; ++r)
        d += stg_d1<ORD>(r) * (t[i + (r + P) * s] - t[i - (r + M) * s]);

    return d * h;
}

//
// Staggered 1D derivative of adjoint.
// @tparam RIGHT Staggered grid shift with respect to regular grid.
// @param h Grid step.
// @param i Global index.
// @param s Stride.
//
template <int ORD, bool RIGHT>
__device__ __forceinline__ float2
stg(float const* __restrict__ field1, float const* __restrict__ field2, float const* __restrict__ irho,
    float const* __restrict__ model1, float const* __restrict__ model2, float const* __restrict__ model3,
    float const* __restrict__ model4, float const* __restrict__ ra, float h, int i, int s)
{
    constexpr int P = RIGHT ? 1 : 0;
    constexpr int M = RIGHT ? 0 : 1;

    float dx = 0.0f;
    float dz = 0.0f;

#pragma unroll
    for (int r = 0; r < ORD / 2; ++r) {
        auto [xp, zp] = adj(field1, field2, irho, model1, model2, model3, model4, ra, i + (r + P) * s);
        auto [xm, zm] = adj(field1, field2, irho, model1, model2, model3, model4, ra, i - (r + M) * s);

        dx += stg_d1<ORD>(r) * (xp - xm);
        dz += stg_d1<ORD>(r) * (zp - zm);
    }

    return { dx * h, dz * h };
}

template <int ORD, int BX, int BY>
__device__ inline void
populate_register_queues(const smStruct<ORD, BX, BY>* sm, RegBase<ORD>& r, int i, int ty, int tx)
{
    constexpr int radius = ORD / 2;
    r.fwd_field1_dz[i] = sm->fwd_field1_dz[ty - radius][tx - radius];
    r.fwd_field2_dz[i] = sm->fwd_field2_dz[ty - radius][tx - radius];
    r.adj_field1_z[i] = sm->adj_field1_z[ty - radius][tx - radius];
    r.adj_field2_z[i] = sm->adj_field2_z[ty - radius][tx - radius];
}

template <int ORD, int BX, int BY>
__device__ inline void
compute_z_values(RegBase<ORD>& r, float hz, int i, int g_ig) //i = ORD/2
{
    // Calculate forward second derivatives.
    r.d_fwd_field1_dz = stg<ORD, true>(r.fwd_field1_dz, hz, i, 1);
    r.d_fwd_field2_dz = stg<ORD, true>(r.fwd_field2_dz, hz, i, 1);

    // Calculate adjoint second derivatives.
    r.d_adj_field1_z = stg<ORD, false>(r.adj_field1_z, hz, i, 1);
    r.d_adj_field2_z = stg<ORD, false>(r.adj_field2_z, hz, i, 1);
}

template <int ORD>
__device__ inline void
shift_register_queues(RegBase<ORD>& r)
{
#pragma unroll
    for (int i = 1; i < ORD + 1; ++i) {
        r.adj_field1_z[i - 1] = r.adj_field1_z[i];
        r.adj_field2_z[i - 1] = r.adj_field2_z[i];

        r.fwd_field1_dz[i - 1] = r.fwd_field1_dz[i];
        r.fwd_field2_dz[i - 1] = r.fwd_field2_dz[i];
        if (i < ORD / 2 + 1) {
            r.grad_rhs_xy[i - 1] = r.grad_rhs_xy[i];
            r.irho[i - 1] = r.irho[i];
        }
    }

    r.adj_field1_z[ORD] = 0.0f;
    r.adj_field2_z[ORD] = 0.0f;

    r.fwd_field1_dz[ORD] = 0.0f;
    r.fwd_field2_dz[ORD] = 0.0f;
    r.grad_rhs_xy[ORD / 2] = 0.0f;
    r.irho[ORD / 2] = 0.0f;
}

template <int ORD, int BX, int BY>
__device__ __forceinline__ void
compute_xy_contributions(smStruct<ORD, BX, BY>* sm, float hy, float hx, int iy, int ix, float& d_fwd_field1_dx,
                         float& d_fwd_field2_dx, float& d_fwd_field1_dy, float& d_fwd_field2_dy, float& d_adj_field1_x,
                         float& d_adj_field2_x, float& d_adj_field1_y, float& d_adj_field2_y)
{

    //Our ix and iy indices will start from 0,0 here.  The shared mem arrays we are computing will
    //either have an X halo or a Y halo of size ORD / 2, but not both.  So we shift the
    //appropriate index to ensure that if there are X halos then the x index includes them
    //so we are computing only in the interior region.
    constexpr int radius = ORD / 2;
    int ix_halo = ix + radius;
    int iy_halo = iy + radius;

    //No Y halos
    d_fwd_field1_dx = stg<ORD, true>(&sm->fwd_field1_dx[0][0], hx, (iy * (BX + ORD)) + ix_halo, 1);
    d_fwd_field2_dx = stg<ORD, true>(&sm->fwd_field2_dx[0][0], hx, (iy * (BX + ORD)) + ix_halo, 1);

    //No X halos
    d_fwd_field1_dy = stg<ORD, true>(&sm->fwd_field1_dy[0][0], hy, (iy_halo * (BX)) + ix, BX);
    d_fwd_field2_dy = stg<ORD, true>(&sm->fwd_field2_dy[0][0], hy, (iy_halo * (BX)) + ix, BX);

    //No Y halos
    d_adj_field1_x = stg<ORD, false>(&sm->adj_field1_x[0][0], hx, (iy * (BX + ORD)) + ix_halo, 1);
    d_adj_field2_x = stg<ORD, false>(&sm->adj_field2_x[0][0], hx, (iy * (BX + ORD)) + ix_halo, 1);

    //No X halos
    d_adj_field1_y = stg<ORD, false>(&sm->adj_field1_y[0][0], hy, (iy_halo * (BX)) + ix, BX);
    d_adj_field2_y = stg<ORD, false>(&sm->adj_field2_y[0][0], hy, (iy_halo * (BX)) + ix, BX);
}

template <int ORD, int BX, int BY, int BZ>
__global__
__launch_bounds__(BX* BY) void compute_grad_model10(
    float* __restrict__ grad, float const* __restrict__ fwd_field1_dx, float const* __restrict__ fwd_field2_dx,
    float const* __restrict__ fwd_field1_dy, float const* __restrict__ fwd_field2_dy, float const* __restrict__ fwd_field1_dz,
    float const* __restrict__ fwd_field2_dz, float const* __restrict__ field1, float const* __restrict__ field2,
    float const* __restrict__ irho, float const* __restrict__ model1, float const* __restrict__ model2,
    float const* __restrict__ model3, float const* __restrict__ model4, float const* __restrict__ rx,
    float const* __restrict__ ry, float const* __restrict__ rz, int nx, int ny, int nz, int ldimx, int ldimy,
    int stride_y, int stride_z, int g_stride_y, int g_stride_z, float hx, float hy, float hz)
{
    RegBase<ORD> r;

    const int ix = blockIdx.x * BX + threadIdx.x;
    const int iy = blockIdx.y * BY + threadIdx.y;
    const int iz = blockIdx.z * BZ + threadIdx.z;

    // Global position of the thread block, using 2D thread blocks
    const int ixblock = blockIdx.x * BX;
    const int iyblock = blockIdx.y * BY;
    const int izblock = blockIdx.z * BZ;
    const int stride = ldimx * ldimy;
    constexpr int radius = ORD / 2;
    const bool active = ix < nx && iy < ny;

    // Dynamic shared memory. Using a non-templated pointer to get the base address,
    // because we can't have a templated extern declaration with different template args,
    // and we need dynamic shared memory to use > 48KB per thread block.
    extern __shared__ char dsmem[];
    // Now we can cast the base pointer to the proper templated struct
    smStruct<ORD, BX, BY>* sm = reinterpret_cast<smStruct<ORD, BX, BY>*>(dsmem);

    // Local thread block shared memory indexes.
    const int lx = threadIdx.x + ORD / 2; // Skip past halo.
    const int ly = threadIdx.y + ORD / 2; // Skip past halo.
    const int lz = threadIdx.z;

    // Gradient (output) global index.
    int g_ig = iz * g_stride_z + iy * g_stride_y + ix;

    //----------------------------------
    //From adjoin simulator2 kernel
    //remap threads to indices so we can load entire shared mem tile
    //with float4 in one load.

    // Load the shared memory as float4, and remap the threads from (bx, by) into (bx/2, by*2)
    // These reads are offset by -radius from the block offset, to read the halos.
    const int tid = threadIdx.y * BX + threadIdx.x;
    static_assert(BX >= ORD && BY >= ORD, "Incompatible block size for this order");
    const int txload = 4 * (tid % (BX / 2));
    const int tyload = tid / (BX / 2);
    const int ixload = ixblock + txload - radius;
    const int iyload = iyblock + tyload - radius;
    const bool loader = ((BX == ORD || txload < BX + ORD) && (BY == ORD || tyload < BY + ORD)) &&
                        (ixload < nx + radius && iyload < ny + radius);

    int index_load = ((izblock - radius) * ldimy + iyload) * ldimx + ixload;

    // The threads will compute the temp shared arrays with the same float4 thread mapping.
    // We need to identify the threads which are not in the X or Y halos
    const bool center_x = txload >= radius && txload < BX + radius;
    const bool center_y = tyload >= radius && tyload < BY + radius;

    //---------------------------------------------------------------------------------
    //Begin Gradient computations

    async_load(sm, index_load, loader, center_x, center_y, txload, tyload, field1, field2, model1, model2, model3, model4, rx, ry, rz,
               irho, fwd_field1_dx, fwd_field1_dy, fwd_field1_dz, fwd_field2_dx, fwd_field2_dy, fwd_field2_dz);
    index_load += stride;

    // Prime the register queues
    // Loop must be unrolled to allow proper inlining and resolution of FD coefficients.
    int nzloop = min(BZ, nz - izblock);
#pragma unroll
    for (int i = 0; i < ORD + 1; i++) {
        //----------------------------------------------------------------
        //Wait for async load and then sync threads
        __pipeline_wait_prior(0);
        __syncthreads();

        compute_temp_adj_arrays<ORD, BX, BY>(sm, loader, txload, tyload, center_x, center_y, i < radius);
        __syncthreads();

        populate_register_queues(sm, r, i, ly, lx);

        // Compute the X Y contributions of the temp arrays once we're out of the Z halo
        if (i >= radius) {
            compute_xy_contributions(sm, hy, hx, threadIdx.y, threadIdx.x, r.d_fwd_field1_dx, r.d_fwd_field2_dx,
                                     r.d_fwd_field1_dy, r.d_fwd_field2_dy, r.d_adj_field1_x, r.d_adj_field2_x, r.d_adj_field1_y,
                                     r.d_adj_field2_y);

            r.grad_rhs_xy[i - ORD / 2] =
                sm->adj_field1_x[ly - radius][lx] * r.d_fwd_field1_dx + sm->adj_field2_x[ly - radius][lx] * r.d_fwd_field2_dx +
                sm->adj_field1_y[ly][lx - radius] * r.d_fwd_field1_dy + sm->adj_field2_y[ly][lx - radius] * r.d_fwd_field2_dy +
                r.d_adj_field1_x * sm->fwd_field1_dx[ly - radius][lx] + r.d_adj_field2_x * sm->fwd_field2_dx[ly - radius][lx] +
                r.d_adj_field1_y * sm->fwd_field1_dy[ly][lx - radius] + r.d_adj_field2_y * sm->fwd_field2_dy[ly][lx - radius];
            r.irho[i - ORD / 2] = sm->irho[ly][lx];
            __syncthreads();
        }
        // If this zblock has only 1 plane, we have to skip the last out of bounds load
        if (!(i == ORD && nzloop == 1)) {
            async_load(sm, index_load, loader, center_x, center_y, txload, tyload, field1, field2, model1, model2, model3, model4, rx, ry,
                       rz, irho, fwd_field1_dx, fwd_field1_dy, fwd_field1_dz, fwd_field2_dx, fwd_field2_dy, fwd_field2_dz);
            index_load += stride;
        }
    }

    // Loop on the Z dimension, except the last one
    for (int izloop = 0; izloop < nzloop - 1; izloop++) {

        // Compute the Z values
        compute_z_values<ORD, BX, BY>(r, hz, ORD / 2, g_ig);

        if (active) {
            //write the results
            r.grad_rhs = r.grad_rhs_xy[0] + r.adj_field1_z[ORD / 2] * r.d_fwd_field1_dz +
                         r.adj_field2_z[ORD / 2] * r.d_fwd_field2_dz + r.d_adj_field1_z * r.fwd_field1_dz[ORD / 2] +
                         r.d_adj_field2_z * r.fwd_field2_dz[ORD / 2];

            atomicAdd_block(&grad[g_ig], -r.irho[0] * r.grad_rhs);
        }

        shift_register_queues(r);

        //Wait for async load and then sync threads
        __pipeline_wait_prior(0);
        __syncthreads();

        compute_temp_adj_arrays<ORD, BX, BY>(sm, loader, txload, tyload, center_x, center_y, izloop > nzloop - radius);
        __syncthreads();

        populate_register_queues(sm, r, ORD, ly, lx);

        if (izloop < nzloop - radius) {
            compute_xy_contributions(sm, hy, hx, threadIdx.y, threadIdx.x, r.d_fwd_field1_dx, r.d_fwd_field2_dx,
                                     r.d_fwd_field1_dy, r.d_fwd_field2_dy, r.d_adj_field1_x, r.d_adj_field2_x, r.d_adj_field1_y,
                                     r.d_adj_field2_y);
            r.grad_rhs_xy[ORD / 2] =
                sm->adj_field1_x[ly - radius][lx] * r.d_fwd_field1_dx + sm->adj_field2_x[ly - radius][lx] * r.d_fwd_field2_dx +
                sm->adj_field1_y[ly][lx - radius] * r.d_fwd_field1_dy + sm->adj_field2_y[ly][lx - radius] * r.d_fwd_field2_dy +
                r.d_adj_field1_x * sm->fwd_field1_dx[ly - radius][lx] + r.d_adj_field2_x * sm->fwd_field2_dx[ly - radius][lx] +
                r.d_adj_field1_y * sm->fwd_field1_dy[ly][lx - radius] + r.d_adj_field2_y * sm->fwd_field2_dy[ly][lx - radius];
            r.irho[ORD / 2] = sm->irho[ly][lx];

            __syncthreads();
        }

        if (izloop < nzloop - 2) {
            async_load(sm, index_load, loader, center_x, center_y, txload, tyload, field1, field2, model1, model2, model3, model4, rx, ry,
                       rz, irho, fwd_field1_dx, fwd_field1_dy, fwd_field1_dz, fwd_field2_dx, fwd_field2_dy, fwd_field2_dz);
        }

        index_load += stride;
        g_ig += g_stride_z;
    }

    // Compute the final Z value
    compute_z_values<ORD, BX, BY>(r, hz, ORD / 2, g_ig);

    if (active) {
        //write the results
        r.grad_rhs = r.grad_rhs_xy[0] + r.adj_field1_z[ORD / 2] * r.d_fwd_field1_dz + r.adj_field2_z[ORD / 2] * r.d_fwd_field2_dz +
                     r.d_adj_field1_z * r.fwd_field1_dz[ORD / 2] + r.d_adj_field2_z * r.fwd_field2_dz[ORD / 2];
        atomicAdd_block(&grad[g_ig], -r.irho[0] * r.grad_rhs);
    }
}

template <int ORD, int BX, int BY, int BZ>
__global__
__launch_bounds__(BY* BX* BZ) void compute_grad_model10_simple(
    float* __restrict__ grad, float const* __restrict__ fwd_field1_dx, float const* __restrict__ fwd_field2_dx,
    float const* __restrict__ fwd_field1_dy, float const* __restrict__ fwd_field2_dy, float const* __restrict__ fwd_field1_dz,
    float const* __restrict__ fwd_field2_dz, float const* __restrict__ field1, float const* __restrict__ field2,
    float const* __restrict__ irho, float const* __restrict__ model1, float const* __restrict__ model2,
    float const* __restrict__ model3, float const* __restrict__ model4, float const* __restrict__ rx,
    float const* __restrict__ ry, float const* __restrict__ rz, int nx, int ny, int nz, int stride_y, int stride_z,
    int g_stride_y, int g_stride_z, float hx, float hy, float hz)
{
    int const tx = threadIdx.x;
    int const ty = threadIdx.y;
    int const tz = threadIdx.z;
    int const ix = blockIdx.x * BX + tx;
    int const iy = blockIdx.y * BY + ty;
    int const iz = blockIdx.z * BZ + tz;

    if (ix >= nx || iy >= ny || iz >= nz)
        return;

    int const ig = iz * stride_z + iy * stride_y + ix;
    int const g_ig = iz * g_stride_z + iy * g_stride_y + ix;

    auto [adj_field1_x, adj_field2_x] = adj(field1, field2, irho, model1, model2, model3, model4, rx, ig);
    auto [adj_field1_y, adj_field2_y] = adj(field1, field2, irho, model1, model2, model3, model4, ry, ig);
    auto [adj_field1_z, adj_field2_z] = adj(field1, field2, irho, model1, model2, model3, model4, rz, ig);

    float d_fwd_field1_dx = stg<ORD, true>(fwd_field1_dx, hx, ig, 1);
    float d_fwd_field2_dx = stg<ORD, true>(fwd_field2_dx, hx, ig, 1);
    float d_fwd_field1_dy = stg<ORD, true>(fwd_field1_dy, hy, ig, stride_y);
    float d_fwd_field2_dy = stg<ORD, true>(fwd_field2_dy, hy, ig, stride_y);
    float d_fwd_field1_dz = stg<ORD, true>(fwd_field1_dz, hz, ig, stride_z);
    float d_fwd_field2_dz = stg<ORD, true>(fwd_field2_dz, hz, ig, stride_z);

    auto [d_adj_field1_x, d_adj_field2_x] = stg<ORD, false>(field1, field2, irho, model1, model2, model3, model4, rx, hx, ig, 1);
    auto [d_adj_field1_y, d_adj_field2_y] = stg<ORD, false>(field1, field2, irho, model1, model2, model3, model4, ry, hy, ig, stride_y);
    auto [d_adj_field1_z, d_adj_field2_z] = stg<ORD, false>(field1, field2, irho, model1, model2, model3, model4, rz, hz, ig, stride_z);

    float grad_rhs = adj_field1_x * d_fwd_field1_dx + adj_field2_x * d_fwd_field2_dx + adj_field1_y * d_fwd_field1_dy +
                     adj_field2_y * d_fwd_field2_dy + adj_field1_z * d_fwd_field1_dz + adj_field2_z * d_fwd_field2_dz +
                     d_adj_field1_x * fwd_field1_dx[ig] + d_adj_field2_x * fwd_field2_dx[ig] + d_adj_field1_y * fwd_field1_dy[ig] +
                     d_adj_field2_y * fwd_field2_dy[ig] + d_adj_field1_z * fwd_field1_dz[ig] + d_adj_field2_z * fwd_field2_dz[ig];

    float grad_rhs_xy = adj_field1_x * d_fwd_field1_dx + adj_field2_x * d_fwd_field2_dx + adj_field1_y * d_fwd_field1_dy +
                        adj_field2_y * d_fwd_field2_dy + d_adj_field1_x * fwd_field1_dx[ig] + d_adj_field2_x * fwd_field2_dx[ig] +
                        d_adj_field1_y * fwd_field1_dy[ig] + d_adj_field2_y * fwd_field2_dy[ig];

    atomicAdd_block(&grad[g_ig], -irho[ig] * grad_rhs);
}

//
// Launches kernel.
//
template <int ORD>
void
launch_rho(float* grad, float const* fwd_field1_dx, float const* fwd_field2_dx, float const* fwd_field1_dy,
           float const* fwd_field2_dy, float const* fwd_field1_dz, float const* fwd_field2_dz, float const* field1,
           float const* field2, float const* irho, float const* model1, float const* model2, float const* model3, float const* model4,
           float const* rx, float const* ry, float const* rz, int nx, int ny, int nz, int ldimx, int ldimy,
           int stride_y, int stride_z, int g_stride_y, int g_stride_z, float hx, float hy, float hz, bool simple_kernel,
           cudaStream_t stream)
{
    if (simple_kernel) {
        constexpr int BX = 32;
        constexpr int BY = 4;
        constexpr int BZ = 4;

        unsigned gx = (nx + BX - 1) / BX;
        unsigned gy = (ny + BY - 1) / BY;
        unsigned gz = (nz + BZ - 1) / BZ;

        constexpr auto KERNEL = compute_grad_model10_simple<ORD, BX, BY, BZ>;
        KERNEL<<<dim3{ gx, gy, gz }, dim3{ BX, BY, BZ }, 0, stream>>>(
            grad, fwd_field1_dx, fwd_field2_dx, fwd_field1_dy, fwd_field2_dy, fwd_field1_dz, fwd_field2_dz, field1, field2, irho, model1, model2, model3,
            model4, rx, ry, rz, nx, ny, nz, stride_y, stride_z, g_stride_y, g_stride_z, hx, hy, hz);
        CUDA_CHECK_ERROR(__FILE__, __LINE__);
    } else {
        // A block size of 32 x 16 should work well for both 8th order and 16th order.
        // There are register spills for the 16th order kernel with these block sizes,
        // but profiling results showed these were still more efficient than other block sizes.
        constexpr int BX = 32;
        constexpr int BY = 16;
        constexpr int BZ = 64;

        unsigned gx = (nx + BX - 1) / BX;
        unsigned gy = (ny + BY - 1) / BY;
        unsigned gz = (nz + BZ - 1) / BZ;

        constexpr auto KERNEL = compute_grad_model10<ORD, BX, BY, BZ>;

        size_t shm = sizeof(smStruct<ORD, BX, BY>);
        cudaFuncSetAttribute(KERNEL, cudaFuncAttributeMaxDynamicSharedMemorySize, shm);

        KERNEL<<<dim3{ gx, gy, gz }, dim3{ BX, BY, 1 }, shm, stream>>>(
            grad, fwd_field1_dx, fwd_field2_dx, fwd_field1_dy, fwd_field2_dy, fwd_field1_dz, fwd_field2_dz, field1, field2, irho, model1, model2, model3,
            model4, rx, ry, rz, nx, ny, nz, ldimx, ldimy, stride_y, stride_z, g_stride_y, g_stride_z, hx, hy, hz);
        CUDA_CHECK_ERROR(__FILE__, __LINE__);
    }
}

constexpr decltype(&launch_rho<8>) LAUNCHERS_RHO[3] = { launch_rho<8>, launch_rho<12>, launch_rho<16> };

void
launch_update_rho_gradient_rho(float* grad, float const* fwd_field1_dx, float const* fwd_field2_dx, float const* fwd_field1_dy,
                               float const* fwd_field2_dy, float const* fwd_field1_dz, float const* fwd_field2_dz,
                               float const* field1, float const* field2, float const* irho, float const* model1,
                               float const* model2, float const* model3, float const* model4, float const* rx, float const* ry,
                               float const* rz, int ixbeg, int ixend, int iybeg, int iyend, int izbeg, int izend,
                               int ldimx, int ldimy, int grad_ldimx, int grad_ldimy, double dx, double dy, double dz,
                               int order, bool simple_kernel, cudaStream_t stream)
{
    EMSL_VERIFY(order == 8 || order == 12 || order == 16);
    if (ixend < ixbeg || iyend < iybeg || izend < izbeg)
        return;

    //Number of points
    int nx = ixend - ixbeg + 1;
    int ny = iyend - iybeg + 1;
    int nz = izend - izbeg + 1;

    // Input Y and Z strides
    int stride_y = ldimx;
    int stride_z = ldimx * ldimy;

    // Gradient (output) Y and Z strides
    int g_stride_y = grad_ldimx;
    int g_stride_z = grad_ldimx * grad_ldimy;

    //Shift for cart vol data we are reading/
    int shift = izbeg * stride_z + iybeg * stride_y + ixbeg;
    // grad is not shifted.
    fwd_field1_dx += shift;
    fwd_field2_dx += shift;
    fwd_field1_dy += shift;
    fwd_field2_dy += shift;
    fwd_field1_dz += shift;
    fwd_field2_dz += shift;
    field1 += shift;
    field2 += shift;
    irho += shift;
    model1 += shift;
    model2 += shift;
    model3 += shift;
    model4 += shift;
    rx += shift;
    ry += shift;
    rz += shift;

    unsigned launcher_id = 0;
    if (order == 12)
        launcher_id |= 0b01;
    if (order == 16)
        launcher_id |= 0b10;

    float hx, hy, hz, unused;
    kernel_utils::compute_fd_const(dx, dy, dz, hx, hy, hz, unused, unused, unused, unused, unused, unused);

    LAUNCHERS_RHO[launcher_id](grad, fwd_field1_dx, fwd_field2_dx, fwd_field1_dy, fwd_field2_dy, fwd_field1_dz, fwd_field2_dz, field1, field2,
                               irho, model1, model2, model3, model4, rx, ry, rz, nx, ny, nz, ldimx, ldimy, stride_y, stride_z,
                               g_stride_y, g_stride_z, hx, hy, hz, simple_kernel, stream);
}