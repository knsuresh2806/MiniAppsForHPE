#include <cuda_pipeline.h>
#include "cuda_utils.h"
#include "helper_kernels_gpu.h"
#include "kernel_type2_fwd_drv.h"

// Internal stucts and functions
namespace {

/*

Strategy:
Load (y_threads+order) * (x_threads+order) elements from the zplanes of field1, field2 starting from zhalos

Gather partial contributions of z derivatives at each plane
Finalize z derivative when all the order number of contributions are computed

Fianlize xy derivatives starting from izbeg
Load (y_threads) * (x_threads) elements from the zplanes of Model11 starting from izbeg

Multiply the derivatives with Model11 and write the results

*/

template <int order, int x_threads, int y_threads>
struct shmem_volumes
{
    // + order points to include field1, field2 halos
    float field1[y_threads + order][x_threads + order];
    float field2[y_threads + order][x_threads + order];
    // Model11 is not required to be put in shared memory as no values are shared among threads
    // but async loaded to increase bandwidth usage
    float Model11[y_threads][x_threads];
};

// contributions of the shared memory at iz+ir, with ir between [-order/2 , order/2 - 1]
template <FD_COEF_TYPE coeff_type, int order, int x_threads, int y_threads>
__device__ inline void
compute_dz_negative_shift(const shmem_volumes<order, x_threads, y_threads>* sm, const int ir, const float invdz,
                          float& dfield1_dz_rq, float& dfield2_dz_rq, const float prev_dfield1_dz_rq,
                          const float prev_dfield2_dz_rq)
{
    const int radius = order / 2;
    // field1, field2 have halos so shift by radius points
    const int tx = threadIdx.x + radius;
    const int ty = threadIdx.y + radius;
    float coeff = 0.0f;
    if (ir >= 0) {
        coeff = helper_kernels_gpu::df_coef<order, coeff_type>(ir + 1);
        dfield1_dz_rq = prev_dfield1_dz_rq + coeff * sm->field1[ty][tx];
        dfield2_dz_rq = prev_dfield2_dz_rq + coeff * sm->field2[ty][tx];
    } else {
        coeff = helper_kernels_gpu::df_coef<order, coeff_type>(abs(ir));
        dfield1_dz_rq = prev_dfield1_dz_rq - coeff * sm->field1[ty][tx];
        dfield2_dz_rq = prev_dfield2_dz_rq - coeff * sm->field2[ty][tx];
    }

    // apply the grid coefficients to the final contribution
    if (ir == radius - 1) {
        dfield1_dz_rq *= invdz;
        dfield2_dz_rq *= invdz;
    }
}

template <FD_COEF_TYPE coeff_type, int order, int x_threads, int y_threads>
__device__ inline void
compute_dx_dy_negative_shift(const shmem_volumes<order, x_threads, y_threads>* sm, const float invdx, const float invdy,
                             float& dfield1_dx_rq, float& dfield1_dy_rq, float& dfield2_dx_rq, float& dfield2_dy_rq)
{
    const int radius = order / 2;
    // field1, field2 have halos so shift by radius points
    const int tx = threadIdx.x + radius;
    const int ty = threadIdx.y + radius;
    // since shift is -0.5 and the index is offset by -1
    const int displaced_tx = tx - 1;
    const int displaced_ty = ty - 1;

    dfield1_dx_rq = 0.0f;
    dfield1_dy_rq = 0.0f;
    dfield2_dx_rq = 0.0f;
    dfield2_dy_rq = 0.0f;

#pragma unroll
    for (int i = 1; i <= radius; ++i) {
        float coeff = helper_kernels_gpu::df_coef<order, coeff_type>(i);
        dfield1_dx_rq += coeff * (sm->field1[ty][displaced_tx + i] - sm->field1[ty][displaced_tx - (i - 1)]);
        dfield1_dy_rq += coeff * (sm->field1[displaced_ty + i][tx] - sm->field1[displaced_ty - (i - 1)][tx]);

        dfield2_dx_rq += coeff * (sm->field2[ty][displaced_tx + i] - sm->field2[ty][displaced_tx - (i - 1)]);
        dfield2_dy_rq += coeff * (sm->field2[displaced_ty + i][tx] - sm->field2[displaced_ty - (i - 1)][tx]);
    }

    dfield1_dx_rq *= invdx;
    dfield1_dy_rq *= invdy;
    dfield2_dx_rq *= invdx;
    dfield2_dy_rq *= invdy;
}

template <int order, int x_threads, int y_threads>
__device__ inline void
multiply_inv_model10(const shmem_volumes<order, x_threads, y_threads>* sm, float& dfield1_dx_final_result,
                 float& dfield1_dy_final_result, float& dfield1_dz_final_result, float& dfield2_dx_final_result,
                 float& dfield2_dy_final_result, float& dfield2_dz_final_result, const float dfield1_dx_rq,
                 const float dfield1_dy_rq, const float dfield1_dz_rq, const float dfield2_dx_rq, const float dfield2_dy_rq,
                 const float dfield2_dz_rq)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    // Model11 has NO halos so NO shift by radius points
    const float rho = sm->Model11[ty][tx];
    dfield1_dx_final_result = dfield1_dx_rq * rho;
    dfield1_dy_final_result = dfield1_dy_rq * rho;
    dfield1_dz_final_result = dfield1_dz_rq * rho;
    dfield2_dx_final_result = dfield2_dx_rq * rho;
    dfield2_dy_final_result = dfield2_dy_rq * rho;
    dfield2_dz_final_result = dfield2_dz_rq * rho;
}

template <bool fields12_only = false, int order, int x_threads, int y_threads>
__device__ inline void
async_load(shmem_volumes<order, x_threads, y_threads>* sm, const int index_load_nohalos, const int index_load_halos,
           const bool valid_load_nohalos, const bool valid_load_halos, const int txload, const int tyload,
           const float* field1, const float* field2, const float* Model11)
{
    if (valid_load_halos) {
        __pipeline_memcpy_async(&sm->field1[tyload][txload], field1 + index_load_halos, sizeof(float4));
        __pipeline_memcpy_async(&sm->field2[tyload][txload], field2 + index_load_halos, sizeof(float4));
    }
    if (!fields12_only && valid_load_nohalos) {
        __pipeline_memcpy_async(&sm->Model11[tyload][txload], Model11 + index_load_nohalos, sizeof(float4));
    }
    __pipeline_commit();
}

template <int order, FD_COEF_TYPE coeff_type, int x_threads, int y_threads, int z_threads>
__global__
__launch_bounds__(x_threads* y_threads) void fwd_main_loop_kernel(
    const float* __restrict__ field1, const float* __restrict__ field2, const float* __restrict__ Model11,
    float* __restrict__ dfield1dx, float* __restrict__ dfield1dy, float* __restrict__ dfield1dz, float* __restrict__ dfield2dx,
    float* __restrict__ dfield2dy, float* __restrict__ dfield2dz, const float invdx, const float invdy, const float invdz,
    const int ldimx, const int ldimy, const int ixbeg, const int nx, const int ny, const int nz)
{
    static_assert(order == 8 || order == 16, "Only 8th and 16th orders are supported.");
    static_assert(x_threads >= order && y_threads >= order, "Incompatible block size for this order.");

    constexpr int radius = order / 2;
    const int ix_block_global = blockIdx.x * x_threads;
    const int iy_block_global = blockIdx.y * y_threads;
    const int iz_block_global = blockIdx.z * z_threads;
    const int ix_thread_global = ix_block_global + threadIdx.x;
    const int iy_thread_global = iy_block_global + threadIdx.y;

    // inside write region if not in halos sections of the axis
    const bool active_region = ix_thread_global >= ixbeg && ix_thread_global < ixbeg + nx && iy_thread_global < ny;
    // we are moving by xy plane offset
    const int z_stride = ldimx * ldimy;
    const int y_stride = ldimx;
    int global_index = (iz_block_global * z_stride) + (iy_thread_global * y_stride) + ix_thread_global;

    // shared memory structure to hold field1, field2, Model11
    __shared__ shmem_volumes<order, x_threads, y_threads> shmem_vols[2];
    // shmem_tid is the 1D indexing method into the 2D arrays of field1, field2, Model11 inside shmem_vols structure
    const int shmem_tid = threadIdx.y * x_threads + threadIdx.x;
    // reshape the x_threads and y_threads to support loading 4 float values by each thread
    // assuming x_threads = 32
    // ***************************************************
    /*
        | shmem_tid   | shmem_xload     | shmem_yload
        ----------------------------------------------------
        | 0           | 0               | 0
        | 1           | 4               | 0
        | 2           | 8               | 0
        | .           | .               | .
        | .           | .               | .
        | .           | .               | .
        | 7           | 28              | 0
        | 8           | 32              | 0 (skipped by valid_load* check)
        | .           | .               | . (skipped by valid_load* check)
        | .           | .               | . (skipped by valid_load* check)
        | .           | .               | . (skipped by valid_load* check)
        | 15          | 60              | 0
        | 16          | 0               | 1
        | 17          | 4               | 1
        | .           | .               | .
        | .           | .               | .
    */
    // ****************************************************
    const int shmem_xload = 4 * (shmem_tid % (x_threads / 2));
    const int shmem_yload = shmem_tid / (x_threads / 2);

    // where to start reading values from the volume with respect to float4 mapping
    // skipping halos
    const int ixload = ix_block_global + shmem_xload;
    const int iyload = iy_block_global + shmem_yload;
    const bool valid_load_nohalos =
        (shmem_xload < x_threads && shmem_yload < y_threads) && (ixload < ldimx && iyload < ny);
    int index_load_nohalos = (iz_block_global * z_stride) + (iyload * y_stride) + ixload;

    // where to start reading values from the volume with respect to float4 mapping
    // including halos
    const int ixload_halos = ixload - radius;
    const int iyload_halos = iyload - radius;
    const bool valid_load_halos = ((x_threads == order || shmem_xload < x_threads + order) &&
                                   (y_threads == order || shmem_yload < y_threads + order)) &&
                                  (ixload_halos >= 0 && ixload_halos < ldimx && iyload_halos < ny + radius);
    int index_load_halos = ((iz_block_global - radius) * z_stride) + (iyload_halos * y_stride) + ixload_halos;

    // rq -- register queue
    float dfield1_dx_rq[radius - 1] = { 0.0f };
    float dfield1_dy_rq[radius - 1] = { 0.0f };
    float dfield1_dz_rq[order - 1] = { 0.0f };
    float dfield2_dx_rq[radius - 1] = { 0.0f };
    float dfield2_dy_rq[radius - 1] = { 0.0f };
    float dfield2_dz_rq[order - 1] = { 0.0f };

    float dfield1_dx_final_result = 0.0f;
    float dfield1_dy_final_result = 0.0f;
    float dfield1_dz_final_result = 0.0f;
    float dfield2_dx_final_result = 0.0f;
    float dfield2_dy_final_result = 0.0f;
    float dfield2_dz_final_result = 0.0f;

    // ism -- index into the shmem_volumes array
    int ism = 0;
    // asynchronously load the field1, field2 arrays
    async_load<true>(shmem_vols + ism, index_load_nohalos, index_load_halos, valid_load_nohalos, valid_load_halos,
                     shmem_xload, shmem_yload, field1, field2, Model11);
    index_load_halos += z_stride;
    // flip to async load the next buffer
    ism ^= 1;

#pragma unroll
    for (int i = 0; i < order - 1; ++i) {
        // protect the shared memory between iterations
        __syncthreads();
        // prefetch the arrays for next iteration
        if (i < order - 2) {
            async_load<true>(shmem_vols + ism, index_load_nohalos, index_load_halos, valid_load_nohalos,
                             valid_load_halos, shmem_xload, shmem_yload, field1, field2, Model11);
        } else {
            // prefetch the arrays including Model11 during the last iteration
            async_load<false>(shmem_vols + ism, index_load_nohalos, index_load_halos, valid_load_nohalos,
                              valid_load_halos, shmem_xload, shmem_yload, field1, field2, Model11);
            index_load_nohalos += z_stride;
        }
        index_load_halos += z_stride;

        // wait for the previously issued async load to complete
        __pipeline_wait_prior(1);
        __syncthreads();
        // flip again to read volumes from the previous load
        ism ^= 1;

        // update the partial contributions
#pragma unroll
        for (int j = 1; j <= i; ++j) {
            compute_dz_negative_shift<coeff_type>(shmem_vols + ism, j - radius, invdz, dfield1_dz_rq[i - j],
                                                  dfield2_dz_rq[i - j], dfield1_dz_rq[i - j], dfield2_dz_rq[i - j]);
        }
        compute_dz_negative_shift<coeff_type>(shmem_vols + ism, -radius, invdz, dfield1_dz_rq[i], dfield2_dz_rq[i], 0.0f,
                                              0.0f);

        if (i >= radius) {
            compute_dx_dy_negative_shift<coeff_type>(shmem_vols + ism, invdx, invdy, dfield1_dx_rq[i - radius],
                                                     dfield1_dy_rq[i - radius], dfield2_dx_rq[i - radius],
                                                     dfield2_dy_rq[i - radius]);
        }
    }

    int nzloop = min(z_threads, nz - iz_block_global);
    // Loop on the Z dimension, except the last one
    for (int izloop = 0; izloop < nzloop - 1; ++izloop) {
        // Protect the shared memory between iterations
        __syncthreads();
        // Prefetch the next arrays in shared memory
        async_load<false>(shmem_vols + ism, index_load_nohalos, index_load_halos, valid_load_nohalos, valid_load_halos,
                          shmem_xload, shmem_yload, field1, field2, Model11);
        index_load_nohalos += z_stride;
        index_load_halos += z_stride;
        // Wait until the previous shared memory loads are complete
        __pipeline_wait_prior(1);
        __syncthreads();
        ism ^= 1;

        if (active_region) {
            // Finalize the oldest derivatives with Z
            // the interval is [-radius, radius-1]
            compute_dz_negative_shift<coeff_type>(shmem_vols + ism, radius - 1, invdz, dfield1_dz_rq[0], dfield2_dz_rq[0],
                                                  dfield1_dz_rq[0], dfield2_dz_rq[0]);
            multiply_inv_model10(shmem_vols + ism, dfield1_dx_final_result, dfield1_dy_final_result, dfield1_dz_final_result,
                             dfield2_dx_final_result, dfield2_dy_final_result, dfield2_dz_final_result, dfield1_dx_rq[0],
                             dfield1_dy_rq[0], dfield1_dz_rq[0], dfield2_dx_rq[0], dfield2_dy_rq[0], dfield2_dz_rq[0]);
            // write results
            dfield1dx[global_index] = dfield1_dx_final_result;
            dfield1dy[global_index] = dfield1_dy_final_result;
            dfield1dz[global_index] = dfield1_dz_final_result;
            dfield2dx[global_index] = dfield2_dx_final_result;
            dfield2dy[global_index] = dfield2_dy_final_result;
            dfield2dz[global_index] = dfield2_dz_final_result;

            // Update the final contributions, rotate them (read from i write to i-1)
#pragma unroll
            for (int i = 1; i < order - 1; ++i) {
                compute_dz_negative_shift<coeff_type>(shmem_vols + ism, radius - 1 - i, invdz, dfield1_dz_rq[i - 1],
                                                      dfield2_dz_rq[i - 1], dfield1_dz_rq[i], dfield2_dz_rq[i]);
            }
            compute_dz_negative_shift<coeff_type>(shmem_vols + ism, -radius, invdz, dfield1_dz_rq[order - 2],
                                                  dfield2_dz_rq[order - 2], 0.0f, 0.0f);

#pragma unroll
            // Rotate the register queues without Z
            for (int i = 1; i < radius - 1; ++i) {
                dfield1_dx_rq[i - 1] = dfield1_dx_rq[i];
                dfield1_dy_rq[i - 1] = dfield1_dy_rq[i];
                dfield2_dx_rq[i - 1] = dfield2_dx_rq[i];
                dfield2_dy_rq[i - 1] = dfield2_dy_rq[i];
            }

            // Compute new derivatives in X and Y at the end of the register queue
            compute_dx_dy_negative_shift<coeff_type>(shmem_vols + ism, invdx, invdy, dfield1_dx_rq[radius - 2],
                                                     dfield1_dy_rq[radius - 2], dfield2_dx_rq[radius - 2],
                                                     dfield2_dy_rq[radius - 2]);
        }
        global_index += z_stride;
    }

    // Last iteration (no prefetch)
    // Wait until the last shared memory loads are complete
    __pipeline_wait_prior(0);
    __syncthreads();
    ism ^= 1;

    if (active_region) {
        // Finalize the oldest derivatives with Z
        // the interval is [-radius, radius-1]
        compute_dz_negative_shift<coeff_type>(shmem_vols + ism, radius - 1, invdz, dfield1_dz_rq[0], dfield2_dz_rq[0],
                                              dfield1_dz_rq[0], dfield2_dz_rq[0]);
        multiply_inv_model10(shmem_vols + ism, dfield1_dx_final_result, dfield1_dy_final_result, dfield1_dz_final_result,
                         dfield2_dx_final_result, dfield2_dy_final_result, dfield2_dz_final_result, dfield1_dx_rq[0], dfield1_dy_rq[0],
                         dfield1_dz_rq[0], dfield2_dx_rq[0], dfield2_dy_rq[0], dfield2_dz_rq[0]);
        // write results
        dfield1dx[global_index] = dfield1_dx_final_result;
        dfield1dy[global_index] = dfield1_dy_final_result;
        dfield1dz[global_index] = dfield1_dz_final_result;
        dfield2dx[global_index] = dfield2_dx_final_result;
        dfield2dy[global_index] = dfield2_dy_final_result;
        dfield2dz[global_index] = dfield2_dz_final_result;
    }
}

} // end anonymous namespace

namespace kernel3_gpu_kernels {

void
launch_fwd_kernel2_drv(const float* field1, const float* field2, const float* Model11, float* dfield1dx, float* dfield1dy,
                      float* dfield1dz, float* dfield2dx, float* dfield2dy, float* dfield2dz, const int ixbeg, const int ixend,
                      const int iybeg, const int iyend, const int izbeg, const int izend, const float invdx,
                      const float invdy, const float invdz, const int ldimx, const int ldimy, const int order,
                      cudaStream_t stream)
{
    constexpr FD_COEF_TYPE coeff_type = _FD_COEF_LEAST_SQUARE;
    const int nx = ixend - ixbeg + 1;
    const int ny = iyend - iybeg + 1;
    const int nz = izend - izbeg + 1;
    const int radius = order / 2;
    off_t offset = (izbeg * ldimx * ldimy) + (iybeg * ldimx);
    constexpr int z_threads = 64;

    if (order == 8) {
        constexpr int x_threads = 32;
        constexpr int y_threads = 32;

        dim3 threads(x_threads, y_threads);
        dim3 blocks((nx + radius - 1) / x_threads + 1, (ny - 1) / y_threads + 1, (nz - 1) / z_threads + 1);

        fwd_main_loop_kernel<8, coeff_type, x_threads, y_threads, z_threads>
            <<<blocks, threads, 0, stream>>>(field1 + offset, field2 + offset, Model11 + offset, dfield1dx + offset,
                                             dfield1dy + offset, dfield1dz + offset, dfield2dx + offset, dfield2dy + offset,
                                             dfield2dz + offset, invdx, invdy, invdz, ldimx, ldimy, ixbeg, nx, ny, nz);
    }
    if (order == 16) {
        constexpr int x_threads = 32;
        constexpr int y_threads = 16;

        dim3 threads(x_threads, y_threads);
        dim3 blocks((nx + radius - 1) / x_threads + 1, (ny - 1) / y_threads + 1, (nz - 1) / z_threads + 1);

        fwd_main_loop_kernel<16, coeff_type, x_threads, y_threads, z_threads>
            <<<blocks, threads, 0, stream>>>(field1 + offset, field2 + offset, Model11 + offset, dfield1dx + offset,
                                             dfield1dy + offset, dfield1dz + offset, dfield2dx + offset, dfield2dy + offset,
                                             dfield2dz + offset, invdx, invdy, invdz, ldimx, ldimy, ixbeg, nx, ny, nz);
    }

    CUDA_TRY(cudaPeekAtLastError());
}

} // namespace kernel3_gpu_kernels
